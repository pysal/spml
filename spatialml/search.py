from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn import metrics


class BandwidthSearch:
    """Optimal bandwidth search for geographically weighted estimators.

    Reports scores from multiple models with varying bandwidth and identifies
    the optimal one.  When using golden section search, it minimizes (or
    maximizes) the chosen ``criterion``.

    The search supports two broad families of models:

    * **Linear / logistic models** (:class:`~spatialml.linear_model.GWLinearRegression`,
      :class:`~spatialml.linear_model.GWLogisticRegression`): information criteria
      ``"aicc"``, ``"aic"``, ``"bic"`` are valid and recommended.  They are included
      in ``metrics_`` automatically.
    * **Non-linear models** (random forest, gradient boosting, …): information
      criteria are *not* valid (no closed-form log-likelihood or hat matrix).
      Use ``"rmse"`` / ``"mae"`` for regression or ``"log_loss"`` combined with
      ``"prediction_rate"`` for classification instead.

    When using classification models with a defined ``min_proportion``, keep in
    mind that some locations may be excluded from the final model.  In such a
    case, even the valid information criteria are not comparable across
    bandwidths and ``"log_loss"`` should be preferred.

    Parameters
    ----------
    model : type
        A geographically weighted estimator class (e.g.
        :class:`spatialml.linear_model.GWLogisticRegression`) that can be
        instantiated as
        ``model(bandwidth=..., fixed=..., kernel=..., n_jobs=..., ...)``.
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest
        neighbor) bandwidth, by default ``False``
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default
        ``"bisquare"``
    coplanar : {"raise", "jitter", "clique"}, optional
        Method for handling coplanar points with adaptive kernels. Options are
        ``"raise"`` (raising an exception when coplanar points are present),
        ``"jitter"`` (randomly displace coplanar points to produce uniqueness),
        and ``"clique"`` (induce fully-connected sub-cliques for coplanar
        points). By default ``"raise"``
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all
        processors, by default ``-1``
    search_method : {"golden_section", "interval"}, optional
        Method used to search for optimal bandwidth. When using
        ``"golden_section"``, the Golden section optimization is used to find
        the optimal bandwidth while attempting to minimize or maximise
        ``criterion``. When using ``"interval"``, fits all models within the
        specified bandwidths at a set interval without any attempt to optimize
        the selection. By default ``"golden_section"``.
    criterion : str, optional
        Criterion used to select the optimal bandwidth.

        Built-in special values:

        * ``"aicc"``, ``"aic"``, ``"bic"`` — information criteria;
          **only valid for linear / logistic models**.
        * ``"log_loss"`` — cross-entropy loss; for classifiers only.
        * ``"prediction_rate"`` — proportion of fitted locations; classifiers.
        * ``"rmse"`` — root mean squared error of focal residuals; regressors.
        * ``"mae"`` — mean absolute error of focal residuals; regressors.

        Any other string ``m`` is interpreted as an attribute name and
        retrieved from the fitted model as ``getattr(model, m + "_")``.
        By default ``"aicc"`` for linear models and ``"rmse"`` for non-linear.
    metrics : list[str] | None, optional
        Additional metrics to report for each bandwidth.  Follow the same
        conventions as ``criterion``.  By default ``None``.
    minimize : bool, optional
        Minimize or maximize the ``criterion``.  For information criteria and
        error metrics the optimum is the lowest value; for ``"prediction_rate"``
        or accuracy-like metrics it is the highest.  By default ``True``.
    min_bandwidth : int | float | None, optional
        Minimum bandwidth to consider, by default ``None``
    max_bandwidth : int | float | None, optional
        Maximum bandwidth to consider, by default ``None``
    interval : int | float | None, optional
        Interval for bandwidth search when using ``"interval"`` method, by
        default ``None``
    max_iterations : int, optional
        Maximum number of iterations for golden section search, by default
        ``100``
    tolerance : float, optional
        Tolerance for convergence in golden section search, by default ``1e-2``
    verbose : bool | int, optional
        Verbosity level, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialization

    Attributes
    ----------
    scores_ : pd.Series
        Series of criterion scores for each bandwidth tested (index is
        bandwidth).
    metrics_ : pd.DataFrame
        DataFrame of additional metrics for each bandwidth tested.  For
        linear/logistic models, columns ``"aicc"``, ``"aic"``, ``"bic"`` are
        always present; they are omitted for non-linear models.
    optimal_bandwidth_ : int | float
        The optimal bandwidth found by the search method.

    Examples
    --------
    Interval search over a small set of candidate bandwidths:

    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spatialml.linear_model import GWLogisticRegression
    >>> from spatialml.search import BandwidthSearch

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> search = BandwidthSearch(
    ...     GWLogisticRegression,
    ...     fixed=False,
    ...     search_method="interval",
    ...     criterion="aicc",
    ...     min_bandwidth=20,
    ...     max_bandwidth=80,
    ...     interval=10,
    ...     max_iter=200,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> search.optimal_bandwidth_
    40
    """

    def __init__(
        self,
        model,
        *,
        fixed: bool = False,
        kernel: Literal[
            "triangular",
            "parabolic",
            # "gaussian",
            "bisquare",
            "tricube",
            "cosine",
            "boxcar",
            # "exponential",
        ]
        | Callable = "bisquare",
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        n_jobs: int = -1,
        search_method: Literal["golden_section", "interval"] = "golden_section",
        criterion: str | None = None,
        metrics: list | None = None,
        minimize: bool = True,
        min_bandwidth: int | float | None = None,
        max_bandwidth: int | float | None = None,
        interval: int | float | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-2,
        verbose: bool | int = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.kernel = kernel
        self.fixed = fixed
        self.coplanar = coplanar
        self._model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.search_method = search_method
        self.criterion = criterion
        self.minimize = minimize
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.interval = interval
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.metrics = metrics
        self.verbose = verbose
        # Probe model type once at construction to know whether IC is valid.
        self._supports_ic = model()._supports_ic

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "BandwidthSearch":
        """
        Fit the searcher by evaluating candidate bandwidths on the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used to evaluate candidate bandwidths (rows are samples).
        y : pd.Series
            Target values corresponding to X.
        geometry : gpd.GeoSeries
            Geographic location of the observations in the sample. Used to determine the
            spatial interaction weight based on specification by ``bandwidth``,
            ``fixed``, ``kernel``, and ``include_focal`` keywords.

        Returns
        -------
        self
            The fitted instance.

        Notes
        -----
        The optimal bandwidth is selected as the index of the minimum score if
        ``minimize=True``, otherwise as the index of the maximum score.
        """
        self.geometry = geometry

        if self.criterion is None:
            if self._supports_ic:
                self.criterion = "aicc"
            else:
                self.criterion = "rmse"
                if self.metrics is not None and self.criterion not in self.metrics:
                    self.metrics.append(self.criterion)
                else:
                    self.metrics = [self.criterion]

        _ic_criteria = {"aicc", "aic", "bic"}
        if self.criterion in _ic_criteria and not self._supports_ic:
            raise ValueError(
                f"criterion='{self.criterion}' requires information criteria "
                f"(AIC/AICc/BIC) which are not valid for "
                f"'{self.model.__name__}'. "
                f"For regression models use criterion='rmse' or criterion='mae'; "
                f"for classification models use criterion='log_loss' or "
                f"criterion='prediction_rate'."
            )

        if self.search_method == "interval":
            self._interval(X=X, y=y)
        elif self.search_method == "golden_section":
            self._golden_section(X=X, y=y, tolerance=self.tolerance)

        self.optimal_bandwidth_ = (
            self.scores_.idxmin() if self.minimize else self.scores_.idxmax()
        )

        return self

    @property
    def _ic_metrics(self) -> list[str]:
        """IC metric names included automatically when the model supports them."""
        return ["aicc", "aic", "bic"] if self._supports_ic else []

    def _score(
        self, X: pd.DataFrame, y: pd.Series, bw: int | float
    ) -> tuple[float, list[float]]:
        """Fit the model and report criterion score.

        In case of invariant y in a local model, returns np.inf
        """
        met = self._ic_metrics.copy()
        if self.metrics is not None:
            met += self.metrics

        if len(np.unique(y)) == 1:
            return (np.inf, [np.nan] * len(met))

        gwm = self.model(
            bandwidth=bw,
            fixed=self.fixed,
            kernel=self.kernel,
            coplanar=self.coplanar,
            n_jobs=self.n_jobs,
            fit_global_model=False,
            strict=False,
            verbose=self.verbose == 2,
            **self._model_kwargs,
        ).fit(X=X, y=y, geometry=self.geometry)

        if hasattr(gwm, "prediction_rate_") and gwm.prediction_rate_ == 0:
            # prediction rate should report 0, everything else is undefined
            if self.criterion == "prediction_rate":
                score = gwm.prediction_rate_
            else:
                score = np.inf if self.minimize else -np.inf

            all_metrics = []
            for m in met:
                if m == "prediction_rate":
                    all_metrics.append(gwm.prediction_rate_)
                elif m in ("log_loss", "rmse", "mae"):
                    all_metrics.append(np.inf)
                else:
                    all_metrics.append(np.nan)

            return score, all_metrics

        all_metrics = []
        for m in met:
            if m == "log_loss":
                if not hasattr(gwm, "proba_"):
                    raise ValueError(
                        "criterion='log_loss' requires a classifier with 'proba_' "
                        f"but '{type(gwm).__name__}' is a regressor."
                    )
                mask = gwm.proba_.isna().any(axis=1)
                y_masked = y[~mask]
                if len(np.unique(y_masked)) < 2:
                    all_metrics.append(np.inf)
                else:
                    all_metrics.append(
                        metrics.log_loss(
                            y_masked, gwm.proba_[~mask], labels=np.unique(y)
                        )
                    )
            elif m == "rmse":
                if not hasattr(gwm, "resid_"):
                    raise ValueError(
                        "criterion='rmse' requires a regressor with 'resid_' "
                        f"but '{type(gwm).__name__}' is a classifier."
                    )
                all_metrics.append(np.sqrt(np.nanmean(gwm.resid_**2)))
            elif m == "mae":
                if not hasattr(gwm, "resid_"):
                    raise ValueError(
                        "criterion='mae' requires a regressor with 'resid_' "
                        f"but '{type(gwm).__name__}' is a classifier."
                    )
                all_metrics.append(np.nanmean(np.abs(gwm.resid_)))
            else:
                all_metrics.append(getattr(gwm, m + "_"))

        assert self.criterion is not None
        return all_metrics[met.index(self.criterion)], all_metrics

    def _interval(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit models using the equal interval search.

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        """
        if not (
            isinstance(self.min_bandwidth, float | int)
            and isinstance(self.max_bandwidth, float | int)
            and isinstance(self.interval, float | int)
        ):
            raise ValueError(
                "All 'min_bandwidth', 'max_bandwidth' and 'interval' need "
                "to be set when using interval search method."
            )

        scores = {}
        metrics = {}
        bw = self.min_bandwidth
        while bw <= self.max_bandwidth:
            score, metric = self._score(X=X, y=y, bw=bw)
            scores[bw] = score
            metrics[bw] = metric
            if self.verbose:
                print(f"Bandwidth: {bw:.2f}, {self.criterion}: {scores[bw]:.3f}")
            bw += self.interval
        self.scores_ = pd.Series(scores, name=self.criterion)
        self.metrics_ = pd.DataFrame(
            metrics,
            index=pd.Index(
                self._ic_metrics + self.metrics if self.metrics else self._ic_metrics
            ),
        ).T

    def _golden_section(self, X: pd.DataFrame, y: pd.Series, tolerance: float) -> None:
        delta = 0.38197
        if self.fixed:
            pairwise_distance = pdist(self.geometry.get_coordinates())
            min_dist = np.min(pairwise_distance)
            max_dist = np.max(pairwise_distance)

            a = min_dist / 2.0
            c = max_dist * 2.0
        else:
            a = 40 + 2 * X.shape[1]
            c = len(self.geometry)

        if self.min_bandwidth:
            a = self.min_bandwidth
        if self.max_bandwidth:
            c = self.max_bandwidth

        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)

        diff = 1.0e9
        iters = 0
        scores = {}
        metrics = {}
        while diff > tolerance and iters < self.max_iterations and a != np.inf:
            if not self.fixed:  # ensure we use int as possible bandwidth
                b = int(b)
                d = int(d)

            if b in scores:
                score_b = scores[b]
            else:
                score_b, metric_b = self._score(X=X, y=y, bw=b)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{b:.2f}'.rstrip('0').rstrip('.')}, "
                        f"score: {score_b:.3f}"
                    )
                scores[b] = score_b
                metrics[b] = metric_b

            if d in scores:
                score_d = scores[d]
            else:
                score_d, metric_d = self._score(X=X, y=y, bw=d)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{d:.2f}'.rstrip('0').rstrip('.')}, "
                        f"score: {score_d:.3f}"
                    )
                scores[d] = score_d
                metrics[d] = metric_d

            if self.minimize:
                if score_b <= score_d:
                    c = d
                    d = b
                    b = a + delta * np.abs(c - a)

                else:
                    a = b
                    b = d
                    d = c - delta * np.abs(c - a)

                diff = np.abs(score_b - score_d)

            else:
                if score_b >= score_d:
                    c = d
                    d = b
                    b = a + delta * np.abs(c - a)
                else:
                    a = b
                    b = d
                    d = c - delta * np.abs(c - a)

                diff = np.abs(score_b - score_d)

            iters += 1

        self.scores_ = pd.Series(scores)
        self.metrics_ = pd.DataFrame(
            metrics,
            index=pd.Index(
                self._ic_metrics + self.metrics if self.metrics else self._ic_metrics
            ),
        ).T
