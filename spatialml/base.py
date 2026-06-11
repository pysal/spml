import inspect
import warnings
from collections.abc import Callable, Hashable
from numbers import Integral, Real
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import dump, load
from libpysal import graph
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.parallel import Parallel, delayed

__all__ = ["BaseClassifier", "BaseRegressor"]


def _triangular(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u**2


def _gaussian(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-((u / 2) ** 2))


def _bisquare(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (1 - u**2) ** 2


def _cosine(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return np.cos(np.pi / 2 * u)


def _exponential(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    r = (distances < bandwidth).astype(int)
    return r


def _tricube(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (1 - u**3) ** 3


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    # "gaussian": _gaussian,
    "bisquare": _bisquare,
    "tricube": _tricube,
    "cosine": _cosine,
    "boxcar": _boxcar,
    # "exponential": _exponential,
}


class _BaseModel(BaseEstimator):
    """Base class for geographically weighted models"""

    def __init__(
        self,
        model,
        *,
        bandwidth: float | None = None,
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
        include_focal: bool = False,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.include_focal = include_focal
        self.graph = graph
        self.fixed = fixed
        self._model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.fit_global_model = fit_global_model
        self.strict = strict
        if isinstance(keep_models, str):
            keep_models = Path(keep_models)
        self.keep_models = keep_models
        self.temp_folder = temp_folder
        self.batch_size = batch_size
        self.coplanar = coplanar
        self.verbose = verbose
        self._model_type = None

    def _validate_geometry(self, geometry):
        """Validate that geometry contains only Point geometries"""
        if not isinstance(geometry, gpd.GeoSeries):
            raise ValueError(
                f"geometry needs to be geopandas.GeoSeries. Got {type(geometry)}."
            )
        if geometry is not None and not (geometry.geom_type == "Point").all():
            raise ValueError(
                "Unsupported geometry type. Only point geometry is allowed."
            )

    def _build_weights(self) -> graph.Graph:
        """Build spatial weights graph"""
        if not isinstance(self.bandwidth, Real):
            raise ValueError(
                "Bandwidth is not a valid value. Needs to be float or int, "
                f"got {self.bandwidth}."
            )

        kernel = (
            _kernel_functions[self.kernel]
            if isinstance(self.kernel, str)
            else self.kernel
        )

        if self.fixed:  # fixed distance
            weights = graph.Graph.build_kernel(
                self.geometry,
                kernel=kernel,
                bandwidth=self.bandwidth,
                coplanar=self.coplanar,
            )
        else:  # adaptive KNN
            weights = graph.Graph.build_kernel(
                self.geometry,
                kernel="identity",
                k=self.bandwidth - 1 if self.include_focal else self.bandwidth,
                coplanar=self.coplanar,
            )
            # post-process identity weights by the selected kernel
            # and kernel bandwidth derived from each neighborhood
            # the epsilon comes from MGWR to avoid division by zero
            bandwidth = weights._adjacency.groupby(level=0).transform("max") * 1.0000001
            weights = graph.Graph(
                adjacency=kernel(weights._adjacency, bandwidth),
                is_sorted=True,
            )
        if self.include_focal:
            weights = weights.assign_self_weight(1)
        return weights

    def _setup_model_storage(self):
        """Setup model storage directory if needed"""
        if isinstance(self.keep_models, Path):
            self.keep_models.mkdir(exist_ok=True)

    def _fit_models_batch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: graph.Graph,
    ) -> list:
        """Fit models in batches or all at once"""
        if self.batch_size:
            training_output = []
            num_groups = len(y)
            indices = X.index
            for i in range(0, num_groups, self.batch_size):
                if self.verbose:
                    print(
                        f"Processing batch {i // self.batch_size + 1} "
                        f"out of {(num_groups // self.batch_size) + 1}."
                    )

                batch_indices = indices[i : i + self.batch_size]
                subset_weights = weights._adjacency.loc[batch_indices, :]

                index = subset_weights.index
                _weight = subset_weights.values
                X_focals = X.loc[batch_indices].values

                batch_training_output = self._batch_fit(X, y, index, _weight, X_focals)
                training_output.extend(batch_training_output)
        else:
            index = weights._adjacency.index
            _weight = weights._adjacency.values
            X_focals = X.values

            training_output = self._batch_fit(X, y, index, _weight, X_focals)

        return training_output

    def _batch_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        index: pd.MultiIndex,
        _weight: np.ndarray,
        X_focals: np.ndarray,
    ) -> list:
        """Fit a batch of local models"""
        data = X.copy()
        data["_y"] = y
        data = data.loc[index.get_level_values(1)]
        data["_weight"] = _weight
        grouper = data.groupby(index.get_level_values(0), sort=False)

        invariant = grouper["_y"].nunique() == 1
        if invariant.any():
            if self.strict:
                raise ValueError(
                    f"y at locations {invariant.index[invariant]} is invariant."
                )
            elif self.strict is None:
                warnings.warn(
                    f"y at locations {invariant.index[invariant]} is invariant.",
                    stacklevel=3,
                )

        return Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(self._fit_local)(
                self.model,
                group,
                name,
                focal_x,
                self._model_kwargs,
            )
            for (name, group), focal_x in zip(grouper, X_focals, strict=False)
        )

    def _fit_global_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit global baseline model"""
        if self._model_type == "random_forest":
            self._model_kwargs["oob_score"] = True
        # fit global model as a baseline
        if "n_jobs" in inspect.signature(self.model).parameters:
            self.global_model = self.model(n_jobs=self.n_jobs, **self._model_kwargs)
        else:
            self.global_model = self.model(**self._model_kwargs)

        # see gh#44 - remove filter once oldest sklearn is 1.10
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'n_jobs' has no effect since 1.8 and will be removed in 1.10.",
                category=FutureWarning,
            )
            self.global_model.fit(X=X, y=y)

    def _store_model(self, local_model, name: Hashable):
        """Store or serialize local model"""
        if self.keep_models is True:  # if True, models are kept in memory
            return local_model
        elif isinstance(self.keep_models, Path):  # if Path, models are saved to disk
            p = f"{self.keep_models.joinpath(f'{name}.joblib')}"
            with open(p, "wb") as f:
                dump(local_model, f, protocol=5)
            del local_model
            return p
        else:
            del local_model
            return None

    @property
    def _supports_ic(self) -> bool:
        """True for model types for which information criteria are valid.

        Information criteria (AIC, AICc, BIC) and the hat-matrix-based
        ``effective_df_`` are theoretically grounded only for maximum-likelihood
        models with a linear structure: ``"linear"`` (Gaussian) and
        ``"logistic"`` (Bernoulli / GLM).  Tree-based ensembles
        (``"random_forest"``, ``"gradient_boosting"``) do not satisfy these
        preconditions, so those attributes are simply not computed or exposed.
        """
        return self._model_type in {"linear", "logistic"}

    def _compute_hat_value(
        self, X: pd.DataFrame, weights: np.ndarray, focal_x: np.ndarray
    ) -> float:
        """
        Compute the hat value (leverage) for the focal point.

        For classification problems, this is an approximation rather than an ideal
        solution but it should be good enough for bandwidth search.

        Parameters:
        -----------
        X : pd.DataFrame
            Design matrix of the local neighborhood
        weights : numpy.ndarray
            Spatial weights for the neighborhood
        focal_x : numpy.ndarray
            Feature vector of the focal point

        Returns:
        --------
        float : hat value for the focal point
        """
        try:
            # Add intercept if not present
            if not (X.iloc[:, 0] == 1).all():
                X_with_intercept = np.column_stack([np.ones(len(X)), X.values])
                focal_with_intercept = np.concatenate([[1], focal_x.flatten()])
            else:
                X_with_intercept = X.values
                focal_with_intercept = focal_x.flatten()

            # Compute (X^T W X)^(-1)
            XtWX = X_with_intercept.T @ np.diag(weights) @ X_with_intercept
            XtWX_inv = np.linalg.pinv(
                XtWX
            )  # Use pseudo-inverse for numerical stability

            # Hat value: h_ii = x_i^T (X^T W X)^(-1) x_i
            # w_ii is omitted: for include_focal=True with compact kernels,
            # kernel(0, bw) = 1 so the full formula gives the same result.
            hat_value = focal_with_intercept.T @ XtWX_inv @ focal_with_intercept

            return hat_value

        except (np.linalg.LinAlgError, ValueError):
            # Return NaN if computation fails (singular matrix, etc.)
            return np.nan

    def _compute_information_criteria(self):
        """Compute AIC, AICc, and BIC from the global log-likelihood.

        Only called when :attr:`_supports_ic` is ``True`` (i.e. linear or
        logistic models).  Must not be invoked for tree-based estimators.
        """
        n = (
            self._n_fitted_models
            if hasattr(self, "_n_fitted_models")
            else len(self.pred_)
        )

        # Use effective degrees of freedom as the number of parameters
        k = self.effective_df_

        if not np.isnan(self.log_likelihood_) and not np.isnan(k):
            # Akaike Information Criterion
            # k+1 counts effective_df_ plus one scale parameter (σ² for linear
            # regression). For logistic/non-linear models this adds a small constant
            # that cancels in comparative use (bandwidth search).
            self.aic_ = 2 * (k + 1) - 2 * self.log_likelihood_

            # Bayesian Information Criterion
            # Cast n to float to avoid overload resolution issues with numpy.log
            self.bic_ = (
                np.log(float(n)) * (k + 1)  # ty:ignore[invalid-argument-type]
                - 2 * self.log_likelihood_
            )

            # Corrected AIC — GWR/MGWR form (Fotheringham et al. 2002).
            # Uses p = k+1 consistently in both the AIC and the small-sample
            # correction, so it is internally consistent:
            #   AICc = -2ℓ + 2n(k+1)/(n-k-2)
            #        = AIC + 2(k+1)(k+2)/(n-k-2)
            # Compare the Burnham & Anderson (2002) general formula which would give
            # AIC + 2p(p+1)/(n-p-1) with p = k+1: same result.
            if n - k - 2 > 0:
                self.aicc_ = (
                    -2.0 * self.log_likelihood_
                    + 2.0
                    * n  # ty:ignore[unsupported-operator]
                    * (k + 1.0)
                    / (n - k - 2.0)
                )
            else:
                self.aicc_ = np.nan
        else:
            self.aic_ = np.nan
            self.bic_ = np.nan
            self.aicc_ = np.nan

    def _prepare_prediction_nearest(self, geometry: gpd.GeoSeries) -> np.ndarray:
        """
        Prepare nearest-neighbor mapping from target geometries to local model ids.

        Parameters
        ----------
        geometry : geopandas.GeoSeries
            Point geometries for new observations.

        Returns
        -------
        np.ndarray:
            1-D array of local model identifiers corresponding to the nearest
            training geometry for each input geometry. The order aligns with the input
            GeoSeries.
        """

        self._validate_geometry(geometry)

        if not (isinstance(self.geometry, gpd.GeoSeries)):
            raise ValueError("Geometry needs to be specified to enable prediction.")
        indices_array = self.geometry.sindex.nearest(geometry, return_all=False)[1]
        local_ids = self._local_models.index[indices_array.flatten()].to_numpy()
        return local_ids

    def _prepare_prediction_neighborhoods(
        self, geometry: gpd.GeoSeries, bandwidth: float | int | None = None
    ) -> tuple[list, list]:
        """
        Prepare neighborhood information for prediction on new observations.

        Parameters
        ----------
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : float
            Custom bandwidth overriding self.bandwidth

        Returns
        -------
        tuple
            - local_model_ids: list of arrays with local model indices per observation
            - distances: list of arrays with kernel weights per observation
        """
        self._validate_geometry(geometry)

        if not (
            (isinstance(self.bandwidth, Real) or bandwidth)
            and isinstance(self.geometry, gpd.GeoSeries)
        ):
            raise ValueError(
                "Bandwidth and geometry need to be specified to enable prediction."
            )

        bw = bandwidth if bandwidth is not None else self.bandwidth

        if (bw is None or not isinstance(bw, Real)) or np.isnan(bw) or (bw <= 0):
            raise ValueError(f"Bandwidth must be a positive scalar number. Got '{bw}'.")

        if not self.fixed and not isinstance(bw, Integral):
            raise ValueError("Adaptive bandwidth (fixed=False) must be an integer.")

        kernel = (
            _kernel_functions[self.kernel]
            if isinstance(self.kernel, str)
            else self.kernel
        )

        if self.fixed:
            input_ids, indices_array = self.geometry.sindex.query(
                geometry, predicate="dwithin", distance=self.bandwidth
            )
            local_ids = self._local_models.index[indices_array.flatten()].to_numpy()
            distance = kernel(
                self.geometry.iloc[indices_array].distance(
                    geometry.iloc[input_ids], align=False
                ),
                bw,
            )
        else:
            training_coords = self.geometry.get_coordinates()
            tree = KDTree(training_coords)
            query_coords = geometry.get_coordinates()

            distances, indices_array = tree.query(query_coords, k=bw)

            # Flatten arrays for consistent format
            input_ids = np.repeat(np.arange(len(geometry)), bw)
            local_ids = self._local_models.index[indices_array.flatten()].to_numpy()
            distances = distances.flatten()

            # For adaptive KNN, determine the bandwidth for each neighborhood
            # by finding the max distance in each neighborhood
            kernel_bandwidth = (
                pd.Series(distances).groupby(input_ids).transform("max") + 1e-6
            )  # can't have 0
            distance = kernel(distances, kernel_bandwidth)

        split_indices = np.where(np.diff(input_ids))[0] + 1
        local_model_ids = np.split(local_ids, split_indices)
        distances = np.split(np.asarray(distance), split_indices)

        return local_model_ids, distances

    def _predict_local_ensemble(
        self,
        x_: pd.DataFrame,
        models_: np.ndarray,
        distances_: np.ndarray,
    ):
        """
        Make prediction for a single observation using local models.

        Must be implemented by subclasses.

        Parameters
        ----------
        x_ : pd.DataFrame
            Single-row DataFrame with features for the observation.
        models_ : np.ndarray
            Array of local model indices to use for prediction.
        distances_ : np.ndarray
            Array of kernel weights for each local model.

        Returns
        -------
        Prediction result (type depends on subclass implementation).
        """
        raise NotImplementedError("Subclasses must implement _predict_local")

    def _validate_fit_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        geometry: gpd.GeoSeries | None,
    ) -> None:
        """
        Validate input data and configuration parameters before model fitting.

        This method performs structural and spatial consistency checks to ensure that:
        - Feature matrix `X` and target vector `y` have matching lengths.
        - At least one spatial structure (`geometry` or `graph`) is provided.
        - The provided geometry, if any, matches the number of observations in `X`.
        - Bandwidth is positive when specified.
        - Adaptive bandwidth (`fixed=False`) is an integer.

        Raises
        ------
        ValueError
            If any of the validation conditions fail.
        """
        # Length checks
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}."
            )

        # Geometry presence
        if self.graph is None and geometry is None:
            raise ValueError("Either geometry or graph must be provided.")

        # Geometry checks
        if geometry is not None:
            if len(X) != len(geometry):
                raise ValueError(
                    f"X and geometry must have the same length. "
                    f"Got {len(X)} and {len(geometry)}."
                )
            self._validate_geometry(geometry)

        # Bandwidth validation
        bw = self.bandwidth
        if bw is not None:
            # must be scalar and not NaN
            if (
                not np.isscalar(bw)
                or pd.isna(bw)
                or not isinstance(bw, Real)
                or bw <= 0
            ):
                raise ValueError("Bandwidth must be a positive scalar number.")

            if not self.fixed and not isinstance(bw, Integral):
                raise ValueError("Adaptive bandwidth (fixed=False) must be an integer.")

        if isinstance(self.kernel, str):
            if self.kernel not in _kernel_functions:
                raise ValueError(
                    f"Invalid kernel '{self.kernel}'. "
                    f"Supported kernels are: {list(_kernel_functions.keys())} "
                    "or a callable."
                )
        elif not callable(self.kernel):
            raise ValueError(
                "kernel must be either a valid string or a callable function."
            )

    # Abstract methods that subclasses must implement
    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list[Hashable]:
        raise NotImplementedError("Subclasses must implement _fit_local")

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None):
        raise NotImplementedError("Subclasses must implement fit")

    def _get_score_data(
        self,
        local_model: BaseEstimator,  # noqa: ARG002
        X: pd.DataFrame,  # noqa: ARG002
        y: pd.Series,  # noqa: ARG002
    ) -> float | tuple:
        """Subclasses should implement custom function"""
        return np.nan

    def local_metric(self, func: Callable, *args, **kwargs) -> np.ndarray:
        """Compute a metric per fitted local model.

        Parameters
        ----------
        func : callable
            Callable with a signature ``func(y_true, y_pred, *args, **kwargs)``.

        Returns
        -------
        numpy.ndarray
            One value per focal location (NaN for skipped / unfitted local models).
        """
        results = []

        for y, y_pred in zip(self._y_local, self._pred_local, strict=True):
            if y.shape[0] == 0:
                results.append(np.nan)
            else:
                results.append(func(y, y_pred, *args, **kwargs))
        return np.array(results)


class BaseClassifier(ClassifierMixin, _BaseModel):
    """Generic geographically weighted classification meta-estimator.

    This class wraps a scikit-learn-compatible *classifier class* and fits one local
    model per focal observation using spatially varying sample weights.

    The spatial interaction is defined either by (a) ``geometry`` + bandwidth/kernel
    settings or (b) a precomputed :class:`libpysal.graph.Graph` passed via ``graph``.

    Notes
    -----
    - ``y`` must be binary (``{0, 1}`` or boolean).
    - To enable prediction on new data via :meth:`predict`/:meth:`predict_proba`, you
      must set ``keep_models=True`` (store in memory) or ``keep_models=Path(...)``
      (serialize to disk).
    - Only point geometries are supported.

    Parameters
    ----------
    model : ClassifierMixin
        Class implementing the scikit-learn classifier API (e.g.
        :class:`sklearn.linear_model.LogisticRegression`). The class (not an instance)
        is instantiated internally for each local model.
    bandwidth : float | int | None
        Bandwidth for defining neighborhoods.

        - If ``fixed=True``, this is a distance threshold.
        - If ``fixed=False``, this is the number of nearest neighbors used to form the
          local neighborhood.

        If ``graph`` is provided, ``bandwidth`` is ignored.
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows assessment of
        geographically weighted metrics on unseen data without a need for train/test
        split, hence providing value for all samples. This is needed for further spatial
        analysis of the model performance (and generalises to models that do not support
        OOB scoring). However, it leaves out the most representative sample. By default
        False
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations in the sample. If given, it is used directly and ``bandwidth``,
        ``fixed``, ``kernel``, and ``include_focal`` keywords are ignored. Either
        ``geometry`` or ``graph`` need to be specified. To allow prediction, it is
        required to specify ``geometry``. Potentially, both can be specified where
        ``graph`` encodes spatial interaction between observations in ``geometry``.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors by
        default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alongside the
        geographically weighted, by default True
    strict : bool | None, optional
        Do not fit any models if at least one neighborhood has invariant ``y``, by
        default False. None is treated as False but provides a warning if there are
        invariant models.
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that for
        some models, like random forests, the objects can be large. If string or Path is
        provided, the local models are not held in memory but serialized to the disk
        from which they are loaded in prediction.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``, by default
        None
    batch_size : int | None, optional
        Number of models to process in each batch. Specify batch_size if your models do
        not fit into memory. By default None
    min_proportion : float, optional
        Minimum proportion of minority class for a model to be fitted, by default 0.2
    undersample : bool | float, optional
        Whether to apply random undersampling to balance classes.

        If ``True``, undersample the majority class to match the minority class
        (i.e., minority/majority ratio = 1.0).

        If a float ``alpha > 0``, target a minority/majority ratio of ``alpha`` after
        resampling, i.e. ``alpha = N_min / N_resampled_majority``.
        By default False
    leave_out : float | int, optional
        Leave out a fraction (when float) or a set number (when int) of random
        observations from each local model to be used to measure out-of-sample log loss
        based on pooled samples from all the models. This is useful for bandwidth
        selection for cases where some local models are not fitted due to local
        invariance and resulting information criteria are not comparable.
    random_state : int | None, optional
        Random seed for reproducibility, by default None
    coplanar: "raise", "jitter", "clique", optional
        Method for handling coplanar points with adaptive kernel. Options are
        ``'raise'`` (raising an exception when coplanar points are present),
        ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
        ``'clique'`` (induce fully-connected sub cliques for coplanar points).
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation

    Attributes
    ----------
    proba_ : pd.DataFrame
        Probability predictions for focal locations based on a local model trained
        around the point itself.
    pred_ : pd.Series
        Binary predictions for focal locations based on a local model trained around the
        location itself.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of the hat matrix).
        Only available for logistic models.
    effective_df_ : float
        Effective degrees of freedom (sum of hat values).
        Only available for logistic models.
    log_likelihood_ : float
        Global log-likelihood of the model.
        Only available for logistic models.
    aic_ : float
        Akaike information criterion.
        Only available for logistic models.
    aicc_ : float
        Corrected Akaike information criterion (small-sample correction).
        Only available for logistic models.
    bic_ : float
        Bayesian information criterion.
        Only available for logistic models.
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
    local_class_support_: pd.Series
        Number of distinct class labels in each local neighborhood.
    left_out_y_ : numpy.ndarray
        Array of ``y`` values left out when ``leave_out`` is set.
    left_out_proba_ : numpy.ndarray
        Array of probabilites on left out observations in local models when
        ``leave_out`` is set.
    left_out_w_ : numpy.ndarray
        Array of weights on left out observations in local models when
        ``leave_out`` is set.

    Examples
    --------
    Fit a geographically weighted logistic regression by passing a scikit-learn
    classifier class.

    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from sklearn.linear_model import LogisticRegression
    >>> from spatialml.base import BaseClassifier

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> gw = BaseClassifier(
    ...     LogisticRegression,
    ...     bandwidth=30,
    ...     fixed=False,
    ...     include_focal=True,
    ...     keep_models=True,
    ...     max_iter=200,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> gw.pred_.head()
    0     True
    1    False
    2    False
    3     True
    4     True
    dtype: boolean
    """

    def __init__(
        self,
        model,
        *,
        bandwidth: float | None = None,
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
        include_focal: bool = False,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        min_proportion: float = 0.2,
        undersample: bool | float = False,
        leave_out: float | int | None = None,
        random_state: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )
        self.min_proportion = min_proportion
        self.undersample = undersample
        self.random_state = random_state
        self.leave_out = leave_out
        self._empty_score_data = (np.array([]), np.array([]))
        self._empty_feature_imp = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "BaseClassifier":
        """Fit geographically weighted local classification models.

        This fits one local model per focal observation (subject to local invariance
        checks and ``min_proportion``) and stores focal predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Binary target encoded as boolean or ``{0, 1}``.
        geometry : geopandas.GeoSeries | None
            Geographic location of the observations in the sample. Used to determine the
            spatial interaction weight based on specification by ``bandwidth``,
            ``fixed``, ``kernel``, and ``include_focal`` keywords.  If `None`,
            a precomputed ``graph`` needs to be specified. To allow prediction,
            it is required to specify ``geometry``. If both ``graph`` and ``geometry``
            are specified, ``graph`` is used at the fit time, while ``geometry`` is
            used for prediction.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        The neighborhood definition comes from either ``self.graph`` or from
        ``geometry`` + (``bandwidth``, ``fixed``, ``kernel``, ``include_focal``).
        """
        self._start = time()

        def _is_binary(series: pd.Series) -> bool:
            """Check if a pandas Series encodes a binary variable (bool or 0/1)."""
            unique_values = set(np.unique(series))

            # Check for boolean type
            if series.dtype == bool or unique_values.issubset({True, False}):
                return True

            # Check for 0, 1 encoding
            return bool(unique_values.issubset({0, 1}))

        if not _is_binary(y):
            raise ValueError("Only binary dependent variable is supported.")
        self._validate_fit_inputs(X, y, geometry)
        self.geometry = geometry

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Building weights")
        weights = self.graph if self.graph is not None else self._build_weights()

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Weights ready")
        self._setup_model_storage()

        self._global_classes = np.unique(y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])

        # fit the models
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Fitting the models")
        training_output = self._fit_models_batch(X, y, weights)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Models fitted")

        if self.keep_models:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,
                left_out_proba,
                models,
            ) = zip(*training_output, strict=False)
            self._local_models = pd.Series(models, index=self._names)
        else:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,
                left_out_proba,
            ) = zip(*training_output, strict=False)

        self._n_labels = pd.Series(self._n_labels, index=self._names)
        self.local_class_support_ = self._n_labels.copy()
        self.proba_ = pd.DataFrame(focal_proba, index=self._names)

        # Hat values and IC are only valid for linear/logistic models.
        if self._supports_ic:
            self.hat_values_ = pd.Series(hat_values, index=self._names)
            self.effective_df_ = np.nansum(self.hat_values_)

        # support both bool and 0, 1 encoding of binary variable
        col = True if True in self.proba_.columns else 1
        # global GW accuracy
        nan_mask = self.proba_[col].isna()

        self.pred_ = pd.Series(pd.NA, index=y.index, dtype="boolean")
        self.pred_.loc[~nan_mask] = self.proba_[col][~nan_mask] > 0.5

        self._n_fitted_models = (~self.proba_[col].isna()).sum()
        self.prediction_rate_ = self._n_fitted_models / nan_mask.shape[0]
        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        if self.leave_out and self.prediction_rate_ > 0:
            self.left_out_y_ = np.concatenate([arr[1] for arr in left_out_proba])
            self.left_out_proba_ = np.concatenate([arr[0] for arr in left_out_proba])
            self.left_out_w_ = np.concatenate([arr[2] for arr in left_out_proba])

        if self.fit_global_model:
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Fitting global model")
            self._fit_global_model(X, y)

        # Log-likelihood and IC are only valid for linear/logistic models.
        if self._supports_ic:
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Computing global likelihood")
            self.log_likelihood_ = self._compute_global_log_likelihood(y)

            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Computing information criteria")
            self._compute_information_criteria()

        return self

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        return y.to_numpy(), local_model.predict(X)

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list[Hashable]:
        """Fit individual local model"""

        if self.undersample:
            from .undersample import BinaryRandomUnderSampler

        vc = data["_y"].value_counts()
        n_labels = len(vc)
        skip = n_labels == 1
        if n_labels > 1:
            skip = (vc.iloc[1] / vc.iloc[0]) < self.min_proportion

        # empty data for skipped models
        score_data = self._empty_score_data
        feature_imp = self._empty_feature_imp
        output = [
            name,
            n_labels,
            score_data,
            feature_imp,
            pd.Series(np.nan, index=self._global_classes),
            np.nan,
            (np.zeros(shape=(0, 2)), data["_y"].iloc[:0], data["_weight"].iloc[:0]),
        ]
        if self.keep_models:
            output.append(None)

        if skip:
            return output

        if "random_state" in inspect.signature(self.model).parameters:
            local_model = model(random_state=self.random_state, **model_kwargs)
        else:
            local_model = model(**model_kwargs)

        if self.undersample:
            if isinstance(self.undersample, float):
                rus = BinaryRandomUnderSampler(
                    sampling_strategy=self.undersample, random_state=self.random_state
                )
            else:
                rus = BinaryRandomUnderSampler(random_state=self.random_state)
            data, _ = rus.fit_resample(data, data["_y"])

        if self.leave_out:
            try:
                data, left_out_data = train_test_split(
                    data, test_size=self.leave_out, stratify=data["_y"]
                )
            except ValueError:  # only 1 observation of True
                return output
            if len(data["_y"].value_counts()) == 1:
                return output

        X = data.drop(columns=["_y", "_weight"])
        y = data["_y"]

        local_model.fit(
            X=X,
            y=y,
            sample_weight=data["_weight"],
        )
        focal_x_df = pd.DataFrame(focal_x.reshape(1, -1), columns=X.columns)
        focal_proba = pd.Series(
            local_model.predict_proba(focal_x_df).flatten(), index=local_model.classes_
        )

        # Hat value is only meaningful for linear/logistic models.
        hat_value = (
            self._compute_hat_value(X, data["_weight"], focal_x)
            if self._supports_ic
            else np.nan
        )

        if self.leave_out:
            left_out_proba = local_model.predict_proba(
                left_out_data.drop(columns=["_y", "_weight"])
            )
            left_out_proba = (
                left_out_proba,
                left_out_data["_y"],
                left_out_data["_weight"],
            )
        else:
            left_out_proba = None

        output = [
            name,
            n_labels,
            self._get_score_data(local_model, X, y),
            getattr(local_model, "feature_importances_", None),
            focal_proba,
            hat_value,
            left_out_proba,
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

    def _compute_global_log_likelihood(self, y: pd.Series) -> float:
        """Compute a pooled (global) log-likelihood from focal probabilities."""
        # Get valid predictions (non-NaN)
        mask = ~self.proba_.isna().any(axis=1)

        if not mask.any():
            return np.nan

        y_valid = y[mask]
        proba_valid = self.proba_[mask]

        # Handle both boolean and 0/1 encoding
        if True in proba_valid.columns:
            p = proba_valid[True]
            y_binary = y_valid.astype(int) if y_valid.dtype == bool else y_valid
        else:
            p = proba_valid[1]
            y_binary = y_valid

        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        log_likelihood = np.sum(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p))

        return log_likelihood

    def predict_proba(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
        global_model_weight: float = 0,
    ) -> pd.DataFrame:
        """Predict class probabilities for new observations.

        Prediction can be retrieved either from the nearest local model or based on
        the ensemble of local models. In the latter case, the prediction process works
        as follows:

        1. For a new location on which you want a prediction, identify local models
           within the bandwidth used to train the model.
        2. Apply the kernel function used to train the model to derive weights of
           each of the local models.
        3. Make prediction using each of the local models in the bandwidth.
        4. Make weighted average of predictions based on the kernel weights.
        5. Normalize the result to ensure sum of probabilities is 1.

        The results from the nearest and ensemble predictions are typically similar,
        with the ensemble being significantly slower due to the required number of
        inference calls.

        Further the prediction can be a result of a fusion of local and global models
        when ``global_model_weight`` is set to a non-zero value, following
        :cite:t:`georganos2021Geographical`.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : "nearest", float or None
            Prediction method. Nearest uses the nearest location available at the fit
            time and does prediction using its single model. When set to a numeric
            value, uses an ensemble of local models available within the bandwidth, with
            predictions from individual models being weighted based on the distance and
            a set kernel. When ``None``, uses the bandwidth set at the fit time.
        global_model_weight : float
            Weight of the prediction from the global model. When non-zero, the
            resulting prediction is a weighted average of the values from local model(s)
            and from global model, where local prediction has a weight of 1 and global
            model has a weight equal to ``global_model_weight``.

        Returns
        -------
        pandas.DataFrame
            Predicted probabilities with columns equal to the global classes observed
            during fit.

        Notes
        -----
        Requires the estimator to have been fit with ``keep_models=True`` (or a
        ``Path``) so local models can be used at prediction time.
        """
        data = [X.iloc[[i]] for i in range(len(X))]

        probabilities = []
        if bandwidth == "nearest":
            local_model_ids = self._prepare_prediction_nearest(geometry)

            for x_, model_id in zip(data, local_model_ids, strict=True):
                probabilities.append(self._predict_local_nearest(x_, model_id))
        else:
            local_model_ids, distances = self._prepare_prediction_neighborhoods(
                geometry, bandwidth=bandwidth
            )

            for x_, models_, distances_ in zip(
                data, local_model_ids, distances, strict=True
            ):
                probabilities.append(
                    self._predict_local_ensemble(x_, models_, distances_)
                )

        proba = pd.DataFrame(probabilities, columns=self._global_classes, index=X.index)

        if global_model_weight:
            global_proba = self.global_model.predict_proba(X)
            local = proba.values
            # where local is NaN, use global; otherwise use weighted average
            mask = np.isnan(local)
            denom = 1.0 + global_model_weight
            combined = np.empty_like(local, dtype=float)
            combined[~mask] = (
                local[~mask] + global_model_weight * global_proba[~mask]
            ) / denom
            combined[mask] = global_proba[mask]
            proba = pd.DataFrame(combined, columns=self._global_classes, index=X.index)

        return proba

    def _predict_local_ensemble(
        self,
        x_: pd.DataFrame,
        models_: np.ndarray,
        distances_: np.ndarray,
    ) -> pd.Series:
        pred = []
        for i in models_:
            local_model = self._local_models[i]
            if isinstance(local_model, str):
                with open(local_model, "rb") as f:
                    local_model = load(f)

            if not pd.isna(local_model):
                pred.append(
                    pd.Series(
                        local_model.predict_proba(x_).flatten(),
                        index=local_model.classes_,
                    )
                )
            else:
                pred.append(
                    pd.Series(
                        np.nan,
                        index=self._global_classes,
                    )
                )
        pred = pd.DataFrame(pred)

        mask = pred.isna().any(axis=1)
        if mask.all():
            return pd.Series(np.nan, index=pred.columns)

        weighted = np.average(pred[~mask], axis=0, weights=distances_[~mask])

        # normalize
        weighted = weighted / weighted.sum()
        return pd.Series(weighted, index=pred.columns)

    def _predict_local_nearest(self, x_: pd.DataFrame, model_id: Hashable) -> pd.Series:
        local_model = self._local_models[model_id]
        if isinstance(local_model, str):
            with open(local_model, "rb") as f:
                local_model = load(f)

        if not pd.isna(local_model):
            return pd.Series(
                local_model.predict_proba(x_).flatten(),
                index=local_model.classes_,
            )
        else:
            return pd.Series(
                np.nan,
                index=self._global_classes,
            )

    def predict(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
        global_model_weight: float = 0,
    ) -> pd.Series:
        """Predict classes for new observations.

        This is equivalent to ``predict_proba(...).idxmax(axis=1)``.

        Prediction can be retrieved either from the nearest local model or based on
        the ensemble of local models. In the latter case, the prediction process works
        as follows:

        1. For a new location on which you want a prediction, identify local models
           within the bandwidth used to train the model.
        2. Apply the kernel function used to train the model to derive weights of
           each of the local models.
        3. Make prediction using each of the local models in the bandwidth.
        4. Make weighted average of predictions based on the kernel weights.
        5. Normalize the result to ensure sum of probabilities is 1.

        The results from the nearest and ensemble predictions are typically similar,
        with the ensemble being significantly slower due to the required number of
        inference calls.

        Further the prediction can be a result of a fusion of local and global models
        when ``global_model_weight`` is set to a non-zero value, following
        :cite:t:`georganos2021Geographical`.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : "nearest", float or None
            Prediction method. Nearest uses the nearest location available at the fit
            time and does prediction using its single model. When set to a numeric
            value, uses an ensemble of local models available within the bandwidth, with
            predictions from individual models being weighted based on the distance and
            a set kernel. When ``None``, uses the bandwidth set at the fit time.
        global_model_weight : float
            Weight of the prediction from the global model. When non-zero, the
            resulting prediction is a weighted average of the values from local model(s)
            and from global model, where local prediction has a weight of 1 and global
            model has a weight equal to ``global_model_weight``.

        Returns
        -------
        pandas.Series
            Predicted class.

        Notes
        -----
        Requires the estimator to have been fit with ``keep_models=True`` (or a
        ``Path``) so local models can be used at prediction time.
        """
        proba = self.predict_proba(
            X, geometry, bandwidth=bandwidth, global_model_weight=global_model_weight
        )

        mask = proba.iloc[:, 0].notna()
        if not mask.all():
            r = pd.Series(pd.NA, index=proba.index, dtype="boolean")
            r[mask] = proba[mask].idxmax(axis=1)
            return r

        return proba.idxmax(axis=1)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
        global_model_weight: float = 0,
    ) -> float:  # ty:ignore[invalid-method-override]
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        y : pandas.Series
            True labels for X.
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : "nearest", float or None
            Prediction method. See predict().
        global_model_weight : float
            Weight of the prediction from the global model.

        Returns
        -------
        float
            Mean accuracy of self.predict(X, geometry).
        """
        y_pred = self.predict(
            X, geometry, bandwidth=bandwidth, global_model_weight=global_model_weight
        )
        # Handle missing predictions (pd.NA)
        mask = ~pd.isna(y_pred)
        if not mask.any():
            return float("nan")
        return (y_pred[mask] == y[mask]).mean()


class BaseRegressor(_BaseModel, RegressorMixin):
    """Generic geographically weighted regression meta-estimator.

    This class wraps a scikit-learn-compatible *regressor class* and fits one local
    model per focal observation using spatially varying sample weights.

    The fitted object exposes focal predictions (``pred_``,  in-sample if
    ``include_focal=True``) and local goodness-of-fit summaries.

    Notes
    -----
    - Only point geometries are supported.

    Parameters
    ----------
    model : RegressorMixin
        Class implementing the scikit-learn regressor API (e.g.
        :class:`sklearn.linear_model.LinearRegression`). The class (not an instance)
        is instantiated internally for each local model.
    bandwidth : float | int | None
        Bandwidth for defining neighborhoods.

        - If ``fixed=True``, this is a distance threshold.
        - If ``fixed=False``, this is the number of nearest neighbors used to form the
          local neighborhood.

        If ``graph`` is provided, ``bandwidth`` is ignored.
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows
        assessment of geographically weighted metrics on unseen data without a need for
        train/test split, hence providing value for all samples. This is needed for
        further spatial analysis of the model performance (and generalises to models
        that do not support OOB scoring). However, it leaves out the most representative
        sample. By default False
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations in the sample. If given, it is used directly and ``bandwidth``,
        ``fixed``, ``kernel``, and ``include_focal`` keywords are ignored. Either
        ``geometry`` or ``graph`` need to be specified. To allow prediction, it is
        required to specify ``geometry``. Potentially, both can be specified where
        ``graph`` encodes spatial interaction between observations in ``geometry``.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors by
        default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alongside
        the geographically weighted, by default True
    strict : bool | None, optional
        Do not fit any models if at least one neighborhood has invariant ``y``,
        by default False. None is treated as False but provides a warning if there are
        invariant models.
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that
        for some models, like random forests, the objects can be large. If string or
        Path is provided, the local models are not held in memory but serialized to
        the disk from which they are loaded in prediction.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``, by default
        None
    batch_size : int | None, optional
        Number of models to process in each batch. Specify batch_size if your models do
        not fit into memory. By default None
    random_state : int | None, optional
        Random seed for reproducibility, by default None
    coplanar: "raise", "jitter", "clique", optional
        Method for handling coplanar points with adaptive kernel. Options are
        ``'raise'`` (raising an exception when coplanar points are present),
        ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
        ``'clique'`` (induce fully-connected sub cliques for coplanar points).
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation

    Attributes
    ----------
    pred_ : pd.Series
        Focal predictions for each location.
    resid_ : pd.Series
        Residuals for each location (``y`` - ``pred_``).
    RSS_ : pd.Series
        Residual sum of squares for each location.
    TSS_ : pd.Series
        Total sum of squares for each location.
    y_bar_ : pd.Series
        Weighted mean of y for each location.
    local_r2_ : pd.Series
        Local R2 for each location.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of the hat matrix).
        Only available for linear models.
    effective_df_ : float
        Effective degrees of freedom (sum of hat values).
        Only available for linear models.
    log_likelihood_ : float
        Global log-likelihood of the model (Gaussian assumption).
        Only available for linear models.
    aic_ : float
        Akaike information criterion.
        Only available for linear models.
    aicc_ : float
        Corrected Akaike information criterion (small-sample correction).
        Only available for linear models.
    bic_ : float
        Bayesian information criterion.
        Only available for linear models.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from sklearn.linear_model import LinearRegression
    >>> from spatialml.base import BaseRegressor

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Suicids"]

    >>> gwr = BaseRegressor(
    ...     LinearRegression,
    ...     bandwidth=30,
    ...     fixed=False,
    ...     include_focal=True,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> gwr.local_r2_.head()
    0    0.614715
    1    0.488495
    2    0.599862
    3    0.662435
    4    0.662276
    dtype: float64
    """

    def __init__(
        self,
        model,
        *,
        bandwidth: float | None = None,
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
        include_focal: bool = False,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        random_state: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )
        self.random_state = random_state
        self._empty_score_data = (np.array([]), np.array([]))

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "BaseRegressor":
        """Fit geographically weighted local regression models.

        Fits one local model per focal observation and stores focal (in-sample if
        ``include_focal=True``) predictions in ``pred_``.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target values.
        geometry : geopandas.GeoSeries | None
            Geographic location of the observations in the sample. Used to determine the
            spatial interaction weight based on specification by ``bandwidth``,
            ``fixed``, ``kernel``, and ``include_focal`` keywords.  If `None`,
            a precomputed ``graph`` needs to be specified. To allow prediction,
            it is required to specify ``geometry``. If both ``graph`` and ``geometry``
            are specified, ``graph`` is used at the fit time, while ``geometry`` is
            used for prediction.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        The neighborhood definition comes from either ``self.graph`` or from
        ``geometry`` + (``bandwidth``, ``fixed``, ``kernel``, ``include_focal``).
        """
        self._validate_fit_inputs(X, y, geometry)
        if self.graph is None:
            self._validate_geometry(geometry)

        self.geometry = geometry

        weights = self.graph if self.graph is not None else self._build_weights()
        self._setup_model_storage()

        # fit the models
        training_output = self._fit_models_batch(X, y, weights)

        if self.keep_models:
            (
                self._names,
                focal_pred,
                y_bar,
                tss,
                hat_values,
                self._score_data,
                self._feature_importances,
                models,
            ) = zip(*training_output, strict=False)
            self._local_models = pd.Series(models, index=self._names)
        else:
            (
                self._names,
                focal_pred,
                y_bar,
                tss,
                hat_values,
                self._score_data,
                self._feature_importances,
            ) = zip(*training_output, strict=False)

        self.pred_ = pd.Series(np.nan, index=y.index)
        self.pred_.loc[np.array(self._names)] = focal_pred
        self.resid_ = y - self.pred_
        resids_ = (
            weights.adjacency.values
            * self.resid_.loc[weights.adjacency.index.get_level_values(1)] ** 2
        )
        self.RSS_ = resids_.groupby(weights.adjacency.index.get_level_values(0)).sum()
        self.TSS_ = pd.Series(tss, index=self._names)
        self.y_bar_ = pd.Series(y_bar, index=self._names)
        self.local_r2_ = (self.TSS_ - self.RSS_) / self.TSS_
        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        if self.fit_global_model:
            self._fit_global_model(X, y)

        # Hat values, log-likelihood, and IC are only valid for linear models.
        if self._supports_ic:
            self.hat_values_ = pd.Series(hat_values, index=self._names)
            self.effective_df_ = np.nansum(self.hat_values_)
            self.log_likelihood_ = self._compute_global_log_likelihood()
            self._compute_information_criteria()

        return self

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        return y.to_numpy(), local_model.predict(X)

    def predict(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
        global_model_weight: float = 0,
    ) -> pd.Series:
        """Predict target values for new observations.

        Prediction can be retrieved either from the nearest local model or based on
        the ensemble of local models. In the latter case, the prediction process works
        as follows:

        1. For a new location on which you want a prediction, identify local models
           within the bandwidth used to train the model.
        2. Apply the kernel function used to train the model to derive weights of
           each of the local models.
        3. Make prediction using each of the local models in the bandwidth.
        4. Make weighted average of predictions based on the kernel weights.

        The results from the nearest and ensemble predictions are typically similar,
        with the ensemble being significantly slower due to the required number of
        inference calls.

        Further the prediction can be a result of a fusion of local and global models
        when ``global_model_weight`` is set to a non-zero value, following
        :cite:t:`georganos2021Geographical`.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : "nearest", float or None
            Prediction method. Nearest uses the nearest location available at the fit
            time and does prediction using its single model. When set to a numeric
            value, uses an ensemble of local models available within the bandwidth, with
            predictions from individual models being weighted based on the distance and
            a set kernel. When ``None``, uses the bandwidth set at the fit time.
        global_model_weight : float
            Weight of the prediction from the global model. When non-zero, the
            resulting prediction is a weighted average of the values from local model(s)
            and from global model, where local prediction has a weight of 1 and global
            model has a weight equal to ``global_model_weight``.

        Returns
        -------
        pandas.Series
            Predicted values.

        Notes
        -----
        Requires the estimator to have been fit with ``keep_models=True`` (or a
        ``Path``) so local models can be used at prediction time.
        """
        data = [X.iloc[[i]] for i in range(len(X))]

        predictions = []
        if bandwidth == "nearest":
            local_model_ids = self._prepare_prediction_nearest(geometry)

            for x_, model_id in zip(data, local_model_ids, strict=True):
                predictions.append(self._predict_local_nearest(x_, model_id))
        else:
            local_model_ids, distances = self._prepare_prediction_neighborhoods(
                geometry, bandwidth=bandwidth
            )

            for x_, models_, distances_ in zip(
                data, local_model_ids, distances, strict=True
            ):
                predictions.append(
                    self._predict_local_ensemble(x_, models_, distances_)
                )

        pred = pd.Series(predictions, index=X.index)

        if global_model_weight:
            global_pred = self.global_model.predict(X)
            combined_pred = np.column_stack([pred, global_pred])
            pred = np.average(combined_pred, axis=1, weights=[1, global_model_weight])

            return pd.Series(pred, index=X.index)

        return pred

    def _predict_local_ensemble(
        self,
        x_: pd.DataFrame,
        models_: np.ndarray,
        distances_: np.ndarray,
    ) -> float:
        pred = []
        for i in models_:
            local_model = self._local_models[i]
            if isinstance(local_model, str):
                with open(local_model, "rb") as f:
                    local_model = load(f)

            if local_model is not None:
                pred.append(local_model.predict(x_).flatten()[0])
            else:
                pred.append(np.nan)

        pred = np.array(pred)
        mask = np.isnan(pred)
        if mask.all():
            return np.nan

        weighted = np.average(pred[~mask], weights=distances_[~mask])
        return weighted

    def _predict_local_nearest(self, x_: pd.DataFrame, model_id: Hashable) -> float:
        local_model = self._local_models[model_id]
        if isinstance(local_model, str):
            with open(local_model, "rb") as f:
                local_model = load(f)

        if local_model is not None:
            return local_model.predict(x_).flatten()[0]
        return np.nan

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> list[Hashable]:
        if "random_state" in inspect.signature(self.model).parameters:
            local_model = model(random_state=self.random_state, **model_kwargs)
        else:
            local_model = model(**model_kwargs)

        X = data.drop(columns=["_y", "_weight"])
        y = data["_y"]

        local_model.fit(
            X=X,
            y=y,
            sample_weight=data["_weight"],
        )
        focal_x_df = pd.DataFrame(focal_x.reshape(1, -1), columns=X.columns)
        focal_pred = local_model.predict(focal_x_df).flatten()[0]

        y_bar = self._y_bar(y, data["_weight"])
        tss = self._tss(y, y_bar, data["_weight"])

        # Hat value is only meaningful for linear/logistic models.
        hat_value = (
            self._compute_hat_value(X, data["_weight"], focal_x)
            if self._supports_ic
            else np.nan
        )

        output = [
            name,
            focal_pred,
            y_bar,
            tss,
            hat_value,  # Add hat value to output
            self._get_score_data(local_model, X, y),
            getattr(local_model, "feature_importances_", None),
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

    def _compute_global_log_likelihood(self) -> float:
        """
        Compute the global log likelihood for the entire GWR model assuming
        Gaussian errors (appropriate for linear regression).

        Uses only observations with non-NaN residuals so that models with
        partially-missing focal predictions still return a valid likelihood.
        The fitted sample size is stored in ``self._n_fitted_models`` so that
        :meth:`_compute_information_criteria` uses a consistent ``n``.

        Returns
        -------
        float
            Global log-likelihood value.
        """
        residuals = self.resid_.dropna()
        n = len(residuals)
        # Store so _compute_information_criteria uses the same n.
        self._n_fitted_models = n

        if n == 0:
            return np.nan

        # MLE estimate of σ (maximises the Gaussian likelihood)
        sigma2 = np.sum(residuals**2) / n
        sigma = np.sqrt(sigma2)

        if sigma <= 0:
            return np.nan

        # Global log-likelihood assuming Gaussian errors
        log_likelihood = (
            -n / 2.0 * np.log(2 * np.pi)
            - n / 2.0 * np.log(sigma2)
            - np.sum(residuals**2) / (2.0 * sigma2)
        )

        return log_likelihood

    def _y_bar(self, y, w_i):
        """weighted mean of y"""
        sum_yw = np.sum(y * w_i)
        return sum_yw / np.sum(w_i)

    def _tss(self, y, y_bar, w_i):
        """geographically weighted total sum of squares"""
        return np.sum(w_i * (y - y_bar) ** 2)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
        global_model_weight: float = 0,
    ) -> float:  # ty:ignore[invalid-method-override]
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        y : pandas.Series
            True values for X.
        geometry : geopandas.GeoSeries
            Point geometries for new observations.
        bandwidth : "nearest", float or None
            Prediction method. See predict().
        global_model_weight : float
            Weight of the prediction from the global model.

        Returns
        -------
        float
            R^2 of self.predict(X, geometry).
        """
        y_pred = self.predict(
            X, geometry, bandwidth=bandwidth, global_model_weight=global_model_weight
        )
        # Handle missing predictions (np.nan)
        mask = ~pd.isna(y_pred)
        if not mask.any():
            return float("nan")
        y_true = y[mask]
        y_pred = y_pred[mask]
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
