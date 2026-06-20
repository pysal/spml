from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from .base import BaseClassifier, BaseRegressor


class GWLogisticRegression(BaseClassifier):
    """Geographically weighted logistic regression

    Fits one :class:`sklearn.linear_model.LogisticRegression` per focal observation
    using spatially varying sample weights.

    Notes
    -----
    - ``y`` must be binary (``{0, 1}`` or boolean).
    - To enable prediction on new data via :meth:`predict`/:meth:`predict_proba`, you
      must set ``keep_models=True`` (store in memory) or ``keep_models=Path(...)``
      (serialize to disk).
    - Only point geometries are supported.

    Parameters
    ----------
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
        sample. By default True
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
        Binary predictions for focal locations based on a local model trained around
        the location itself.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of hat matrix)
    effective_df_ : float
        Effective degrees of freedom (sum of hat values)
    log_likelihood_ : float
        Global log likelihood of the model
    aic_ : float
        Akaike information criterion of the model
    aicc_ : float
        Corrected Akaike information criterion to account for model
        complexity (smaller bandwidths)
    bic_ : float
        Bayesian information criterion
    local_coef_ : pd.DataFrame
        Local coefficient of the features in the decision function for each feature at
        each location
    local_intercept_ : pd.Series
        Local intercept values at each location
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
    local_class_support_: pd.Series
        Number of distinct class labels in each local neighborhood.
    left_out_y_ : np.ndarray
        Array of ``y`` values left out when ``leave_out`` is set.
    left_out_proba_ : np.ndarray
        Array of probabilites on left out observations in local models when
        ``leave_out`` is set.
    left_out_w_ : np.ndarray
        Array of weights on left out observations in local models when
        ``leave_out`` is set.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spatialml.linear_model import GWLogisticRegression

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> gw = GWLogisticRegression(
    ...     bandwidth=30,
    ...     fixed=False,
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

    # TODO: score_ should be an alias of pooled_score_ - this is different from MGWR
    def __init__(
        self,
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
        include_focal: bool = True,
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
            model=LogisticRegression,
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
            min_proportion=min_proportion,
            undersample=undersample,
            leave_out=leave_out,
            random_state=random_state,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "logistic"

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])

        self._empty_score_data = (
            np.array([]),  # true
            np.array([]),  # pred
            pd.Series(np.nan, index=self.feature_names_in_),  # local coefficients
            np.array([np.nan]),
        )  # intercept

        super().fit(X=X, y=y, geometry=geometry)

        self.local_coef_ = pd.concat(
            [x[2] for x in self._score_data], axis=1, keys=self._names
        ).T
        self.local_intercept_ = pd.Series(
            np.concatenate([x[3] for x in self._score_data]), index=self._names
        )

        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        del self._score_data

        # Check for empty arrays before concatenation to avoid unexpected shapes
        if self._y_local and any(arr.size > 0 for arr in self._y_local):
            self.y_pooled_ = np.concatenate(
                [arr for arr in self._y_local if arr.size > 0]
            )
        else:
            self.y_pooled_ = np.array([])
        if self._pred_local and any(arr.size > 0 for arr in self._pred_local):
            self.pred_pooled_ = np.concatenate(
                [arr for arr in self._pred_local if arr.size > 0]
            )
        else:
            self.pred_pooled_ = np.array([])

        return self

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        local_proba = pd.DataFrame(
            local_model.predict_proba(X), columns=local_model.classes_
        )
        return (
            y,
            local_proba.idxmax(axis=1),
            pd.Series(
                local_model.coef_.flatten(),
                index=local_model.feature_names_in_,
            ),  # coefficients
            local_model.intercept_,  # intercept
        )


class GWLinearRegression(BaseRegressor):
    """Geographically weighted linear regression

    Fits one :class:`sklearn.linear_model.LinearRegression` per focal observation
    using spatially varying sample weights.

    The fitted object exposes focal predictions (``pred_``,  in-sample if
    ``include_focal=True``) and local goodness-of-fit summaries.

    Prediction for new (out-of-sample) observations is not currently implemented for
    regressors.

    Parameters
    ----------
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
        sample. By default True
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
    coplanar: "raise", "jitter", "clique", optional
        Method for handling coplanar points with adaptive kernel. Options are
        ``'raise'`` (raising an exception when coplanar points are present),
        ``'jitter'`` (randomly displace coplanar points to produce uniqueness), &
        ``'clique'`` (induce fully-connected sub cliques for coplanar points).
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``sklearn.linear_model.LinearRegression``
        initialisation

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
        Hat values for each location (diagonal elements of hat matrix).
    effective_df_ : float
        Effective degrees of freedom (sum of hat values).
    log_likelihood_ : float
        Global log likelihood of the model.
    aic_ : float
        Akaike information criterion of the model.
    aicc_ : float
        Corrected Akaike information criterion to account for model
        complexity (smaller bandwidths).
    bic_ : float
        Bayesian information criterion.
    local_coef_ : pd.DataFrame
        Local coefficient of the features in the decision function for each feature at
        each location
    local_intercept_ : pd.Series
        Local intercept values at each location

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spatialml.linear_model import GWLinearRegression

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Suicids"]

    >>> gwr = GWLinearRegression(
    ...     bandwidth=30,
    ...     fixed=False,
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
        include_focal: bool = True,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        strict: bool | None = False,
        keep_models: bool = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=LinearRegression,
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

        self._model_type = "linear"

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,  # noqa: ARG002
        y: pd.Series,  # noqa: ARG002
    ) -> tuple:
        return (
            pd.Series(
                local_model.coef_.flatten(),
                index=local_model.feature_names_in_,
            ),  # coefficients
            local_model.intercept_,  # intercept
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])
        self._empty_score_data = (
            pd.Series(np.nan, index=self.feature_names_in_),  # local coefficients
            np.array([np.nan]),
        )  # intercept

        super().fit(X=X, y=y, geometry=geometry)

        self.local_coef_ = pd.concat(
            [x[0] for x in self._score_data], axis=1, keys=self._names
        ).T
        self.local_intercept_ = pd.Series(
            [x[1] for x in self._score_data], index=self._names
        )

        return self
