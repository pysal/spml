from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from .base import BaseClassifier, BaseRegressor


class GWRandomForestClassifier(BaseClassifier):
    """Geographically weighted random forest classifier.

    Fits one :class:`sklearn.ensemble.RandomForestClassifier` per focal observation
    using spatially varying sample weights.

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
        type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows assessment of
        geographically weighted metrics on unseen data without a need for train/test
        split, hence providing value for all samples. This is needed for futher spatial
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
        Determines if the global baseline model shall be fitted alognside the
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
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
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
    oob_y_pooled_ : numpy.ndarray
        Pooled out-of-bag (OOB) true labels across all fitted local models.
    oob_pred_pooled_ : numpy.ndarray
        Pooled out-of-bag (OOB) predictions/scores across all fitted local models.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.ensemble import GWRandomForestClassifier

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> gw = GWRandomForestClassifier(
    ...     bandwidth=30,
    ...     fixed=False,
    ...     random_state=0,
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
            model=RandomForestClassifier,
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

        self._model_type = "random_forest"
        self._model_kwargs["oob_score"] = self._get_oob_score_data

        self._empty_score_data = (np.array([]).reshape(-1, 1), np.array([]))

    def _get_oob_score_data(self, true, pred):
        """Callback used by scikit-learn to collect OOB targets/predictions."""
        return true, pred

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "GWRandomForestClassifier":
        """Fit geographically weighted random forests.

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
        GWRandomForestClassifier
            Fitted estimator.

        Notes
        -----
        In addition to the base classifier outputs, this method also populates
        ``oob_y_pooled_`` and ``oob_pred_pooled_`` by pooling OOB values across all
        fitted local models.
        """
        self._empty_feature_imp = np.array([np.nan] * (X.shape[1]))
        super().fit(X=X, y=y, geometry=geometry)

        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        del self._score_data

        # Filter out empty arrays before concatenation
        non_empty_y = [arr for arr in self._y_local if arr.size > 0]
        non_empty_pred = [arr for arr in self._pred_local if arr.size > 0]
        if non_empty_y:
            self.oob_y_pooled_ = np.concatenate(non_empty_y)
        else:
            # Set to empty array with same dtype as y
            self.oob_y_pooled_ = np.array([], dtype=y.dtype)
        if non_empty_pred:
            self.oob_pred_pooled_ = np.concatenate(non_empty_pred)
        else:
            # Set to empty array with float dtype (typical for predictions)
            self.oob_pred_pooled_ = np.array([], dtype=float)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,  # noqa: ARG002
        y: pd.Series,  # noqa: ARG002
    ) -> tuple:
        return local_model.oob_score_


class GWGradientBoostingClassifier(BaseClassifier):
    """Geographically weighted gradient boosting classifier.

    Fits one :class:`sklearn.ensemble.GradientBoostingClassifier` per focal observation
    using spatially varying sample weights.

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
        type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows
        assessment of geographically weighted metrics on unseen data without a need for
        train/test split, hence providing value for all samples. This is needed for
        futher spatial analysis of the model performance (and generalises to models
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
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
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
        Number of models to process in each batch. Specify batch_size fi your models do
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
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.ensemble import GWGradientBoostingClassifier

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> gw = GWGradientBoostingClassifier(
    ...     bandwidth=30,
    ...     fixed=False,
    ...     random_state=0,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> gw.pred_.head()
    0    False
    1    False
    2    False
    3     True
    4     True
    dtype: boolean
    """

    def __init__(
        self,
        *,
        bandwidth: int | float | None = None,
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
        random_state: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=GradientBoostingClassifier,
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
            random_state=random_state,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "gradient_boosting"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "GWGradientBoostingClassifier":
        """Fit geographically weighted gradient boosting classifiers.

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
        GWGradientBoostingClassifier
            Fitted estimator.

        Notes
        -----
        Populates ``feature_importances_`` from the fitted local models.
        """
        self._empty_feature_imp = np.array([np.nan] * (X.shape[1]))
        super().fit(X=X, y=y, geometry=geometry)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self


class GWRandomForestRegressor(BaseRegressor):
    """Geographically weighted random forest regressor.

    Fits one :class:`sklearn.ensemble.RandomForestRegressor` per focal observation
    using spatially varying sample weights.

    The spatial interaction is defined either by (a) ``geometry`` + bandwidth/kernel
    settings or (b) a precomputed :class:`libpysal.graph.Graph` passed via ``graph``.

    Notes
    -----
    - To enable prediction on new data via :meth:`predict`, you must set
      ``keep_models=True`` (store in memory) or ``keep_models=Path(...)`` (serialize
      to disk).
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
        type of kernel function used to weight observations, by default "bisquare"
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
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
    oob_y_pooled_ : numpy.ndarray
        Pooled out-of-bag (OOB) true values across all fitted local models.
    oob_pred_pooled_ : numpy.ndarray
        Pooled out-of-bag (OOB) predictions across all fitted local models.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.ensemble import GWRandomForestRegressor

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Suicids"]

    >>> gw = GWRandomForestRegressor(
    ...     bandwidth=30,
    ...     fixed=False,
    ...     random_state=0,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> gw.pred_.head()
    0    104647.21
    1     18963.73
    2     28642.92
    3     23943.21
    4     57140.26
    dtype: float64
    """

    def __init__(
        self,
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
            model=RandomForestRegressor,
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
            random_state=random_state,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "random_forest"
        self._model_kwargs["oob_score"] = self._get_oob_score_data

        self._empty_score_data = (np.array([]).reshape(-1, 1), np.array([]))

    def _get_oob_score_data(self, true, pred):
        """Callback used by scikit-learn to collect OOB targets/predictions."""
        return true, pred

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "GWRandomForestRegressor":
        """Fit geographically weighted random forests.

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
        GWRandomForestRegressor
            Fitted estimator.

        Notes
        -----
        In addition to the base regressor outputs, this method also populates
        ``oob_y_pooled_`` and ``oob_pred_pooled_`` by pooling OOB values across all
        fitted local models.
        """
        super().fit(X=X, y=y, geometry=geometry)

        # Handle OOB data
        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        del self._score_data

        # Filter out empty arrays before concatenation
        non_empty_y = [arr for arr in self._y_local if arr.size > 0]
        non_empty_pred = [arr for arr in self._pred_local if arr.size > 0]
        if non_empty_y:
            self.oob_y_pooled_ = np.concatenate(non_empty_y)
        else:
            # Set to empty array with same dtype as y
            self.oob_y_pooled_ = np.array([], dtype=y.dtype)
        if non_empty_pred:
            self.oob_pred_pooled_ = np.concatenate(non_empty_pred)
        else:
            # Set to empty array with float dtype (typical for predictions)
            self.oob_pred_pooled_ = np.array([], dtype=float)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self

    def _get_score_data(
        self,
        local_model: BaseEstimator,
        X: pd.DataFrame,  # noqa: ARG002
        y: pd.Series,  # noqa: ARG002
    ) -> tuple:
        return local_model.oob_score_


class GWGradientBoostingRegressor(BaseRegressor):
    """Geographically weighted gradient boosting regressor.

    Fits one :class:`sklearn.ensemble.GradientBoostingRegressor` per focal observation
    using spatially varying sample weights.

    The spatial interaction is defined either by (a) ``geometry`` + bandwidth/kernel
    settings or (b) a precomputed :class:`libpysal.graph.Graph` passed via ``graph``.

    Notes
    -----
    - To enable prediction on new data via :meth:`predict`, you must set
      ``keep_models=True`` (store in memory) or ``keep_models=Path(...)`` (serialize
      to disk).
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
        type of kernel function used to weight observations, by default "bisquare"
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
        some models, like gradient boosting, the objects can be large. If string or Path
        is provided, the local models are not held in memory but serialized to the disk
        from which they are loaded in prediction.
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
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.ensemble import GWGradientBoostingRegressor

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Suicids"]

    >>> gw = GWGradientBoostingRegressor(
    ...     bandwidth=30,
    ...     fixed=False,
    ...     random_state=0,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> gw.pred_.head()
    0    74314.578640
    1    14362.093351
    2    24158.876462
    3    21085.659844
    4    43375.134041
    dtype: float64
    """

    def __init__(
        self,
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
            model=GradientBoostingRegressor,
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
            random_state=random_state,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "gradient_boosting"
        self._empty_score_data = np.nan

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries | None = None
    ) -> "GWGradientBoostingRegressor":
        """Fit geographically weighted gradient boosting regressors.

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
        GWGradientBoostingRegressor
            Fitted estimator.

        Notes
        -----
        Populates ``feature_importances_`` from the fitted local models.
        """
        super().fit(X=X, y=y, geometry=geometry)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self
