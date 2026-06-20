from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from libpysal import graph
from sklearn.base import TransformerMixin
from sklearn.utils.parallel import Parallel, delayed

from ..base import _BaseModel

__all__ = ["BaseDecomposition"]


class BaseDecomposition(TransformerMixin, _BaseModel):
    """Geographically weighted decomposition meta-estimator (scaffold).

    Base class for unsupervised geographically weighted decompositions
    (e.g. ``GWPCA``, ``RobustGWPCA``, ``GWFA``).  Subclasses implement
    :meth:`_fit_local`, returning, for each focal location, the local
    components, scores, mean and winning-variable vector.  The shared
    spatial-weighting machinery, batch dispatch and output unpacking are
    inherited from :class:`_BaseModel` (with the ``y=None`` path enabled
    in :meth:`_batch_fit`).

    Notes
    -----
    No supervised baseline is fitted: ``fit_global_model`` is forced to
    ``False`` and any ``y`` passed to :meth:`fit` is ignored (the argument
    exists for scikit-learn pipeline / ``check_estimator`` compatibility).

    Parameters
    ----------
    n_components : int | None, optional
        Number of components retained per local fit.  ``None`` defers the
        decision to the subclass.  By default ``None``.
    bandwidth : float | int | None, optional
        Bandwidth for defining neighbourhoods.  See :class:`BaseClassifier`
        for the full semantics of ``fixed`` / ``kernel`` / ``include_focal``.
    fixed : bool, optional
        Distance-based (``True``) versus adaptive KNN (``False``)
        bandwidth, by default ``False``.
    kernel : str | Callable, optional
        Weighting kernel, by default ``"bisquare"``.
    include_focal : bool, optional
        Include the focal observation in its own local fit, by default
        ``False``.
    graph : libpysal.graph.Graph | None, optional
        Pre-computed spatial graph (overrides ``bandwidth`` / ``kernel``).
    n_jobs : int, optional
        Parallelism for the per-location fits, by default ``-1``.
    keep_models : bool | str | Path, optional
        Whether to retain local fitted objects (kept here for API symmetry
        with the supervised bases; subclasses may ignore).  By default
        ``False``.
    temp_folder : str | None, optional
        Joblib temp folder for memmapping, by default ``None``.
    batch_size : int | None, optional
        Batched dispatch size, by default ``None``.
    coplanar : {"raise", "jitter", "clique"}, optional
        Coplanar-point handling for adaptive kernels, by default ``"raise"``.
    verbose : bool, optional
        Progress logging, by default ``False``.
    **kwargs
        Additional keyword arguments forwarded to the per-local estimator
        instantiation (subclass-dependent).

    Attributes
    ----------
    components_ : numpy.ndarray
        Local loadings, shape ``(n_locations, n_features, n_components)``.
    scores_ : numpy.ndarray
        Focal-point projections, shape ``(n_locations, n_components)``.
    local_means_ : numpy.ndarray
        Weighted local means, shape ``(n_locations, n_features)``.
    winning_variable_ : numpy.ndarray
        Index of the maximum-absolute-loading variable per component,
        shape ``(n_locations, n_components)``.
    """

    def __init__(
        self,
        *,
        n_components: int | None = None,
        bandwidth: float | None = None,
        fixed: bool = False,
        kernel: Literal[
            "triangular",
            "parabolic",
            "bisquare",
            "tricube",
            "cosine",
            "boxcar",
        ]
        | Callable = "bisquare",
        include_focal: bool = False,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
        # ``strict`` is accepted so that BandwidthSearch (which passes it to
        # every model it creates) does not raise a TypeError.  Decompositions
        # have no notion of invariant y, so the value is always ignored.
        strict: bool | None = False,  # noqa: ARG002
        **kwargs,
    ):
        super().__init__(
            model=None,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            strict=False,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )
        self.n_components = n_components

    @property
    def _is_decomposition(self) -> bool:
        return True

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002 — sklearn API compatibility
        geometry: gpd.GeoSeries | None = None,
    ) -> "BaseDecomposition":
        """Fit local decompositions, one per focal location.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : None
            Ignored.  Present only so the estimator slots into scikit-learn
            pipelines and ``check_estimator``.
        geometry : geopandas.GeoSeries | None
            Point geometries.  Required unless ``self.graph`` was provided.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        Subclasses must implement :meth:`_fit_local` and return, for each
        focal location, a tuple of the form
        ``(name, components, scores, local_mean, winning_variable)``.
        """
        self._start = time()
        self._validate_fit_inputs(X, None, geometry)
        if self.n_components is not None and self.n_components > X.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} cannot be greater than the "
                f"number of features={X.shape[1]}."
            )
        self.geometry = geometry
        self.n_features_in_ = X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Building weights")
        weights = self.graph if self.graph is not None else self._build_weights()

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Weights ready")
        self._setup_model_storage()

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Fitting local decompositions")
        training_output = self._fit_models_batch(X, None, weights)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Local fits complete")

        (
            self._names,
            components,
            eigenvalues,
            scores,
            local_means,
        ) = zip(*training_output, strict=False)

        self._components = np.stack(components)
        self._eigenvalues = np.stack(eigenvalues)
        self._scores = np.stack(scores)
        self._local_means = np.stack(local_means)
        self._names = np.asarray(self._names)

        return self

    @property
    def components_(self) -> np.ndarray:
        """Local loadings, shape ``(n_locations, n_features, n_components)``."""
        return self._components

    @property
    def explained_variance_(self) -> np.ndarray:
        """Local eigenvalues. Shape ``(n, q)``."""
        return self._eigenvalues

    @property
    def condition_number_(self) -> pd.Series:
        """Condition number (max/min eigenvalue) per location."""
        cond = self._eigenvalues[:, 0] / (self._eigenvalues[:, -1] + 1e-10)
        return pd.Series(cond, index=self._names)

    @property
    def scores_(self) -> np.ndarray:
        """Focal-point projections onto the local components."""
        return self._scores

    @property
    def local_means_(self) -> np.ndarray:
        """Weighted local mean per location, shape ``(n_locations, n_features)``."""
        return self._local_means

    @property
    def winning_variable_(self) -> pd.Series:
        """Index of the maximum-absolute-loading variable per component."""
        idx = np.argmax(np.abs(self._components[:, :, 0]), axis=1)
        return pd.Series(self.feature_names_in_[idx], index=self._names)

    def transform(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries | None = None,
    ) -> np.ndarray:
        """Project ``X`` onto local components via nearest-neighbour lookup.

        Each row of ``X`` is mapped to the closest training geometry; the
        corresponding local mean is subtracted and the result is projected
        onto that location's components.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix for new observations.
        geometry : geopandas.GeoSeries | None
            Geometries for ``X``.  When ``None``, the stored in-sample
            :attr:`scores_` is returned (in-sample transform).

        Returns
        -------
        numpy.ndarray
            Shape ``(len(X), n_components)``.
        """
        if geometry is None:
            raise ValueError("geometry is required to transform data.")

        if not isinstance(self.geometry, gpd.GeoSeries):
            raise ValueError("Geometry needs to be specified at fit time to transform.")
        self._validate_geometry(geometry)

        indices = self.geometry.sindex.nearest(geometry, return_all=False)[1]
        X_values = X.values if hasattr(X, "values") else np.asarray(X)
        n_components = self._components.shape[2]
        out = np.empty((len(X_values), n_components))
        for i, idx in enumerate(indices):
            loc_components = self._components[idx]
            loc_mean = self._local_means[idx]
            out[i] = (X_values[i] - loc_mean) @ loc_components
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002 — sklearn API compatibility
        geometry: gpd.GeoSeries | None = None,
        **fit_params,  # noqa: ARG002
    ) -> np.ndarray:
        """Fit the model and return the in-sample focal-point scores."""
        return self.fit(X, geometry=geometry).scores_
