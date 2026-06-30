from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from sklearn.base import TransformerMixin

from ..base import _BaseModel

__all__ = ["BaseDecomposition"]


class BaseDecomposition(TransformerMixin, _BaseModel):
    """Generic geographically weighted decomposition meta-estimator.

    This class provides the shared machinery for decomposition estimators that
    fit one local model per focal observation using spatially varying sample
    weights. Subclasses provide the local decomposition routine; the base class
    handles spatial weights, batching, and result assembly.

    Notes
    -----
    - Decomposition estimators fit from ``X`` and spatial weights; ``y`` is
      not used.
    - Only point geometries are supported.

    Parameters
    ----------
    n_components : int | None, optional
        Number of components retained in each local fit. If ``None``, the
        subclass decides how many components to keep. By default ``None``.
    bandwidth : float | int | None, optional
        Bandwidth used to define local neighborhoods.

        - If ``fixed=True``, this is a distance threshold.
        - If ``fixed=False``, this is the number of nearest neighbors used
          for each local fit.

        If ``graph`` is provided, ``bandwidth`` is ignored.
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest
        neighbor) bandwidth, by default ``False``.
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default
        ``"bisquare"``.
    include_focal : bool, optional
        Whether to include the focal observation in its own local fit, by
        default ``False``.
    graph : libpysal.graph.Graph | None, optional
        Precomputed spatial graph. If provided, it is used directly and
        ``bandwidth``, ``fixed``, ``kernel``, and ``include_focal`` are
        ignored.
    n_jobs : int, optional
        Number of jobs to run in parallel. ``-1`` uses all processors.
    fit_global_model : bool, optional
        Determines if the global baseline decomposition shall be fitted
        alongside the geographically weighted decomposition, by default
        ``True``.
    keep_models : bool | str | Path, optional
        Whether to retain local fitted objects, by default ``False``.
        Subclasses may ignore this parameter.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing
        memory with worker processes. Passed to ``joblib.Parallel``, by
        default ``None``.
    batch_size : int | None, optional
        Number of models to process in each batch, by default ``None``.
    coplanar : {"raise", "jitter", "clique"}, optional
        Method for handling coplanar points in adaptive kernels.
    verbose : bool, optional
        Whether to print progress information, by default ``False``.
    **kwargs
        Additional keyword arguments passed to subclass-specific local fitting.

    Attributes
    ----------
    components_ : pandas.DataFrame
        Local component loadings indexed by focal location, with MultiIndex
        columns ``(Component, Feature)``.
    explained_variance_ : pandas.DataFrame
        Local eigenvalues for retained components.
    scores_ : pandas.DataFrame
        Focal observations projected onto local components.
    local_means_ : pandas.DataFrame
        Weighted local means of ``X`` at each focal location.
    condition_number_ : pandas.Series
        Ratio of largest to smallest retained eigenvalue per location.
    winning_variable_ : pandas.Series
        Feature with the largest absolute loading on the first component.
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
        fit_global_model: bool = True,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        coplanar: Literal["raise", "jitter", "clique"] = "raise",
        verbose: bool = False,
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
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            coplanar=coplanar,
            verbose=verbose,
            **kwargs,
        )
        self.n_components = n_components

    @property
    def _requires_y(self) -> bool:
        """Whether the estimator requires a target vector during fitting."""
        return False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002
        geometry: gpd.GeoSeries | None = None,
    ) -> BaseDecomposition:
        """Fit one local decomposition model per focal location.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series | None
            Not used by decomposition estimators.
        geometry : geopandas.GeoSeries | None
            Point geometries aligned to ``X``. Required unless ``self.graph``
            was provided.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        Subclasses must implement :meth:`_fit_local` and return, for each
        focal location, a tuple of the form
        ``(name, components, eigenvalues, scores, local_mean)``.
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

        if self.fit_global_model:
            self._fit_global_model(X)

        return self

    @property
    def components_(self) -> pd.DataFrame:
        """Local component loadings with MultiIndex columns."""
        n, p, q = self._components.shape

        reshaped = np.transpose(self._components, (0, 2, 1)).reshape(n, q * p)

        comp_labels = [f"PC{i}" for i in range(q)]
        columns = pd.MultiIndex.from_product(
            [comp_labels, self.feature_names_in_],
            names=["Component", "Feature"],
        )

        return pd.DataFrame(reshaped, index=self._names, columns=columns)

    @property
    def explained_variance_(self) -> pd.DataFrame:
        """Local eigenvalues for retained components."""
        cols = [f"PC{i}" for i in range(self._eigenvalues.shape[1])]
        return pd.DataFrame(self._eigenvalues, index=self._names, columns=cols)

    @property
    def condition_number_(self) -> pd.Series:
        """Local condition number as largest/smallest eigenvalue."""
        cond = self._eigenvalues[:, 0] / (self._eigenvalues[:, -1] + 1e-10)
        return pd.Series(cond, index=self._names)

    @property
    def scores_(self) -> pd.DataFrame:
        """Focal observations projected onto local components."""
        cols = [f"PC{i}" for i in range(self._scores.shape[1])]
        return pd.DataFrame(self._scores, index=self._names, columns=cols)

    @property
    def local_means_(self) -> pd.DataFrame:
        """Weighted local mean of ``X`` at each focal location."""
        return pd.DataFrame(
            self._local_means, index=self._names, columns=self.feature_names_in_
        )

    @property
    def winning_variable_(self) -> pd.Series:
        """Feature with largest absolute loading on the first component."""
        idx = np.argmax(np.abs(self._components[:, :, 0]), axis=1)
        return pd.Series(self.feature_names_in_[idx], index=self._names)

    def transform(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries | None = None,
    ) -> np.ndarray:
        """Project observations onto fitted local components.

        New observations are matched to the nearest fitted location before
        projection. This is a nearest-neighbor assignment, not a weighted
        interpolation across multiple local models.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix to transform.
        geometry : geopandas.GeoSeries | None
            Point geometries aligned to ``X``. Required for transformation.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_components)``.
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
        y: pd.Series | None = None,  # noqa: ARG002
        geometry: gpd.GeoSeries | None = None,
        **fit_params,  # noqa: ARG002
    ) -> np.ndarray:
        """Fit the model and return in-sample local component scores."""
        return self.fit(X, geometry=geometry).scores_
