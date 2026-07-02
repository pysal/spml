from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from scipy.spatial import KDTree
from sklearn.base import TransformerMixin

from ..base import _BaseModel, _kernel_functions

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
        if geometry is not None:
            self._validate_geometry_alignment(X, geometry)
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
        self._name_to_position = {
            name: pos for pos, name in enumerate(self._names.tolist())
        }

        if self.fit_global_model:
            self._fit_global_model(X, None)
        elif hasattr(self, "global_model"):
            delattr(self, "global_model")

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
        eigenvalues = getattr(self, "_all_eigenvalues", self._eigenvalues)
        cond = eigenvalues[:, 0] / (eigenvalues[:, -1] + 1e-10)
        return pd.Series(cond, index=self._names)

    @property
    def scores_(self) -> pd.DataFrame:
        """Focal observations projected onto local components."""
        cols = [f"PC{i}" for i in range(self._scores.shape[1])]
        return pd.DataFrame(self._scores, index=self._names, columns=cols)

    @property
    def winning_variable_(self) -> pd.Series:
        """Feature with largest absolute loading on the first component."""
        loadings = np.abs(self._components[:, :, 0])
        valid = ~np.isnan(loadings).all(axis=1)
        idx = np.argmax(np.where(np.isnan(loadings), -np.inf, loadings), axis=1)
        labels = self.feature_names_in_[idx].astype(object)
        labels[~valid] = pd.NA
        return pd.Series(labels, index=self._names)

    def _prepare_transform_nearest(self, geometry: gpd.GeoSeries) -> np.ndarray:
        """Map target geometries to the nearest fitted decomposition."""
        self._validate_geometry(geometry)

        if not isinstance(self.geometry, gpd.GeoSeries):
            raise ValueError("Geometry needs to be specified at fit time to transform.")

        indices_array = self.geometry.sindex.nearest(geometry, return_all=False)[1]
        return self._names[indices_array.flatten()]

    def _prepare_transform_neighborhoods(
        self, geometry: gpd.GeoSeries, bandwidth: float | int | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Prepare local decomposition ids and weights for transformation."""
        self._validate_geometry(geometry)

        if not (
            (isinstance(self.bandwidth, Real) or bandwidth)
            and isinstance(self.geometry, gpd.GeoSeries)
        ):
            raise ValueError(
                "Bandwidth and geometry need to be specified to enable transform."
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
                geometry, predicate="dwithin", distance=bw
            )
            local_ids = self._names[indices_array.flatten()]
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

            input_ids = np.repeat(np.arange(len(geometry)), bw)
            local_ids = self._names[indices_array.flatten()]
            distances = distances.flatten()

            kernel_bandwidth = (
                pd.Series(distances).groupby(input_ids).transform("max") + 1e-6
            )
            distance = kernel(distances, kernel_bandwidth)

        split_indices = np.where(np.diff(input_ids))[0] + 1
        local_model_ids = np.split(np.asarray(local_ids), split_indices)
        distances = np.split(np.asarray(distance), split_indices)

        return local_model_ids, distances

    def _transform_local_nearest(self, x_: np.ndarray, local_id) -> np.ndarray:
        """Project a single observation using the nearest local decomposition."""
        idx = self._name_to_position[local_id]
        return (x_ - self._local_means[idx]) @ self._components[idx]

    def _transform_local_ensemble(
        self,
        x_: np.ndarray,
        local_ids: np.ndarray,
        distances_: np.ndarray,
    ) -> np.ndarray:
        """Project a single observation using a weighted ensemble of local fits."""
        projections = []
        valid_weights = []

        for local_id, weight in zip(local_ids, distances_, strict=True):
            idx = self._name_to_position[local_id]
            components = self._components[idx]
            if np.isnan(components).any():
                continue
            projections.append((x_ - self._local_means[idx]) @ components)
            valid_weights.append(weight)

        if not projections:
            return np.full(self._components.shape[2], np.nan)

        return np.average(np.vstack(projections), axis=0, weights=valid_weights)

    @staticmethod
    def _validate_geometry_alignment(
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
    ) -> None:
        """Require geometry to align with the feature matrix by index and order."""
        if hasattr(X, "index") and not X.index.equals(geometry.index):
            raise ValueError(
                "X and geometry must have matching indexes in the same order."
            )

    def _prepare_transform_data(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
    ) -> np.ndarray:
        """Validate transform inputs and align columns to the fitted feature order."""
        self._validate_geometry_alignment(X, geometry)

        if isinstance(X, pd.DataFrame):
            expected = pd.Index(self.feature_names_in_)
            missing = expected.difference(X.columns)
            extra = X.columns.difference(expected)
            if len(missing) or len(extra):
                problems = []
                if len(missing):
                    problems.append(f"missing columns: {missing.tolist()}")
                if len(extra):
                    problems.append(f"unexpected columns: {extra.tolist()}")
                detail = "; ".join(problems)
                raise ValueError(
                    "Transform input columns must match the fitted feature set; "
                    f"{detail}."
                )
            return X.loc[:, expected].to_numpy()

        X_values = np.asarray(X)
        if X_values.shape[1] != len(self.feature_names_in_):
            raise ValueError(
                "Transform input must have the same number of features used "
                f"during fit. Expected {len(self.feature_names_in_)}, got "
                f"{X_values.shape[1]}."
            )
        return X_values

    def transform(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries | None = None,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
    ) -> pd.DataFrame:
        """Project observations onto fitted local components.

        Transformation can use either the nearest fitted local decomposition or
        a weighted combination of local decompositions within a bandwidth, in
        the same spirit as estimator prediction.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix to transform.
        geometry : geopandas.GeoSeries | None
            Point geometries aligned to ``X``. Required for transformation.
        bandwidth : {"nearest"} | int | float | None, optional
            Transformation mode. ``"nearest"`` uses the nearest fitted local
            decomposition. Numeric values use a weighted ensemble of local
            decompositions within the supplied bandwidth. ``None`` reuses the
            bandwidth from fit.

        Returns
        -------
        pandas.DataFrame
            Transformed values with one column per retained component.
        """
        if geometry is None:
            raise ValueError("geometry is required to transform data.")
        X_values = self._prepare_transform_data(X, geometry)
        columns = [f"PC{i}" for i in range(self._components.shape[2])]

        transformed = []
        if bandwidth == "nearest":
            local_ids = self._prepare_transform_nearest(geometry)
            for x_, local_id in zip(X_values, local_ids, strict=True):
                transformed.append(self._transform_local_nearest(x_, local_id))
        else:
            local_ids, distances = self._prepare_transform_neighborhoods(
                geometry, bandwidth=bandwidth
            )
            for x_, ids, dist in zip(X_values, local_ids, distances, strict=True):
                transformed.append(self._transform_local_ensemble(x_, ids, dist))

        return pd.DataFrame(transformed, index=X.index, columns=columns)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002
        geometry: gpd.GeoSeries | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Fit the model and return in-sample local component scores."""
        return self.fit(X, y=y, geometry=geometry, **fit_params).scores_
