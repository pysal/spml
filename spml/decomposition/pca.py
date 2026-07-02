from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import libpysal.graph as graph
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._base import BaseDecomposition

__all__ = ["GWPCA"]


class GWPCA(BaseDecomposition):
    """Geographically weighted principal components analysis.

    Fits a local PCA at each spatial location using a kernel-weighted
    covariance matrix. Produces local components, eigenvalues, scores, and
    diagnostics for spatially varying feature structure.

    Parameters
    ----------
    n_components : int | None, optional
        Number of components to retain per location. If ``None``, all
        components are retained. By default ``None``.
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
        Whether to include the focal observation in its own neighborhood,
        by default ``True``.
    graph : libpysal.graph.Graph | None, optional
        Precomputed spatial graph. If provided, it is used directly and
        ``bandwidth``, ``fixed``, ``kernel``, and ``include_focal`` are
        ignored.
    n_jobs : int, optional
        Number of jobs to run in parallel. ``-1`` uses all processors, by
        default ``-1``.
    fit_global_model : bool, optional
        Whether to fit a global PCA baseline alongside the geographically
        weighted PCA, by default ``True``.
    keep_models : bool | str | Path, optional
        Whether to retain local fitted objects, by default ``False``.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing
        memory with worker processes. Passed to ``joblib.Parallel``, by
        default ``None``.
    batch_size : int | None, optional
        Number of models to process in each batch, by default ``None``.
    verbose : bool, optional
        Whether to print progress information, by default ``False``.
    sign_convention : {"first_positive", "max_abs", "none"}, optional
        Rule used to orient eigenvector signs consistently across locations.

    Attributes
    ----------
    components_ : pandas.DataFrame
        Local component loadings indexed by focal location.
    explained_variance_ : pandas.DataFrame
        Local eigenvalues for retained components.
    explained_variance_ratio_ : pandas.DataFrame
        Share of local variance explained by each retained component.
    scores_ : pandas.DataFrame
        Local component scores for each focal observation.
    winning_variable_ : pandas.Series
        Feature with largest absolute loading on the first component.
    condition_number_ : pandas.Series
        Local covariance condition number.
    cv_score_ : float | None
        Leave-one-out reconstruction error when ``cv=True`` in :meth:`fit`.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spml.decomposition import GWPCA

    >>> gdf = gpd.read_file(get_path("geoda.guerry")).set_geometry(lambda g: g.centroid)
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth", "Donatns", "Infants"]]
    >>> X = (X - X.mean()) / X.std()
    >>> model = GWPCA(n_components=3, bandwidth=50).fit(X, geometry=gdf.geometry)
    >>> model.explained_variance_ratio_.shape
    (85, 3)
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        bandwidth: float | None = None,
        fixed: bool = False,
        kernel: Literal[
            "triangular", "parabolic", "bisquare", "tricube", "cosine", "boxcar"
        ]
        | Callable = "bisquare",
        include_focal: bool = True,
        graph: graph.Graph | None = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
        sign_convention: Literal[
            "first_positive", "max_abs", "none"
        ] = "first_positive",
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
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
            verbose=verbose,
            **kwargs,
        )
        self.n_components = n_components
        self.sign_convention = sign_convention

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002
        geometry: gpd.GeoSeries | None = None,
        cv: bool = False,
    ) -> GWPCA:
        """Fit a local PCA model at each spatial location.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Standardize before calling.
        y : pd.Series | None
            Not used by GWPCA.
        geometry : gpd.GeoSeries | None
            Point geometries aligned to ``X``. Required unless ``graph``
            was supplied during initialization.
        cv : bool
            Compute leave-one-out reconstruction error.

        Returns
        -------
        self
            Fitted estimator.
        """
        super().fit(X, y=None, geometry=geometry)

        self._all_eigenvalues = self._eigenvalues.copy()
        q = self.n_components
        if q is not None:
            self._eigenvalues = self._eigenvalues[:, :q]

        self.cv_score_ = None
        if cv:
            self.cv_score_ = self._compute_cv_score(X)

        return self

    def _fit_global_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,  # noqa: ARG002
    ):
        """Fit a global PCA baseline model."""
        from sklearn.decomposition import PCA

        self.global_model = PCA(n_components=self.n_components)
        self.global_model.fit(X)

    @property
    def explained_variance_ratio_(self) -> pd.DataFrame:
        """Share of total local variance explained by each component."""
        totals = self._all_eigenvalues.sum(axis=1, keepdims=True)
        evr = np.where(totals > 0, self._eigenvalues / totals, 0.0)
        cols = [f"PC{i}" for i in range(evr.shape[1])]
        return pd.DataFrame(evr, index=self._names, columns=cols)

    def _fit_local(
        self,
        model,  # noqa: ARG002
        data: pd.DataFrame,
        name,
        focal_x: np.ndarray,
        model_kwargs: dict,  # noqa: ARG002
    ) -> list:
        """Fit the local PCA for a single focal location."""
        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)

        if wt.sum() < 1e-12 or len(wt) < 2:
            p = X_local.shape[1]
            q = self.n_components or p
            nan_vec = np.full(p, np.nan)
            return [
                name,
                np.full((p, q), np.nan),
                np.full(p, np.nan),
                np.full(q, np.nan),
                nan_vec,
            ]

        weighted_mean = np.average(X_local, axis=0, weights=wt)
        cov = np.cov(X_local.T, aweights=wt, ddof=0)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 0, None)

        if eigenvalues.max() < 1e-12:
            p = X_local.shape[1]
            q = self.n_components or p
            nan_vec = np.full(p, np.nan)
            return [
                name,
                np.full((p, q), np.nan),
                np.full(p, np.nan),
                np.full(q, np.nan),
                nan_vec,
            ]

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        sc = self.sign_convention
        if sc == "first_positive":
            signs = np.sign(eigenvectors[0, :])
            signs[signs == 0] = 1.0
            eigenvectors = eigenvectors * signs
        elif sc == "max_abs":
            dom = np.argmax(np.abs(eigenvectors), axis=0)
            signs = np.sign(eigenvectors[dom, np.arange(eigenvectors.shape[1])])
            eigenvectors = eigenvectors * signs

        q = self.n_components
        if q is not None:
            eigenvectors = eigenvectors[:, :q]

        focal_score = (focal_x - weighted_mean) @ eigenvectors

        return [name, eigenvectors, eigenvalues, focal_score, weighted_mean]

    def _compute_cv_score(self, X: pd.DataFrame) -> float:
        """Compute leave-one-out reconstruction error."""
        weights = self.graph if self.graph is not None else self._build_weights()

        adjacency = weights._adjacency
        X_vals = X.values.astype(float)
        q = self.n_components

        def _cv_local(focal_id, focal_x):
            nbr_weights = adjacency.loc[focal_id].copy()
            if focal_id in nbr_weights.index:
                nbr_weights[focal_id] = 0.0

            use_mask = nbr_weights > 0
            if use_mask.sum() < 2:
                return np.nan

            nbr_ids = nbr_weights.index[use_mask]
            wt = nbr_weights[use_mask].values.astype(float)

            loc_positions = [X.index.get_loc(idx) for idx in nbr_ids]
            X_nbr = X_vals[loc_positions]

            w_mean = np.average(X_nbr, axis=0, weights=wt)
            cov = np.cov(X_nbr.T, aweights=wt, ddof=0)

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            if eigvals.max() < 1e-12:
                return np.nan
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            if q is not None:
                eigvecs = eigvecs[:, :q]

            x_i = focal_x - w_mean
            reconstructed = x_i @ eigvecs @ eigvecs.T
            return float(np.sum((x_i - reconstructed) ** 2))

        cv_scores = Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(_cv_local)(fid, X_vals[X.index.get_loc(fid)]) for fid in self._names
        )

        valid = [s for s in cv_scores if not np.isnan(s)]
        return float(np.sum(valid)) if valid else np.inf

    def stationarity_test(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        component: int = 0,
        n_permutations: int = 99,
        random_state: int | None = None,
    ) -> dict:
        """Monte Carlo permutation test for eigenvalue nonstationarity.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used for each permutation refit.
        geometry : gpd.GeoSeries
            Point geometries aligned to ``X``.
        component : int
            Zero-based component index to test.
        n_permutations : int
            Number of random geometry permutations.
        random_state : int | None
            Random seed used for permutation sampling.

        Returns
        -------
        dict
            Dictionary with keys ``"true_sd"``, ``"permuted_sds"``, and
            ``"p_value"``.
        """
        rng = np.random.default_rng(random_state)

        true_sd = float(np.nanstd(self._eigenvalues[:, component]))

        permuted_sds = []
        for _ in range(n_permutations):
            perm_geom = geometry.iloc[rng.permutation(len(geometry))].set_axis(
                geometry.index
            )
            perm_model = GWPCA(
                n_components=self.n_components,
                bandwidth=self.bandwidth,
                fixed=self.fixed,
                kernel=self.kernel,
                include_focal=self.include_focal,
                n_jobs=self.n_jobs,
                fit_global_model=False,
            ).fit(X, geometry=perm_geom)
            permuted_sds.append(float(np.nanstd(perm_model._eigenvalues[:, component])))

        permuted_sds = np.array(permuted_sds)
        p_value = float(np.mean(permuted_sds >= true_sd))

        return {
            "true_sd": true_sd,
            "permuted_sds": permuted_sds,
            "p_value": p_value,
        }

    def identify_collinear_locations(
        self,
        threshold: float = 30.0,
    ) -> pd.DataFrame:
        """Flag locations with high local collinearity.

        Parameters
        ----------
        threshold : float
            Condition-number threshold used to flag collinear locations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``condition_number``, ``pc1_evr``,
            ``last_pc_evr``, and ``is_collinear``.
        """
        if not hasattr(self, "_eigenvalues"):
            raise ValueError("Call fit() before identify_collinear_locations().")

        ev = np.abs(getattr(self, "_all_eigenvalues", self._eigenvalues))
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(
                ev[:, -1] > 0,
                ev[:, 0] / ev[:, -1],
                np.inf,
            )
        total = ev.sum(axis=1)
        pc1_evr = np.where(total > 0, ev[:, 0] / total, np.nan)
        last_evr = np.where(total > 0, ev[:, -1] / total, np.nan)

        return pd.DataFrame(
            {
                "condition_number": cond,
                "pc1_evr": pc1_evr,
                "last_pc_evr": last_evr,
                "is_collinear": cond > threshold,
            },
            index=self._names,
        )
