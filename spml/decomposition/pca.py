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
    """Geographically Weighted Principal Components Analysis.

    Fits a local PCA at each spatial location via a kernel-weighted covariance
    matrix. Produces a surface of local eigenvectors, eigenvalues, and scores.
    Follows Harris, Brunsdon & Charlton (2011).

    Parameters
    ----------
    n_components : int | None
        Components to retain per location. None keeps all.
    bandwidth : float | int | None
        KNN count (fixed=False) or distance threshold (fixed=True).
    fixed : bool
        Adaptive KNN (False) or fixed distance (True). Default False.
    kernel : str | Callable
        Weight function. Default ``"bisquare"``.
    include_focal : bool
        Include the focal point in its own neighbourhood. Default True.
    graph : libpysal.graph.Graph | None
        Pre-built spatial weights. Overrides bandwidth/kernel if given.
    n_jobs : int
        Joblib parallelism. -1 uses all CPUs.
    fit_global_model : bool
        Also fit a global sklearn PCA as a baseline.
    sign_convention : {"first_positive", "max_abs", "none"}
        Eigenvector sign rule. ``"first_positive"`` matches GWmodel.

    Attributes
    ----------
    components_ : ndarray (n, p, q)
        Local eigenvectors.
    explained_variance_ratio_ : ndarray (n, q)
        Local fraction of variance per component.
    scores_ : ndarray (n, q)
        Local PC scores for each focal observation.
    local_means_ : ndarray (n, p)
        Geographically weighted mean at each location.
    winning_variable_ : pd.Series
        Variable with the highest PC1 loading at each location.
    condition_number_ : pd.Series
        Local covariance condition number (collinearity diagnostic).
    cv_score_ : float | None
        LOO reconstruction error. Set when ``cv=True`` is passed to ``fit``.

    Examples
    --------
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
        """Fit GWPCA at every spatial location.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Standardise before calling.
        y : None
            Ignored. Present for sklearn Pipeline compatibility.
        geometry : gpd.GeoSeries | None
            Point geometry. Required unless ``graph`` was supplied at init.
        cv : bool
            Compute LOO CV reconstruction error (Harris et al. 2011, §4.1).

        Returns
        -------
        self
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

    def _fit_global_model(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: ARG002
        from sklearn.decomposition import PCA

        self.global_model = PCA(n_components=self.n_components)
        self.global_model.fit(X)

    @property
    def explained_variance_ratio_(self) -> pd.DataFrame:
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
        """Fit one local PCA at focal point ``name``."""
        X_local = data.drop(columns=["_weight"]).values.astype(float)
        wt = data["_weight"].values.astype(float)

        if wt.sum() < 1e-12 or len(wt) < 2:
            p = X_local.shape[1]
            q = self.n_components or p
            nan_vec = np.full(p, np.nan)
            return [
                name,
                np.full((p, q), np.nan),
                np.full(q, np.nan),
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
                np.full(q, np.nan),
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
        """LOO cross-validation reconstruction error (Harris et al. 2011, §4.1)."""
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
        """Monte Carlo permutation test for eigenvalue nonstationarity
        (Harris 2011, §4.2).

        Returns dict with keys ``"true_sd"``, ``"permuted_sds"``, ``"p_value"``.
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
        """Flag locations with condition number above ``threshold`` (Harris 2011, §4.5).

        Returns DataFrame with columns: condition_number, pc1_evr,
        last_pc_evr, is_collinear.
        """
        if not hasattr(self, "_eigenvalues"):
            raise ValueError("Call fit() before identify_collinear_locations().")

        ev = np.abs(self._eigenvalues)  # (n, q)
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
