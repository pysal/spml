"""Geographically weighted matrix decomposition.

This module adds an *unsupervised* branch to the geographically weighted
framework.  Where :mod:`spml.linear_model` and :mod:`spml.ensemble` fit one
supervised ``scikit-learn`` estimator per focal location, the estimators here
operate on a feature matrix ``X`` alone and expose a *local* decomposition of
its (spatially weighted) covariance structure.

Currently implemented
---------------------
:class:`GWPCA`
    Geographically weighted principal components analysis, following
    :cite:t:`harris2011` and :cite:t:`harris2015`.  At every location a
    kernel-weighted covariance (or correlation) matrix is formed and
    eigendecomposed, giving spatially varying loadings and explained variance.

Design notes / open questions (see gh#79)
----------------------------------------
* ``sklearn.decomposition.PCA`` has no ``sample_weight``; the weighted
  covariance is therefore built and eigendecomposed directly with NumPy.
* Eigenvectors are only defined up to sign, and eigenvalue *order* can swap
  between neighbouring locations.  A per-location sign convention is applied
  (largest-magnitude loading forced positive); cross-location alignment
  (e.g. Procrustes rotation toward the global solution) is left as a TODO.
* The number of meaningful components is itself spatially varying.  For now
  ``n_components`` is fixed; adaptive selection is future work.
* ``BaseDecomposition`` vs ``BaseGWDecomposition``-style naming (matching the
  ``BaseClassifier`` / ``BaseRegressor`` precedent) and whether a ``model=``
  parameter is kept are still to be settled with the API owners.

Deliberate parity with ``GWmodel::gwpca`` (R), including one of its bugs
-------------------------------------------------------------------------
This implementation intentionally reproduces the *exact* numerical behaviour
of the reference R implementation (package ``GWmodel``, function ``gwpca``),
not just its published methodology:

* ``local_scores_`` matches ``GWmodel``'s ``gwpca.scores``: a score for
  *every neighbour* in each location's neighbourhood, computed from the
  **raw, uncentered** (only globally standardized if ``standardize=True``)
  feature values dotted with the local loadings — *not* a single
  mean-centered score for the focal observation. See
  ``GWPCA._decompose_local``.
* **Known upstream bug, replicated on purpose (documented here "for later
  reference" per gh#79):** ``GWmodel::gwpca`` normalizes every location's
  reported eigenvalues by ``sum(wt)`` taken from whichever neighbourhood was
  *last processed in the input data's row order* (a stray/reused loop
  variable in the R source), not that location's own neighbourhood
  weight-sum. This is only harmless when every neighbourhood has the same
  weight-sum, which in practice means an adaptive (KNN) ``boxcar`` kernel.
  Any distance-decay kernel (including the default adaptive ``bisquare``,
  whose weight-sums depend on how the neighbours are spaced) or a
  fixed-distance kernel over irregularly spaced points silently mis-scales the
  reported eigenvalue *magnitudes* for every location except the last one
  processed. Loadings
  (eigenvectors) are never affected, and neither is
  ``explained_variance_ratio_``/``proportion_total_var_`` (the shared divisor
  cancels out of any within-row ratio) — only ``local_eigenvalues_`` and
  ``explained_variance_`` carry it, and, as a direct consequence, their
  values depend on ``X``'s row order at fit time. See
  ``GWPCA._scale_eigenvalues`` (the base class divides by each location's own
  weight sum), and
  ``test_gwpca_eigenvalue_scale_depends_on_row_order`` in the test suite for
  a concrete demonstration.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from sklearn.decomposition import PCA
from sklearn.utils.parallel import Parallel, delayed

from .base import _BaseModel

__all__ = ["BaseGWDecomposition", "GWPCA"]


class BaseGWDecomposition(_BaseModel):
    """Generic geographically weighted decomposition meta-estimator.

    Holds the machinery shared by every local decomposition, in the same way
    :class:`~spml.base.BaseRegressor` and :class:`~spml.base.BaseClassifier`
    do for the supervised estimators: input validation, optional global
    standardization, building the spatial weights, the parallel loop over
    neighbourhoods, local model storage, and assembly of the generic
    per-location outputs. For each neighbourhood, :meth:`_fit_local` computes
    the kernel weights and the weighted local mean, then hands the data to
    :meth:`_decompose_local`, which concrete subclasses such as :class:`GWPCA`
    must implement.

    Parameters
    ----------
    model : type
        Global ``scikit-learn`` decomposition class, used for the global
        baseline when ``fit_global_model=True`` (e.g.
        :class:`sklearn.decomposition.PCA`). It is instantiated as
        ``model(n_components=n_components, **kwargs)``; the local fits never
        use it.
    n_components : int
        Number of components to retain per location.
    bandwidth : float | int | None
        Bandwidth for defining neighbourhoods.  Distance threshold when
        ``fixed=True``; number of nearest neighbours when ``fixed=False``.
        Ignored when ``graph`` is provided.
    fixed : bool, optional
        Distance-based (``True``) vs adaptive nearest-neighbour (``False``)
        bandwidth, by default ``False``.
    kernel : str | Callable, optional
        Kernel used to weight observations, by default ``"bisquare"``.
    include_focal : bool, optional
        Include the focal observation in its own neighbourhood, by default
        ``True`` (the usual choice for GWPCA).
    standardize : bool, optional
        When ``True``, globally z-score the features once (mean and population
        sd across all observations) before forming local weighted covariance
        matrices. ``GWmodel::gwpca`` has no scaling argument of its own; this
        reproduces its documented workflow of calling ``scale()`` on the data
        before ``gwpca()``, except that ``scale()`` uses the sample sd
        (``n - 1``). The two choices differ only by a common factor on every
        feature, so loadings are identical, while ``scale()``-based
        eigenvalues are smaller by a factor ``(n - 1) / n`` and scores by its
        square root. The result is *not* a true local correlation matrix (the
        local diagonal is not forced to 1); it only converges to one in the
        large-bandwidth limit. ``False`` decomposes the raw local covariance
        matrix. By default ``True``.
    graph : libpysal.graph.Graph, optional
        Precomputed spatial interaction graph.  When given, ``bandwidth``,
        ``fixed``, ``kernel`` and ``include_focal`` are ignored.
    n_jobs : int, optional
        Parallel jobs, ``-1`` for all processors, by default ``-1``.
    fit_global_model : bool, optional
        Also fit a global ``model`` baseline, by default ``True``.
    keep_models : bool | str | Path, optional
        Persist per-location means and loadings (required for
        :meth:`transform`), by default ``False``.
    batch_size : int | None, optional
        Process neighbourhoods in batches of this size, by default ``None``.
    temp_folder : str | None, optional
        ``joblib`` memmap folder, by default ``None``.
    coplanar : {"raise", "jitter", "clique"}, optional
        Coplanar-point handling for adaptive kernels, by default ``"raise"``.
    verbose : bool, optional
        Print progress, by default ``False``.
    **kwargs
        Passed to the global ``model`` only.

    Attributes
    ----------
    components_ : numpy.ndarray
        Local components, shape ``(n_samples, n_components, n_features)``.
        ``components_[i]`` is oriented like ``sklearn`` — rows are components,
        columns are features.
    local_eigenvalues_ : pandas.DataFrame
        All local eigenvalues (descending), shape ``(n_samples, n_features)``,
        scaled by :meth:`_scale_eigenvalues`.
    explained_variance_ : pandas.DataFrame
        Local eigenvalues of the retained components,
        shape ``(n_samples, n_components)``.
    explained_variance_ratio_ : pandas.DataFrame
        Local proportion of variance per retained component.
    global_model_ : sklearn estimator | None
        Fitted global baseline (``None`` when ``fit_global_model=False``).
    """

    #: Prefix of the component column labels (``C1``, ``C2``, ...).
    _component_prefix: str = "C"

    def __init__(
        self,
        model,
        n_components: int = 2,
        *,
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
        include_focal: bool = True,
        standardize: bool = True,
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
            model=model,
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
        self.standardize = standardize

    # ------------------------------------------------------------------ fit ---
    def fit(
        self, X: pd.DataFrame, geometry: gpd.GeoSeries | None = None
    ) -> BaseGWDecomposition:
        """Fit local decompositions.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        geometry : geopandas.GeoSeries | None
            Point geometries of the observations.  Required unless ``graph``
            was supplied; required in any case to enable :meth:`transform`.
        """
        X = self._validate_decomposition_inputs(X, geometry)
        self.geometry = geometry
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        if not (1 <= self.n_components <= self.n_features_in_):
            raise ValueError(
                f"n_components must be in [1, {self.n_features_in_}], "
                f"got {self.n_components}."
            )

        if self.standardize:
            Xs, self._global_mean_, self._global_sd_ = self._standardize(
                X.to_numpy(dtype=float), return_moments=True
            )
            X_fit = pd.DataFrame(Xs, columns=X.columns, index=X.index)
        else:
            self._global_mean_ = None
            self._global_sd_ = None
            X_fit = X

        weights = self.graph if self.graph is not None else self._build_weights()

        self._setup_model_storage()
        training_output = self._fit_local_batch(X_fit, weights)
        self._collect(training_output, X_fit.index)

        if self.fit_global_model:
            self.global_model_ = self.model(
                n_components=self.n_components, **self._model_kwargs
            )
            self.global_model_.fit(X_fit.to_numpy())
        else:
            self.global_model_ = None

        return self

    # -------------------------------------------------------------- helpers ---
    def _validate_decomposition_inputs(
        self, X: pd.DataFrame, geometry: gpd.GeoSeries | None
    ) -> pd.DataFrame:
        """Structural / spatial checks; a ``y``-free cousin of
        :meth:`_BaseModel._validate_fit_inputs`."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))

        if self.graph is None and geometry is None:
            raise ValueError("Either geometry or graph must be provided.")

        if geometry is not None:
            if len(X) != len(geometry):
                raise ValueError(
                    f"X and geometry must have the same length. "
                    f"Got {len(X)} and {len(geometry)}."
                )
            self._validate_geometry(geometry)

        if self.graph is None:
            bw = self.bandwidth
            if bw is not None:
                if (
                    not np.isscalar(bw)
                    or pd.isna(bw)
                    or not isinstance(bw, Real)
                    or bw <= 0
                ):
                    raise ValueError("Bandwidth must be a positive scalar number.")
                if not self.fixed and not isinstance(bw, Integral):
                    raise ValueError(
                        "Adaptive bandwidth (fixed=False) must be an integer."
                    )
            elif not callable(self.kernel):
                raise ValueError(
                    "kernel must be either a valid string or a callable function."
                )

        return X

    @staticmethod
    def _standardize(
        a: np.ndarray, return_moments: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Column-wise z-score with population sd (``ddof=0``).

        Zero-variance columns get ``sd = 1`` so they centre to 0 instead of
        dividing by zero. With ``return_moments=True`` also return the
        ``(mean, sd)`` used, so callers can reapply them to new data.
        """
        mean = a.mean(axis=0)
        sd = a.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        z = (a - mean) / sd
        return (z, mean, sd) if return_moments else z

    def _component_labels(self, n: int) -> list[str]:
        return [f"{self._component_prefix}{j + 1}" for j in range(n)]

    def _fit_local_batch(self, X: pd.DataFrame, weights: graph.Graph) -> list:
        """Iterate neighbourhoods and dispatch :meth:`_fit_local` in parallel.

        Mirrors :meth:`_BaseModel._batch_fit` without the ``_y`` column.
        """
        adjacency = weights._adjacency
        if self.batch_size:
            # TODO: batch like _BaseModel._fit_models_batch for large samples.
            warnings.warn(
                "batch_size is not yet honoured by BaseGWDecomposition; "
                "fitting all neighbourhoods at once.",
                stacklevel=2,
            )

        index = adjacency.index
        _weight = adjacency.values

        data = X.loc[index.get_level_values(1)].copy()
        data["_weight"] = _weight
        grouper = data.groupby(index.get_level_values(0), sort=False)

        return Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(self._fit_local)(group, name) for name, group in grouper
        )

    def _fit_local(self, data: pd.DataFrame, name: Hashable) -> tuple:
        """Fit a single neighbourhood.

        Computes the kernel weights and the weighted local mean, centres the
        data, and delegates the decomposition itself to
        :meth:`_decompose_local`.

        Returns ``(name, components, eigvals_raw, wsum, neighbor_names,
        local_extra, stored)``, where ``components``, ``eigvals_raw`` and
        ``local_extra`` come from :meth:`_decompose_local` and ``stored`` is
        the payload kept for :meth:`transform` when ``keep_models`` is truthy,
        else ``None``.
        """
        w = data["_weight"].to_numpy(dtype=float)
        Xn = data.drop(columns="_weight").to_numpy(dtype=float)
        neighbor_names = data.index.to_numpy()

        wsum = w.sum()
        mean = (w[:, None] * Xn).sum(axis=0) / wsum
        Xc = Xn - mean

        components, eigvals_raw, local_extra = self._decompose_local(Xn, Xc, w)

        stored = None
        if self.keep_models:
            payload = {
                "mean": mean,
                "sd": self._global_sd_ if self.standardize else None,
                "components": components[: self.n_components],
            }
            stored = self._store_model(payload, name)

        return (
            name,
            components,
            eigvals_raw,
            wsum,
            neighbor_names,
            local_extra,
            stored,
        )

    def _decompose_local(
        self, Xn: np.ndarray, Xc: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Decompose one neighbourhood. Implemented by subclasses.

        Parameters
        ----------
        Xn : numpy.ndarray
            Neighbourhood feature values, shape ``(n_neighbors, n_features)``
            (globally standardized if ``standardize=True``).
        Xc : numpy.ndarray
            ``Xn`` centred on the kernel-weighted local mean.
        w : numpy.ndarray
            Kernel weights, shape ``(n_neighbors,)``.

        Returns
        -------
        components : numpy.ndarray
            All local components, shape ``(n_features, n_features)``, rows are
            components in descending eigenvalue order.
        eigvals_raw : numpy.ndarray
            Matching eigenvalues, shape ``(n_features,)``, before
            :meth:`_scale_eigenvalues` is applied.
        local_extra : object
            Any subclass-specific per-location output (e.g. scores); collected
            into ``self._local_extra`` for the subclass's ``fit`` to use.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _decompose_local."
        )

    def _scale_eigenvalues(
        self,
        eig_raw: np.ndarray,
        wsums: np.ndarray,
        names: list,  # noqa: ARG002 - used by subclass overrides
        row_order: pd.Index,  # noqa: ARG002 - used by subclass overrides
    ) -> np.ndarray:
        """Turn raw eigenvalues into reported ones.

        By default each location's eigenvalues are divided by that location's
        own kernel-weight sum, giving the eigenvalues of its weighted
        covariance matrix. Subclasses may override this (see
        :meth:`GWPCA._scale_eigenvalues`).
        """
        return eig_raw / wsums[:, None]

    def _collect(self, training_output: list, row_order: pd.Index) -> None:
        """Assemble per-location results into the generic public attributes.

        Subclass-specific outputs are left in ``self._neighbor_names`` and
        ``self._local_extra`` for the subclass's ``fit`` to turn into its own
        attributes.
        """
        names, comps, eigs_raw, wsums, neighbor_names, local_extra, stored = zip(
            *training_output, strict=False
        )
        self._names = list(names)
        k = self.n_components

        comps = np.stack(comps)  # (n_loc, n_features, n_features)
        self.components_ = comps[:, :k, :]

        eig = self._scale_eigenvalues(
            np.stack(eigs_raw), np.asarray(wsums, dtype=float), self._names, row_order
        )  # (n_loc, n_features)

        self.local_eigenvalues_ = pd.DataFrame(
            eig, index=self._names, columns=self._component_labels(eig.shape[1])
        )
        self.explained_variance_ = self.local_eigenvalues_.iloc[:, :k].copy()
        ratio = eig / eig.sum(axis=1, keepdims=True)
        self.explained_variance_ratio_ = pd.DataFrame(
            ratio[:, :k], index=self._names, columns=self._component_labels(k)
        )

        self._neighbor_names = list(neighbor_names)
        self._local_extra = list(local_extra)

        if self.keep_models:
            self._local_models = pd.Series(stored, index=self._names)

    # -------------------------------------------------------------- transform ---
    def transform(
        self,
        X: pd.DataFrame,
        geometry: gpd.GeoSeries,
        bandwidth: Literal["nearest"] | int | float | None = "nearest",
    ) -> pd.DataFrame:
        """Project new observations onto their local components.

        Not yet implemented.  Intended behaviour: for each new point pick the
        nearest fitted local model (``bandwidth="nearest"``) or a
        kernel-weighted ensemble of nearby local models, centre with that
        location's stored mean (and sd), and project onto the stored loadings.
        Requires ``keep_models`` to have been set at fit time.
        """
        raise NotImplementedError(
            "BaseGWDecomposition.transform is not implemented yet — see gh#79. "
            "Reuse _prepare_prediction_nearest / _prepare_prediction_neighborhoods "
            "from _BaseModel and the payloads stored in self._local_models."
        )


class GWPCA(BaseGWDecomposition):
    """Geographically weighted principal components analysis.

    At every location a kernel-weighted covariance matrix of the features
    (optionally globally standardized first, see ``standardize``) is formed and
    eigendecomposed, yielding spatially varying loadings, eigenvalues and
    component scores :cite:p:`harris2011`.

    Parameters are those of :class:`BaseGWDecomposition` except ``model``,
    which is fixed to :class:`sklearn.decomposition.PCA` for the global
    baseline.

    Attributes
    ----------
    components_ : numpy.ndarray
        Local loadings, shape ``(n_samples, n_components, n_features)``.
    local_loadings_ : pandas.DataFrame
        The same loadings as a tidy frame, MultiIndexed by
        ``(location, component)`` with one column per feature.
    explained_variance_ : pandas.DataFrame
        Local eigenvalues of the retained components,
        shape ``(n_samples, n_components)``. **Row-order dependent** — see
        the module-level "Deliberate parity with GWmodel::gwpca" note: the
        normalizing divisor is shared across all locations, taken from
        whichever neighbourhood is last in ``X``'s row order at fit time,
        replicating an upstream ``GWmodel::gwpca`` bug.
    explained_variance_ratio_ : pandas.DataFrame
        Local proportion of variance per retained component. Row-order
        *invariant* — the shared divisor above cancels out of this ratio.
    local_eigenvalues_ : pandas.DataFrame
        All local eigenvalues (descending), shape ``(n_samples, n_features)``.
        Row-order dependent; see ``explained_variance_``.
    proportion_total_var_ : pandas.Series
        Proportion of total local variation captured by the first
        ``n_components`` components (a.k.a. PTV). Row-order invariant.
    local_scores_ : pandas.DataFrame
        Matches ``GWmodel``'s ``gwpca.scores``: a score for *every neighbour*
        of each location (not just the focal observation), computed from
        **raw, uncentered** feature values dotted with the local loadings.
        MultiIndexed by ``(location, neighbor)``, one column per retained
        component. Ragged — the number of ``neighbor`` rows per ``location``
        equals that location's neighbourhood size, which varies unless the
        kernel gives every location the same neighbourhood.
    winning_variable_ : pandas.Series
        Feature with the largest absolute loading on the first local component
        at each location — the classic GWPCA map.
    global_model_ : sklearn.decomposition.PCA | None
        Fitted global baseline (``None`` when ``fit_global_model=False``).

    Examples
    --------
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> from geodatasets import get_path  # doctest: +SKIP
    >>> from spml.decomposition import GWPCA  # doctest: +SKIP
    >>> gdf = gpd.read_file(get_path("geoda.guerry"))  # doctest: +SKIP
    >>> gdf = gdf.set_geometry(gdf.centroid)  # doctest: +SKIP
    >>> X = gdf[["Crm_prs", "Litercy", "Wealth", "Donatns"]]  # doctest: +SKIP
    >>> gwpca = GWPCA(n_components=2, bandwidth=30, fixed=False).fit(  # doctest: +SKIP
    ...     X, geometry=gdf.geometry
    ... )
    >>> gwpca.winning_variable_.value_counts()  # doctest: +SKIP
    """

    _component_prefix = "PC"

    def __init__(
        self,
        n_components: int = 2,
        *,
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
        include_focal: bool = True,
        standardize: bool = True,
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
            model=PCA,
            n_components=n_components,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            standardize=standardize,
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
        self._model_type = "gwpca"

    def fit(self, X: pd.DataFrame, geometry: gpd.GeoSeries | None = None) -> GWPCA:
        """Fit GWPCA.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        geometry : geopandas.GeoSeries | None
            Point geometries of the observations.  Required unless ``graph``
            was supplied; required in any case to enable :meth:`transform`.
        """
        super().fit(X, geometry=geometry)

        feat = self.feature_names_in_
        k = self.n_components
        pc_labels = self._component_labels(k)

        self.proportion_total_var_ = self.explained_variance_ratio_.sum(axis=1)

        # Build the tidy (location, neighbor) / (location, component) frames from
        # stacked arrays in one shot rather than concatenating one small
        # DataFrame per location -- the per-location construction loop's
        # overhead otherwise dominates the actual (fully vectorized)
        # eigendecomposition cost once n_loc reaches the thousands typical of
        # this framework's use cases.
        neighbor_counts = [len(nn) for nn in self._neighbor_names]
        self.local_scores_ = pd.DataFrame(
            np.concatenate(self._local_extra, axis=0),
            index=pd.MultiIndex.from_arrays(
                [
                    np.repeat(self._names, neighbor_counts),
                    np.concatenate(self._neighbor_names),
                ],
                names=["location", "neighbor"],
            ),
            columns=pc_labels,
        )
        del self._neighbor_names, self._local_extra

        self.local_loadings_ = pd.DataFrame(
            self.components_.reshape(-1, self.components_.shape[-1]),
            index=pd.MultiIndex.from_arrays(
                [np.repeat(self._names, k), np.tile(pc_labels, len(self._names))],
                names=["location", "component"],
            ),
            columns=feat,
        )

        self.winning_variable_ = pd.Series(
            feat[np.argmax(np.abs(self.components_[:, 0, :]), axis=1)],
            index=self._names,
            name="winning_variable",
        )

        return self

    def _decompose_local(
        self, Xn: np.ndarray, Xc: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eigendecompose one neighbourhood's weighted scatter matrix.

        ``eigvals_raw`` are eigenvalues of the *unnormalized* weighted scatter
        matrix ``Xc.T @ diag(w) @ Xc`` (not divided by the weight sum) —
        matching ``GWmodel``'s ``wpca`` internals, which likewise defer that
        division to :meth:`_scale_eigenvalues`. The third element returned is
        the GWmodel-style neighbour scores.
        """
        scatter = (Xc * w[:, None]).T @ Xc  # unnormalized weighted scatter

        eigvals_raw, eigvecs = np.linalg.eigh(scatter)  # ascending
        eigvals_raw = eigvals_raw[::-1]
        eigvecs = eigvecs[:, ::-1]

        # per-location sign convention: largest |loading| of each component > 0
        # TODO(gh#79): align order/sign *across* locations (Procrustes to global).
        for k in range(eigvecs.shape[1]):
            j = np.argmax(np.abs(eigvecs[:, k]))
            if eigvecs[j, k] < 0:
                eigvecs[:, k] *= -1

        # GWmodel::gwpca's `gwpca.scores`: raw (uncentered) neighbourhood
        # values dotted with the retained loadings — one score per neighbour,
        # not a single mean-centered focal score. See the module docstring.
        neighbor_scores = Xn @ eigvecs[:, : self.n_components]

        return eigvecs.T, eigvals_raw, neighbor_scores

    def _scale_eigenvalues(
        self,
        eig_raw: np.ndarray,
        wsums: np.ndarray,
        names: list,
        row_order: pd.Index,
    ) -> np.ndarray:
        """GWmodel-faithful eigenvalue scaling, bug included (gh#79).

        ``GWmodel::gwpca`` divides every location's eigenvalues by the
        weight sum of whichever neighbourhood is *last* in the input data's
        row order, not each location's own (see the module docstring). That
        neighbourhood is looked up by ``X``'s own last index label rather than
        assumed to be the last one fitted: the neighbourhoods are enumerated in
        the spatial graph's internal build order (geometry's row order when
        built via ``_build_weights``, or a supplied ``graph``'s own order),
        which only coincides with ``X``'s row order when ``X``, geometry and
        ``graph`` are all ordered alike.
        """
        wsum_by_name = dict(zip(names, wsums, strict=True))
        last_name = row_order[-1]
        if last_name not in wsum_by_name:
            warnings.warn(
                "X's last row has no fitted neighbourhood (e.g. it was "
                "skipped as an isolate); falling back to the last "
                "neighbourhood processed instead for the shared "
                "eigenvalue-scaling divisor.",
                stacklevel=2,
            )
            last_name = names[-1]
        self._last_neighborhood_wsum_ = wsum_by_name[last_name]
        return eig_raw / self._last_neighborhood_wsum_
