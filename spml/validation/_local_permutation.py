"""Local permutations for Spatial Model Validation

References
----------
.. [1] Kim et al. "Local Permutation Tests for Conditional
        Independence." *Annals of Statistics* 50(6): 3388-3414.
        https://doi.org/10.1214/22-AOS2233
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from ._utils import (
    LIBPYSAL_KERNEL_MAP,
    _get_coords,
    _idx_and_is_geo,
)

_COPLANAR_OPTIONS = ("raise", "jitter", "clique")


class LocalPermutation(BaseEstimator):
    """Spatially-constrained permutation without replacement (derangement).

    Shuffles the rows of an n-site dataset so that each row moves only
    within a local neighbourhood defined by *bandwidth*, *k*, or a
    pre-built *graph*.  Every row appears exactly once -- it is a true
    permutation, analogous to :class:`LocalBootstrap` but **without**
    replacement.

    Optionally enforced as a **derangement**: no row may remain at its
    original site.

    Parameters
    ----------
    bandwidth : float or None
        Neighbourhood radius in the same units as the input distances.
        Required unless *k* or *graph* is provided.
        Mutually exclusive with *k*.
    k : int or None
        Number of nearest neighbours.  Mutually exclusive with
        *bandwidth*.
    kernel : str, default 'uniform'
        One of ``'uniform'``, ``'bisquare'``, ``'triangular'``,
        ``'gaussian'``, ``'exponential'``, ``'parabolic'``.
        ``'uniform'`` gives a hard cutoff (binary adjacency);
        smooth kernels weight swap proposals by proximity.
    derangement : bool, default True
        If True, every value must move (no fixed points).
    n_permutations : int, default 99
        Number of permutations to generate.
    n_burn : int or None
        Proposed Markov-chain steps between each yielded permutation.
        Defaults to ``10 * n``.
    graph : libpysal.graph.Graph or None
        Pre-built spatial weights (must expose ``.sparse``).  Edge
        weights drive proposals.  Overrides *bandwidth* and *k*.
    coplanar : str, default 'raise'
        How to handle coincident (duplicate-location) points when *k* is
        given, passed straight through to ``libpysal.graph.Graph.build_kernel``.
    random_state : int, RandomState instance, or None

    Raises
    ------
    ValueError
        If no valid derangement exists for the given constraints, or
        if neither *bandwidth*, *k*, nor *graph* is supplied.

    Examples
    --------
    Bandwidth with default uniform kernel (hard cutoff):

    >>> lp = LocalPermutation(bandwidth=50_000, random_state=0)
    >>> for perm in lp.sample(gdf):
    ...     model.fit(X[perm], y[perm])

    Bandwidth with a smooth kernel (nearby pairs proposed more often):

    >>> lp = LocalPermutation(bandwidth=50_000, kernel='bisquare',
    ...                       random_state=0)
    >>> for perm in lp.sample(gdf):
    ...     model.fit(X[perm], y[perm])
    """

    def __init__(
        self,
        bandwidth: float | str | None = None,
        k: int | str | None = None,
        kernel: str = "uniform",
        derangement: bool = True,
        n_permutations: int = 99,
        n_burn: int | None = None,
        graph=None,
        coplanar: str = "raise",
        random_state=None,
    ):
        if bandwidth is not None and k is not None:
            raise ValueError(
                "Specify at most one of 'bandwidth' or 'k'. "
                "They are mutually exclusive."
            )
        if isinstance(bandwidth, str) and bandwidth != "auto":
            raise ValueError(
                f"bandwidth must be a number, None, or 'auto'; got {bandwidth!r}"
            )
        if isinstance(k, str) and k != "auto":
            raise ValueError(f"k must be an integer, None, or 'auto'; got {k!r}")
        self.bandwidth = bandwidth
        self.k = k
        self.kernel = kernel
        self.derangement = derangement
        self.n_permutations = n_permutations
        self.n_burn = n_burn
        self.graph = graph
        self.coplanar = coplanar
        self.random_state = random_state

    def _resolve_auto(self, X, y):
        """Return (bandwidth, k) resolving 'auto' via correlogram/knn range."""
        bw, k = self.bandwidth, self.k
        if bw == "auto" or k == "auto":
            if y is None:
                raise ValueError(
                    "Call fit(X, y) before sample() when bandwidth='auto' or k='auto'."
                )
            from ._range import correlogram_range, knn_range

            if bw == "auto":
                bw = correlogram_range(X, y)
                self.bandwidth_ = bw
            if k == "auto":
                k = knn_range(X, y)
                self.k_ = k
        return bw, k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Build the spatial adjacency structure.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Locations.
        y : array-like or None
            Response variable; required when ``bandwidth='auto'`` or
            ``k='auto'``.

        Returns
        -------
        self
        """
        bw, k = self._resolve_auto(X, y)

        if self.graph is not None:
            n = self._n_from_graph(self.graph)
            adj_csr, adj_sets, edge_i, edge_j, cumw = self._adj_from_graph(
                self.graph, n
            )
            self.graph_ = self.graph
        elif bw is not None or k is not None:
            if self.kernel not in LIBPYSAL_KERNEL_MAP:
                raise ValueError(
                    f"Unknown kernel '{self.kernel}'. "
                    f"Choose from: {sorted(LIBPYSAL_KERNEL_MAP)}."
                )
            if self.coplanar not in _COPLANAR_OPTIONS:
                raise ValueError(
                    f"coplanar must be one of {_COPLANAR_OPTIONS}; "
                    f"got {self.coplanar!r}."
                )
            adj_csr, adj_sets, edge_i, edge_j, cumw = self._kernel_adj(X, bw, k)
        else:
            raise ValueError("Specify one of 'bandwidth', 'k', or a pre-built 'graph'.")

        self._adj_csr_ = adj_csr
        self._adj_sets_ = adj_sets
        self._edge_i_ = edge_i
        self._edge_j_ = edge_j
        self._cumw_ = cumw
        return self

    def sample(self, X):
        """Yield constrained permutation index arrays.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Locations.  When *graph* is provided coordinates are used
            only to determine n.

        Yields
        ------
        perm : ndarray of shape (n,)
            ``perm[i]`` is the label of the row assigned to position i,
            using the input's index when X is a GeoDataFrame/GeoSeries,
            or integer positions for raw arrays.
        """
        from sklearn.exceptions import NotFittedError

        rng = check_random_state(self.random_state)

        if not hasattr(self, "_adj_csr_"):
            if self.bandwidth == "auto" or self.k == "auto":
                raise NotFittedError(
                    f"This {type(self).__name__} instance has bandwidth='auto' "
                    "but fit() has not been called. Call fit(X, y) first."
                )
            self.fit(X)

        idx, _ = _idx_and_is_geo(X)
        n = self._adj_csr_.shape[0]
        n_burn = self.n_burn if self.n_burn is not None else 10 * n
        perm = self._initial_permutation(self._adj_csr_, n, rng)

        for _ in range(self.n_permutations):
            perm = self._markov_mix(
                perm,
                self._adj_sets_,
                self._edge_i_,
                self._edge_j_,
                self._cumw_,
                n_burn,
                rng,
            )
            positions = perm.copy()
            yield idx[positions] if idx is not None else positions

    # ------------------------------------------------------------------
    # Adjacency builders
    # ------------------------------------------------------------------

    def _kernel_adj(self, X, bw, k):
        """Build a kernel-weighted adjacency via libpysal.graph.Graph.build_kernel.

        1-D/time-series input is duplicated into a second column so the
        same 2-D machinery applies unchanged; Euclidean distance on that
        duplicated array is a constant (sqrt(2)) multiple of the 1-D
        distance, which cancels exactly under the k-NN adaptive bandwidth
        and is compensated for explicitly for a fixed *bw*.
        """
        from libpysal.graph import Graph

        coords = _get_coords(X)
        n = len(coords)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], coords[:, 0]])
            if k is None:
                bw = bw * np.sqrt(2)

        if k is not None:
            directed = Graph.build_kernel(
                coords,
                k=k,
                bandwidth="adaptive",
                kernel=LIBPYSAL_KERNEL_MAP[self.kernel],
                coplanar=self.coplanar,
            )
            # Symmetrise: union of both directions, summing weights where
            # both directions are already present. Vectorised sparse add --
            # much faster than Graph.make_symmetric().
            W = directed.sparse
            sym = (W + W.T).tocoo()
            graph = Graph.from_arrays(sym.row, sym.col, sym.data)
        else:
            graph = Graph.build_kernel(
                coords,
                bandwidth=bw,
                kernel=LIBPYSAL_KERNEL_MAP[self.kernel],
                coplanar=self.coplanar,
            )
        return self._adj_from_graph(graph, n)

    def _adj_from_graph(self, graph, n: int):
        """Weighted adjacency from a pre-built graph."""
        try:
            W = graph.sparse.tocsr().astype(float)
        except AttributeError as e:
            raise TypeError(
                "Expected a libpysal Graph with a '.sparse' attribute "
                "(scipy sparse matrix)."
            ) from e

        W_coo = W.tocoo()
        mask = W_coo.data != 0
        off_diag = W_coo.row != W_coo.col
        mask &= off_diag

        row = W_coo.row[mask]
        col = W_coo.col[mask]
        weights = W_coo.data[mask]

        upper = row < col
        edge_i = row[upper]
        edge_j = col[upper]
        cumw = np.cumsum(weights[upper])

        if not self.derangement:
            diag = np.arange(n)
            row = np.concatenate([row, diag])
            col = np.concatenate([col, diag])

        data = np.ones(len(row), dtype=bool)
        adj_csr = csr_array((data, (row, col)), shape=(n, n))
        adj_csr.sum_duplicates()
        adj_sets = self._sets_from_pairs(row, col, n)
        return adj_csr, adj_sets, edge_i, edge_j, cumw

    @staticmethod
    def _sets_from_pairs(rows, cols, n) -> list:
        """Build a list-of-sets adjacency for O(1) Markov-chain lookups."""
        adj_sets: list[set] = [set() for _ in range(n)]
        for i, j in zip(rows, cols, strict=True):
            adj_sets[i].add(j)
        return adj_sets

    @staticmethod
    def _n_from_graph(graph) -> int:
        try:
            return graph.n
        except AttributeError as e:
            raise TypeError(
                "Expected a libpysal Graph with a '.sparse' attribute."
            ) from e

    # ------------------------------------------------------------------
    # Initial permutation -- sparse assignment
    # ------------------------------------------------------------------

    def _initial_permutation(self, adj_csr, n: int, rng) -> np.ndarray:
        """Find a random feasible matching via min_weight_full_bipartite_matching.

        Assigns i.i.d. Uniform[0,1] weights to feasible pairs so the
        minimum-weight solution is effectively a random feasible matching.
        The sparse cost matrix is never expanded to a dense (n, n) array.
        """
        rows, cols = adj_csr.nonzero()
        weights = rng.uniform(0.0, 1.0, len(rows)).astype(float)
        cost_csr = csr_array((weights, (rows, cols)), shape=(n, n))

        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(cost_csr)
        except ValueError as e:
            kind = "derangement" if self.derangement else "permutation"
            src = (
                "the supplied graph"
                if self.graph is not None
                else f"bandwidth={self.bandwidth}"
                if self.bandwidth is not None
                else f"k={self.k}"
            )
            raise ValueError(
                f"No valid constrained {kind} exists within {src}.  "
                "Increase bandwidth/k, use a denser graph, or set "
                "derangement=False."
            ) from e

        perm = np.empty(n, dtype=int)
        perm[row_ind] = col_ind
        return perm

    # ------------------------------------------------------------------
    # Markov chain -- edge-weighted proposals
    # ------------------------------------------------------------------

    def _markov_mix(
        self,
        perm: np.ndarray,
        adj_sets: list,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        cumw: np.ndarray,
        n_steps: int,
        rng,
    ) -> np.ndarray:
        """Random-transposition Markov chain with edge-weighted proposals.

        Samples candidate pair (i, j) proportional to graph edge weight
        rather than uniformly over all n*(n-1)/2 pairs.  On a sparse
        graph this eliminates the O(k/n) acceptance rate of uniform
        sampling -- every proposal is already a connected pair, so the
        only remaining rejections are cases where the values perm[i]/
        perm[j] have drifted outside each other's adjacency sets.
        """
        perm = perm.copy()
        total_w = float(cumw[-1])

        for _ in range(n_steps):
            idx = np.searchsorted(cumw, rng.uniform() * total_w)
            i, j = int(edge_i[idx]), int(edge_j[idx])
            vi, vj = perm[i], perm[j]

            if vj not in adj_sets[i] or vi not in adj_sets[j]:
                continue
            if self.derangement and (vj == i or vi == j):
                continue

            perm[i], perm[j] = vj, vi

        return perm
