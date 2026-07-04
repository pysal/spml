"""Local Bootstraps for spatial/temporal Model Validation

References
----------
.. [1] Statham, Thomas. *The Global Inconsistencies of Gridded
        Population Data At Different Spatial Scales.* Doctoral
        Dissertation, University of Bristol.
"""

import numpy
import geopandas
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from scipy.sparse import diags
from scipy.spatial.distance import cdist

from ._utils import (
    _get_coords,
    _to_point_gdf,
    _idx_and_is_geo,
    KERNELS,
    LIBPYSAL_KERNEL_MAP,
)


class LocalBootstrap(BaseEstimator):
    """Locally-weighted bootstrap -- resampling with spatial (or temporal) proximity weighting.

    Generates *n_bootstraps* resampled datasets each of size n.  At every
    site i the observation placed there is drawn **with replacement** from all
    n observations; the probability is proportional to a kernel weight
    K(d_{ij} / bandwidth):

        p_{ij}  ∝  K(d_{ij} / bandwidth)

    so nearby observations are drawn more often.

    Works for **spatial** data (GeoDataFrame / (n, 2) coordinates) and for
    **time-series** data (pass a 1-D array of time indices; distance is then
    the absolute time lag).

    Passing a pre-built ``libpysal.graph.Graph`` via *graph* overrides
    *bandwidth* and *kernel*.

    Parameters
    ----------
    n_bootstraps : int, default 100
        Number of bootstrap resamples to generate.
    bandwidth : float or None
        Kernel bandwidth in the same units as the input distances.  Required
        unless *graph* is provided.
    kernel : str, default 'gaussian'
        One of ``'gaussian'``, ``'exponential'``, ``'bisquare'``,
        ``'triangular'``, ``'uniform'``, ``'parabolic'``.
    graph : libpysal.graph.Graph or None
        Pre-built spatial weights (must expose ``.sparse``).
        Overrides *bandwidth* / *kernel*.
    random_state : int, RandomState instance, or None

    Notes
    -----
    When input is a GeoDataFrame/GeoSeries and a supported kernel is given,
    a libpysal Graph is built internally -- the weight matrix stays sparse.
    The dense O(n**2) path is used only for raw array inputs or when kernel is
    ``'exponential'`` (not supported by libpysal).

    Explored initially in

    Statham, Thomas A. The Global Inconsistencies of Gridded Population Data
    at Different Spatial Scales. Diss. University of Bristol, 2024.

    Examples
    --------
    Spatial use (GeoDataFrame):

    >>> lb = LocalBootstrap(n_bootstraps=200, bandwidth=5_000,
    ...                     kernel='bisquare', random_state=0)
    >>> for indices in lb.sample(gdf):
    ...     boot = gdf.iloc[indices].copy()
    ...     boot.geometry = gdf.geometry.values   # restore original locations
    ...     model.fit(boot[features], y[indices])

    Time-series use (1-D array of time steps):

    >>> t = numpy.arange(len(df))
    >>> lb = LocalBootstrap(n_bootstraps=100, bandwidth=12, random_state=0)
    >>> for indices in lb.sample(t):
    ...     model.fit(X[indices], y[indices])
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        bandwidth: float | str | None = None,
        k: int | str | None = None,
        kernel: str = "gaussian",
        graph=None,
        random_state=None,
    ):
        if bandwidth is not None and k is not None:
            raise ValueError(
                "Specify at most one of 'bandwidth' (radius-based) or "
                "'k' (k-nearest-neighbour).  They are mutually exclusive."
            )
        if isinstance(bandwidth, str) and bandwidth != "auto":
            raise ValueError(
                f"bandwidth must be a number, None, or 'auto'; got {bandwidth!r}"
            )
        if isinstance(k, str) and k != "auto":
            raise ValueError(f"k must be an integer, None, or 'auto'; got {k!r}")
        self.n_bootstraps = n_bootstraps
        self.bandwidth = bandwidth
        self.k = k
        self.kernel = kernel
        self.graph = graph
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

    def fit(self, X, y=None):
        """Build the spatial weight graph.

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
        _, is_geo = _idx_and_is_geo(X)

        if self.graph is not None:
            self.graph_ = self.graph
            return self

        if bw is None and k is None:
            raise ValueError("Specify one of 'bandwidth', 'k', or a pre-built 'graph'.")
        if self.kernel not in KERNELS:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. Choose from: {sorted(KERNELS)}."
            )

        if k is not None:
            from libpysal.graph import Graph

            coords = _get_coords(X)
            W_csr = self._knn_weight_csr(coords, coords, k, skip_self=True)
            coo = W_csr.tocoo()
            self.graph_ = Graph.from_arrays(coo.row, coo.col, coo.data)
            return self

        if is_geo:
            from libpysal.graph import Graph

            point_gdf = _to_point_gdf(X)
            self.graph_ = Graph.build_kernel(
                point_gdf,
                bandwidth=bw,
                kernel=LIBPYSAL_KERNEL_MAP[self.kernel],
            )
            return self

        coords = _get_coords(X)
        self._W_dense_ = self._dense_weight_matrix(coords, bw)
        return self

    def sample(self, X, donor=None):
        """Yield locally-weighted bootstrap samples.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Target locations at which to draw samples.  Pass a 1-D array of
            time indices for time-series data.
        donor : GeoDataFrame, GeoSeries, ndarray, or None
            Donor pool from which observations are drawn (same types as ``X``).
            When None (default), observations are drawn from ``X`` itself
            (standard bootstrap).

            When provided, the weight matrix is (|X|, |donor|): row i weights
            every donor observation by its kernel distance from ``X[i]``, so each
            new location draws locally from the donor pool.

            If both X and donor are GeoDataFrames, each yielded value is a
            GeoDataFrame of donor rows with index and geometry overwritten to
            match X -- i.e. donor attributes placed at new locations.
            Otherwise each yielded value is an array of donor index labels
            (or positional integers when donor has no pandas index).

            Incompatible with a pre-built ``graph``.

        Yields
        ------
        result : GeoDataFrame or ndarray of shape (n,)
            When donor is a GeoDataFrame and X is a GeoDataFrame: a
            GeoDataFrame of selected donor rows with X's index and geometry.
            Otherwise: ``result[i]`` is the label (or position) of the donor
            row drawn for location i.
        """
        from sklearn.exceptions import NotFittedError

        rng = check_random_state(self.random_state)

        if donor is not None:
            if self.graph is not None:
                raise ValueError(
                    "donor= cannot be combined with a pre-built graph. "
                    "Use bandwidth= for cross-geometry sampling."
                )
            bw = getattr(self, "bandwidth_", self.bandwidth)
            k = getattr(self, "k_", self.k)
            if bw == "auto" or k == "auto":
                raise NotFittedError(
                    f"This {type(self).__name__} instance has bandwidth='auto' "
                    "but fit() has not been called. Call fit(X, y) first."
                )
            return self._sample_cross(X, donor, rng, bw, k)

        _fitted = hasattr(self, "graph_") or hasattr(self, "_W_dense_")
        if not _fitted:
            if self.bandwidth == "auto" or self.k == "auto":
                raise NotFittedError(
                    f"This {type(self).__name__} instance has bandwidth='auto' "
                    "but fit() has not been called. Call fit(X, y) first."
                )
            self.fit(X)

        idx, is_geo = _idx_and_is_geo(X)

        def _out(positions):
            return idx[positions] if idx is not None else positions

        if hasattr(self, "_W_dense_"):
            return self._yield_dense(self._W_dense_, rng, _out)

        W_csr = self._sparse_weights_from_graph(self.graph_)
        return self._yield_csr(W_csr, rng, _out)

    # ------------------------------------------------------------------
    # Cross-geometry sampling
    # ------------------------------------------------------------------

    def _sample_cross(self, X, donor, rng, bw, k):
        """Set up and return a generator for cross-geometry sampling."""
        if bw is None and k is None:
            raise ValueError(
                "Specify one of 'bandwidth' or 'k' for cross-geometry sampling."
            )
        if self.kernel not in KERNELS:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. Choose from: {sorted(KERNELS)}."
            )

        X_coords = _get_coords(X)
        donor_coords = _get_coords(donor)
        n_donor = len(donor_coords)

        if k is not None:
            W_csr = self._knn_weight_csr(X_coords, donor_coords, k)
        else:
            W_csr = self._radius_weight_csr(X_coords, donor_coords, bw)

        return_df = isinstance(X, geopandas.GeoDataFrame) and isinstance(
            donor, geopandas.GeoDataFrame
        )
        donor_idx, _ = _idx_and_is_geo(donor)

        return self._yield_cross(W_csr, X, donor, donor_idx, return_df, n_donor, rng)

    def _yield_cross(self, W_csr, X, donor, donor_idx, return_df, n_donor, rng):
        for _ in range(self.n_bootstraps):
            positions = self._sample_csr(W_csr, rng, n_cols=n_donor)
            if return_df:
                result = donor.iloc[positions].copy()
                result.index = X.index
                result["geometry"] = X.geometry.values
                yield result
            elif donor_idx is not None:
                yield donor_idx[positions]
            else:
                yield positions

    def _yield_csr(self, W_csr, rng, out_fn):
        for _ in range(self.n_bootstraps):
            yield out_fn(self._sample_csr(W_csr, rng))

    def _yield_dense(self, W, rng, out_fn):
        for _ in range(self.n_bootstraps):
            yield out_fn(self._sample_dense(W, rng))

    # ------------------------------------------------------------------
    # Sparse weight matrix builders
    # ------------------------------------------------------------------

    def _knn_weight_csr(
        self,
        X_coords: numpy.ndarray,
        donor_coords: numpy.ndarray,
        k: int,
        skip_self: bool = False,
    ):
        """Sparse (|X|, |donor|) CSR weight matrix via k-NN query.

        Kernel weights use an adaptive bandwidth equal to the distance to
        the k-th nearest neighbour, so the nearest neighbour always receives
        the maximum weight regardless of absolute distance.
        When skip_self=True the self-hit (distance 0) is dropped, which is
        correct for the self-sampling case where X == donor.
        """
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix

        n_query = k + (1 if skip_self else 0)
        tree = cKDTree(donor_coords)
        distances, indices = tree.query(X_coords, k=n_query)

        if skip_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        n_X = len(X_coords)
        n_donor = len(donor_coords)

        # Adaptive bandwidth: distance to the k-th (farthest) neighbour
        bw = distances[:, -1:] + 1e-10
        weights = KERNELS[self.kernel](distances / bw)  # (n_X, k)

        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        weights = weights / row_sums

        rows = numpy.repeat(numpy.arange(n_X), k)
        cols = indices.ravel()
        data = weights.ravel()
        W = csr_matrix((data, (rows, cols)), shape=(n_X, n_donor))
        W.eliminate_zeros()
        return W

    def _radius_weight_csr(
        self,
        X_coords: numpy.ndarray,
        donor_coords: numpy.ndarray,
        bandwidth: float,
    ):
        """Sparse (|X|, |donor|) CSR weight matrix via radius search.

        Uses cKDTree.sparse_distance_matrix to find all donor-X pairs
        within *bandwidth* without materialising the full dense matrix.
        """
        from scipy.spatial import cKDTree

        tree_X = cKDTree(X_coords)
        tree_donor = cKDTree(donor_coords)

        # sparse_distance_matrix returns a dok_matrix with shape (|X|, |donor|)
        D = tree_X.sparse_distance_matrix(
            tree_donor, max_distance=bandwidth, output_type="coo_matrix"
        )
        data = KERNELS[self.kernel](D.data / bandwidth)
        mask = data > 0

        from scipy.sparse import csr_matrix

        n_X, n_donor = len(X_coords), len(donor_coords)
        W = csr_matrix((data[mask], (D.row[mask], D.col[mask])), shape=(n_X, n_donor))
        row_sums = numpy.asarray(W.sum(axis=1)).ravel()
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        W = diags(1.0 / row_sums) @ W
        W.eliminate_zeros()
        return W

    # ------------------------------------------------------------------
    # Weight matrix builders
    # ------------------------------------------------------------------

    def _dense_weight_matrix(
        self, coords: numpy.ndarray, bandwidth: float
    ) -> numpy.ndarray:
        """Build a row-normalised (n, n) dense weight matrix."""
        if coords.shape[1] == 1:
            D = numpy.abs(coords - coords.T)
        else:
            D = cdist(coords, coords)
        W = KERNELS[self.kernel](D / bandwidth)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        return W / row_sums

    def _sparse_weights_from_graph(self, graph):
        """Return a row-normalised CSR sparse matrix -- stays sparse throughout."""
        try:
            W = graph.sparse.tocsr().astype(float)
        except AttributeError:
            raise TypeError(
                "Expected a libpysal Graph with a '.sparse' attribute "
                "(scipy sparse matrix)."
            )
        row_sums = numpy.asarray(W.sum(axis=1)).ravel()
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        W_norm = diags(1.0 / row_sums) @ W
        W_norm.eliminate_zeros()
        return W_norm

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_csr(W_csr, rng, n_cols=None) -> numpy.ndarray:
        """Draw one index per row from a row-normalised CSR weight matrix.

        Uses ``numpy.searchsorted`` on each row's cumulative weights

        Basic idea here is that "u" is how we select the random candidate,
        cumw increases quickly when an candidate has a lot of weight, and
        slowly if it doesnt. since "u" is U(0,1), observations with high
        weight will be selected more frequently, because "u" is more likely
        to fall in their range in the cumsum. It's like you're taking the row
        standardized weight

        0                                                                    1
        |--------------------------------------------------------------------|
        |----A: .2----|---------------B: .6----------------|-C: .1-||-D: .1--|
        u1:.4                       *
        u2:.7                                      *
        u3:.2      *

        Since we drop all zero-weight entries, there are no ties at zero,
        which would result in the edge weight being picked arbitrarly often
        (this was a fun bug to debug). Sampling with replacement, this selects
        BBA as the ensemble.
        """
        n_rows = W_csr.shape[0]
        # n_cols is only supplied for cross-geometry (|X|, |donor|) matrices.
        # When None (self-sampling) an isolated node falls back to drawing from
        # itself, preserving the original behaviour.
        cross = n_cols is not None
        if n_cols is None:
            n_cols = n_rows
        u = rng.uniform(size=n_rows)
        result = numpy.empty(n_rows, dtype=numpy.intp)
        for i in range(n_rows):
            start = int(W_csr.indptr[i])
            end = int(W_csr.indptr[i + 1])
            if start == end:
                result[i] = int(rng.uniform() * n_cols) if cross else i
                continue
            cumw = numpy.cumsum(W_csr.data[start:end])
            idx = numpy.searchsorted(cumw, u[i] * cumw[-1])
            result[i] = W_csr.indices[start + min(idx, end - start - 1)]
        return result

    @staticmethod
    def _sample_dense(W, rng) -> numpy.ndarray:
        """Draw one index per row from a row-normalised dense weight matrix."""
        n = W.shape[0]
        u = rng.uniform(size=n)
        cumW = W.cumsum(axis=1)
        return numpy.array(
            [numpy.searchsorted(cumW[i], u[i]) for i in range(n)], dtype=numpy.intp
        )
