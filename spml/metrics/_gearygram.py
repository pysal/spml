"""Geary correlogram for univariate and multivariate spatial data.

References
----------
.. [1] Geary, R. C. (1954). The contiguity ratio and statistical mapping.
   *The Incorporated Statistician*, 5(3), 115-145.
   https://doi.org/10.2307/2986645

.. [2] Anselin, L. (2019). A local indicator of multivariate spatial
   association: Extending Geary's C. *Geographical Analysis*, 51(2),
   133-150. https://doi.org/10.1111/gean.12164
"""

from __future__ import annotations

import warnings

import numpy
from sklearn.utils import Bunch

from ..validation._utils import KERNELS, _get_coords

_COMPACT_KERNELS = {"bisquare", "triangular", "uniform", "parabolic"}


def _weighted_geary(Z, focal, neighbor, weights, p):
    """Weighted multivariate Geary's C for explicit pairs and weights."""
    W = float(weights.sum())
    if W == 0:
        return numpy.nan
    DZ = Z[focal] - Z[neighbor]  # (m, p)
    return float(numpy.dot(weights, (DZ**2).sum(axis=1)) / (2.0 * W * p))


def gearygram(
    X,
    geometry=None,
    n_bins: int = 15,
    max_distance: float | None = None,
    max_k: int | None = None,
    kernel: str | None = None,
    nonparametric: bool = False,
) -> Bunch:
    """Compute a Geary correlogram for univariate or multivariate spatial data.

    Three modes:

    **Bandwidth** (default, *n_bins*): evaluates the weighted Geary's C at
    *n_bins* increasing kernel bandwidths.  At each bandwidth *h* every pair
    (*i*, *j*) receives weight :math:`K(d_{ij}/h)`.  Compact kernels use a
    sparse distance matrix; non-compact kernels compute all pairwise distances.
    Defaults to ``kernel="gaussian"``.

    **kNN** (*max_k*): evaluates Geary's C for cumulative k-NN graphs at
    k = 1 … *max_k*.  Each observation's k-th nearest-neighbour distance is
    used as an adaptive bandwidth so *kernel* controls how the k included
    neighbours are weighted by distance.  Defaults to ``kernel="uniform"``,
    which gives equal binary weights (standard kNN Geary).

    **Nonparametric** (*nonparametric=True*): fits a LOWESS curve of the
    per-pair Geary contribution :math:`\\|\\mathbf{z}_i-\\mathbf{z}_j\\|^2/(2p)`
    against squared spatial distance :math:`d_{ij}^2`, then evaluates the
    smooth at *n_bins* equally spaced distance values.

    In all parametric modes the multivariate statistic :cite:t:`anselin2019` is:

    .. math::

        C_m = \\frac{\\sum_{(i,j)} w_{ij}\\,\\|\\mathbf{z}_i - \\mathbf{z}_j\\|^2}
                    {2\\,W\\,p}

    where :math:`\\mathbf{z}` is column-standardised *X*, *W* is the weight
    sum, and *p* is the number of variables.  For *p* = 1 this reduces to
    the standard weighted Geary's C :cite:t:`geary1954`.

    Parameters
    ----------
    X : array-like of shape (n,) or (n, p)
        Observed values.  A 1-D array is treated as a single variable.
        If *X* is a GeoDataFrame with a ``geometry`` column and *geometry*
        is not provided, locations are read from that column.
    geometry : GeoDataFrame | GeoSeries | (n, 2) ndarray or None
        Locations.  Required if *X* has no ``geometry`` attribute.
    n_bins : int, default 15
        Number of bandwidths (bandwidth/nonparametric mode).
        Ignored for kNN mode.
    max_distance : float or None
        Maximum bandwidth (bandwidth/nonparametric mode).  Defaults to the
        maximum pairwise distance.  Ignored for kNN mode.
    max_k : int or None
        If set, use kNN mode with k = 1 … *max_k*.
    kernel : str or None
        Kernel weighting.  One of ``"gaussian"``, ``"exponential"``,
        ``"bisquare"``, ``"triangular"``, ``"uniform"``, ``"parabolic"``.
        Defaults to ``"uniform"`` for kNN mode and ``"gaussian"`` for
        bandwidth mode.  Ignored for nonparametric mode.
    nonparametric : bool, default False
        If True, fit a LOWESS curve instead of computing kernel-weighted
        bins.  Ignored when *max_k* is set.

    Returns
    -------
    sklearn.utils.Bunch
        ``bin_centers`` : ndarray
            Bandwidth values (distance units) or k indices.
        ``C`` : ndarray
            Geary's C per lag.  Approaches 1 under spatial independence,
            < 1 for positive autocorrelation, > 1 for negative.
        ``n_pairs`` : ndarray or None
            Number of pairs per lag (None for nonparametric mode).

    Examples
    --------
    Bandwidth correlogram (Gaussian kernel):

    >>> result = gearygram(Y, gdf)

    kNN correlogram with Gaussian kernel weighting:

    >>> result = gearygram(Y, gdf, max_k=20, kernel="gaussian")

    Nonparametric LOWESS correlogram:

    >>> result = gearygram(Y, gdf, nonparametric=True)

    Notes
    -----
    Bandwidth bins with fewer than 2 pairs are returned as ``NaN``.
    """
    from scipy.spatial import cKDTree

    if geometry is None:
        if hasattr(X, "geometry"):
            geometry = X.geometry
            X = X.drop("geometry", axis=1)
        else:
            raise ValueError(
                "geometry must be provided when X has no geometry attribute."
            )

    X = numpy.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    if len(geometry) != n:
        raise ValueError(f"geometry has {len(geometry)} observations but X has {n}.")

    std = X.std(axis=0)
    if numpy.any(std == 0):
        zero_cols = numpy.where(std == 0)[0].tolist()
        raise ValueError(
            f"Column(s) {zero_cols} of X are constant; Geary's C is undefined."
        )
    Z = (X - X.mean(axis=0)) / std  # (n, p)

    if kernel is None:
        kernel = "uniform" if max_k is not None else "gaussian"
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {list(KERNELS)}, got {kernel!r}.")

    coords = _get_coords(geometry)
    tree = cKDTree(coords)

    # --- kNN path -------------------------------------------------------------
    if max_k is not None:
        dists_k, idx_k = tree.query(coords, k=max_k + 1)  # col 0 = self

        bin_centers = numpy.arange(1, max_k + 1, dtype=float)
        C = numpy.empty(max_k)
        n_pairs_arr = numpy.empty(max_k, dtype=int)

        for k in range(1, max_k + 1):
            focal = numpy.repeat(numpy.arange(n), k)
            neighbor = idx_k[:, 1 : k + 1].ravel()
            d_pairs = dists_k[:, 1 : k + 1].ravel()
            # adaptive bandwidth: k-th NN distance + epsilon keeps the
            # boundary neighbour just inside the kernel support
            bw = numpy.repeat(dists_k[:, k] * (1.0 + 1e-6), k)
            weights = KERNELS[kernel](d_pairs / bw)
            C[k - 1] = _weighted_geary(Z, focal, neighbor, weights, p)
            n_pairs_arr[k - 1] = int((weights > 0).sum())

        return Bunch(bin_centers=bin_centers, C=C, n_pairs=n_pairs_arr)

    # --- nonparametric LOWESS path --------------------------------------------
    if nonparametric:
        from scipy.spatial.distance import pdist as _pdist
        from statsmodels.nonparametric.smoothers_lowess import lowess

        dists = _pdist(coords)
        iu = numpy.triu_indices(n, k=1)

        if max_distance is not None and max_distance < dists.max():
            keep = dists <= max_distance
            dists_lw = dists[keep]
            DZ = Z[iu[0][keep]] - Z[iu[1][keep]]
        else:
            dists_lw = dists
            DZ = Z[iu[0]] - Z[iu[1]]

        x_lw = dists_lw**2
        y_lw = (DZ**2).sum(axis=1) / (2.0 * p)

        smoothed = lowess(y_lw, x_lw, return_sorted=True)  # (m, 2)

        x_grid_sq = numpy.linspace(0.0, float(x_lw.max()), n_bins)
        C_smooth = numpy.interp(x_grid_sq, smoothed[:, 0], smoothed[:, 1])

        return Bunch(bin_centers=numpy.sqrt(x_grid_sq), C=C_smooth, n_pairs=None)

    # --- bandwidth path -------------------------------------------------------
    if max_distance is None:
        raise ValueError(
            "max_distance is required for bandwidth mode. "
            "Pass max_k for kNN mode or nonparametric=True for LOWESS."
        )

    bins = numpy.linspace(0.0, max_distance, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    C = numpy.full(n_bins, numpy.nan)
    n_pairs_arr = numpy.zeros(n_bins, dtype=int)

    if kernel in _COMPACT_KERNELS:
        # only pairs within max_distance ever get nonzero weight; build once
        sp = tree.sparse_distance_matrix(
            tree, max_distance=max_distance, output_type="coo_matrix"
        )
        rows = numpy.asarray(sp.row)
        cols = numpy.asarray(sp.col)
        d = numpy.asarray(sp.data, dtype=float)
        ut = rows < cols  # upper triangle — each pair once
        rows, cols, d = rows[ut], cols[ut], d[ut]

        for b, h in enumerate(bin_centers):
            weights = KERNELS[kernel](d / h)
            nz = weights > 0
            m = int(nz.sum())
            if m >= 2:
                C[b] = _weighted_geary(Z, rows[nz], cols[nz], weights[nz], p)
                n_pairs_arr[b] = m
            elif m == 1:
                warnings.warn(
                    f"Bandwidth bin {b} (h={h:.4g}) has only 1 pair with "
                    "nonzero weight; returning NaN.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        # non-compact kernels: all pairs contribute at every bandwidth
        from scipy.spatial.distance import pdist as _pdist

        dists = _pdist(coords)
        iu = numpy.triu_indices(n, k=1)

        for b, h in enumerate(bin_centers):
            weights = KERNELS[kernel](dists / h)
            nz = weights > 0
            m = int(nz.sum())
            if m >= 2:
                C[b] = _weighted_geary(Z, iu[0][nz], iu[1][nz], weights[nz], p)
                n_pairs_arr[b] = m
            elif m == 1:
                warnings.warn(
                    f"Bandwidth bin {b} (h={h:.4g}) has only 1 pair with "
                    "nonzero weight; returning NaN.",
                    UserWarning,
                    stacklevel=2,
                )

    return Bunch(bin_centers=bin_centers, C=C, n_pairs=n_pairs_arr)
