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
import scipy
from packaging.version import Version
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
    geometry,
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
        A 2-D array results in multivariate statistics.
    geometry : GeoDataFrame | GeoSeries | (n, 2) ndarray
        Locations.
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
        bins.  Ignored when *max_k* is set. Requires ``statsmodels`` package.

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
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spml.metrics import gearygram

    >>> gdf = gpd.read_file(get_path('geoda.nyc'))

    Bandwidth correlogram (Gaussian kernel):

    >>> gearygram(gdf["forhis06"], gdf)
    {'bin_centers': array([  5189.69739564,  15569.09218692,  25948.4869782 ,  36327.88176948,
             46707.27656076,  57086.67135204,  67466.06614332,  77845.4609346 ,
             88224.85572587,  98604.25051715, 108983.64530843, 119363.04009971,
            129742.43489099, 140121.82968227, 150501.22447355]),
     'C': array([0.40951283, 0.77018579, 0.94478368, 0.97883804, 0.98355345,
            0.98583428, 0.98896993, 0.99248711, 0.99586266, 0.998867  ,
            1.00144838, 1.00363258, 1.00547127, 1.00701953, 1.00832734]),
     'n_pairs': array([1485, 1485, 1485, 1485, 1485, 1485, 1485, 1485, 1485, 1485, 1485,
            1485, 1485, 1485, 1485])}

    kNN correlogram with Gaussian kernel weighting:

    >>> gearygram(gdf["forhis06"], gdf, max_k=20, kernel="gaussian")
    {'bin_centers': array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
            14., 15., 16., 17., 18., 19., 20.]),
     'C': array([0.41510402, 0.47727678, 0.4995121 , 0.51334353, 0.51458923,
            0.55075855, 0.56997683, 0.62989703, 0.66767808, 0.70706257,
            0.73832148, 0.75745566, 0.76105857, 0.78276971, 0.80870455,
            0.83424149, 0.86179572, 0.88380642, 0.89642786, 0.9066637 ]),
     'n_pairs': array([  55,  110,  165,  220,  275,  330,  385,  440,  495,  550,  605,
             660,  715,  770,  825,  880,  935,  990, 1045, 1100])}

    Nonparametric LOWESS correlogram:

    >>> gearygram(gdf["forhis06"], gdf, nonparametric=True)
    {'bin_centers': array([     0.        ,  41610.14913325,  58845.63723661,  72070.89240931,
            83220.2982665 ,  93043.12201585, 101923.63349757, 110090.10662289,
            117691.27447322, 124830.44739975, 131582.84504035, 138005.25214572,
            144141.78481861, 150027.52627964, 155690.92186919]),
    'C': array([0.50732906, 0.6164969 , 0.54898884, 0.52667128, 0.60943206,
            0.71562291, 0.82443414, 0.96403992, 1.12606386, 1.26639777,
            1.39781745, 1.53063871, 1.66770033, 1.8083954 , 1.95048593]),
    'n_pairs': None}

    Notes
    -----
    Bandwidth bins with fewer than 2 pairs are returned as ``NaN``.
    """  # noqa: E501
    from scipy.spatial import KDTree
    from scipy.spatial.distance import pdist as _pdist

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
    tree = KDTree(coords)

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
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError as e:
            raise ImportError("Nonparametric gearygram require `statsmodels`.") from e

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
        max_distance = float(_pdist(coords).max())

    bins = numpy.linspace(0.0, max_distance, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    C = numpy.full(n_bins, numpy.nan)
    n_pairs_arr = numpy.zeros(n_bins, dtype=int)

    if kernel in _COMPACT_KERNELS:
        # only pairs within max_distance ever get nonzero weight; build once
        if Version(scipy.__version__) >= Version("1.18"):
            sp = tree.sparse_distance_matrix(
                tree, max_distance=max_distance, output_type="coo_array"
            )
        else:
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
