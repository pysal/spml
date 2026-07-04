"""Estimate the range of spatial autocorrelation for bandwidth selection."""

from __future__ import annotations

import warnings

import numpy
from sklearn.utils import check_random_state

from ._utils import _get_coords


def correlogram_range(
    X,
    y,
    n_bins: int = 15,
    max_distance: float | None = None,
    n_sample: int | None = 1000,
    random_state=None,
) -> float:
    """Estimate the distance at which spatial autocorrelation drops to zero.

    Builds an empirical spatial correlogram -- the average normalised
    cross-product ``(y_i - y_bar)(y_j - y_bar) / σ**2`` for all pairs at each
    distance lag -- and returns the lag at which it first crosses zero,
    found by linear interpolation between adjacent bin centres.

    Works for 2-D spatial data and 1-D time-series (pass a 1-D array of
    time indices; distance is then the absolute lag).

    Parameters
    ----------
    X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) ndarray
        Locations or time indices.
    y : array-like of shape (n,)
        Observed values.
    n_bins : int, default 15
        Number of distance bins for the correlogram.
    max_distance : float or None
        Upper limit of the correlogram.  Defaults to half the maximum
        pairwise distance.
    n_sample : int or None
        Subsample to this many points before computing all pairwise
        distances (O(n**2)).  ``None`` uses all points.
    random_state : int, RandomState, or None

    Returns
    -------
    float
        Estimated range.  Returns ``max_distance`` if the correlogram does
        not cross zero within the window.

    Examples
    --------
    >>> d = correlogram_range(gdf, y)
    >>> lb = LocalBootstrap(bandwidth=d)
    >>> lp = LocalPermutation(threshold=d)
    """
    from scipy.spatial.distance import pdist

    rng = check_random_state(random_state)
    coords = _get_coords(X)
    y = numpy.asarray(y, dtype=float)

    if len(coords) != len(y):
        raise ValueError("X and y must have the same length.")

    n = len(coords)
    if n_sample is not None and n > n_sample:
        idx = rng.choice(n, size=n_sample, replace=False)
        coords, y = coords[idx], y[idx]
        n = n_sample

    std = y.std()
    if std == 0:
        raise ValueError("y is constant -- autocorrelation range is undefined.")
    y_z = (y - y.mean()) / std

    # Condensed pairwise distances
    dists = pdist(coords)

    if max_distance is None:
        max_distance = float(dists.max()) / 2.0

    # Pairwise z-score products for the upper triangle
    iu = numpy.triu_indices(n, k=1)
    prods = numpy.outer(y_z, y_z)[iu]

    bins = numpy.linspace(0.0, max_distance, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_corrs = numpy.full(n_bins, numpy.nan)

    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (dists >= lo) & (dists < hi)
        if mask.sum() >= 2:
            bin_corrs[k] = float(prods[mask].mean())

    # Linear interpolation of first zero crossing
    for i in range(n_bins - 1):
        c1, c2 = bin_corrs[i], bin_corrs[i + 1]
        if numpy.isnan(c1) or numpy.isnan(c2):
            continue
        if c1 >= 0.0 > c2:
            d1, d2 = bin_centers[i], bin_centers[i + 1]
            return float(d1 + (d2 - d1) * c1 / (c1 - c2))

    warnings.warn(
        "correlogram_range: the correlogram did not cross zero within "
        f"max_distance={max_distance:.4g}. Consider increasing max_distance "
        "or n_bins. Returning max_distance.",
        UserWarning,
        stacklevel=2,
    )
    return float(max_distance)


def knn_range(X, y, max_k: int = 30) -> int:
    """Estimate the number of nearest neighbours at which spatial
    autocorrelation drops to zero.

    For k = 1, 2, ..., *max_k*, computes the Pearson correlation between
    ``y`` and its k-NN spatial lag (the mean of each observation's k
    nearest neighbours).  Returns the last k at which the correlation is
    still positive, interpolating linearly between that k and the next to
    find the fractional crossing (then rounding down).

    Parameters
    ----------
    X : GeoDataFrame | GeoSeries | (n, 2) ndarray
        Locations.  1-D time-series inputs are not supported.
    y : array-like of shape (n,)
        Observed values.
    max_k : int, default 30
        Maximum k to test.

    Returns
    -------
    int
        Estimated k at the autocorrelation range.  Returns *max_k* if
        correlation never drops to zero within the tested range.

    Examples
    --------
    >>> k = knn_range(gdf, y)
    >>> from libpysal.graph import Graph
    >>> graph = Graph.build_knn(gdf, k=k)
    >>> lb = LocalBootstrap(graph=graph)
    """
    from scipy.spatial import cKDTree

    coords = _get_coords(X)
    if coords.shape[1] == 1:
        raise ValueError(
            "knn_range requires 2-D spatial coordinates, not a 1-D time index."
        )

    y = numpy.asarray(y, dtype=float)
    if len(coords) != len(y):
        raise ValueError("X and y must have the same length.")

    std = y.std()
    if std == 0:
        raise ValueError("y is constant -- autocorrelation range is undefined.")
    y_z = (y - y.mean()) / std

    n = len(coords)
    max_k = min(max_k, n - 1)

    tree = cKDTree(coords)
    # Query all neighbours at once; first column is the point itself
    _, all_idx = tree.query(coords, k=max_k + 1)
    all_idx = all_idx[:, 1:]  # drop self -> shape (n, max_k)

    prev_r, prev_k = None, None

    for k in range(1, max_k + 1):
        lag = y_z[all_idx[:, :k]].mean(axis=1)
        r = float(numpy.corrcoef(y_z, lag)[0, 1])

        if k == 1 and r <= 0:
            warnings.warn(
                "knn_range: correlation is non-positive even at k=1. "
                "The data may lack detectable spatial autocorrelation. "
                "Returning k=1.",
                UserWarning,
                stacklevel=2,
            )
            return 1

        if prev_r is not None and prev_r > 0 and r <= 0:
            return prev_k  # last k with positive correlation

        prev_r, prev_k = r, k

    warnings.warn(
        f"knn_range: correlation did not drop to zero within max_k={max_k}. "
        "Consider increasing max_k. Returning max_k.",
        UserWarning,
        stacklevel=2,
    )
    return max_k
