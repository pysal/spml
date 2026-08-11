"""Hilbert Curve Crossvalidation

References
----------
.. [1] Lister, Andrew and Scott, Charles T. "Use of space-fillting
       curves to select sample locations in natural resource monitoring
       studies." (2009). *Environment Monitoring and Assessment* 149: 71-80.
       https://doi.org/10.1007/s10661-008-0184-y
"""

import numpy
import geopandas
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from ._utils import _get_coords


def _hilbert_sort(X, level: int = 16) -> numpy.ndarray:
    """Return indices that sort X along the Hilbert curve.

    For GeoDataFrame/GeoSeries input, calls ``hilbert_distance()`` directly
    on the original geometries so that polygon extents inform the Hilbert
    grid bounds -- not the centroid point cloud.  For array/Series input,
    falls back to a centroid-point sort (2-D) or a plain value sort (1-D).
    """
    from geopandas.array import GeometryArray

    if isinstance(X, geopandas.GeoDataFrame):
        hd = X.geometry.hilbert_distance(level=level).values
        return numpy.argsort(hd, kind="stable")

    if isinstance(X, (geopandas.GeoSeries, GeometryArray)):
        gs = X if isinstance(X, geopandas.GeoSeries) else geopandas.GeoSeries(X)
        hd = gs.hilbert_distance(level=level).values
        return numpy.argsort(hd, kind="stable")

    coords = _get_coords(X)
    if coords.shape[1] == 1:
        return numpy.argsort(coords[:, 0])
    gs = geopandas.GeoSeries(geopandas.points_from_xy(coords[:, 0], coords[:, 1]))
    return numpy.argsort(gs.hilbert_distance(level=level).values, kind="stable")


class HilbertKFold(BaseEstimator):
    """Spatially balanced k-fold cross-validator via Hilbert curve ordering. [1]

    Partitions observations into *n_splits* folds such that each fold is a
    **spatially spread-out** subsample covering the entire study area.
    Points are sorted along the **Hilbert curve** (via
    ``GeoSeries.hilbert_distance``); every k-th point in this ordering is
    assigned to the same fold.

    Because the Hilbert curve is continuous with no quadrant-boundary jumps,
    spatially nearby points always receive consecutive codes, so cycling
    through the sorted order reliably sends neighbours to *different* folds.
    The result is analogous to a quasi-random spatial sample: each fold is a
    low-discrepancy subsample of the full dataset rather than a compact
    geographic block.

    This maximises the **mean within-fold nearest-neighbour distance** and
    ensures each fold's training set is geographically representative of the
    whole region -- the spatial equivalent of stratified random sampling.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    level : int (1-16), default 16
        Hilbert curve precision.  Higher values give finer spatial
        discrimination; 16 is sufficient for any practical dataset.
    random_state : int, RandomState instance, or None
        Controls within-band shuffle (breaks ties among points with the
        same Hilbert code).

    Examples
    --------
    >>> skf = HilbertKFold(n_splits=5, random_state=0)
    >>> for train_idx, test_idx in skf.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(self, n_splits: int = 5, level: int = 16, random_state=None):
        self.n_splits = n_splits
        self.level = level
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each spatial fold.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Locations.  Pass a 1-D array of time indices for time-series data.
        y, groups : ignored, present for sklearn API compatibility.

        Yields
        ------
        train : ndarray of int
        test  : ndarray of int
        """
        n = len(X)
        k = self.n_splits
        if n < k:
            raise ValueError(f"n_splits={k} cannot exceed n_samples={n}.")

        rng = check_random_state(self.random_state)
        order = _hilbert_sort(X, level=self.level)

        # Assign fold IDs in bands of k along the Hilbert curve, with a
        # random within-band shuffle for reproducible but non-deterministic
        # fold membership.
        band_labels = numpy.empty(n, dtype=int)
        for start in range(0, n, k):
            end = min(start + k, n)
            band = numpy.arange(end - start)
            rng.shuffle(band)
            band_labels[start:end] = band

        labels = numpy.empty(n, dtype=int)
        labels[order] = band_labels

        indices = numpy.arange(n)
        for fold in range(k):
            test = indices[labels == fold]
            train = indices[labels != fold]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
