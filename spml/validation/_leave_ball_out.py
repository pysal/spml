"""Buffered leave-one-out cross-validation for spatial data.

References
----------
.. [1] Ploton et al. "Spatial validation reveals poor
       predictive performance of large-scale ecological
       mapping models." (2020). *Nature Communications*
       11:4454. https://doi.org/10.1038/s41467-020-18321-y
"""

import numpy
from sklearn.base import BaseEstimator

from ._utils import _get_coords


class LeaveBallOut(BaseEstimator):
    """Buffered leave-one-out cross-validator.

    For each observation i the test set is ``{i}`` and the training set
    is every observation farther than *radius* from i.  Points inside
    the buffer zone are dropped entirely -- they are neither in train
    nor in test.

    Excluding nearby training points prevents spatial autocorrelation
    leakage: observations that are close enough to give the model an
    unfair advantage are withheld when predicting point i.

    Parameters
    ----------
    radius : float
        Exclusion radius in the same units as the input coordinates.
        All points within this distance of the test point are excluded
        from the training set for that iteration.

    Notes
    -----
    For dense datasets or large radii, many training points will be
    dropped per iteration and some folds may have very small training
    sets.  Consider ``BallKFold`` when a guaranteed minimum training
    size matters more than strict per-point buffering.

    Examples
    --------
    >>> lbo = LeaveBallOut(radius=2000)
    >>> for train_idx, test_idx in lbo.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(self, radius: float):
        self.radius = radius

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each observation.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations.
        y, groups : ignored, present for sklearn API compatibility.

        Yields
        ------
        train : ndarray of int
            All observations farther than *radius* from the test point.
        test : ndarray of int
            Single-element array containing the index of point i.
        """
        from scipy.spatial import cKDTree

        coords = _get_coords(X)
        n = len(coords)
        r = float(self.radius)
        tree = cKDTree(coords)
        indices = numpy.arange(n)

        for i in range(n):
            # query_ball_point includes i itself (distance 0 < r)
            buffered = numpy.array(tree.query_ball_point(coords[i], r))
            train_mask = numpy.ones(n, dtype=bool)
            train_mask[buffered] = False
            yield indices[train_mask], numpy.array([i])

    def get_n_splits(self, X, y=None, groups=None) -> int:
        return len(X)
