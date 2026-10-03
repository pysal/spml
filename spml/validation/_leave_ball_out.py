"""Buffered leave-one-out cross-validation for spatial data.

References
----------
.. [1] Ploton et al. "Spatial validation reveals poor
       predictive performance of large-scale ecological
       mapping models." (2020). *Nature Communications*
       11:4454. https://doi.org/10.1038/s41467-020-18321-y
"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from ._utils import _get_coords


class LeaveBallOut(BaseCrossValidator):
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
    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from spml.validation import LeaveBallOut

    >>> gdf = gpd.read_file(get_path('geoda.nyc'))
    >>> lbo = LeaveBallOut(radius=50000)
    >>> for i, (train_index, test_index) in zip(range(5), (lbo.split(gdf))):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 18 19 20 21 22 23 24 25 26 27
     28 29 30 31 32 33 34 35 36 37 38 39 42 43 44 45 46]
      Test:  index=[0]
    Fold 1:
      Train: index=[ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
     27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 51 52]
      Test:  index=[1]
    Fold 2:
      Train: index=[ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
     27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
     51 52 53 54]
      Test:  index=[2]
    Fold 3:
      Train: index=[ 0  1  2 15 16 45 48 49 50 51 52 53 54]
      Test:  index=[3]
    Fold 4:
      Train: index=[ 0  1  2 15 16 32 36 49 52 53 54]
      Test:  index=[4]
    """  # noqa: E501

    def __init__(self, radius: float):
        self.radius = radius

    def split(self, X, y=None, groups=None):  # noqa: ARG002
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
        from scipy.spatial import KDTree

        coords = _get_coords(X)
        n = len(coords)
        r = float(self.radius)
        tree = KDTree(coords)
        indices = np.arange(n)

        for i in range(n):
            # query_ball_point includes i itself (distance 0 < r)
            buffered = np.array(tree.query_ball_point(coords[i], r))
            train_mask = np.ones(n, dtype=bool)
            train_mask[buffered] = False
            yield indices[train_mask], np.array([i])

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        if X is None:
            raise ValueError("X cannot be None.")
        return len(X)
