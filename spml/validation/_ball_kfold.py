"""Distance-based exclusion for ball k-fold crossvalidation

References
----------
.. [1] Ploton et al. "Spatial validation reveals poor
        predictive performance of large-scale ecological
        mapping models. (2020). *Nature Communications*
        11:45450. https://doi.org/10.1038/s41467-020-18321-y
"""

import warnings

import geopandas as gpd
import numpy as np
from libpysal import weights

from ._utils import _get_coords


class BallKFold:
    """Spatially exclusive k-fold cross-validator.

    Assigns observations to folds such that no two observations in the
    same fold are within radius *r* of each other.  Each fold's test set
    is a spatial independent set in the conflict graph whose edges
    connect pairs of points within *r*.

    The number of folds is determined by greedy graph colouring and
    equals at most the maximum number of points within *r* of any single
    point plus one.  Colouring uses a count-balanced algorithm,
    which produces the folds or comparable sizes.

    Parameters
    ----------
    radius : float or None
        Exclusion radius in the same units as the input coordinates.
        Two points within this distance cannot share a fold.
        Mutually exclusive with *n_splits*.
    n_splits : int or None
        Target number of folds.  The implied radius is computed as the
        minimum n_splits-th nearest-neighbour distance across all points
        (set by the densest region), stored as ``radius_`` after
        ``split()`` is called.  Mutually exclusive with *radius*.

    Notes
    -----
    When *n_splits* is given, the actual number of folds returned by
    ``split()`` may be less than *n_splits* if the conflict graph is
    sparse enough to colour with fewer colours.  It will not exceed
    *n_splits* by construction (the radius choice guarantees max degree
    <= n_splits - 1, and greedy colouring uses at most max_degree + 1
    colours).
    """

    def __init__(self, radius=None, n_splits=None):
        if (radius is None) == (n_splits is None):
            raise ValueError("Specify exactly one of 'radius' or 'n_splits'.")
        self.radius = radius
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):  # noqa: ARG002
        """Yield ``(train_indices, test_indices)`` for each fold.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations.

        Yields
        ------
        train : ndarray of int
        test  : ndarray of int
        """

        coords = _get_coords(X)
        n = len(coords)

        if self.radius is not None:
            r = float(self.radius)
        else:
            if self.n_splits is None:
                raise ValueError("Specify exactly one of 'radius' or 'n_splits'.")
            # r = minimum n_splits-th NN distance across all points.
            # Every point then has at most n_splits-1 neighbours within r,
            # so greedy colouring uses at most n_splits colours.
            from scipy.spatial import KDTree

            tree = KDTree(coords)
            distances, _ = tree.query(coords, k=self.n_splits + 1)
            r = float(distances[:, self.n_splits].min())
            self.radius_ = r

        colors = self._color(X, coords, r)
        self.n_splits_ = int(colors.max() + 1)

        indices = np.arange(n)
        for fold in range(self.n_splits_):
            test = indices[colors == fold]
            train = indices[colors != fold]
            yield train, test

    def _color(self, X, coords, r):
        """Return integer color array using mapclassify's balanced greedy colouring."""

        import mapclassify

        if not isinstance(X, (gpd.GeoDataFrame, gpd.GeoSeries)):
            X = gpd.GeoSeries.from_xy(coords[:, 0], coords[:, 1])

        sparr = X.sindex.query(
            X, predicate="dwithin", distance=r, output_format="sparse"
        )
        sparr.setdiag(0)

        # TODO: let mapclassify.greedy understand Graph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            W = weights.W.from_sparse(sparr)
        return mapclassify.greedy(X, sw=W, strategy="balanced", balance="count").values

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: ARG002
        if hasattr(self, "n_splits_"):
            return self.n_splits_
        raise ValueError("Call split(X) first to determine the number of folds.")
