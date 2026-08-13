"""KNeighborsFeatures transformer."""

import numpy
import pandas
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._utils import _get_feature_coords


class KNeighborsFeatures(TransformerMixin, BaseEstimator):
    """Expand a spatial dataset with k nearest-neighbor feature columns.

    For each observation and each input feature column, adds *k* new columns
    holding the feature value at the 1st, 2nd, ..., k-th nearest training-set
    neighbor.  Output columns are named ``<feature>_nn1``, ``<feature>_nn2``,
    ..., ``<feature>_nnk``.

    Parameters
    ----------
    k : int, default 5
        Number of neighbor ranks to add.
    metric : str, default "euclidean"
        Distance metric passed to ``scipy.spatial.cKDTree``.

    Notes
    -----
    If *X* is a ``GeoDataFrame`` the geometry column is used automatically and
    excluded from the feature block.  Otherwise pass *geometry* explicitly to
    ``fit`` and ``transform``.

    When ``transform`` is called on the training coordinates (e.g. during
    ``fit_transform``), each observation excludes itself from its own neighbor
    list — detected by a zero nearest-neighbor distance.
    """

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y=None, geometry=None):
        """Build a KD-tree over the training coordinates.

        Parameters
        ----------
        X : GeoDataFrame or DataFrame or array-like of shape (n, p)
            Training features.  Geometry is extracted from a GeoDataFrame
            automatically; otherwise pass *geometry* explicitly.
        y : ignored
        geometry : GeoSeries, GeoDataFrame, or (n, 2) array, optional
            Coordinates.  Required when *X* is not a GeoDataFrame.

        Returns
        -------
        self
        """
        coords, feature_df = _get_feature_coords(X, geometry)
        self._tree = cKDTree(coords)
        self._train_values = feature_df.to_numpy(dtype=float)
        self.feature_names_in_ = numpy.asarray(feature_df.columns)
        self.n_features_in_ = len(feature_df.columns)
        return self

    def transform(self, X, geometry=None):
        """Add k nearest-neighbor columns to *X*.

        Parameters
        ----------
        X : GeoDataFrame or DataFrame or array-like of shape (n, p)
            Features to transform.
        geometry : GeoSeries, GeoDataFrame, or (n, 2) array, optional
            Coordinates.  Required when *X* is not a GeoDataFrame.

        Returns
        -------
        DataFrame of shape (n, p + k * p)
            Original feature columns followed by ``<col>_nn1`` ...
            ``<col>_nnk`` for each input feature column.
        """
        check_is_fitted(self, "_tree")
        coords, feature_df = _get_feature_coords(X, geometry)

        # Query k+1 neighbors; if the nearest has distance 0 it is self — skip it.
        dists, idxs = self._tree.query(coords, k=self.k + 1)
        has_self = dists[:, 0] == 0.0
        selected = numpy.where(
            has_self[:, None],
            idxs[:, 1:self.k + 1],
            idxs[:, :self.k],
        )

        new_cols = {}
        for j, col in enumerate(self.feature_names_in_):
            for rank in range(self.k):
                new_cols[f"{col}_nn{rank + 1}"] = (
                    self._train_values[selected[:, rank], j]
                )

        return pandas.concat(
            [feature_df, pandas.DataFrame(new_cols, index=feature_df.index)],
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for ``set_output`` compatibility."""
        check_is_fitted(self, "feature_names_in_")
        names = []
        for col in self.feature_names_in_:
            names.append(col)
        for col in self.feature_names_in_:
            for rank in range(1, self.k + 1):
                names.append(f"{col}_nn{rank}")
        return numpy.asarray(names, dtype=object)
