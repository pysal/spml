"""RadiusNeighborsFeatures transformer."""

import numpy
import pandas
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._utils import _get_feature_coords, _AGG_FUNCS, _fmt_r


class RadiusNeighborsFeatures(TransformerMixin, BaseEstimator):
    """Expand a spatial dataset with distance-band aggregated feature columns.

    For each observation and each input feature column, adds one new column per
    radius band containing a summary statistic of the feature values of all
    training-set neighbors in that band.

    Parameters
    ----------
    radii : sequence of float
        Band edges in the same units as the coordinates.  Need not be sorted.
    func : str or callable, default "mean"
        Aggregation applied within each band.  Accepted strings: ``"mean"``,
        ``"sum"``, ``"median"``, ``"min"``, ``"max"``.  A callable must accept
        a 1-D numpy array and return a scalar.
    exclusive : bool, default True
        If ``True``, bands are non-overlapping annular rings:
        ``(0, r0]``, ``(r0, r1]``, ...
        If ``False``, bands are cumulative buffers:
        ``[0, r0]``, ``[0, r1]``, ...
    metric : str, default "euclidean"
        Distance metric passed to ``scipy.spatial.cKDTree``.

    Notes
    -----
    If *X* is a ``GeoDataFrame`` the geometry column is used automatically and
    excluded from the feature block.  Otherwise pass *geometry* explicitly to
    ``fit`` and ``transform``.

    Bands with no neighbors produce ``NaN`` in that column.

    When ``transform`` is called on the training coordinates (e.g. during
    ``fit_transform``), each observation excludes itself from its own band
    aggregation — detected by a zero nearest-neighbor distance.

    Output column names:

    * ``exclusive=False`` (cumulative): ``<col>_r<radius>``
    * ``exclusive=True`` (annular rings): ``<col>_r<inner>_<outer>``;
      the first ring's inner edge is ``0``.
    """

    def __init__(
        self,
        radii=(500, 1000, 5000),
        func="mean",
        exclusive=True,
        metric="euclidean",
    ):
        self.radii = radii
        self.func = func
        self.exclusive = exclusive
        self.metric = metric

    def fit(self, X, y=None, geometry=None):
        """Build a KD-tree over the training coordinates.

        Parameters
        ----------
        X : GeoDataFrame or DataFrame or array-like of shape (n, p)
            Training features.
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
        """Add distance-band aggregate columns to *X*.

        Parameters
        ----------
        X : GeoDataFrame or DataFrame or array-like of shape (n, p)
            Features to transform.
        geometry : GeoSeries, GeoDataFrame, or (n, 2) array, optional
            Coordinates.  Required when *X* is not a GeoDataFrame.

        Returns
        -------
        DataFrame of shape (n, p + len(radii) * p)
            Original feature columns followed by one aggregated column per
            radius band per input feature.
        """
        check_is_fitted(self, "_tree")
        coords, feature_df = _get_feature_coords(X, geometry)
        agg = self.func if callable(self.func) else _AGG_FUNCS[self.func]
        sorted_radii = sorted(self.radii)
        n = len(coords)

        # Cumulative neighbor lists for each radius
        neighbors_by_r = [
            self._tree.query_ball_point(coords, r, workers=-1)
            for r in sorted_radii
        ]

        # Exclude self (zero-distance) from every radius list
        min_dists, self_idxs = self._tree.query(coords, k=1)
        for i in numpy.where(min_dists == 0.0)[0]:
            si = int(self_idxs[i])
            for nbrs in neighbors_by_r:
                try:
                    nbrs[i].remove(si)
                except ValueError:
                    pass

        # Build per-band neighbor lists
        if self.exclusive:
            bands = [list(neighbors_by_r[0])]
            for k in range(1, len(sorted_radii)):
                inner_sets = [set(nb) for nb in neighbors_by_r[k - 1]]
                bands.append([
                    [nb for nb in neighbors_by_r[k][i] if nb not in inner_sets[i]]
                    for i in range(n)
                ])
            inner_edges = [0] + list(sorted_radii[:-1])
            col_suffixes = [
                f"r{_fmt_r(inn)}_{_fmt_r(out)}"
                for inn, out in zip(inner_edges, sorted_radii)
            ]
        else:
            bands = neighbors_by_r
            col_suffixes = [f"r{_fmt_r(r)}" for r in sorted_radii]

        # Aggregate each band for each feature column
        new_cols = {}
        for j, feat in enumerate(self.feature_names_in_):
            train_col = self._train_values[:, j]
            for band_nbrs, suffix in zip(bands, col_suffixes):
                vals = numpy.array([
                    agg(train_col[nbrs]) if nbrs else numpy.nan
                    for nbrs in band_nbrs
                ])
                new_cols[f"{feat}_{suffix}"] = vals

        return pandas.concat(
            [feature_df, pandas.DataFrame(new_cols, index=feature_df.index)],
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for ``set_output`` compatibility."""
        check_is_fitted(self, "feature_names_in_")
        sorted_radii = sorted(self.radii)
        if self.exclusive:
            inner_edges = [0] + list(sorted_radii[:-1])
            suffixes = [
                f"r{_fmt_r(inn)}_{_fmt_r(out)}"
                for inn, out in zip(inner_edges, sorted_radii)
            ]
        else:
            suffixes = [f"r{_fmt_r(r)}" for r in sorted_radii]

        names = list(self.feature_names_in_)
        for col in self.feature_names_in_:
            for suffix in suffixes:
                names.append(f"{col}_{suffix}")
        return numpy.asarray(names, dtype=object)
