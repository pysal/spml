"""Shared helpers for the preprocessing subpackage."""

import numpy
import pandas
import geopandas


_AGG_FUNCS = {
    "mean":   numpy.mean,
    "sum":    numpy.sum,
    "median": numpy.median,
    "min":    numpy.min,
    "max":    numpy.max,
}


def _get_feature_coords(X, geometry=None):
    """Return ``(coords, feature_df)`` from *X* and optional *geometry*.

    If *X* is a GeoDataFrame the geometry column is used automatically and
    excluded from *feature_df*.  Otherwise *geometry* must be provided as a
    GeoSeries, GeoDataFrame, or ``(n, 2)`` coordinate array.
    """
    if isinstance(X, geopandas.GeoDataFrame):
        if geometry is not None:
            raise ValueError("Do not pass geometry when X is a GeoDataFrame.")
        geom = X.geometry.centroid
        coords = numpy.column_stack([geom.x.to_numpy(), geom.y.to_numpy()])
        feature_df = X.drop(columns=X.geometry.name)
        return coords, feature_df

    if geometry is None:
        raise ValueError(
            "geometry must be provided when X is not a GeoDataFrame."
        )

    if isinstance(geometry, geopandas.GeoDataFrame):
        c = geometry.geometry.centroid
        coords = numpy.column_stack([c.x.to_numpy(), c.y.to_numpy()])
    elif isinstance(geometry, geopandas.GeoSeries):
        c = geometry.centroid
        coords = numpy.column_stack([c.x.to_numpy(), c.y.to_numpy()])
    else:
        arr = numpy.asarray(geometry, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "geometry must be a GeoSeries, GeoDataFrame, or (n, 2) array."
            )
        coords = arr

    feature_df = X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X)
    return coords, feature_df


def _fmt_r(r):
    """Format a radius value without trailing '.0' for integer values."""
    return str(int(r)) if r == int(r) else str(r)
