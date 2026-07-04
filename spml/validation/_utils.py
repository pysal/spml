"""Shared helpers for the validation subpackage."""

import numpy

import geopandas

# Maps our kernel names to the equivalent libpysal kernel name.
LIBPYSAL_KERNEL_MAP: dict[str, str] = {
    "gaussian": "gaussian",
    "exponential": "exponential",
    "bisquare": "bisquare",
    "triangular": "triangular",
    "uniform": "boxcar",
    "parabolic": "parabolic",
}


def _to_point_gdf(X) -> geopandas.GeoDataFrame:
    from geopandas.array import GeometryArray

    if isinstance(X, GeometryArray):
        gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(X))
    elif isinstance(X, geopandas.GeoSeries):
        gdf = geopandas.GeoDataFrame(geometry=X)
    else:
        gdf = X.copy()

    if not (gdf.geom_type == "Point").all():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.centroid

    return gdf


def _idx_and_is_geo(X):
    import pandas

    from geopandas.array import GeometryArray

    if isinstance(X, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
        return X.index.to_numpy(), True

    if isinstance(X, GeometryArray):
        return None, True

    if isinstance(X, pandas.Series):
        return X.index.to_numpy(), False

    return None, False


def _get_coords(X) -> numpy.ndarray:
    import pandas

    from geopandas.array import GeometryArray

    if isinstance(X, geopandas.GeoDataFrame):
        c = X.geometry.centroid
        return numpy.column_stack([c.x, c.y])

    if isinstance(X, geopandas.GeoSeries):
        c = X.centroid
        return numpy.column_stack([c.x, c.y])

    if isinstance(X, GeometryArray):
        c = geopandas.GeoSeries(X).centroid
        return numpy.column_stack([c.x, c.y])

    if isinstance(X, pandas.Series):
        idx = X.index
        if isinstance(idx, pandas.DatetimeIndex):
            t = (idx - idx[0]).total_seconds().to_numpy()
        else:
            t = numpy.asarray(idx, dtype=float)
        return t.reshape(-1, 1)

    arr = numpy.asarray(X, dtype=float)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if arr.ndim == 2 and arr.shape[1] in (1, 2):
        return arr

    raise ValueError(
        "Array-like input must be shape (n,), (n, 1), or (n, 2). "
        "Pass a 1-D array for time-series data (distance = absolute lag)."
    )


def _assign_noise_to_nearest(coords, labels, noise_label, clusterer=None):
    """Return a copy of *labels* with noise points reassigned to their nearest non-noise cluster.

    If *clusterer* exposes a ``metric`` attribute the same metric is used for
    the nearest-neighbour search.  Haversine is handled specially: because
    ``_get_coords`` returns ``(lon, lat)`` in degrees while sklearn's haversine
    implementation expects ``(lat, lon)`` in radians, the coordinates are
    converted automatically.
    """
    from sklearn.neighbors import NearestNeighbors

    noise_mask = labels == noise_label
    if not noise_mask.any():
        return labels.copy()

    metric = "euclidean"
    metric_params = {}
    fit_coords = coords

    if clusterer is not None and hasattr(clusterer, "metric"):
        metric = clusterer.metric
        metric_params = getattr(clusterer, "metric_params", None) or {}
        if metric == "haversine":
            # _get_coords returns (lon, lat) in degrees; haversine needs (lat, lon) in radians
            fit_coords = numpy.radians(coords[:, [1, 0]])

    non_noise_mask = ~noise_mask
    nn_kw = dict(n_neighbors=1, metric=metric)
    if metric_params:
        nn_kw["metric_params"] = metric_params

    nbrs = NearestNeighbors(**nn_kw).fit(fit_coords[non_noise_mask])
    _, idx = nbrs.kneighbors(fit_coords[noise_mask])

    new_labels = labels.copy()
    new_labels[noise_mask] = labels[non_noise_mask][idx.ravel()]
    return new_labels


KERNELS: dict = {
    "gaussian": lambda t: numpy.exp(-0.5 * t ** 2),
    "exponential": lambda t: numpy.exp(-t),
    "bisquare": lambda t: numpy.where(t < 1.0, (1.0 - t ** 2) ** 2, 0.0),
    "triangular": lambda t: numpy.where(t < 1.0, 1.0 - t, 0.0),
    "uniform": lambda t: numpy.where(t < 1.0, 1.0, 0.0),
    "parabolic": lambda t: numpy.where(t < 1.0, 0.75 * (1.0 - t ** 2), 0.0),
}
