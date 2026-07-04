import numpy
import pandas as pd
import geopandas
import shapely
from sklearn.utils import check_random_state

from ._base import BasePointSampler
from ._utils import _sample_geometry, _allocate_proportionally, _open_raster


class StratifiedClassSampler(BasePointSampler):
    """Sample *n_samples* points total, allocated proportionally across classes.

    When *weights* are provided the allocation for each class is proportional
    to the **sum** of the weight values within that class.  The integer counts
    are resolved by the largest-remainder (Hamilton) method so the total equals
    *n_samples* exactly.

    When *weights* are omitted the sampler falls back to **uniform random
    sampling** over the full union of all class geometries and then assigns
    labels by spatial containment -- equivalent to :class:`PointSampler`.

    Parameters
    ----------
    n_samples : int, default 500
    quasi_random : str or None
    random_state : int, RandomState instance, or None

    Examples
    --------
    GeoDataFrame path::

        pts = StratifiedClassSampler(n_samples=500).sample(
            gdf.geometry, gdf["lc_class"], gdf["area_ha"]
        )

    Raster path -- read bands yourself::

        with rasterio.open("landcover.tif") as ds:
            pts = StratifiedClassSampler(n_samples=500).sample(
                ds, ds.read(1), ds.read(2)   # class band, value band
            )
    """

    def __init__(
        self,
        n_samples: int = 500,
        quasi_random: str | None = None,
        random_state=None,
    ):
        self.n_samples = n_samples
        self.quasi_random = quasi_random
        self.random_state = random_state

    def sample(self, geometry, labels=None, weights=None) -> geopandas.GeoDataFrame:
        """Generate proportionally allocated class samples.

        Parameters
        ----------
        geometry : GeoSeries | GeoDataFrame | rasterio.DatasetReader | path-like
        labels : array-like of shape (n,) or 2-D ndarray, required
            Class label per geometry (GDF) or 2-D integer class array (raster).
        weights : array-like of shape (n,) or 2-D ndarray, optional
            Per-geometry weight (GDF) or 2-D value array (raster).  When
            *None*, falls back to uniform random sampling.

        Returns
        -------
        geopandas.GeoDataFrame
            Columns: ``geometry``, ``class_label``.
        """
        rng = check_random_state(self.random_state)

        if labels is None:
            raise ValueError(
                "labels must be provided.  "
                "Pass a class-label Series/array for GeoDataFrame input, "
                "or ds.read(band) for raster input."
            )

        if isinstance(geometry, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
            geoseries = geometry.geometry if isinstance(geometry, geopandas.GeoDataFrame) else geometry
            labels_arr  = numpy.asarray(labels)
            weights_arr = numpy.asarray(weights, dtype=float) if weights is not None else None
            return self._sample_gdf(geoseries, labels_arr, weights_arr, self.n_samples, rng)

        labels_arr  = numpy.asarray(labels)
        weights_arr = numpy.asarray(weights, dtype=float) if weights is not None else None
        return self._sample_raster(geometry, labels_arr, weights_arr, self.n_samples, rng)

    # ------------------------------------------------------------------
    # GeoDataFrame path
    # ------------------------------------------------------------------

    def _sample_gdf(self, geoseries, labels_arr, weights_arr, n_samples, rng):
        geoms = numpy.asarray(geoseries)
        crs = geoseries.crs

        if weights_arr is None:
            # Uniform: sample from the full union, assign labels by containment.
            union = shapely.union_all(geoms)
            pts = _sample_geometry(union, n_samples, rng, self.quasi_random)
            pts_arr = numpy.asarray(pts)
            out_labels = numpy.empty(n_samples, dtype=object)
            for label in numpy.unique(labels_arr):
                geom_union = shapely.union_all(geoms[labels_arr == label])
                shapely.prepare(geom_union)
                out_labels[shapely.covers(geom_union, pts_arr)] = label
            return geopandas.GeoDataFrame({"class_label": out_labels, "geometry": pts}, crs=crs)

        # Weighted: allocate proportionally to per-class weight sum.
        unique_labels = numpy.unique(labels_arr)
        class_weights = numpy.array([
            weights_arr[labels_arr == label].sum() for label in unique_labels
        ])
        counts = _allocate_proportionally(class_weights / class_weights.sum(), n_samples)

        frames = []
        for label, count in zip(unique_labels, counts):
            if count == 0:
                continue
            geom_union = shapely.union_all(geoms[labels_arr == label])
            pts = _sample_geometry(geom_union, count, rng, self.quasi_random)
            frames.append(geopandas.GeoDataFrame({"class_label": label, "geometry": pts}, crs=crs))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Raster path
    # ------------------------------------------------------------------

    def _sample_raster(self, source, labels_arr, weights_arr, n_samples, rng):
        import rasterio.transform

        with _open_raster(source) as ds:
            nodata = ds.nodata
            transform = ds.transform
            crs = ds.crs
            res_x, res_y = ds.res

        class_data = labels_arr if labels_arr.ndim == 2 else labels_arr.squeeze()

        if weights_arr is None:
            # Uniform: sample from all valid pixels.
            valid_mask = (class_data != nodata) if nodata is not None \
                         else numpy.ones_like(class_data, dtype=bool)
            valid_rows, valid_cols = numpy.where(valid_mask)
            replace = valid_rows.size < n_samples
            idx = rng.choice(valid_rows.size, n_samples, replace=replace)
            r, c = valid_rows[idx], valid_cols[idx]

            xs, ys = rasterio.transform.xy(transform, r, c)
            xs = numpy.asarray(xs, dtype=float) + rng.uniform(-res_x / 2, res_x / 2, n_samples)
            ys = numpy.asarray(ys, dtype=float) + rng.uniform(-res_y / 2, res_y / 2, n_samples)
            pts = list(shapely.points(xs, ys))
            return geopandas.GeoDataFrame(
                {"class_label": class_data[r, c].astype(int), "geometry": pts}, crs=crs
            )

        # Weighted: allocate proportionally to per-class pixel sum.
        val_data = (weights_arr if weights_arr.ndim == 2 else weights_arr.squeeze()).copy()
        if nodata is not None:
            val_data[class_data == nodata] = 0.0
        val_data = numpy.clip(val_data, 0, None)

        classes = numpy.unique(class_data)
        if nodata is not None:
            classes = classes[classes != nodata]

        class_weights = numpy.array([val_data[class_data == cls].sum() for cls in classes])
        counts = _allocate_proportionally(class_weights / class_weights.sum(), n_samples)

        frames = []
        for cls, count in zip(classes, counts):
            if count == 0:
                continue
            rows, cols = numpy.where(class_data == cls)
            if rows.size == 0:
                continue
            replace = rows.size < count
            idx = rng.choice(rows.size, count, replace=replace)
            r, c = rows[idx], cols[idx]

            xs, ys = rasterio.transform.xy(transform, r, c)
            xs = numpy.asarray(xs, dtype=float) + rng.uniform(-res_x / 2, res_x / 2, count)
            ys = numpy.asarray(ys, dtype=float) + rng.uniform(-res_y / 2, res_y / 2, count)
            pts = list(shapely.points(xs, ys))
            frames.append(geopandas.GeoDataFrame({"class_label": int(cls), "geometry": pts}, crs=crs))
        return pd.concat(frames, ignore_index=True)
