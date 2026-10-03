import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from sklearn.utils import check_random_state

from ._base import BasePointSampler
from ._utils import _collect_geometries, _open_raster, _sample_geometry


class MultinomialSampler(BasePointSampler):
    r"""Sample ``n_samples`` points via two-stage multinomial allocation.

    **Stage 1 -- allocate sample counts across classes:**

    Sum ``weights`` within each ``labels`` group to get a per-class total weight
    :math:`W_k`.  Draw class sample counts jointly from

    .. math::

        (n_1, n_2, \ldots, n_K) \sim
        \operatorname{Multinomial}\left(
            n_{\mathrm{samples}}, \frac{W_k}{\sum_k W_k}
        \right)

    so counts are stochastic but always sum exactly to ``n_samples``.

    **Stage 2 -- place points within each class:**

    Sample :math:`n_k` points uniformly at random from within the union of
    geometries belonging to class :math:`k`.

    When ``weights`` are omitted every geometry / pixel is assigned weight 1,
    so :math:`W_k` equals the number of observations in class :math:`k`.

    This differs from :class:`StratifiedClassSampler`, which uses the
    deterministic largest-remainder (Hamilton) method for allocation.

    Parameters
    ----------
    n_samples : int, default 500
        Total number of points to generate.
    quasi_random : str or None
        # TODO: this should use the same wording as elsewhere. This is super opaque.
        Low-discrepancy sequence for within-class coordinate generation.
    random_state : int, RandomState instance, or None
        Seed / random state passed to ``sklearn.utils.check_random_state``.
        TODO: all random_state docstrings across spml shoud say the same

    Examples
    --------
    GeoDataFrame path::

        pts = MultinomialSampler(n_samples=500).sample(
            gdf.geometry, gdf["lc_class"], gdf["area_ha"]
        )

    Raster path -- read bands yourself::

        with rasterio.open("landcover.tif") as ds:
            pts = MultinomialSampler(n_samples=500).sample(
                ds, ds.read(1), ds.read(2)   # class band, weight band
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

    def sample(self, geometry, labels=None, weights=None) -> gpd.GeoDataFrame:  # ty:ignore[invalid-method-override]
        """Generate multinomially allocated class samples.

        Parameters
        ----------
        geometry : GeoSeries | GeoDataFrame | rasterio.DatasetReader | path-like
            Spatial source.
        labels : array-like of shape (n,) or 2-D ndarray, required
            Class label per geometry (GeoDataFrame) or 2-D integer class array with
            shape ``(nrows, ncols)`` (raster, e.g. ``ds.read(1)``).
        weights : array-like of shape (n,) or 2-D ndarray, optional
            Non-negative weight per geometry (GeoDataFrame) or per pixel (raster).
            When *None*, all weights default to 1 so class counts are
            proportional to class size.

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

        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geoseries = (
                geometry.geometry
                if isinstance(geometry, gpd.GeoDataFrame)
                else geometry
            )
            labels_arr = np.asarray(labels)
            weights_arr = (
                np.asarray(weights, dtype=float)
                if weights is not None
                else np.ones(len(labels_arr), dtype=float)
            )
            return self._sample_gdf(
                geoseries, labels_arr, weights_arr, self.n_samples, rng
            )

        labels_arr = np.asarray(labels)
        weights_arr = np.asarray(weights, dtype=float) if weights is not None else None
        return self._sample_raster(
            geometry, labels_arr, weights_arr, self.n_samples, rng
        )

    # ------------------------------------------------------------------
    # GeoDataFrame path
    # ------------------------------------------------------------------

    def _sample_gdf(self, geoseries, labels_arr, weights_arr, n_samples, rng):
        weights_arr = np.clip(weights_arr, 0.0, None)
        geoms = np.asarray(geoseries)
        crs = geoseries.crs

        unique_labels = np.unique(labels_arr)
        class_totals = np.array(
            [weights_arr[labels_arr == lbl].sum() for lbl in unique_labels]
        )

        total = class_totals.sum()
        if total == 0:
            raise ValueError(
                "All weights are zero -- cannot define class probabilities."
            )

        probs = class_totals / total
        counts = rng.multinomial(n_samples, probs)

        frames = []
        for lbl, count in zip(unique_labels, counts, strict=True):
            if count == 0:
                continue
            geom_union = _collect_geometries(geoms[labels_arr == lbl])
            pts = _sample_geometry(geom_union, count, rng, self.quasi_random)
            frames.append(
                gpd.GeoDataFrame({"class_label": lbl, "geometry": pts}, crs=crs)
            )

        if not frames:
            return gpd.GeoDataFrame(
                {"geometry": [], "class_label": []},
                crs=crs,
            )
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

        classes = np.unique(class_data)
        if nodata is not None:
            classes = classes[classes != nodata]

        if weights_arr is None:
            # Default: weight = 1 per pixel -> class weight = pixel count
            class_totals = np.array(
                [np.sum(class_data == cls) for cls in classes], dtype=float
            )
        else:
            val_data = (
                weights_arr if weights_arr.ndim == 2 else weights_arr.squeeze()
            ).copy()
            if nodata is not None:
                val_data[class_data == nodata] = 0.0
            val_data = np.clip(val_data, 0.0, None)
            class_totals = np.array(
                [val_data[class_data == cls].sum() for cls in classes]
            )

        total = class_totals.sum()
        if total == 0:
            raise ValueError(
                "All weights are zero -- cannot define class probabilities."
            )

        probs = class_totals / total
        counts = rng.multinomial(n_samples, probs)

        frames = []
        for cls, count in zip(classes, counts, strict=True):
            if count == 0:
                continue
            rows, cols = np.where(class_data == cls)
            if rows.size == 0:
                continue
            replace = rows.size < count
            idx = rng.choice(rows.size, count, replace=replace)
            r, c = rows[idx], cols[idx]

            xs, ys = rasterio.transform.xy(transform, r, c)
            xs = np.asarray(xs, dtype=float) + rng.uniform(-res_x / 2, res_x / 2, count)
            ys = np.asarray(ys, dtype=float) + rng.uniform(-res_y / 2, res_y / 2, count)
            pts = list(shapely.points(xs, ys))
            frames.append(
                gpd.GeoDataFrame({"class_label": int(cls), "geometry": pts}, crs=crs)
            )

        if not frames:
            return gpd.GeoDataFrame(
                {"geometry": [], "class_label": []},
                crs=crs,
            )
        return pd.concat(frames, ignore_index=True)
