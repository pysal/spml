import numpy
import pandas as pd
import geopandas
import shapely
from sklearn.utils import check_random_state

from ._base import BasePointSampler
from ._utils import _sample_geometry, _open_raster


class ConstantClassSampler(BasePointSampler):
    """Sample exactly *n_per_class* points from each class.

    Accepts a :class:`geopandas.GeoSeries` / :class:`geopandas.GeoDataFrame`
    paired with a *labels* vector, or a rasterio dataset paired with a 2-D
    numpy array of class labels read from the desired band.

    Parameters
    ----------
    n_per_class : int, default 100
        Exact number of points to generate per class.
    quasi_random : str or None
    random_state : int, RandomState instance, or None

    Examples
    --------
    GeoDataFrame path::

        pts = ConstantClassSampler(n_per_class=50).sample(
            gdf.geometry, gdf["lc_class"]
        )

    Raster path -- read the band yourself, pass it as *labels*::

        with rasterio.open("landcover.tif") as ds:
            pts = ConstantClassSampler(n_per_class=50).sample(ds, ds.read(1))
    """

    def __init__(
        self,
        n_per_class: int = 100,
        quasi_random: str | None = None,
        random_state=None,
    ):
        self.n_per_class = n_per_class
        self.quasi_random = quasi_random
        self.random_state = random_state

    def sample(self, geometry, labels=None) -> geopandas.GeoDataFrame:
        """Generate balanced class samples.

        Parameters
        ----------
        geometry : GeoSeries | GeoDataFrame | rasterio.DatasetReader | path-like
            Spatial source.  For the raster path a ``DatasetReader`` (or file
            path) is used for its transform / CRS / nodata metadata only --
            no band is read from it here.
        labels : array-like of shape (n,) or 2-D ndarray, required
            Class label for each geometry (GDF path) or a 2-D numpy array of
            integer class labels with shape ``(nrows, ncols)`` (raster path,
            e.g. ``ds.read(1)``).

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
            geoseries = (
                geometry.geometry
                if isinstance(geometry, geopandas.GeoDataFrame)
                else geometry
            )
            return self._sample_gdf(
                geoseries, numpy.asarray(labels), self.n_per_class, rng
            )

        return self._sample_raster(geometry, numpy.asarray(labels), self.n_per_class, rng)

    # ------------------------------------------------------------------
    # GeoDataFrame path
    # ------------------------------------------------------------------

    def _sample_gdf(self, geoseries, labels_arr, n_per_class, rng):
        geoms = numpy.asarray(geoseries)
        crs = geoseries.crs
        frames = []
        for label in numpy.unique(labels_arr):
            union = shapely.union_all(geoms[labels_arr == label])
            pts = _sample_geometry(union, n_per_class, rng, self.quasi_random)
            frames.append(
                geopandas.GeoDataFrame({"class_label": label, "geometry": pts}, crs=crs)
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Raster path
    # ------------------------------------------------------------------

    def _sample_raster(self, source, labels_arr, n_per_class, rng):
        import rasterio.transform

        # Use the dataset only for spatial metadata.
        with _open_raster(source) as ds:
            nodata = ds.nodata
            transform = ds.transform
            crs = ds.crs
            res_x, res_y = ds.res

        data = labels_arr if labels_arr.ndim == 2 else labels_arr.squeeze()
        classes = numpy.unique(data)
        if nodata is not None:
            classes = classes[classes != nodata]

        frames = []
        for cls in classes:
            rows, cols = numpy.where(data == cls)
            if rows.size == 0:
                continue
            replace = rows.size < n_per_class
            idx = rng.choice(rows.size, n_per_class, replace=replace)
            r, c = rows[idx], cols[idx]

            xs, ys = rasterio.transform.xy(transform, r, c)
            xs = numpy.asarray(xs, dtype=float) + rng.uniform(
                -res_x / 2, res_x / 2, n_per_class
            )
            ys = numpy.asarray(ys, dtype=float) + rng.uniform(
                -res_y / 2, res_y / 2, n_per_class
            )

            pts = list(shapely.points(xs, ys))
            frames.append(
                geopandas.GeoDataFrame({"class_label": int(cls), "geometry": pts}, crs=crs)
            )

        return pd.concat(frames, ignore_index=True)
