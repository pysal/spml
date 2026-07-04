"""
Inhomogeneous Poisson point process (IPPP) sampler.

Four intensity specifications are supported:

  callable       - λ(x, y) -> array  (Lewis-Shedler thinning)
  rasterio.DatasetReader / 2-D ndarray
                 - pixel image  (pixel-selection or bilinear-thinning)
  1-D numeric array aligned to polygon geometry
                 - per-polygon values burned to a raster
  point pattern  - GeoSeries / GeoDataFrame / (N, 2) ndarray
                 -> kernel-density intensity estimate

Unlike the fixed-n samplers, the output count N follows
Poisson(∫∫_W λ(x,y) dA) and is itself random.  Use *n_expected* to
pin the expected count; otherwise the raw integral of the intensity
surface determines E[N].
"""

from __future__ import annotations

import numpy
import geopandas
import shapely

from ._base import BasePointSampler


class PoissonSampler(BasePointSampler):
    """
    Inhomogeneous Poisson point process sampler.

    The number of returned points is random -- N ~ Poisson(∫∫_W λ(x,y) dA).
    Set *n_expected* to control the expected count; the sampler computes the
    normalisation constant K = ∫∫_W λ dA internally and derives
    scale = n_expected / K.  When *n_expected* is None the raw integral of
    the intensity surface sets E[N].

    Four ways to specify the intensity surface λ(x,y) are accepted by
    :meth:`sample`:

    * **callable** ``f(x, y) -> array`` -- intensity in points per unit area.
      Receives two 1-D NumPy arrays of x and y coordinates and must return
      a 1-D array of the same length.  Lewis-Shedler thinning is used.
    * **rasterio.DatasetReader** or **2-D ndarray** -- pixel image.
      With ``interpolation='nearest'`` the pixel-selection algorithm is used.
      With ``interpolation='linear'`` bilinear interpolation + Lewis-Shedler
      thinning.
    * **1-D numeric array / Series** (when *geometry* is a GeoSeries /
      GeoDataFrame of Polygons) -- per-feature values burned into a 512-cell
      raster aligned to *geometry*.  E.g.
      ``sample(df.geometry, df.population)``.
    * **GeoSeries / GeoDataFrame of Points / (N, 2) ndarray** -- an observed
      point pattern; a kernel-density estimate of its intensity is used.

    Parameters
    ----------
    n_expected : float or None
        Expected number of points per call to :meth:`sample`.  Internally
        converted to a scale factor via scale = n_expected / K, where K is
        the normalisation constant of the intensity surface.  When None the
        raw integral K sets E[N].
    bandwidth : float or None
        Kernel bandwidth for KDE mode, in the same units as the CRS.
        ``None`` applies Scott's rule.
    kernel : str
        Kernel for KDE mode.  Passed to
        ``sklearn.neighbors.KernelDensity``; typical values are
        ``'gaussian'`` (default), ``'tophat'``, ``'epanechnikov'``.
    interpolation : {'nearest', 'linear'}
        Pixel-interpolation strategy for raster input.
        ``'nearest'`` uses the pixel-selection algorithm (fast).
        ``'linear'`` uses bilinear interpolation + Lewis-Shedler thinning.
    random_state : int or None
    """

    def __init__(
        self,
        n_expected: float | None = None,
        bandwidth: float | None = None,
        kernel: str = "gaussian",
        interpolation: str = "linear",
        random_state: int | None = None,
    ):
        self.n_expected = n_expected
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.interpolation = interpolation
        self.random_state = random_state

    # -- public ----------------------------------------------------------------

    def sample(self, geometry, intensity):
        """
        Generate an inhomogeneous Poisson point pattern inside *geometry*.

        Parameters
        ----------
        geometry : shapely.Geometry | GeoSeries | GeoDataFrame | None
            Sampling window.  CRS is inferred automatically from GeoSeries /
            GeoDataFrame input.  Pass ``None`` when *intensity* is a
            rasterio DatasetReader -- the window and CRS are then taken
            directly from the raster.
        intensity : callable | DatasetReader | ndarray (2-D) | array-like (1-D) | GeoSeries | ndarray (N, 2)
            Intensity surface (points per unit area):

            * **callable** ``f(x, y)`` -- receives two 1-D NumPy arrays and
              must return a 1-D array of intensities at those locations.
            * **rasterio.DatasetReader** -- first band used.  *geometry* may
              be ``None`` to use the full raster extent.
            * **2-D ndarray** -- pixel values, rows=north->south,
              columns=west->east, domain aligned to *geometry*'s bounding box.
            * **1-D numeric array / Series** (when *geometry* contains
              Polygons) -- per-feature intensity values burned into a raster.
              E.g. ``sample(df.geometry, df.population)``.
            * **GeoSeries / GeoDataFrame of Points / (N, 2) ndarray** --
              a KDE is fitted and used as the intensity.

        Returns
        -------
        geopandas.GeoDataFrame
            Column: ``geometry`` (Point objects).
        """
        rng = numpy.random.default_rng(self.random_state)
        crs = None

        # -- resolve sampling window ----------------------------------------
        # When geometry is None and intensity is a rasterio dataset, derive
        # the window and CRS from the raster itself.
        if geometry is None:
            try:
                import rasterio as _rio
                from shapely.geometry import box as _box
                if not isinstance(intensity, _rio.DatasetReader):
                    raise ValueError(
                        "geometry=None is only supported when intensity is a "
                        "rasterio DatasetReader."
                    )
                crs = intensity.crs.to_string() if intensity.crs else None
                window = _box(*intensity.bounds)
            except ImportError:
                raise ValueError(
                    "geometry=None requires rasterio to be installed."
                )
        elif isinstance(geometry, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
            crs = geometry.crs
            window = geometry.union_all()
            if window.area == 0:
                window = window.convex_hull
        else:
            window = geometry

        # -- detect polygon-geometry case -----------------------------------
        if isinstance(geometry, geopandas.GeoDataFrame):
            geo_geoms = geometry.geometry
        elif isinstance(geometry, geopandas.GeoSeries):
            geo_geoms = geometry
        else:
            geo_geoms = None

        is_poly_geometry = (
            geo_geoms is not None
            and geo_geoms.geom_type.isin({"Polygon", "MultiPolygon"}).any()
        )

        # -- dispatch on intensity type -------------------------------------
        try:
            import rasterio as _rio

            _is_ds = isinstance(intensity, _rio.DatasetReader)
        except ImportError:
            _is_ds = False

        if (
            is_poly_geometry
            and not callable(intensity)
            and not _is_ds
            and not isinstance(intensity, (geopandas.GeoDataFrame, geopandas.GeoSeries))
        ):
            values = numpy.asarray(intensity, dtype=float)
            coords = self._from_polygons(window, geo_geoms, values, rng)
        elif callable(intensity):
            coords = self._thinning(window, intensity, rng)
        elif _is_ds:
            coords = self._from_raster_ds(window, intensity, rng)
        elif (
            isinstance(intensity, numpy.ndarray)
            and intensity.ndim == 2
            and intensity.shape[1] != 2
        ):
            coords = self._from_raster_array(window, intensity, rng)
        else:
            coords = self._from_kde(window, intensity, rng)

        if len(coords) == 0:
            if self.n_expected:
                import warnings
                warnings.warn(
                    "PoissonSampler returned no points despite n_expected="
                    f"{self.n_expected}. The intensity surface is zero or "
                    "negative everywhere inside the window.",
                    UserWarning,
                    stacklevel=2,
                )
            pts = geopandas.GeoSeries([], dtype="geometry", crs=crs)
        else:
            pts = geopandas.GeoSeries(
                geopandas.points_from_xy(coords[:, 0], coords[:, 1]), crs=crs
            )
        return geopandas.GeoDataFrame({"geometry": pts})

    # -- scale helper ----------------------------------------------------------

    def _effective_scale(self, K: float) -> float:
        """Return scale = n_expected / K, or 1.0 when n_expected is None."""
        if self.n_expected is None or K <= 0.0:
            return 1.0
        return float(self.n_expected) / K

    # -- Lewis-Shedler thinning ------------------------------------------------

    def _thinning(self, window, fn, rng):
        """Accept-reject thinning for a vectorised intensity callable.

        K is estimated from a 64x64 coarse grid over the bounding box.
        """
        minx, miny, maxx, maxy = window.bounds
        bbox_area = (maxx - minx) * (maxy - miny)

        g = 64
        gx = numpy.linspace(minx, maxx, g)
        gy = numpy.linspace(miny, maxy, g)
        XX, YY = numpy.meshgrid(gx, gy)
        Z = numpy.asarray(fn(XX.ravel(), YY.ravel()), dtype=float)
        Z = numpy.where(numpy.isfinite(Z) & (Z > 0.0), Z, 0.0)

        K = float(Z.mean()) * bbox_area
        scale = self._effective_scale(K)

        lambda_max = float(Z.max()) * scale * 1.05
        if lambda_max <= 0.0:
            return []

        N = int(rng.poisson(lambda_max * bbox_area))
        if N == 0:
            return []

        xs = rng.uniform(minx, maxx, N)
        ys = rng.uniform(miny, maxy, N)

        lam = numpy.asarray(fn(xs, ys), dtype=float) * scale
        lam = numpy.where(numpy.isfinite(lam), numpy.maximum(lam, 0.0), 0.0)
        keep = rng.random(N) < (lam / lambda_max)
        xs, ys = xs[keep], ys[keep]

        inside = shapely.contains(window, shapely.points(xs, ys))
        return numpy.vstack((xs[inside], ys[inside])).T

    # -- raster input ----------------------------------------------------------

    def _from_raster_ds(self, window, ds, rng):
        from rasterio.mask import mask as rio_mask

        geom_json = [window.__geo_interface__]
        out, transform = rio_mask(ds, geom_json, crop=True, filled=False)
        arr = numpy.ma.filled(out[0], fill_value=0).astype(float)
        arr = numpy.maximum(arr, 0.0)

        if self.interpolation == "linear":
            return self._bilinear_thinning(window, arr, transform, rng)
        return self._pixel_algorithm(window, arr, transform, rng)

    def _from_raster_array(self, window, arr, rng):
        from rasterio.transform import from_bounds

        minx, miny, maxx, maxy = window.bounds
        transform = from_bounds(minx, miny, maxx, maxy, arr.shape[1], arr.shape[0])
        arr = numpy.maximum(arr.astype(float), 0.0)

        if self.interpolation == "linear":
            return self._bilinear_thinning(window, arr, transform, rng)
        return self._pixel_algorithm(window, arr, transform, rng)

    def _pixel_algorithm(self, window, arr, transform, rng):
        """
        Pixel-selection algorithm: N ~ Poisson(Λ), then allocate N points
        to pixels proportionally to intensity and place uniformly within each.

        K = sum(arr_in_window) x pixel_area; scale = n_expected / K.
        Matches the spatstat modern algorithm (spatstat >= 1.42-3).
        """
        nrows, ncols = arr.shape
        col_c = transform.c + (numpy.arange(ncols) + 0.5) * transform.a
        row_c = transform.f + (numpy.arange(nrows) + 0.5) * transform.e
        CX, CY = numpy.meshgrid(col_c, row_c)

        flat_pts = shapely.points(CX.ravel(), CY.ravel())
        in_win = shapely.covers(window, flat_pts)

        flat_int = arr.ravel() * in_win
        pixel_area = abs(transform.a * transform.e)
        K = float(flat_int.sum()) * pixel_area
        scale = self._effective_scale(K)

        flat_int = flat_int * scale
        total = float(flat_int.sum())
        if total <= 0.0:
            return []

        Lambda = total * pixel_area
        N = int(rng.poisson(Lambda))
        if N == 0:
            return []

        probs = flat_int / flat_int.sum()
        px_idx = rng.choice(len(probs), size=N, p=probs)
        rows_s, cols_s = numpy.unravel_index(px_idx, arr.shape)

        xs = transform.c + (cols_s + rng.random(N)) * transform.a
        ys = transform.f + (rows_s + rng.random(N)) * transform.e

        inside = shapely.covers(window, shapely.points(xs, ys))
        return numpy.vstack((xs[inside], ys[inside])).T

    def _bilinear_thinning(self, window, arr, transform, rng):
        """Bilinear RegularGridInterpolator over pixel grid -> Lewis-Shedler."""
        from scipy.interpolate import RegularGridInterpolator

        nrows, ncols = arr.shape
        xs_grid = transform.c + (numpy.arange(ncols) + 0.5) * transform.a
        ys_grid = transform.f + (numpy.arange(nrows) + 0.5) * transform.e

        if transform.e < 0:  # north-up raster (most common)
            ys_grid = ys_grid[::-1]
            arr = arr[::-1, :]

        interp = RegularGridInterpolator(
            (ys_grid, xs_grid),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        def fn(x, y):
            return numpy.maximum(interp(numpy.column_stack([y, x])), 0.0)

        return self._thinning(window, fn, rng)

    # -- polygon intensity -----------------------------------------------------

    def _from_polygons(
        self, window, geoms: geopandas.GeoSeries, values: numpy.ndarray, rng
    ):
        """Rasterize polygon geometries with per-feature values, then sample."""
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError(
                "rasterio is required to use polygon intensity. "
                "Install it with: pip install rasterio"
            )

        minx, miny, maxx, maxy = window.bounds
        w, h = maxx - minx, maxy - miny

        cells = 512
        if w >= h:
            ncols = cells
            nrows = max(1, int(round(cells * h / w)))
        else:
            nrows = cells
            ncols = max(1, int(round(cells * w / h)))

        transform = from_bounds(minx, miny, maxx, maxy, ncols, nrows)
        shapes = (
            (geom, float(val))
            for geom, val in zip(geoms, values)
            if geom is not None and not geom.is_empty and numpy.isfinite(val)
        )
        arr = rasterize(
            shapes,
            out_shape=(nrows, ncols),
            transform=transform,
            fill=0.0,
            dtype=float,
        )
        return self._pixel_algorithm(window, arr, transform, rng)

    # -- KDE intensity ---------------------------------------------------------

    def _from_kde(self, window, points, rng):
        """Fit a 2-D kernel-density estimate, use it as the process intensity."""
        from sklearn.neighbors import KernelDensity

        if isinstance(points, geopandas.GeoDataFrame):
            points = points.geometry
        if isinstance(points, geopandas.GeoSeries):
            xy = numpy.column_stack([points.x, points.y])
        elif isinstance(points, numpy.ndarray):
            xy = numpy.atleast_2d(points)
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(
                    "For KDE mode pass an (N, 2) array of [x, y] coordinates."
                )
        else:
            raise TypeError(
                f"Cannot interpret intensity of type {type(points).__name__!r} "
                "as a point pattern; expected GeoSeries, GeoDataFrame, or (N,2) ndarray."
            )

        n_obs = len(xy)
        if n_obs == 0:
            return []

        bw = self.bandwidth
        if bw is None:
            std = numpy.std(xy, axis=0)
            bw = float(n_obs ** (-1.0 / 6.0) * numpy.sqrt(max(std[0] * std[1], 1e-12)))

        kde = KernelDensity(bandwidth=bw, kernel=self.kernel)
        kde.fit(xy)

        # fn integrates to n_obs over all space (KDE integrates to 1)
        def fn(x, y):
            log_dens = kde.score_samples(numpy.column_stack([x, y]))
            return numpy.exp(log_dens) * n_obs

        return self._thinning(window, fn, rng)
