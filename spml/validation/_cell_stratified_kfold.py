"""Discrete global grid system stratified k-fold cross-validation."""

import numpy
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold


_GRIDS = ("h3", "a5", "healpix", "s2")

# Resolution sequences to scan when auto-detecting (coarse -> fine)
_RES_RANGES = {
    "h3":      range(0, 16),
    "a5":      range(0, 31),
    "s2":      range(0, 31),
    "healpix": range(0, 14),   # resolution n means nside = 2**n
}


def _to_lonlat(X):
    """Return (n, 2) float64 array of [longitude, latitude] in WGS84 degrees."""
    import geopandas

    if isinstance(X, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
        geom = X.geometry if isinstance(X, geopandas.GeoDataFrame) else X
        if geom.crs is not None and not geom.crs.equals("EPSG:4326"):
            geom = geom.to_crs("EPSG:4326")
        pts = geom.centroid
        return numpy.column_stack([pts.x, pts.y])
    arr = numpy.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise ValueError("Array input must be (n, 2) with columns [longitude, latitude].")


def _assign_cells(lonlat, grid, resolution):
    if grid == "h3":
        import h3
        return numpy.array([
            h3.latlng_to_cell(lat, lon, resolution) for lon, lat in lonlat
        ])
    if grid == "a5":
        from a5 import lonlat_to_cell
        return numpy.array([
            lonlat_to_cell([lon, lat], resolution) for lon, lat in lonlat
        ])
    if grid == "healpix":
        import healpy
        nside = 1 << resolution  # 2**resolution
        theta = numpy.radians(90.0 - lonlat[:, 1])
        phi = numpy.radians(lonlat[:, 0] % 360.0)
        return healpy.ang2pix(nside, theta, phi)
    if grid == "s2":
        import s2sphere
        cells = []
        for lon, lat in lonlat:
            cid = s2sphere.CellId.from_lat_lng(
                s2sphere.LatLng.from_degrees(lat, lon)
            ).parent(resolution)
            cells.append(cid.id())
        return numpy.array(cells)


class CellStratifiedKFold(BaseEstimator):
    """Stratified k-fold cross-validator using a discrete global grid system.

    Each observation is indexed to a DGGS cell.  Cell membership becomes
    the stratum label for a standard StratifiedKFold split, so every fold
    receives a proportional mix of observations from across the study area.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    grid : {"h3", "a5", "healpix", "s2"}, default "h3"
        DGGS backend.  The corresponding package must be installed
        (``h3``, ``pya5``, ``healpy``, ``s2sphere``).
    resolution : int or None, default None
        Grid resolution.  Meaning varies by backend:

        - **h3**: 0 (coarsest) -- 15 (finest)
        - **a5**: 0 (coarsest) -- 30 (finest)
        - **s2**: 0 (coarsest) -- 30 (finest)
        - **healpix**: log2(nside), so 0 = nside 1, 1 = nside 2, etc.

        If *None*, the coarsest resolution with at least *n_splits*
        occupied cells is chosen automatically and stored as
        ``resolution_`` after calling :meth:`split`.
    shuffle : bool, default False
    random_state : int or None, default None

    Attributes
    ----------
    resolution_ : int
        Resolution used (set after first :meth:`split` call).
    cell_ids_ : ndarray of shape (n,)
        DGGS cell ID assigned to each observation.

    Notes
    -----
    Cells with fewer members than *n_splits* are handled by sklearn's
    StratifiedKFold (the minority observations land in fewer folds).
    If this causes an error, increase *resolution* or reduce *n_splits*.

    When ``shuffle=False`` (the default), observations within each cell are
    assigned to folds in their original row order.  If the dataset has a
    spatial sort order (common in GeoDataFrames), this can produce visually
    clustered fold maps even though the stratification is correct.  Pass
    ``shuffle=True, random_state=<int>`` for reproducible, spatially uniform
    within-cell fold assignment.

    Examples
    --------
    >>> cv = CellStratifiedKFold(n_splits=5, grid="h3")
    >>> for train, test in cv.split(gdf):
    ...     model.fit(X[train], y[train])
    ...     score = model.score(X[test], y[test])
    """

    def __init__(self, n_splits=5, grid="h3", resolution=None,
                 shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.grid = grid
        self.resolution = resolution
        self.shuffle = shuffle
        self.random_state = random_state

    def _auto_resolution(self, lonlat):
        for res in _RES_RANGES[self.grid]:
            cells = _assign_cells(lonlat, self.grid, res)
            if len(set(cells)) >= self.n_splits:
                return res
        raise ValueError(
            f"No {self.grid!r} resolution yields >= {self.n_splits} occupied "
            "cells. Try fewer n_splits or a different grid."
        )

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each fold.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations.  GeoDataFrame/GeoSeries are reprojected to WGS84
            automatically; array input must be [longitude, latitude] in degrees.
        y, groups : ignored

        Yields
        ------
        train : ndarray of int
        test  : ndarray of int
        """
        if self.grid not in _GRIDS:
            raise ValueError(f"grid must be one of {_GRIDS}, got {self.grid!r}")
        lonlat = _to_lonlat(X)
        res = self.resolution if self.resolution is not None else self._auto_resolution(lonlat)
        self.resolution_ = res
        cell_ids = _assign_cells(lonlat, self.grid, res)
        self.cell_ids_ = cell_ids
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        yield from skf.split(lonlat, cell_ids)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
