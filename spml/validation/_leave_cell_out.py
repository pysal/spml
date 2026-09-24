"""Discrete global grid system leave-one-cell-out cross-validation."""

import numpy

from sklearn.base import BaseEstimator

from ._cell_stratified_kfold import _GRIDS, _RES_RANGES, _assign_cells, _to_lonlat


class LeaveCellOut(BaseEstimator):
    """Leave-one-DGGS-cell-out cross-validator.

    Each observation is indexed to a discrete global grid system (DGGS) cell.
    The test set for each fold is all observations in one cell; the training
    set is all observations in all other cells.

    This is the spatial block cross-validation analogue of
    :class:`CellStratifiedKFold`: where that class distributes observations
    from each cell across all folds, this class holds out one cell at a time.
    Block size and shape are controlled by the grid resolution.

    Parameters
    ----------
    grid : {"h3", "a5", "healpix", "s2"}, default "h3"
        DGGS backend. The corresponding package must be installed
        (``h3``, ``pya5``, ``healpy``, ``s2sphere``).
    resolution : int or None, default None
        Grid resolution. Meaning varies by backend:

        - **h3**: 0 (coarsest) -- 15 (finest)
        - **a5**: 0 (coarsest) -- 30 (finest)
        - **s2**: 0 (coarsest) -- 30 (finest)
        - **healpix**: log2(nside), so 0 = nside 1, 1 = nside 2, etc.

        If *None*, the coarsest resolution with at least 2 occupied cells is
        chosen automatically and stored as ``resolution_`` after calling
        :meth:`split`.
    min_test_size : int, default 1
        Cells with fewer than *min_test_size* observations are folded into
        every training set and never used as a test fold. Increase this to
        avoid test folds too small to produce reliable scores.

    Attributes
    ----------
    resolution_ : int
        Resolution used (set after first :meth:`split` call).
    cell_ids_ : ndarray of shape (n,)
        DGGS cell ID assigned to each observation.
    n_cells_ : int
        Number of cells used as test folds (set after first :meth:`split` call).

    Examples
    --------
    >>> lco = LeaveCellOut(grid="h3", resolution=3)
    >>> for train_idx, test_idx in lco.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])

    Cells with fewer than 5 observations are kept only in training:

    >>> lco = LeaveCellOut(grid="h3", resolution=4, min_test_size=5)
    """

    def __init__(self, grid: str = "h3", resolution=None, min_test_size: int = 1):
        self.grid = grid
        self.resolution = resolution
        self.min_test_size = min_test_size

    def _auto_resolution(self, lonlat):
        for res in _RES_RANGES[self.grid]:
            cells = _assign_cells(lonlat, self.grid, res)
            if len(set(cells)) >= 2:
                return res
        raise ValueError(
            f"No {self.grid!r} resolution yields >= 2 occupied cells. "
            "Try a different grid."
        )

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each occupied DGGS cell.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations. GeoDataFrame/GeoSeries are reprojected to WGS84
            automatically; array input must be [longitude, latitude] in degrees.
        y, groups : ignored, present for sklearn API compatibility.

        Yields
        ------
        train : ndarray of int
            All observations not in the test cell.
        test : ndarray of int
            All observations in the current cell.
        """
        if self.grid not in _GRIDS:
            raise ValueError(f"grid must be one of {_GRIDS}, got {self.grid!r}")

        lonlat = _to_lonlat(X)
        n = len(lonlat)

        res = (
            self.resolution
            if self.resolution is not None
            else self._auto_resolution(lonlat)
        )
        self.resolution_ = res

        cell_ids = _assign_cells(lonlat, self.grid, res)
        self.cell_ids_ = cell_ids

        unique_cells, counts = numpy.unique(cell_ids, return_counts=True)
        test_cells = unique_cells[counts >= self.min_test_size]
        self.n_cells_ = len(test_cells)

        indices = numpy.arange(n)

        for cell in test_cells:
            test = indices[cell_ids == cell]
            train = indices[cell_ids != cell]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if hasattr(self, "n_cells_"):
            return self.n_cells_
        if X is not None:
            if self.grid not in _GRIDS:
                raise ValueError(f"grid must be one of {_GRIDS}, got {self.grid!r}")
            lonlat = _to_lonlat(X)
            res = (
                self.resolution
                if self.resolution is not None
                else self._auto_resolution(lonlat)
            )
            cell_ids = _assign_cells(lonlat, self.grid, res)
            _, counts = numpy.unique(cell_ids, return_counts=True)
            return int((counts >= self.min_test_size).sum())
        raise ValueError(
            "Call split() first or pass X to determine the number of cells."
        )
