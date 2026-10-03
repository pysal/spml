import geopandas as gpd
import numpy as np
import pytest
from shapely import Point

pytest.importorskip("h3")

from spml.validation import LeaveCellOut


@pytest.fixture
def spread_points():
    """30 points in 3 well-separated PNW city clusters (Seattle, Portland, Vancouver).
    std=0.005 deg (~550 m) keeps each cluster within a single H3 cell at res >= 2.
    """
    rng = np.random.default_rng(0)
    seattle = np.column_stack(
        [rng.normal(-122.3, 0.005, 10), rng.normal(47.6, 0.005, 10)]
    )
    portland = np.column_stack(
        [rng.normal(-122.7, 0.005, 10), rng.normal(45.5, 0.005, 10)]
    )
    vancouver = np.column_stack(
        [rng.normal(-123.1, 0.005, 10), rng.normal(49.3, 0.005, 10)]
    )
    return np.vstack([seattle, portland, vancouver])  # [lon, lat]


@pytest.fixture
def spread_gdf(spread_points):
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in spread_points],
        crs="EPSG:4326",
    )


@pytest.fixture
def points_with_singleton(spread_points):
    """spread_points + 1 isolated point in NYC (always its own H3 cell vs PNW)."""
    nyc = np.array([[-74.0, 40.7]])
    return np.vstack([spread_points, nyc])


# -- Basic split behaviour -----------------------------------------------------


def test_n_folds_equals_n_cells(spread_points):
    lco = LeaveCellOut(grid="h3", resolution=3)
    splits = list(lco.split(spread_points))
    assert len(splits) == lco.n_cells_


def test_train_test_disjoint(spread_points):
    for tr, te in LeaveCellOut(grid="h3", resolution=3).split(spread_points):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_all_indices_covered(spread_points):
    """Every index appears in exactly one test fold."""
    n = len(spread_points)
    test_counts = np.zeros(n, dtype=int)
    for _, te in LeaveCellOut(grid="h3", resolution=3).split(spread_points):
        test_counts[te] += 1
    np.testing.assert_array_equal(test_counts, 1)


def test_test_is_single_cell(spread_points):
    """Each test fold contains observations from exactly one H3 cell."""
    lco = LeaveCellOut(grid="h3", resolution=3)
    for _, te in lco.split(spread_points):
        assert len(np.unique(lco.cell_ids_[te])) == 1


def test_resolution_stored(spread_points):
    lco = LeaveCellOut(grid="h3", resolution=3)
    list(lco.split(spread_points))
    assert lco.resolution_ == 3


# -- min_test_size -------------------------------------------------------------


def test_min_test_size_excludes_small_cells(points_with_singleton):
    """Cell containing only the singleton NYC point is excluded from test folds
    when min_test_size=2."""
    lco = LeaveCellOut(grid="h3", resolution=3, min_test_size=2)
    splits = list(lco.split(points_with_singleton))
    n = len(points_with_singleton)
    singleton_idx = n - 1  # last point is the NYC singleton
    for _, te in splits:
        assert singleton_idx not in te.tolist()


def test_min_test_size_1_includes_singleton(points_with_singleton):
    """With min_test_size=1 (default), every cell appears as a test fold."""
    lco_1 = LeaveCellOut(grid="h3", resolution=3, min_test_size=1)
    lco_2 = LeaveCellOut(grid="h3", resolution=3, min_test_size=2)
    n1 = len(list(lco_1.split(points_with_singleton)))
    n2 = len(list(lco_2.split(points_with_singleton)))
    assert n1 > n2


# -- Auto-resolution -----------------------------------------------------------


def test_auto_resolution_finds_multiple_cells(spread_points):
    lco = LeaveCellOut(grid="h3", resolution=None)
    list(lco.split(spread_points))
    assert hasattr(lco, "resolution_")
    assert lco.n_cells_ >= 2


# -- get_n_splits --------------------------------------------------------------


def test_get_n_splits_before_split_raises():
    lco = LeaveCellOut(grid="h3", resolution=3)
    with pytest.raises(ValueError):
        lco.get_n_splits()


def test_get_n_splits_after_split(spread_points):
    lco = LeaveCellOut(grid="h3", resolution=3)
    list(lco.split(spread_points))
    assert lco.get_n_splits() == lco.n_cells_


def test_get_n_splits_with_x(spread_points):
    lco = LeaveCellOut(grid="h3", resolution=3)
    n = lco.get_n_splits(X=spread_points)
    assert n >= 1


# -- Input types ---------------------------------------------------------------


def test_geodataframe_input(spread_gdf):
    splits = list(LeaveCellOut(grid="h3", resolution=3).split(spread_gdf))
    assert len(splits) >= 1


def test_geoseries_input(spread_gdf):
    splits = list(LeaveCellOut(grid="h3", resolution=3).split(spread_gdf.geometry))
    assert len(splits) >= 1


def test_geodataframe_reprojected(spread_points):
    """GeoDataFrame in a projected CRS should be reprojected to WGS84 transparently."""
    gdf_wgs = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in spread_points],
        crs="EPSG:4326",
    ).to_crs("EPSG:32610")
    lco_proj = LeaveCellOut(grid="h3", resolution=3)
    lco_wgs = LeaveCellOut(grid="h3", resolution=3)
    list(lco_proj.split(gdf_wgs))
    list(lco_wgs.split(spread_points))
    np.testing.assert_array_equal(lco_proj.cell_ids_, lco_wgs.cell_ids_)


# -- Edge cases / errors -------------------------------------------------------


def test_invalid_grid_raises(spread_points):
    with pytest.raises(ValueError, match="grid"):
        list(LeaveCellOut(grid="bogus").split(spread_points))


def test_invalid_array_shape_raises():
    with pytest.raises(ValueError):
        list(LeaveCellOut(grid="h3", resolution=3).split(np.ones((10, 3))))


# -- Other backends (skipped if not installed) ---------------------------------


def test_healpix_basic(spread_points):
    healpy = pytest.importorskip("healpy")  # noqa: F841
    splits = list(LeaveCellOut(grid="healpix", resolution=1).split(spread_points))
    assert len(splits) >= 1


def test_s2_basic(spread_points):
    pytest.importorskip("s2sphere")
    splits = list(LeaveCellOut(grid="s2", resolution=5).split(spread_points))
    assert len(splits) >= 1


def test_a5_basic(spread_points):
    pytest.importorskip("a5")
    splits = list(LeaveCellOut(grid="a5", resolution=3).split(spread_points))
    assert len(splits) >= 1
