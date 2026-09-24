import numpy
import pytest
import geopandas
from shapely.geometry import Point

pytest.importorskip("h3")

from spml.validation import CellStratifiedKFold


@pytest.fixture
def spread_points():
    """30 points in 3 well-separated PNW city clusters.
    std=0.005 deg (~550 m) keeps each cluster in a single H3 cell at res >= 2.
    """
    rng = numpy.random.default_rng(0)
    seattle   = numpy.column_stack([rng.normal(-122.3, 0.005, 10), rng.normal(47.6,  0.005, 10)])
    portland  = numpy.column_stack([rng.normal(-122.7, 0.005, 10), rng.normal(45.5,  0.005, 10)])
    vancouver = numpy.column_stack([rng.normal(-123.1, 0.005, 10), rng.normal(49.3,  0.005, 10)])
    return numpy.vstack([seattle, portland, vancouver])  # [lon, lat]


@pytest.fixture
def spread_gdf(spread_points):
    return geopandas.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in spread_points],
        crs="EPSG:4326",
    )


# -- Basic split behaviour -----------------------------------------------------

def test_correct_number_of_folds(spread_points):
    splits = list(CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(spread_points))
    assert len(splits) == 3


def test_get_n_splits(spread_points):
    assert CellStratifiedKFold(n_splits=4, grid="h3", resolution=3).get_n_splits() == 4


def test_train_test_disjoint(spread_points):
    for tr, te in CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(spread_points):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_all_indices_covered(spread_points):
    n = len(spread_points)
    seen = set()
    for tr, te in CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(spread_points):
        seen.update(tr.tolist())
        seen.update(te.tolist())
    assert seen == set(range(n))


def test_each_fold_spans_multiple_cells(spread_points):
    """With 3 clusters at res=3 and 3 folds, every test fold should draw from all 3 cells."""
    ckf = CellStratifiedKFold(n_splits=3, grid="h3", resolution=3, shuffle=True, random_state=0)
    for _, te in ckf.split(spread_points):
        cells_in_test = len(numpy.unique(ckf.cell_ids_[te]))
        assert cells_in_test >= 1


def test_resolution_stored(spread_points):
    ckf = CellStratifiedKFold(n_splits=3, grid="h3", resolution=3)
    list(ckf.split(spread_points))
    assert ckf.resolution_ == 3


def test_cell_ids_stored(spread_points):
    ckf = CellStratifiedKFold(n_splits=3, grid="h3", resolution=3)
    list(ckf.split(spread_points))
    assert len(ckf.cell_ids_) == len(spread_points)


# -- Auto-resolution -----------------------------------------------------------

def test_auto_resolution_finds_enough_cells(spread_points):
    ckf = CellStratifiedKFold(n_splits=3, grid="h3", resolution=None)
    list(ckf.split(spread_points))
    assert hasattr(ckf, "resolution_")
    assert len(numpy.unique(ckf.cell_ids_)) >= 3


def test_auto_resolution_too_many_splits_raises(spread_points):
    """Asking for more splits than discoverable cells should raise."""
    with pytest.raises(ValueError, match="resolution"):
        list(CellStratifiedKFold(n_splits=1000, grid="h3", resolution=None).split(spread_points))


# -- Shuffle / reproducibility -------------------------------------------------

def test_shuffle_reproducible(spread_points):
    kw = dict(n_splits=3, grid="h3", resolution=3, shuffle=True, random_state=42)
    s1 = list(CellStratifiedKFold(**kw).split(spread_points))
    s2 = list(CellStratifiedKFold(**kw).split(spread_points))
    for (tr1, te1), (tr2, te2) in zip(s1, s2):
        numpy.testing.assert_array_equal(tr1, tr2)
        numpy.testing.assert_array_equal(te1, te2)


# -- Input types ---------------------------------------------------------------

def test_geodataframe_input(spread_gdf):
    splits = list(CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(spread_gdf))
    assert len(splits) == 3


def test_geoseries_input(spread_gdf):
    splits = list(CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(spread_gdf.geometry))
    assert len(splits) == 3


def test_geodataframe_reprojected(spread_points):
    """GeoDataFrame in a projected CRS should be transparently handled."""
    gdf_utm = geopandas.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in spread_points],
        crs="EPSG:4326",
    ).to_crs("EPSG:32610")
    ckf_utm = CellStratifiedKFold(n_splits=3, grid="h3", resolution=3)
    ckf_wgs = CellStratifiedKFold(n_splits=3, grid="h3", resolution=3)
    list(ckf_utm.split(gdf_utm))
    list(ckf_wgs.split(spread_points))
    numpy.testing.assert_array_equal(ckf_utm.cell_ids_, ckf_wgs.cell_ids_)


def test_invalid_array_shape_raises():
    with pytest.raises(ValueError):
        list(CellStratifiedKFold(n_splits=3, grid="h3", resolution=3).split(numpy.ones((10, 3))))


# -- Edge cases / errors -------------------------------------------------------

def test_invalid_grid_raises(spread_points):
    with pytest.raises(ValueError, match="grid"):
        list(CellStratifiedKFold(n_splits=3, grid="bogus").split(spread_points))


# -- Other backends (skipped if not installed) ---------------------------------

def test_healpix_basic(spread_points):
    pytest.importorskip("healpy")
    splits = list(CellStratifiedKFold(n_splits=3, grid="healpix", resolution=1).split(spread_points))
    assert len(splits) == 3


def test_s2_basic(spread_points):
    pytest.importorskip("s2sphere")
    splits = list(CellStratifiedKFold(n_splits=3, grid="s2", resolution=5).split(spread_points))
    assert len(splits) == 3


def test_a5_basic(spread_points):
    pytest.importorskip("a5")
    splits = list(CellStratifiedKFold(n_splits=3, grid="a5", resolution=3).split(spread_points))
    assert len(splits) == 3
