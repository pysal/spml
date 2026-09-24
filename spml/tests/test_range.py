import numpy
import pytest
import geopandas
from shapely.geometry import Point

from spml.validation import correlogram_range, knn_range


@pytest.fixture
def line_data():
    """20 equally-spaced points; y decays with distance from the origin."""
    rng = numpy.random.default_rng(0)
    gdf = geopandas.GeoDataFrame(geometry=[Point(i, 0) for i in range(20)])
    # Strong local autocorrelation: y_i correlated with nearby y_j
    y = numpy.array([numpy.exp(-0.1 * i) + 0.1 * rng.standard_normal() for i in range(20)])
    return gdf, y


@pytest.fixture
def grid_data():
    """5*5 grid; y is a smooth spatial surface."""
    rng = numpy.random.default_rng(1)
    pts = [Point(i, j) for i in range(5) for j in range(5)]
    gdf = geopandas.GeoDataFrame(geometry=pts)
    coords = numpy.array([(p.x, p.y) for p in pts])
    y = numpy.sin(coords[:, 0]) + numpy.cos(coords[:, 1]) + 0.05 * rng.standard_normal(25)
    return gdf, y


# -- correlogram_range ---------------------------------------------------------

@pytest.mark.filterwarnings("ignore:correlogram_range:UserWarning")
def test_correlogram_returns_float(line_data):
    gdf, y = line_data
    d = correlogram_range(gdf, y, random_state=0)
    assert isinstance(d, float)
    assert d > 0


def test_correlogram_range_plausible(line_data):
    gdf, y = line_data
    d = correlogram_range(gdf, y, n_bins=10, random_state=0)
    # Points are spaced 1 unit apart; autocorrelation should drop to zero
    # somewhere within the range of the dataset (0 to 10 units)
    assert 0 < d < 10


def test_correlogram_range_1d_array():
    """1-D time index input (time-series mode)."""
    rng = numpy.random.default_rng(2)
    t = numpy.arange(50, dtype=float)
    y = numpy.exp(-0.2 * t) + 0.05 * rng.standard_normal(50)
    d = correlogram_range(t, y, random_state=0)
    assert isinstance(d, float)
    assert d > 0


def test_correlogram_range_accepts_raw_array(grid_data):
    gdf, y = grid_data
    coords = numpy.array([(p.x, p.y) for p in gdf.geometry])
    d = correlogram_range(coords, y, random_state=0)
    assert isinstance(d, float)


def test_correlogram_constant_y_raises(line_data):
    gdf, _ = line_data
    with pytest.raises(ValueError, match="constant"):
        correlogram_range(gdf, numpy.ones(len(gdf)))


def test_correlogram_length_mismatch_raises(line_data):
    gdf, y = line_data
    with pytest.raises(ValueError, match="same length"):
        correlogram_range(gdf, y[:-1])


def test_correlogram_no_crossing_warns():
    """Uncorrelated white noise may not produce a zero crossing -- should warn."""
    rng = numpy.random.default_rng(99)
    gdf = geopandas.GeoDataFrame(geometry=[Point(i, 0) for i in range(30)])
    y = rng.standard_normal(30)
    # With random noise and a very small max_distance, crossing may not be found
    with pytest.warns(UserWarning, match="did not cross zero"):
        correlogram_range(gdf, y, max_distance=0.1, random_state=0)


def test_correlogram_subsampling(grid_data):
    gdf, y = grid_data
    # n=25, n_sample=10: should still return a float
    d = correlogram_range(gdf, y, n_sample=10, random_state=0)
    assert isinstance(d, float)


# -- knn_range -----------------------------------------------------------------

def test_knn_returns_int(grid_data):
    gdf, y = grid_data
    k = knn_range(gdf, y)
    assert isinstance(k, int)
    assert k >= 1


@pytest.mark.filterwarnings("ignore:knn_range:UserWarning")
def test_knn_range_plausible(grid_data):
    gdf, y = grid_data
    k = knn_range(gdf, y, max_k=10)
    assert 1 <= k <= 10


def test_knn_range_no_crossing_warns():
    """Strongly autocorrelated data over a small dataset: may not drop to zero."""
    gdf = geopandas.GeoDataFrame(geometry=[Point(i, 0) for i in range(10)])
    y = numpy.arange(10, dtype=float)   # perfectly ordered -- very high autocorrelation
    with pytest.warns(UserWarning, match="did not drop to zero"):
        k = knn_range(gdf, y, max_k=5)
    assert k == 5


def test_knn_1d_raises():
    t = numpy.arange(20, dtype=float)
    y = numpy.random.default_rng(0).standard_normal(20)
    with pytest.raises(ValueError, match="2-D spatial"):
        knn_range(t, y)


def test_knn_constant_y_raises(grid_data):
    gdf, _ = grid_data
    with pytest.raises(ValueError, match="constant"):
        knn_range(gdf, numpy.ones(len(gdf)))


def test_knn_length_mismatch_raises(grid_data):
    gdf, y = grid_data
    with pytest.raises(ValueError, match="same length"):
        knn_range(gdf, y[:-1])


# -- integration: use as bandwidth/threshold -----------------------------------

@pytest.mark.filterwarnings("ignore:correlogram_range:UserWarning")
def test_correlogram_range_as_bandwidth(line_data):
    from spml.validation import LocalBootstrap
    gdf, y = line_data
    bw = correlogram_range(gdf, y, random_state=0)
    boots = list(LocalBootstrap(bandwidth=bw, n_bootstraps=5, random_state=0).sample(gdf))
    assert len(boots) == 5
    assert all(b.shape == (len(gdf),) for b in boots)


@pytest.mark.filterwarnings("ignore:correlogram_range:UserWarning")
def test_correlogram_range_as_threshold(line_data):
    from spml.validation import LocalPermutation
    gdf, y = line_data
    d = correlogram_range(gdf, y, random_state=0)
    perms = list(LocalPermutation(bandwidth=d, n_permutations=3, random_state=0).sample(gdf))
    assert len(perms) == 3
