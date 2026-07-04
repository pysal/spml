"""Tests for PoissonSampler."""

import importlib.util

import numpy
import pytest
import geopandas
from shapely.geometry import box

from spml.validation import PoissonSampler

requires_rasterio = pytest.mark.skipif(
    importlib.util.find_spec("rasterio") is None,
    reason="rasterio is required for raster-based sampling",
)


# -- fixtures ------------------------------------------------------------------

@pytest.fixture
def square():
    return box(0, 0, 10, 10)   # area = 100


@pytest.fixture
def obs_points(square):
    """Cluster of observed points in the NE corner of the square."""
    rng = numpy.random.default_rng(0)
    x = rng.uniform(7, 10, 50)
    y = rng.uniform(7, 10, 50)
    return geopandas.GeoSeries(geopandas.points_from_xy(x, y))


# -- callable mode -------------------------------------------------------------

def test_callable_uniform_count(square):
    """Uniform λ=1 -> E[N] = 100. Run many seeds and check the mean."""
    counts = []
    for seed in range(30):
        pts = PoissonSampler(random_state=seed).sample(
            square, lambda x, y: numpy.ones_like(x, dtype=float)
        )
        counts.append(len(pts))
    mean_n = numpy.mean(counts)
    assert 70 < mean_n < 130, f"Expected E[N]≈100, got mean={mean_n:.1f}"


def test_callable_points_inside_window(square):
    pts = PoissonSampler(random_state=0).sample(
        square, lambda x, y: numpy.ones_like(x, dtype=float)
    )
    assert all(square.covers(p) for p in pts.geometry)


def test_callable_gradient_skews_distribution(square):
    """λ(x,y) = x/10 -> more points in the east half (x > 5) than the west."""
    pts = PoissonSampler(random_state=42).sample(
        square, lambda x, y: numpy.asarray(x, dtype=float) / 10.0
    )
    assert len(pts) > 0
    east = (pts.geometry.x > 5).sum()
    west = (pts.geometry.x <= 5).sum()
    assert east > west * 2, f"Expected east >> west, got east={east} west={west}"


def test_callable_zero_intensity_returns_empty(square):
    pts = PoissonSampler(random_state=0).sample(
        square, lambda x, y: numpy.zeros_like(x, dtype=float)
    )
    assert len(pts) == 0


def test_callable_n_expected(square):
    """n_expected pins the expected count regardless of the raw intensity."""
    def fn(x, y):
        return numpy.ones_like(x, dtype=float)
    counts = [
        len(PoissonSampler(n_expected=50, random_state=s).sample(square, fn))
        for s in range(30)
    ]
    assert 20 < numpy.mean(counts) < 80


def test_callable_reproducible(square):
    def fn(x, y):
        return numpy.ones_like(x, dtype=float)
    pts1 = PoissonSampler(random_state=5).sample(square, fn)
    pts2 = PoissonSampler(random_state=5).sample(square, fn)
    assert list(pts1.geometry.x) == list(pts2.geometry.x)
    assert list(pts1.geometry.y) == list(pts2.geometry.y)


def test_callable_returns_geodataframe(square):
    pts = PoissonSampler(random_state=0).sample(
        square, lambda x, y: numpy.ones_like(x, dtype=float)
    )
    assert isinstance(pts, geopandas.GeoDataFrame)
    assert "geometry" in pts.columns


# -- raster / 2-D ndarray mode -------------------------------------------------

@requires_rasterio
def test_raster_array_nearest_count(square):
    """Uniform pixel array of 1s -> E[N] = pixel_sum * pixel_area = array_sum*1."""
    arr = numpy.ones((10, 10), dtype=float)  # 10*10 pixels, each 1*1 unit
    counts = []
    for seed in range(20):
        pts = PoissonSampler(interpolation="nearest", random_state=seed).sample(
            square, arr
        )
        counts.append(len(pts))
    mean_n = numpy.mean(counts)
    # Each pixel: intensity=1, area=1 -> Λ = 100
    assert 60 < mean_n < 140, f"Expected E[N]≈100, got mean={mean_n:.1f}"


@requires_rasterio
def test_raster_array_linear_count(square):
    """Same as nearest but with bilinear interpolation."""
    arr = numpy.ones((10, 10), dtype=float)
    counts = []
    for seed in range(20):
        pts = PoissonSampler(interpolation="linear", random_state=seed).sample(
            square, arr
        )
        counts.append(len(pts))
    mean_n = numpy.mean(counts)
    assert 60 < mean_n < 140, f"Expected E[N]≈100, got mean={mean_n:.1f}"


@requires_rasterio
def test_raster_array_points_inside_window(square):
    arr = numpy.ones((20, 20), dtype=float)
    pts = PoissonSampler(interpolation="nearest", random_state=0).sample(
        square, arr
    )
    assert all(square.covers(p) for p in pts.geometry)


@requires_rasterio
def test_raster_array_gradient_skews_distribution(square):
    """Pixel intensity increases east -> more points in east half."""
    arr = numpy.tile(numpy.linspace(0.1, 2.0, 20), (20, 1))  # west->east gradient
    pts = PoissonSampler(interpolation="nearest", random_state=0).sample(
        square, arr
    )
    assert len(pts) > 0
    east = (pts.geometry.x > 5).sum()
    west = (pts.geometry.x <= 5).sum()
    assert east > west


@requires_rasterio
def test_raster_array_zero_returns_empty(square):
    arr = numpy.zeros((10, 10), dtype=float)
    pts = PoissonSampler(random_state=0).sample(square, arr)
    assert len(pts) == 0


@requires_rasterio
def test_raster_array_negative_clipped_to_zero(square):
    """Negative pixel values should be treated as zero (no points there)."""
    arr = numpy.full((10, 10), -1.0)
    pts = PoissonSampler(random_state=0).sample(square, arr)
    assert len(pts) == 0


# -- KDE mode ------------------------------------------------------------------

def test_kde_points_in_window(square, obs_points):
    pts = PoissonSampler(random_state=0).sample(
        square, obs_points
    )
    assert all(square.covers(p) for p in pts.geometry)


def test_kde_expected_count_near_n_obs(square, obs_points):
    """E[N_output] ≈ N_observed (50) when n_expected is None."""
    counts = []
    for seed in range(30):
        pts = PoissonSampler(random_state=seed).sample(
            square, obs_points
        )
        counts.append(len(pts))
    mean_n = numpy.mean(counts)
    # The KDE puts most mass in the NE corner of the window; expected ≈ 50
    # but the window clips the KDE tails, so allow generous tolerance
    assert 10 < mean_n < 100, f"Expected E[N]≈50, got mean={mean_n:.1f}"


def test_kde_cluster_skews_output(square, obs_points):
    """Points concentrated in NE -> output should be denser in NE."""
    all_pts = []
    for seed in range(10):
        pts = PoissonSampler(random_state=seed).sample(
            square, obs_points
        )
        all_pts.append(pts)
    combined = geopandas.pd.concat(all_pts, ignore_index=True)
    if len(combined) == 0:
        pytest.skip("No points generated -- KDE clipped entirely outside window.")
    ne = ((combined.geometry.x > 5) & (combined.geometry.y > 5)).sum()
    other = len(combined) - ne
    assert ne > other, f"Expected NE cluster to dominate; NE={ne} other={other}"


def test_kde_accepts_ndarray(square):
    rng = numpy.random.default_rng(0)
    xy = rng.uniform(0, 10, (40, 2))
    pts = PoissonSampler(bandwidth=1.0, random_state=0).sample(
        square, xy
    )
    assert isinstance(pts, geopandas.GeoDataFrame)


def test_kde_accepts_geodataframe(square, obs_points):
    gdf = geopandas.GeoDataFrame(geometry=obs_points)
    pts = PoissonSampler(random_state=0).sample(square, gdf)
    assert isinstance(pts, geopandas.GeoDataFrame)


def test_kde_bad_ndarray_raises(square):
    # A 1-D array routes to KDE mode but fails the (N, 2) shape check
    with pytest.raises(ValueError, match="N, 2"):
        PoissonSampler().sample(square, numpy.ones(10))


def test_kde_bad_type_raises(square):
    with pytest.raises(TypeError, match="Cannot interpret"):
        PoissonSampler().sample(square, "not_a_valid_intensity")


# -- GeoSeries window ----------------------------------------------------------

def test_geoseries_window_crs_propagated():
    gs = geopandas.GeoSeries([box(0, 0, 10, 10)], crs="EPSG:4326")
    pts = PoissonSampler(random_state=0).sample(
        gs, lambda x, y: numpy.ones_like(x, dtype=float)
    )
    assert pts.crs is not None
    assert pts.crs.to_epsg() == 4326


# -- sklearn API ---------------------------------------------------------------

def test_get_params():
    s = PoissonSampler(n_expected=200, bandwidth=50, kernel="epanechnikov",
                       interpolation="nearest", random_state=1)
    p = s.get_params()
    assert p["n_expected"] == 200
    assert p["bandwidth"] == 50
    assert p["kernel"] == "epanechnikov"
    assert p["interpolation"] == "nearest"
    assert p["random_state"] == 1


@requires_rasterio
def test_set_params(square):
    s = PoissonSampler()
    s.set_params(interpolation="nearest", random_state=0)
    arr = numpy.ones((5, 5))
    pts = s.sample(square, arr)
    assert isinstance(pts, geopandas.GeoDataFrame)
