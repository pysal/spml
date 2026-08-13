import pytest
import geopandas

from spml.validation import PointSampler


def test_count(square):
    pts = PointSampler(n_samples=50, random_state=0).sample(square)
    assert len(pts) == 50


def test_all_inside(square):
    pts = PointSampler(n_samples=200, random_state=42).sample(square)
    assert all(square.contains(p) for p in pts.geometry)


def test_override_n_samples(square):
    sampler = PointSampler(n_samples=50)
    pts = sampler.sample(square, n_samples=200)
    assert len(pts) == 200


def test_crs_from_geoseries(class_gdf):
    gdf = class_gdf.set_crs("EPSG:4326")
    pts = PointSampler(n_samples=10, random_state=0).sample(gdf)
    assert pts.crs.to_epsg() == 4326


def test_geodataframe_input_dissolved(class_gdf):
    pts = PointSampler(n_samples=50, random_state=0).sample(class_gdf)
    assert len(pts) == 50
    assert "geometry" in pts.columns


def test_concave_geometry():
    """L-shaped geometry -- tests that rejection sampling handles non-convex shapes."""
    from shapely.geometry import Polygon
    l_shape = Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])
    pts = PointSampler(n_samples=100, random_state=7).sample(l_shape)
    assert len(pts) == 100
    assert all(l_shape.contains(p) for p in pts.geometry)


def test_sklearn_get_params():
    sampler = PointSampler(n_samples=100, random_state=42)
    assert sampler.get_params() == {"n_samples": 100, "quasi_random": None, "random_state": 42}


def test_sklearn_set_params(square):
    sampler = PointSampler()
    sampler.set_params(n_samples=77, random_state=3)
    assert sampler.n_samples == 77
    pts = sampler.sample(square)
    assert len(pts) == 77


def test_zero_area_raises():
    from shapely.geometry import Polygon
    with pytest.raises(ValueError, match="zero-area"):
        PointSampler(n_samples=10).sample(Polygon())  # empty / degenerate polygon


# -- Line geometry tests -------------------------------------------------------

def test_linestring_count():
    from shapely.geometry import LineString
    line = LineString([(0, 0), (10, 0)])
    pts = PointSampler(n_samples=200, random_state=0).sample(line)
    assert len(pts) == 200


def test_linestring_points_on_line():
    from shapely.geometry import LineString
    line = LineString([(0, 0), (10, 0)])
    pts = PointSampler(n_samples=500, random_state=7).sample(line)
    # All points lie on y=0, x in [0,10]
    assert all(abs(p.y) < 1e-9 for p in pts.geometry)
    assert all(0.0 <= p.x <= 10.0 for p in pts.geometry)


def test_multilinestring_uniform_arc_length():
    from shapely.geometry import MultiLineString
    # Two segments: length 6 and length 4 -> total 10
    mls = MultiLineString([[(0, 0), (6, 0)], [(10, 0), (14, 0)]])
    pts = PointSampler(n_samples=10_000, random_state=0).sample(mls)
    assert len(pts) == 10_000
    # ~60 % of points should lie on the first segment (x in [0,6])
    on_first = sum(1 for p in pts.geometry if p.x <= 6.0)
    assert 5_500 < on_first < 6_500  # 60 % ± 5 %


def test_diagonal_linestring():
    from shapely.geometry import LineString
    line = LineString([(0, 0), (3, 4)])  # length = 5
    pts = PointSampler(n_samples=100, random_state=1).sample(line)
    assert len(pts) == 100
    for p in pts.geometry:
        # points lie on the line y = (4/3)x
        assert abs(p.y - (4 / 3) * p.x) < 1e-9


def test_linearring():
    from shapely.geometry import LinearRing
    ring = LinearRing([(0, 0), (4, 0), (4, 3), (0, 3)])  # perimeter = 14
    pts = PointSampler(n_samples=150, random_state=2).sample(ring)
    assert len(pts) == 150


def test_line_crs_preserved():
    from shapely.geometry import LineString
    gs = geopandas.GeoSeries([LineString([(0, 0), (1, 0)])], crs="EPSG:32632")
    pts = PointSampler(n_samples=50, random_state=0).sample(gs)
    assert pts.crs.to_epsg() == 32632
