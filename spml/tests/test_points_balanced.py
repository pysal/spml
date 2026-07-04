import pytest
import geopandas
from shapely.geometry import box, LineString

from spml.validation import ConstantClassSampler


def test_equal_counts_per_class(class_gdf):
    pts = ConstantClassSampler(n_per_class=25).sample(class_gdf.geometry, class_gdf["class"])
    counts = pts.groupby("class_label").size()
    assert (counts == 25).all()


def test_class_labels_present(class_gdf):
    pts = ConstantClassSampler(n_per_class=10).sample(class_gdf.geometry, class_gdf["class"])
    assert set(pts["class_label"]) == {"A", "B"}


def test_geometry_column(class_gdf):
    pts = ConstantClassSampler(n_per_class=10).sample(class_gdf.geometry, class_gdf["class"])
    assert "geometry" in pts.columns


def test_geodataframe_input(class_gdf):
    """Passing a full GeoDataFrame (instead of GeoSeries) should also work."""
    pts = ConstantClassSampler(n_per_class=10).sample(class_gdf, class_gdf["class"])
    assert "geometry" in pts.columns


def test_labels_as_numpy_array(class_gdf):
    labels = class_gdf["class"].to_numpy()
    pts = ConstantClassSampler(n_per_class=15).sample(class_gdf.geometry, labels)
    assert set(pts["class_label"]) == {"A", "B"}


def test_no_labels_raises(class_gdf):
    with pytest.raises(ValueError, match="labels"):
        ConstantClassSampler(n_per_class=10).sample(class_gdf.geometry)


def test_sklearn_repr():
    s = ConstantClassSampler(n_per_class=50)
    assert "ConstantClassSampler" in repr(s)


def test_sklearn_get_params():
    p = ConstantClassSampler(n_per_class=30).get_params()
    assert p["n_per_class"] == 30
    assert "class_band" not in p
    assert "class_col" not in p


def test_line_geodataframe():
    gdf = geopandas.GeoDataFrame({
        "class": ["A", "A", "B"],
        "geometry": [
            LineString([(0, 0), (5, 0)]),
            LineString([(5, 0), (10, 0)]),
            LineString([(0, 1), (10, 1)]),
        ],
    })
    pts = ConstantClassSampler(n_per_class=30).sample(gdf.geometry, gdf["class"])
    counts = pts.groupby("class_label").size()
    assert (counts == 30).all()
    for label, group in pts.groupby("class_label"):
        expected_y = 0.0 if label == "A" else 1.0
        assert all(abs(p.y - expected_y) < 1e-9 for p in group.geometry)


def test_many_classes():
    geoms = [box(i, 0, i + 1, 1) for i in range(10)]
    labels = list(range(10))
    gdf = geopandas.GeoDataFrame({"class": labels, "geometry": geoms})
    pts = ConstantClassSampler(n_per_class=20).sample(gdf.geometry, gdf["class"])
    counts = pts.groupby("class_label").size()
    assert len(counts) == 10
    assert (counts == 20).all()
