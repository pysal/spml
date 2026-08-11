import pytest
import geopandas
from shapely.geometry import box, LineString

from spml.validation import StratifiedClassSampler


def test_total_count(class_gdf):
    pts = StratifiedClassSampler(n_samples=100).sample(class_gdf.geometry, class_gdf["class"])
    assert len(pts) == 100


def test_exact_total_always(class_gdf):
    """Total must equal n_samples exactly (largest-remainder allocation)."""
    for n in [1, 7, 50, 99, 100, 101, 200]:
        pts = StratifiedClassSampler(n_samples=n).sample(
            class_gdf.geometry, class_gdf["class"], class_gdf["value"]
        )
        assert len(pts) == n, f"Expected {n} points, got {len(pts)}"


def test_value_weighted_equal_split(class_gdf):
    """class_gdf: A sum(value)=30, B sum(value)=30  ->  50/50 split at n=100."""
    pts = StratifiedClassSampler(n_samples=100, random_state=0).sample(
        class_gdf.geometry, class_gdf["class"], class_gdf["value"]
    )
    counts = pts.groupby("class_label").size()
    assert counts["A"] == 50
    assert counts["B"] == 50


def test_no_weights_is_uniform_not_row_count():
    """Without weights the sampler falls back to uniform random sampling.

    Class A: 1 large polygon (area 9).  Class B: 9 small polygons (area 1 each, total 9).
    Row-count weighting would give A=10 %, B=90 %.
    Uniform spatial sampling gives A≈50 %, B≈50 % because the areas are equal.
    """
    gdf = geopandas.GeoDataFrame({
        "class": ["A"] + ["B"] * 9,
        "geometry": [box(0, 0, 9, 1)]
                  + [box(i, 1, i + 1, 2) for i in range(9)],
    })
    pts = StratifiedClassSampler(n_samples=2000, random_state=0).sample(
        gdf.geometry, gdf["class"]
    )
    assert len(pts) == 2000
    counts = pts.groupby("class_label").size()
    assert 800 < counts.get("A", 0) < 1200
    assert 800 < counts.get("B", 0) < 1200


def test_class_labels_in_output(class_gdf):
    pts = StratifiedClassSampler(n_samples=50).sample(
        class_gdf.geometry, class_gdf["class"]
    )
    assert "class_label" in pts.columns
    assert "geometry" in pts.columns


def test_no_labels_raises(class_gdf):
    with pytest.raises(ValueError, match="labels"):
        StratifiedClassSampler(n_samples=10).sample(class_gdf.geometry)


def test_geodataframe_input(class_gdf):
    """Passing a GeoDataFrame as geometry should work (geometry column is used)."""
    pts = StratifiedClassSampler(n_samples=60, random_state=0).sample(
        class_gdf, class_gdf["class"], class_gdf["value"]
    )
    assert len(pts) == 60


def test_weights_as_numpy_array(class_gdf):
    pts = StratifiedClassSampler(n_samples=100, random_state=0).sample(
        class_gdf.geometry,
        class_gdf["class"].to_numpy(),
        class_gdf["value"].to_numpy(),
    )
    assert len(pts) == 100


def test_line_geodataframe():
    gdf = geopandas.GeoDataFrame({
        "class":  ["A", "B"],
        "length": [6.0, 4.0],
        "geometry": [LineString([(0, 0), (6, 0)]), LineString([(0, 1), (4, 1)])],
    })
    pts = StratifiedClassSampler(n_samples=100).sample(
        gdf.geometry, gdf["class"], gdf["length"]
    )
    assert len(pts) == 100
    counts = pts.groupby("class_label").size()
    assert counts["A"] == 60
    assert counts["B"] == 40


def test_sklearn_get_params():
    p = StratifiedClassSampler(n_samples=200).get_params()
    assert p["n_samples"] == 200
    assert "class_band" not in p
    assert "value_band" not in p
    assert "class_col" not in p
    assert "value_col" not in p
