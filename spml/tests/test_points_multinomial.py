import pytest
import geopandas
from shapely.geometry import box, LineString

from spml.validation import MultinomialSampler


# -- Basic counts --------------------------------------------------------------

def test_total_is_exact(multinomial_gdf):
    """Multinomial always returns exactly n_samples."""
    pts = MultinomialSampler(n_samples=200, random_state=0).sample(
        multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"]
    )
    assert len(pts) == 200


def test_class_label_column(multinomial_gdf):
    pts = MultinomialSampler(n_samples=100, random_state=0).sample(
        multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"]
    )
    assert "class_label" in pts.columns
    assert "geometry" in pts.columns
    assert set(pts["class_label"]) == {"A", "B"}


# -- Multinomial allocation ----------------------------------------------------

def test_equal_class_weights_give_approx_equal_counts(multinomial_gdf):
    """Class A total weight == class B total weight -> roughly 50/50 split."""
    pts = MultinomialSampler(n_samples=2000, random_state=0).sample(
        multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"]
    )
    counts = pts.groupby("class_label").size()
    assert 800 < counts["A"] < 1200
    assert 800 < counts["B"] < 1200


def test_skewed_weights_skew_counts():
    """Class A weight 1, class B weight 9 -> class B gets ~90 % of samples."""
    gdf = geopandas.GeoDataFrame({
        "class":   ["A", "B"],
        "weight":  [1.0, 9.0],
        "geometry": [box(0, 0, 5, 5), box(5, 0, 10, 5)],
    })
    pts = MultinomialSampler(n_samples=2000, random_state=0).sample(
        gdf.geometry, gdf["class"], gdf["weight"]
    )
    counts = pts.groupby("class_label").size()
    assert counts.get("B", 0) > counts.get("A", 0) * 5


def test_no_weights_defaults_to_count(multinomial_gdf):
    """Without weights every geometry counts equally -> class size drives allocation."""
    # Both classes have 2 geometries -> expect ~50/50 split with no weights
    pts = MultinomialSampler(n_samples=2000, random_state=0).sample(
        multinomial_gdf.geometry, multinomial_gdf["class"]
    )
    assert len(pts) == 2000
    counts = pts.groupby("class_label").size()
    assert 800 < counts["A"] < 1200


def test_total_always_exact_multiple_n():
    gdf = geopandas.GeoDataFrame({
        "class":   ["A", "B", "C"],
        "weight":  [1.0, 2.0, 7.0],
        "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
    })
    for n in [1, 7, 50, 99, 100, 200]:
        pts = MultinomialSampler(n_samples=n, random_state=0).sample(
            gdf.geometry, gdf["class"], gdf["weight"]
        )
        assert len(pts) == n


# -- Error handling ------------------------------------------------------------

def test_no_labels_raises(multinomial_gdf):
    with pytest.raises(ValueError, match="labels"):
        MultinomialSampler(n_samples=100).sample(multinomial_gdf.geometry)


def test_all_zero_weights_raises(multinomial_gdf):
    gdf = multinomial_gdf.copy()
    gdf["weight"] = 0.0
    with pytest.raises(ValueError, match="zero"):
        MultinomialSampler(n_samples=100).sample(gdf.geometry, gdf["class"], gdf["weight"])


# -- Input types ---------------------------------------------------------------

def test_geodataframe_input(multinomial_gdf):
    pts = MultinomialSampler(n_samples=50, random_state=0).sample(
        multinomial_gdf, multinomial_gdf["class"], multinomial_gdf["weight"]
    )
    assert len(pts) == 50


def test_numpy_array_inputs(multinomial_gdf):
    pts = MultinomialSampler(n_samples=50, random_state=0).sample(
        multinomial_gdf.geometry,
        multinomial_gdf["class"].to_numpy(),
        multinomial_gdf["weight"].to_numpy(),
    )
    assert len(pts) == 50


def test_line_geodataframe():
    gdf = geopandas.GeoDataFrame({
        "class":   ["A", "B"],
        "weight":  [3.0, 7.0],
        "geometry": [LineString([(0, 0), (5, 0)]), LineString([(0, 1), (5, 1)])],
    })
    pts = MultinomialSampler(n_samples=1000, random_state=0).sample(
        gdf.geometry, gdf["class"], gdf["weight"]
    )
    assert len(pts) == 1000


# -- Reproducibility -----------------------------------------------------------

def test_reproducible(multinomial_gdf):
    s1 = MultinomialSampler(n_samples=500, random_state=99)
    s2 = MultinomialSampler(n_samples=500, random_state=99)
    pts1 = s1.sample(multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"])
    pts2 = s2.sample(multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"])
    assert len(pts1) == len(pts2)
    assert list(pts1.geometry.x) == list(pts2.geometry.x)


# -- sklearn API ---------------------------------------------------------------

def test_sklearn_get_params():
    p = MultinomialSampler(n_samples=300).get_params()
    assert p["n_samples"] == 300
    assert "band" not in p
    assert "intensity_col" not in p
    assert "area_normalized" not in p
