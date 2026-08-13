"""Tests for quasi-random (QRN) sequence support across all four samplers."""

import numpy
import pytest
from shapely.geometry import LineString

from spml.validation import (
    PointSampler,
    ConstantClassSampler,
    StratifiedClassSampler,
    MultinomialSampler,
)


SEQS = ["sobol", "halton", "r2"]


# -- Invalid sequence name -----------------------------------------------------

def test_invalid_sequence_raises(square):
    with pytest.raises(ValueError, match="quasi_random"):
        PointSampler(n_samples=10, quasi_random="bad").sample(square)


# -- PointSampler -- polygon ----------------------------------------------------

@pytest.mark.parametrize("seq", SEQS)
def test_point_polygon_count(square, seq):
    pts = PointSampler(n_samples=100, quasi_random=seq, random_state=0).sample(square)
    assert len(pts) == 100


@pytest.mark.parametrize("seq", SEQS)
def test_point_polygon_all_inside(square, seq):
    pts = PointSampler(n_samples=200, quasi_random=seq, random_state=0).sample(square)
    assert all(square.contains(p) for p in pts.geometry)


@pytest.mark.parametrize("seq", SEQS)
def test_point_concave_polygon(seq):
    from shapely.geometry import Polygon
    l_shape = Polygon([(0,0),(10,0),(10,5),(5,5),(5,10),(0,10)])
    pts = PointSampler(n_samples=150, quasi_random=seq, random_state=0).sample(l_shape)
    assert len(pts) == 150
    assert all(l_shape.contains(p) for p in pts.geometry)


# -- PointSampler -- line -------------------------------------------------------

@pytest.mark.parametrize("seq", SEQS)
def test_point_line_count(seq):
    line = LineString([(0, 0), (10, 0)])
    pts = PointSampler(n_samples=100, quasi_random=seq, random_state=0).sample(line)
    assert len(pts) == 100


@pytest.mark.parametrize("seq", SEQS)
def test_point_line_on_line(seq):
    line = LineString([(0, 0), (10, 0)])
    pts = PointSampler(n_samples=200, quasi_random=seq, random_state=0).sample(line)
    assert all(abs(p.y) < 1e-9 for p in pts.geometry)
    assert all(0.0 <= p.x <= 10.0 for p in pts.geometry)


# -- Reproducibility -----------------------------------------------------------

@pytest.mark.parametrize("seq", SEQS)
def test_reproducible(square, seq):
    pts1 = PointSampler(100, quasi_random=seq, random_state=7).sample(square)
    pts2 = PointSampler(100, quasi_random=seq, random_state=7).sample(square)
    assert list(pts1.geometry.x) == list(pts2.geometry.x)
    assert list(pts1.geometry.y) == list(pts2.geometry.y)


def test_different_seeds_differ(square):
    pts1 = PointSampler(50, quasi_random="halton", random_state=1).sample(square)
    pts2 = PointSampler(50, quasi_random="halton", random_state=2).sample(square)
    assert list(pts1.geometry.x) != list(pts2.geometry.x)


def test_different_sequences_differ(square):
    pts_s = PointSampler(50, quasi_random="sobol",  random_state=0).sample(square)
    pts_h = PointSampler(50, quasi_random="halton", random_state=0).sample(square)
    pts_r = PointSampler(50, quasi_random="r2",     random_state=0).sample(square)
    xs = [list(p.geometry.x) for p in (pts_s, pts_h, pts_r)]
    assert xs[0] != xs[1]
    assert xs[0] != xs[2]
    assert xs[1] != xs[2]


def test_qrn_differs_from_random(square):
    pts_rng = PointSampler(50, quasi_random=None, random_state=0).sample(square)
    pts_qrn = PointSampler(50, quasi_random="r2",  random_state=0).sample(square)
    assert list(pts_rng.geometry.x) != list(pts_qrn.geometry.x)


# -- Uniformity ----------------------------------------------------------------

def test_qrn_more_uniform_than_random(square):
    n, cells = 500, 5
    cell_w = 10 / cells

    def grid_variance(pts):
        counts = numpy.zeros((cells, cells), dtype=int)
        for p in pts.geometry:
            i = min(int(p.x / cell_w), cells - 1)
            j = min(int(p.y / cell_w), cells - 1)
            counts[j, i] += 1
        return counts.var()

    var_rng = grid_variance(PointSampler(n, random_state=0).sample(square))
    var_qrn = grid_variance(PointSampler(n, quasi_random="halton", random_state=0).sample(square))
    assert var_qrn < var_rng


# -- Class samplers ------------------------------------------------------------

@pytest.mark.parametrize("seq", SEQS)
def test_constant_class_sampler(class_gdf, seq):
    pts = ConstantClassSampler(n_per_class=30, quasi_random=seq, random_state=0).sample(
        class_gdf.geometry, class_gdf["class"]
    )
    counts = pts.groupby("class_label").size()
    assert (counts == 30).all()


@pytest.mark.parametrize("seq", SEQS)
def test_stratified_class_sampler(class_gdf, seq):
    pts = StratifiedClassSampler(n_samples=100, quasi_random=seq, random_state=0).sample(
        class_gdf.geometry, class_gdf["class"], class_gdf["value"]
    )
    assert len(pts) == 100
    counts = pts.groupby("class_label").size()
    assert counts["A"] == 50
    assert counts["B"] == 50


@pytest.mark.parametrize("seq", SEQS)
def test_multinomial_sampler(multinomial_gdf, seq):
    pts = MultinomialSampler(n_samples=200, quasi_random=seq, random_state=0).sample(
        multinomial_gdf.geometry, multinomial_gdf["class"], multinomial_gdf["weight"]
    )
    assert len(pts) == 200
    assert "class_label" in pts.columns


# -- sklearn API ---------------------------------------------------------------

def test_get_params_includes_quasi_random():
    s = PointSampler(n_samples=100, quasi_random="sobol", random_state=1)
    p = s.get_params()
    assert p["quasi_random"] == "sobol"


def test_set_params_quasi_random(square):
    s = PointSampler(n_samples=50)
    s.set_params(quasi_random="r2", random_state=0)
    pts = s.sample(square)
    assert len(pts) == 50
