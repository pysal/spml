import numpy
import pytest
import geopandas
from shapely.geometry import Point

from spml.validation import LocalPermutation


@pytest.fixture
def grid_gdf():
    """4 * 4 regular grid with unit spacing -- each site has 2-4 neighbors
    within bandwidth=1.5."""
    return geopandas.GeoDataFrame(
        geometry=[Point(x, y) for x in range(4) for y in range(4)]
    )


@pytest.fixture
def coords(grid_gdf):
    return numpy.column_stack([grid_gdf.geometry.x, grid_gdf.geometry.y])


# -- Bijection -----------------------------------------------------------------

def test_is_valid_permutation(grid_gdf):
    """Output must be a bijection -- each index appears exactly once."""
    n = len(grid_gdf)
    lp = LocalPermutation(bandwidth=1.5, derangement=False,
                           n_permutations=10, random_state=0)
    for perm in lp.sample(grid_gdf):
        assert sorted(perm) == list(range(n)), "Not a valid permutation"


# -- Distance constraint -------------------------------------------------------

def test_distance_constraint(grid_gdf, coords):
    """Every value must move at most *threshold* distance units."""
    lp = LocalPermutation(bandwidth=1.5, derangement=False,
                           n_permutations=20, random_state=0)
    for perm in lp.sample(grid_gdf):
        for i, j in enumerate(perm):
            d = numpy.linalg.norm(coords[i] - coords[j])
            assert d <= 1.5 + 1e-9, f"Constraint violated: site {i} <- site {j}, dist={d:.4f}"


def test_distance_constraint_derangement(grid_gdf, coords):
    lp = LocalPermutation(bandwidth=1.5, derangement=True,
                           n_permutations=20, random_state=0)
    for perm in lp.sample(grid_gdf):
        for i, j in enumerate(perm):
            d = numpy.linalg.norm(coords[i] - coords[j])
            assert d <= 1.5 + 1e-9


# -- Derangement ---------------------------------------------------------------

def test_derangement_no_fixed_points(grid_gdf):
    """With derangement=True, no value may remain at its origin."""
    lp = LocalPermutation(bandwidth=1.5, derangement=True,
                           n_permutations=20, random_state=0)
    for perm in lp.sample(grid_gdf):
        assert not any(perm[i] == i for i in range(len(perm))), \
            "Fixed point found in derangement"


def test_non_derangement_may_have_fixed_points(grid_gdf):
    """With derangement=False, fixed points are permitted (not required)."""
    lp = LocalPermutation(bandwidth=1.5, derangement=False,
                           n_permutations=100, random_state=0)
    has_fixed = False
    for perm in lp.sample(grid_gdf):
        if any(perm[i] == i for i in range(len(perm))):
            has_fixed = True
            break
    # With 100 permutations and a connected grid, at least one should have a fixed point
    assert has_fixed, "Expected at least one permutation with a fixed point"


# -- Distribution --------------------------------------------------------------

def test_values_move(grid_gdf):
    """At least some values should move to different sites."""
    lp = LocalPermutation(bandwidth=1.5, derangement=False,
                           n_permutations=10, random_state=0)
    for perm in lp.sample(grid_gdf):
        assert not numpy.all(perm == numpy.arange(len(perm))), "Identity permutation returned"


def test_permutations_differ(grid_gdf):
    """Consecutive permutations should not all be identical."""
    lp = LocalPermutation(bandwidth=1.5, derangement=True,
                           n_permutations=20, random_state=0)
    perms = list(lp.sample(grid_gdf))
    unique = set(tuple(p) for p in perms)
    assert len(unique) > 1, "All permutations identical -- Markov chain is stuck"


# -- Infeasibility -------------------------------------------------------------

def test_infeasible_derangement_raises():
    """Two isolated points: threshold too small for derangement."""
    gdf = geopandas.GeoDataFrame(geometry=[Point(0, 0), Point(100, 100)])
    lp = LocalPermutation(bandwidth=1.0, derangement=True, n_permutations=1)
    with pytest.raises(ValueError, match="No valid"):
        list(lp.sample(gdf))


def test_infeasible_permutation_raises():
    """A completely isolated point: cannot receive any value within threshold."""
    gdf = geopandas.GeoDataFrame(
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0)]
    )
    # bandwidth=1 -- no pair within range, self-assignment also blocked
    lp = LocalPermutation(bandwidth=0.5, derangement=True, n_permutations=1)
    with pytest.raises(ValueError, match="No valid"):
        list(lp.sample(gdf))


# -- Reproducibility -----------------------------------------------------------

def test_reproducible(grid_gdf):
    lp1 = LocalPermutation(bandwidth=1.5, derangement=True,
                            n_permutations=10, random_state=99)
    lp2 = LocalPermutation(bandwidth=1.5, derangement=True,
                            n_permutations=10, random_state=99)
    for p1, p2 in zip(lp1.sample(grid_gdf), lp2.sample(grid_gdf)):
        numpy.testing.assert_array_equal(p1, p2)


# -- Input types ---------------------------------------------------------------

def test_array_input():
    """Raw ndarray: positional integer output."""
    coords = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lp = LocalPermutation(bandwidth=1.5, derangement=True,
                           n_permutations=5, random_state=0)
    for perm in lp.sample(coords):
        assert sorted(perm) == [0, 1, 2, 3]


def test_1d_time_series_input():
    """1-D ndarray: positional output, each value moves within threshold."""
    t = numpy.arange(10, dtype=float)
    lp = LocalPermutation(bandwidth=2.5, derangement=True,
                           n_permutations=5, random_state=0)
    for perm in lp.sample(t):
        assert sorted(perm) == list(range(10))
        assert all(abs(perm[i] - i) <= 2.5 for i in range(10))


def test_pandas_series_input():
    """pandas.Series: index labels propagated, still a valid permutation."""
    import pandas
    t = pandas.Series(numpy.zeros(10),
                      index=numpy.arange(100, 110, dtype=int))
    lp = LocalPermutation(bandwidth=2.5, n_permutations=3, random_state=0)
    for perm in lp.sample(t):
        assert sorted(perm) == list(range(100, 110))


def test_pandas_series_datetime_index():
    """pandas.Series with DatetimeIndex -- timestamps propagated as labels."""
    import pandas
    idx = pandas.date_range("2020-01-01", periods=10, freq="D")
    t = pandas.Series(numpy.zeros(10), index=idx)
    lp = LocalPermutation(bandwidth=3 * 86400, n_permutations=3, random_state=0)
    for perm in lp.sample(t):
        assert sorted(perm) == sorted(idx)


def test_geometry_array_input():
    """GeometryArray: spatial path, positional integer output."""
    from shapely.geometry import Point
    gdf = geopandas.GeoDataFrame(geometry=[Point(i, 0) for i in range(10)])
    arr = gdf.geometry.values
    lp = LocalPermutation(bandwidth=3.0, n_permutations=3, random_state=0)
    for perm in lp.sample(arr):
        assert sorted(perm) == list(range(10))


# -- sklearn API ---------------------------------------------------------------

def test_no_bandwidth_no_graph_raises():
    gdf = geopandas.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)])
    with pytest.raises(ValueError, match="bandwidth"):
        list(LocalPermutation(derangement=False, n_permutations=1).sample(gdf))


def test_get_params():
    p = LocalPermutation(bandwidth=500.0, derangement=False,
                          n_permutations=99).get_params()
    assert p["bandwidth"] == 500.0
    assert p["derangement"] is False
    assert p["n_permutations"] == 99
