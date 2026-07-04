import numpy
import pytest
import geopandas
from shapely.geometry import Point

from spml.validation import LocalBootstrap


@pytest.fixture
def line_gdf():
    """10 equally-spaced points on the x-axis."""
    return geopandas.GeoDataFrame(geometry=[Point(i, 0) for i in range(10)])


# -- Basic behaviour -----------------------------------------------------------

def test_n_bootstraps(line_gdf):
    boots = list(LocalBootstrap(n_bootstraps=20, bandwidth=2.0, random_state=0).sample(line_gdf))
    assert len(boots) == 20


def test_output_shape(line_gdf):
    n = len(line_gdf)
    for indices in LocalBootstrap(n_bootstraps=5, bandwidth=2.0, random_state=0).sample(line_gdf):
        assert indices.shape == (n,)
        assert numpy.issubdtype(indices.dtype, numpy.integer)


def test_indices_in_range(line_gdf):
    n = len(line_gdf)
    for indices in LocalBootstrap(n_bootstraps=10, bandwidth=2.0, random_state=0).sample(line_gdf):
        assert indices.min() >= 0
        assert indices.max() < n


# -- Kernel weighting ----------------------------------------------------------

def test_large_bandwidth_approaches_uniform(line_gdf):
    n = len(line_gdf)
    counts = numpy.zeros(n)
    for indices in LocalBootstrap(n_bootstraps=2000, bandwidth=1e6,
                                  random_state=0).sample(line_gdf):
        for idx in indices:
            counts[idx] += 1
    assert counts.std() / counts.mean() < 0.3


def test_small_bandwidth_samples_locally(line_gdf):
    site0_draws = []
    for indices in LocalBootstrap(n_bootstraps=500, bandwidth=0.5,
                                  random_state=0).sample(line_gdf):
        site0_draws.append(indices[0])
    assert (numpy.array(site0_draws) <= 1).mean() > 0.9


def test_high_weight_neighbor_sampled_more(line_gdf):
    site5_draws = []
    for indices in LocalBootstrap(n_bootstraps=1000, bandwidth=2.0,
                                  random_state=0).sample(line_gdf):
        site5_draws.append(indices[5])
    site5_draws = numpy.array(site5_draws)
    central = numpy.sum((site5_draws >= 3) & (site5_draws <= 7))
    extreme = numpy.sum((site5_draws < 3) | (site5_draws > 7))
    assert central > extreme


# -- 1-D (time-series) input ---------------------------------------------------

def test_1d_array_input():
    """Pass a 1-D time index -- distance is absolute lag, no index labels."""
    t = numpy.arange(20, dtype=float)
    boots = list(LocalBootstrap(n_bootstraps=10, bandwidth=3.0,
                                random_state=0).sample(t))
    assert len(boots) == 10
    assert boots[0].shape == (20,)
    assert numpy.issubdtype(boots[0].dtype, numpy.integer)


def test_1d_local_sampling():
    """With narrow bandwidth, each time step draws mostly from nearby steps."""
    t = numpy.arange(50, dtype=float)
    draws_at_25 = []
    for indices in LocalBootstrap(n_bootstraps=500, bandwidth=2.0,
                                  random_state=0).sample(t):
        draws_at_25.append(indices[25])
    draws = numpy.array(draws_at_25)
    assert (numpy.abs(draws - 25) <= 5).mean() > 0.85


def test_pandas_series_input():
    """pandas.Series: index labels are propagated to output."""
    import pandas
    t = pandas.Series(numpy.zeros(20),
                      index=numpy.arange(100, 120, dtype=int))
    boots = list(LocalBootstrap(n_bootstraps=5, bandwidth=3.0,
                                random_state=0).sample(t))
    assert len(boots) == 5
    assert boots[0].shape == (20,)
    # All values must be valid index labels
    assert all(v in t.index for v in boots[0])


def test_pandas_series_datetime_index():
    """pandas.Series with DatetimeIndex -- timestamps propagated as labels."""
    import pandas
    idx = pandas.date_range("2020-01-01", periods=20, freq="D")
    t = pandas.Series(numpy.zeros(20), index=idx)
    boots = list(LocalBootstrap(n_bootstraps=5, bandwidth=3.0,
                                random_state=0).sample(t))
    assert all(v in idx for v in boots[0])


def test_geometry_array_input(line_gdf):
    """GeometryArray: spatial path, positional integer output."""
    arr = line_gdf.geometry.values   # GeometryArray
    boots = list(LocalBootstrap(n_bootstraps=5, bandwidth=3.0,
                                random_state=0).sample(arr))
    assert len(boots) == 5
    assert boots[0].shape == (len(line_gdf),)
    assert all(0 <= v < len(line_gdf) for v in boots[0])


# -- Kernel options ------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["gaussian", "bisquare", "triangular",
                                     "uniform", "exponential", "parabolic"])
def test_all_kernels(line_gdf, kernel):
    boots = list(LocalBootstrap(n_bootstraps=5, bandwidth=3.0, kernel=kernel,
                                random_state=0).sample(line_gdf))
    assert len(boots) == 5


def test_invalid_kernel_raises(line_gdf):
    with pytest.raises(ValueError, match="kernel"):
        list(LocalBootstrap(n_bootstraps=1, bandwidth=1.0, kernel="bad").sample(line_gdf))


def test_no_bandwidth_no_graph_raises(line_gdf):
    with pytest.raises(ValueError, match="bandwidth"):
        list(LocalBootstrap(n_bootstraps=1).sample(line_gdf))


# -- Sparse graph path ---------------------------------------------------------

def test_sparse_graph_output_shape(line_gdf):
    """Graph path stays sparse and produces same-shaped output as dense path."""
    # Build a minimal mock graph via libpysal Distance Band
    from libpysal.weights import DistanceBand
    W = DistanceBand.from_dataframe(line_gdf, threshold=2.5)
    W.transform = "r"

    class _MockGraph:
        """Minimal duck-type so we don't need libpysal Graph."""
        def __init__(self, w):
            import scipy.sparse as sp
            self.sparse = sp.csr_matrix(w.full()[0])

    lb = LocalBootstrap(n_bootstraps=10, graph=_MockGraph(W), random_state=0)
    boots = list(lb.sample(line_gdf))
    assert len(boots) == 10
    assert all(b.shape == (len(line_gdf),) for b in boots)
    assert all(b.min() >= 0 and b.max() < len(line_gdf) for b in boots)


def test_sparse_samples_locally(line_gdf):
    """With a narrow graph, site 0 should draw only from its direct neighbours."""
    import scipy.sparse as sp

    n = len(line_gdf)
    # Hand-craft adjacency: site i connects only to i-1 and i+1 (row-normalised)
    rows, cols, data = [], [], []
    for i in range(n):
        nbrs = [j for j in (i - 1, i, i + 1) if 0 <= j < n]
        w = 1.0 / len(nbrs)
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            data.append(w)
    W_csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    class _MockGraph:
        sparse = W_csr

    lb = LocalBootstrap(n_bootstraps=200, graph=_MockGraph(), random_state=0)
    draws_at_0 = [indices[0] for indices in lb.sample(line_gdf)]
    # Site 0 can only draw from {0, 1}
    assert all(d in {0, 1} for d in draws_at_0)


def test_sparse_reproducible(line_gdf):
    import scipy.sparse as sp
    n = len(line_gdf)
    W_csr = sp.eye(n, k=1, format="csr") + sp.eye(n, k=-1, format="csr")
    row_sums = numpy.asarray(W_csr.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    from scipy.sparse import diags
    W_norm = diags(1.0 / row_sums) @ W_csr

    class _MockGraph:
        sparse = W_norm

    b1 = list(LocalBootstrap(n_bootstraps=5, graph=_MockGraph(), random_state=7).sample(line_gdf))
    b2 = list(LocalBootstrap(n_bootstraps=5, graph=_MockGraph(), random_state=7).sample(line_gdf))
    for a, b in zip(b1, b2):
        numpy.testing.assert_array_equal(a, b)




# -- Reproducibility -----------------------------------------------------------

def test_reproducible(line_gdf):
    b1 = list(LocalBootstrap(n_bootstraps=10, bandwidth=2.0, random_state=42).sample(line_gdf))
    b2 = list(LocalBootstrap(n_bootstraps=10, bandwidth=2.0, random_state=42).sample(line_gdf))
    for a, b in zip(b1, b2):
        numpy.testing.assert_array_equal(a, b)


# -- Cross-geometry sampling ---------------------------------------------------

@pytest.fixture
def donor_gdf():
    """5 measured points offset from the line_gdf grid."""
    return geopandas.GeoDataFrame(
        {"value": [10.0, 20.0, 30.0, 40.0, 50.0]},
        geometry=[Point(i * 2 + 0.5, 0) for i in range(5)],
        index=[100, 101, 102, 103, 104],
    )


def test_cross_output_length(line_gdf, donor_gdf):
    boots = list(
        LocalBootstrap(n_bootstraps=10, bandwidth=2.0, random_state=0)
        .sample(line_gdf, donor=donor_gdf)
    )
    assert len(boots) == 10


def test_cross_returns_geodataframe(line_gdf, donor_gdf):
    """Both X and donor GeoDataFrame -> yield GeoDataFrames."""
    boot = next(
        LocalBootstrap(n_bootstraps=1, bandwidth=2.0, random_state=0)
        .sample(line_gdf, donor=donor_gdf)
    )
    assert isinstance(boot, geopandas.GeoDataFrame)
    assert len(boot) == len(line_gdf)
    # Index and geometry must come from X (line_gdf)
    numpy.testing.assert_array_equal(boot.index, line_gdf.index)
    assert list(boot.geometry) == list(line_gdf.geometry)


def test_cross_values_from_donor(line_gdf, donor_gdf):
    """Data values in the result must come from the donor."""
    boot = next(
        LocalBootstrap(n_bootstraps=1, bandwidth=2.0, random_state=0)
        .sample(line_gdf, donor=donor_gdf)
    )
    assert set(boot["value"]).issubset(set(donor_gdf["value"]))


def test_cross_locality(line_gdf, donor_gdf):
    """With narrow bandwidth, X[0] (at x=0) almost always draws from the
    nearest donor (at x=0.5, value=10.0) rather than the far donor (x=8.5)."""
    draws_at_0 = []
    for boot in LocalBootstrap(n_bootstraps=300, bandwidth=0.6, random_state=0).sample(
        line_gdf, donor=donor_gdf
    ):
        draws_at_0.append(boot.iloc[0]["value"])
    assert (numpy.array(draws_at_0) == 10.0).mean() > 0.8


def test_cross_index_array_donor(line_gdf):
    """Donor as raw array -> yield positional integer arrays."""
    donor_coords = numpy.column_stack(
        [numpy.array([0.5, 2.5, 4.5, 6.5, 8.5]), numpy.zeros(5)]
    )
    boots = list(
        LocalBootstrap(n_bootstraps=5, bandwidth=2.0, random_state=0)
        .sample(line_gdf, donor=donor_coords)
    )
    assert all(0 <= v < 5 for b in boots for v in b)


def test_cross_graph_raises(line_gdf, donor_gdf):
    """graph= and donor= together must raise."""
    import scipy.sparse as sp
    class _G:
        sparse = sp.eye(5, format="csr")
    with pytest.raises(ValueError, match="donor"):
        list(
            LocalBootstrap(graph=_G(), random_state=0)
            .sample(line_gdf, donor=donor_gdf)
        )


def test_k_self_sampling(line_gdf):
    """k= path: output shape and index labels correct."""
    boots = list(LocalBootstrap(k=3, n_bootstraps=5, random_state=0).sample(line_gdf))
    assert len(boots) == 5
    assert all(b.shape == (len(line_gdf),) for b in boots)
    assert all(v in line_gdf.index for b in boots for v in b)


def test_k_cross_sampling(line_gdf, donor_gdf):
    """k= with donor: yields GeoDataFrame with X index and donor data."""
    boot = next(
        LocalBootstrap(k=2, n_bootstraps=1, random_state=0)
        .sample(line_gdf, donor=donor_gdf)
    )
    assert isinstance(boot, geopandas.GeoDataFrame)
    numpy.testing.assert_array_equal(boot.index, line_gdf.index)
    assert set(boot["value"]).issubset(set(donor_gdf["value"]))


def test_bandwidth_k_both_raises(line_gdf):
    with pytest.raises(ValueError, match="mutually exclusive"):
        LocalBootstrap(bandwidth=1.0, k=3)


def test_cross_reproducible(line_gdf, donor_gdf):
    b1 = list(LocalBootstrap(n_bootstraps=5, bandwidth=2.0, random_state=3).sample(line_gdf, donor=donor_gdf))
    b2 = list(LocalBootstrap(n_bootstraps=5, bandwidth=2.0, random_state=3).sample(line_gdf, donor=donor_gdf))
    for a, b in zip(b1, b2):
        numpy.testing.assert_array_equal(a.index, b.index)
        numpy.testing.assert_array_equal(a["value"].values, b["value"].values)


# -- sklearn API ---------------------------------------------------------------

def test_get_params():
    p = LocalBootstrap(n_bootstraps=50, bandwidth=100.0, kernel="bisquare").get_params()
    assert p["n_bootstraps"] == 50
    assert p["bandwidth"] == 100.0
    assert p["kernel"] == "bisquare"
