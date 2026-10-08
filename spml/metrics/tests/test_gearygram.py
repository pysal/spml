import geopandas as gpd
import numpy as np
import pytest

from spml.metrics import gearygram

EXPECTED = {
    "gaussian_uni": (
        dict(n_bins=4),
        [12.97879, 38.936369, 64.893949, 90.851529],
        [1.01918058, 0.96608964, 1.0364114, 1.05497031],
        [105, 105, 105, 105],
    ),
    "exponential_multi": (
        dict(n_bins=4, kernel="exponential"),
        [12.97879, 38.936369, 64.893949, 90.851529],
        [0.65017347, 0.8417558, 0.92513933, 0.96498039],
        [105, 105, 105, 105],
    ),
    "bisquare_multi": (
        dict(n_bins=4, kernel="bisquare"),
        [12.97879, 38.936369, 64.893949, 90.851529],
        [0.99721775, 0.57010476, 0.63271656, 0.75261051],
        [4, 32, 66, 98],
    ),
    "triangular_maxdist": (
        dict(n_bins=4, kernel="triangular", max_distance=60),
        [7.5, 22.5, 37.5, 52.5],
        [np.nan, 1.25512693, 0.89126369, 0.85406333],
        [0, 10, 31, 44],
    ),
    "knn_uniform_uni": (
        dict(max_k=4),
        [1.0, 2.0, 3.0, 4.0],
        [1.01292628, 0.74046819, 0.70463979, 0.70199258],
        [15, 30, 45, 60],
    ),
    "knn_gaussian_multi": (
        dict(max_k=4, kernel="gaussian"),
        [1.0, 2.0, 3.0, 4.0],
        [0.6496413, 0.54880744, 0.56571669, 0.59953981],
        [15, 30, 45, 60],
    ),
    "nonparametric": (
        dict(n_bins=4, nonparametric=True),
        [0.0, 59.946462, 84.7771, 103.830319],
        [0.33493292, 0.84120097, 0.59942112, 0.43288955],
        None,
    ),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, (15, 2))
    X = rng.normal(size=(15, 3)) + coords[:, [0]] / 50
    return X, coords


@pytest.mark.filterwarnings("ignore:Bandwidth bin")
@pytest.mark.parametrize("case", EXPECTED)
def test_values(data, case):
    if case == "nonparametric":
        pytest.importorskip("statsmodels")
    X, coords = data
    kwargs, centers, C, n_pairs = EXPECTED[case]
    if not case.endswith("_multi"):
        X = X[:, 0]
    res = gearygram(X, coords, **kwargs)  # ty:ignore[invalid-argument-type]
    np.testing.assert_allclose(res.bin_centers, centers, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.C, C, rtol=1e-6, equal_nan=True)
    if n_pairs is None:
        assert res.n_pairs is None
    else:
        np.testing.assert_array_equal(res.n_pairs, n_pairs)


def test_geoseries_equivalent(data):
    X, coords = data
    gs = gpd.GeoSeries(gpd.points_from_xy(*coords.T))
    a = gearygram(X[:, 0], coords, n_bins=4)
    b = gearygram(X[:, 0], gs, n_bins=4)
    np.testing.assert_allclose(a.C, b.C)


def test_errors(data):
    X, coords = data
    with pytest.raises(ValueError, match="geometry has"):
        gearygram(X[:5, 0], coords)
    with pytest.raises(ValueError, match="constant"):
        gearygram(np.ones(len(coords)), coords)
    with pytest.raises(ValueError, match="kernel must be"):
        gearygram(X[:, 0], coords, kernel="nope")
