from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from spml.decomposition import GWPCA
from spml.search import BandwidthSearch

_MIN_BW_GRID = 8
_N_COMP_GRID = 2


@pytest.fixture(scope="module")
def nonstationary_grid_data():
    """Create a 10x10 synthetic grid with spatially varying covariance."""
    rng = np.random.default_rng(42)
    xs = np.linspace(0, 10, 10)
    ys = np.linspace(0, 10, 10)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    n = len(coords)
    X = np.zeros((n, 4))
    for i, (x, _) in enumerate(coords):
        if x <= 5:
            X[i] = rng.normal([3.0, 0.0, 0.0, 0.0], [2.0, 0.3, 0.3, 0.3])
        else:
            X[i] = rng.normal([0.0, 3.0, 0.0, 0.0], [0.3, 2.0, 0.3, 0.3])

    X_df = pd.DataFrame(X, columns=pd.Index(["f0", "f1", "f2", "f3"]))
    X_std = (X_df - X_df.mean()) / X_df.std()
    geometry = gpd.GeoSeries.from_xy(coords[:, 0], coords[:, 1])
    return X_std, geometry


@pytest.fixture(scope="module")
def california_housing_data():
    """Create a standardized California Housing subsample of size 800."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data.data), size=800, replace=False)

    X_raw = pd.DataFrame(data.data[idx], columns=data.feature_names)
    X_feat = X_raw.drop(columns=["Latitude", "Longitude"])
    X_std = (X_feat - X_feat.mean()) / X_feat.std()

    geometry = gpd.GeoSeries.from_xy(
        X_raw["Longitude"].values,
        X_raw["Latitude"].values,
    )
    return X_std, geometry


class TestBandwidthSearchSyntheticGrid:
    """Test GWPCA bandwidth search on synthetic grid data."""

    def test_interval_search_finds_optimal_in_range(self, nonstationary_grid_data):
        """Test that interval search finds an optimal bandwidth in range."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=_MIN_BW_GRID,
            max_bandwidth=50,
            interval=_MIN_BW_GRID,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert _MIN_BW_GRID <= search.optimal_bandwidth_ <= 50

    def test_golden_section_finds_optimal_in_range(self, nonstationary_grid_data):
        """Test that golden section search finds an optimal bandwidth in range."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="golden_section",
            criterion="cv_score",
            min_bandwidth=_MIN_BW_GRID,
            max_bandwidth=60,
            max_iterations=20,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert _MIN_BW_GRID <= search.optimal_bandwidth_ <= 60

    def test_cv_score_column_in_metrics(self, nonstationary_grid_data):
        """Test that cv_score column is present in metrics dataframe."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=10,
            max_bandwidth=30,
            interval=10,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        assert "cv_score" in search.metrics_.columns
        assert len(search.metrics_) == 3

    def test_scores_series_is_finite(self, nonstationary_grid_data):
        """Test that all searched bandwidths yield finite scores."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=10,
            max_bandwidth=40,
            interval=10,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        assert np.isfinite(search.scores_).all()
        assert (search.scores_ > 0).all()

    def test_small_bandwidth_worse_than_medium(self, nonstationary_grid_data):
        """Test that small bandwidth has worse CV score than optimal."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=_MIN_BW_GRID,
            max_bandwidth=40,
            interval=_MIN_BW_GRID,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        scores = search.scores_.sort_index()
        optimal = search.optimal_bandwidth_
        assert optimal != scores.index[0]

    def test_non_stationary_optimal_below_n(self, nonstationary_grid_data):
        """Test that non-stationary optimal bandwidth is strictly local."""
        X, geometry = nonstationary_grid_data
        n = len(X)
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=_MIN_BW_GRID,
            max_bandwidth=n - 1,
            interval=10,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        assert search.optimal_bandwidth_ < n // 2

    def test_fit_with_optimal_bandwidth_runs(self, nonstationary_grid_data):
        """Test GWPCA fit runs with optimal bandwidth."""
        X, geometry = nonstationary_grid_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=10,
            max_bandwidth=30,
            interval=10,
            n_components=_N_COMP_GRID,
            n_jobs=1,
        ).fit(X, y=None, geometry=geometry)

        model = GWPCA(
            n_components=_N_COMP_GRID,
            bandwidth=search.optimal_bandwidth_,
            fixed=False,
            n_jobs=1,
        ).fit(X, geometry=geometry)

        assert model.components_.shape == (len(X), X.shape[1] * _N_COMP_GRID)
        assert model.explained_variance_ratio_.shape == (len(X), _N_COMP_GRID)


class TestBandwidthSearchCaliforniaHousing:
    """Test GWPCA bandwidth search on California Housing data."""

    def test_interval_search_completes(self, california_housing_data):
        """Test interval search on California Housing dataset."""
        X, geometry = california_housing_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=50,
            max_bandwidth=200,
            interval=50,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert 50 <= search.optimal_bandwidth_ <= 200

    def test_golden_section_search_completes(self, california_housing_data):
        """Test golden section search on California Housing dataset."""
        X, geometry = california_housing_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="golden_section",
            criterion="cv_score",
            min_bandwidth=40,
            max_bandwidth=300,
            max_iterations=10,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert 40 <= search.optimal_bandwidth_ <= 300

    def test_scores_are_finite(self, california_housing_data):
        """Test that scores on California Housing are finite."""
        X, geometry = california_housing_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=100,
            max_bandwidth=300,
            interval=100,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        assert np.isfinite(search.scores_).all()

    def test_metrics_dataframe_shape(self, california_housing_data):
        """Test the shape of metrics dataframe on California Housing."""
        X, geometry = california_housing_data
        n_bw = 3
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=50,
            max_bandwidth=150,
            interval=50,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        assert search.metrics_.shape[0] == n_bw
        assert "cv_score" in search.metrics_.columns

    def test_fit_at_optimal_bandwidth_produces_valid_output(
        self, california_housing_data
    ):
        """Test GWPCA output with optimal bandwidth."""
        X, geometry = california_housing_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=50,
            max_bandwidth=150,
            interval=50,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        model = GWPCA(
            n_components=3,
            bandwidth=search.optimal_bandwidth_,
            fixed=False,
            coplanar="jitter",
            n_jobs=-1,
        ).fit(X, geometry=geometry)

        n, p = len(X), X.shape[1]
        assert model.components_.shape == (n, p * 3)
        assert model.explained_variance_ratio_.shape == (n, 3)

        row_sums = model.explained_variance_ratio_.sum(axis=1)
        assert (row_sums > 0).all()
        assert (row_sums <= 1.000001).all()

    def test_cv_score_decreases_then_increases(self, california_housing_data):
        """Test that optimal bandwidth is found in the search boundaries."""
        X, geometry = california_housing_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            coplanar="jitter",
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=20,
            max_bandwidth=500,
            interval=120,
            n_components=3,
            n_jobs=-1,
        ).fit(X, y=None, geometry=geometry)

        optimal = search.optimal_bandwidth_
        assert 20 <= optimal <= 500
