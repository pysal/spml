import io
from contextlib import redirect_stdout

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from spatialml.linear_model import GWLogisticRegression
from spatialml.search import BandwidthSearch


def test_bandwidth_search_init(sample_data):
    """Test BandwidthSearch initialization with default parameters."""
    X, y, geometry = sample_data
    search = BandwidthSearch(GWLogisticRegression, geometry=geometry)

    # Check default parameters
    assert search.model == GWLogisticRegression
    assert search.fixed is False
    assert search.kernel == "bisquare"
    assert search.n_jobs == -1
    assert search.search_method == "golden_section"
    assert search.criterion is None
    assert search.min_bandwidth is None
    assert search.max_bandwidth is None
    assert search.interval is None
    assert search.max_iterations == 100
    assert search.tolerance == 1e-2
    assert search.verbose is False


def test_bandwidth_search_init_custom_params(sample_data):
    """Test BandwidthSearch initialization with custom parameters."""
    X, y, geometry = sample_data

    search = BandwidthSearch(
        model=GWLogisticRegression,
        geometry=geometry,
        fixed=True,
        kernel="tricube",
        n_jobs=2,
        search_method="interval",
        criterion="bic",
        min_bandwidth=100,
        max_bandwidth=500,
        interval=50,
        max_iterations=50,
        tolerance=1e-3,
        verbose=True,
        max_iter=200,  # Model-specific parameter
    )

    # Check custom parameters
    assert search.model == GWLogisticRegression
    assert search.fixed is True
    assert search.kernel == "tricube"
    assert search.n_jobs == 2
    assert search.search_method == "interval"
    assert search.criterion == "bic"
    assert search.min_bandwidth == 100
    assert search.max_bandwidth == 500
    assert search.interval == 50
    assert search.max_iterations == 50
    assert search.tolerance == 1e-3
    assert search.verbose is True
    assert search._model_kwargs["max_iter"] == 200


def test_interval_search_basic(sample_data):  # noqa: F811
    """Test basic interval search functionality."""
    X, y, geometry = sample_data

    # Use a very small range for faster testing
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,  # Fixed bandwidth for faster testing
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=50000,
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that the search was performed
    assert hasattr(search, "scores_")
    assert isinstance(search.scores_, pd.Series)

    # Check that optimal bandwidth was found
    assert hasattr(search, "optimal_bandwidth_")
    assert (
        search.min_bandwidth
        <= search.optimal_bandwidth_  # ty:ignore[unsupported-operator]
        <= search.max_bandwidth
    )

    # Check the number of bandwidths tested
    expected_n_bandwidths = (
        int(
            (
                search.max_bandwidth - search.min_bandwidth  # ty:ignore[unsupported-operator]
            )
            / search.interval
        )
        + 1
    )
    assert len(search.scores_) == expected_n_bandwidths


def test_golden_section_search_basic(sample_data):  # noqa: F811
    """Test basic golden section search functionality."""
    X, y, geometry = sample_data

    # Configure for a quick golden section search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,  # Fixed bandwidth for faster testing
        search_method="golden_section",
        min_bandwidth=100000,
        max_bandwidth=600000,
        max_iterations=5,  # Limit iterations for faster testing
        tolerance=0.1,  # High tolerance for faster convergence
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that the search was performed
    assert hasattr(search, "scores_")
    assert isinstance(search.scores_, pd.Series)

    # Check that optimal bandwidth was found
    assert hasattr(search, "optimal_bandwidth_")
    assert (
        search.min_bandwidth
        <= search.optimal_bandwidth_  # ty:ignore[unsupported-operator]
        <= search.max_bandwidth
    )
    # Golden section should evaluate fewer points than interval search
    assert len(search.scores_) <= search.max_iterations * 2


@pytest.mark.parametrize("criterion", ["aic", "bic", "aicc"])
def test_different_criteria(sample_data, criterion):  # noqa: F811
    """Test BandwidthSearch with different criteria."""
    X, y, geometry = sample_data

    # Configure for a quick search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,  # Just two points for speed
        criterion=criterion,
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that the search was performed
    assert hasattr(search, "scores_")
    assert search.scores_.name == criterion


def test_adaptive_bandwidth_search(sample_data):  # noqa: F811
    """Test BandwidthSearch with adaptive bandwidth."""
    X, y, geometry = sample_data

    # Configure for a quick adaptive bandwidth search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=False,  # Adaptive bandwidth
        search_method="interval",
        min_bandwidth=5,  # Small number of neighbors
        max_bandwidth=84,
        interval=25,  # Just 3 points for speed
        verbose=False,
        random_state=42,
        max_iter=500,
        include_focal=False,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that the search was performed
    assert hasattr(search, "scores_")
    assert isinstance(search.scores_, pd.Series)

    # Check that optimal bandwidth was found and is an integer (neighbor count)
    assert hasattr(search, "optimal_bandwidth_")
    assert search.optimal_bandwidth_ == 80


def test_lower_criterion_value_is_better(sample_data):  # noqa: F811
    """Test that lower criterion value is considered better."""
    X, y, geometry = sample_data

    # Configure for a controlled test with just two bandwidth values
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,  # Just two points: 100000 and 200000
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that the optimal bandwidth is the one with lower criterion value
    min_score_bw = search.scores_.idxmin()
    assert search.optimal_bandwidth_ == min_score_bw


def test_model_invariant_y_returns_inf(sample_data):  # noqa: F811
    """Test that invariant y in a model returns inf criterion score."""
    X, y, geometry = sample_data

    # Create a search instance
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        verbose=False,
    )

    # Mock a dataset with invariant y
    y_invariant = pd.Series([True] * len(y))

    # Call the internal _score method directly
    search.geometry = geometry
    score = search._score(X, y_invariant, bw=100000)

    # Check that the score is np.inf for invariant y
    assert score[0] == np.inf
    assert isinstance(score[1], list)
    assert len(score[1]) == 3
    assert all(np.isnan(x) for x in score[1])


def test_bandwidth_search_returns_self(sample_data):  # noqa: F811
    """Test that fit returns self for method chaining."""
    X, y, geometry = sample_data

    # Configure for a quick search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,  # Just two points for speed
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit and check return value
    result = search.fit(X, y, geometry)
    assert result is search


def test_bandwidth_search_accepts_model_params(sample_data):  # noqa: F811
    """Test that BandwidthSearch passes model parameters correctly."""
    X, y, geometry = sample_data

    # Set some custom parameters for the model
    custom_params = {
        "C": 0.5,
        "max_iter": 500,
        "solver": "liblinear",
        "random_state": 42,
    }

    # Create search with model parameters
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,
        verbose=False,
        **custom_params,  # type: ignore
    )

    # Check that parameters were stored correctly
    for param, value in custom_params.items():
        assert search._model_kwargs[param] == value

    # Fit to ensure parameters are passed to model instances
    search.fit(X, y, geometry)

    # Since model instances aren't kept, we just check that fit completes without errors
    assert hasattr(search, "optimal_bandwidth_")


def test_bandwidth_search_accepts_coplanar_option():
    """Test that BandwidthSearch exposes and forwards the coplanar option."""
    X = pd.DataFrame({"feat": np.arange(6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    geometry = gpd.GeoSeries(
        [Point(0, 0), Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0), Point(4, 0)]
    )

    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=False,
        coplanar="jitter",
        search_method="interval",
        min_bandwidth=2,
        max_bandwidth=2,
        interval=1,
        criterion="prediction_rate",
        metrics=["prediction_rate"],
        minimize=False,
        verbose=False,
        max_iter=200,
    )

    assert search.coplanar == "jitter"

    search.fit(X, y, geometry)

    assert hasattr(search, "optimal_bandwidth_")


def test_bandwidth_search_verbosity(sample_data):  # noqa: F811
    """Test that the verbose flag in BandwidthSearch produces expected output."""
    X, y, geometry = sample_data

    # Configure for a quick search with verbose=True
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,  # Just two points for speed
        verbose=True,  # Enable verbosity
        random_state=42,
        max_iter=500,
    )

    # Capture standard output
    f = io.StringIO()
    with redirect_stdout(f):
        search.fit(X, y, geometry)

    # Get the captured output
    output = f.getvalue()

    # Check that bandwidth and score information is printed
    assert "Bandwidth: 100000" in output
    assert "Bandwidth: 200000" in output
    assert "aicc:" in output

    # Check with verbose=False (should not produce output)
    search_quiet = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,
        verbose=False,  # Disable verbosity
        random_state=42,
        max_iter=500,
    )

    f_quiet = io.StringIO()
    with redirect_stdout(f_quiet):
        search_quiet.fit(X, y, geometry)

    # Get the captured output (should be minimal)
    output_quiet = f_quiet.getvalue()

    # There should still be some output due to the print in _score method,
    # but no bandwidth/score reports
    assert "Bandwidth:" not in output_quiet

    # Check that both searches produce the same result
    assert search.optimal_bandwidth_ == search_quiet.optimal_bandwidth_


@pytest.mark.parametrize("search_method", ["interval", "golden_section"])
def test_bandwidth_search_metrics(sample_data, search_method):
    """Test that BandwidthSearch handles custom metrics."""
    X, y, geometry = sample_data

    # Define custom metrics to track
    custom_metrics = ["prediction_rate"]

    # Test interval search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method=search_method,
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=100000,  # Just two points for speed
        max_iterations=3,  # Limit iterations for faster testing
        metrics=custom_metrics,
        verbose=False,
        random_state=42,
        max_iter=500,
    )

    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that metrics were tracked correctly
    assert hasattr(search, "metrics_")
    assert isinstance(search.metrics_, pd.DataFrame)
    assert list(search.metrics_.columns) == ["aicc", "aic", "bic"] + custom_metrics
    assert len(search.metrics_) == len(search.scores_)

    # Verify that metrics contain expected types
    assert (search.metrics_["prediction_rate"] > 0).all()
    assert (search.metrics_["prediction_rate"] <= 1).all()


def test_maximize_custom_metric(sample_data):
    """Test that BandwidthSearch handles custom metrics."""
    X, y, geometry = sample_data

    # Define custom metrics to track
    custom_metrics = ["prediction_rate"]

    # Test interval search
    search = BandwidthSearch(
        model=GWLogisticRegression,
        fixed=True,
        search_method="golden_section",
        min_bandwidth=100000,
        max_bandwidth=600000,
        max_iterations=10,  # Limit iterations for faster testing
        tolerance=0.01,
        verbose=False,
        random_state=42,
        max_iter=500,
        metrics=custom_metrics,
        criterion="prediction_rate",  # stupid example but does the trick in the test
        minimize=False,
    )
    # Fit the bandwidth search
    search.fit(X, y, geometry)

    # Check that metrics were tracked correctly
    assert search.optimal_bandwidth_ > 400000  # ty:ignore[unsupported-operator]


def test_bandwidth_search_global_invariant_y():
    """
    Verifies that BandwidthSearch handles a target y that is all 0s or all 1s.
    The fix ensures _score returns a correctly sized list of NaNs instead of crashing.
    """
    # Create 10 points with invariant y (all ones)
    n = 10
    X = pd.DataFrame({"feat": np.random.rand(n)})
    y = pd.Series([1] * n)
    geometry = gpd.GeoSeries([Point(i, i) for i in range(n)])

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=1,
        max_bandwidth=5,
        interval=1,
        verbose=False,
    )

    # This should not crash and should return Inf scores
    search.fit(X, y, geometry)

    assert (search.scores_ == np.inf).all()
    # Ensure metrics DataFrame has correct columns (met) and is all NaNs
    assert search.metrics_.isna().all().all()
    assert len(search.metrics_.columns) == 3  # aicc, aic, bic (default)


def test_bandwidth_search_local_invariant_y():
    """
    Constructs data where target is mixed globally, but the subset of locations
    that successfully fit a local model all have the same global label.

    Data Setup:
    - Cluster A (0-19): y=0. Far away. Will fail (invariant neighborhood).
    - Cluster B (20-24): y=1. At x=100.
    - Cluster C (25-29): y=0. At x=101.
    - Cluster D (30-49): y=0. At x=101.5 (closer to C than B is).

    With adaptive bandwidth k=10:
    - Cluster A points see only y=0. Fail.
    - Cluster B points see B (y=1) and C (y=0). Mixed! Success.
    - Cluster C points see C (y=0) and D (y=0). Invariant! Fail.
    - Cluster D points see only y=0. Fail.

    Only Cluster B points (all y=1) succeed. y_masked is all 1s.
    This triggers the fix in log_loss calculation for invariant subsets.
    """
    # y construction
    y = pd.Series([0] * 20 + [1] * 5 + [0] * 5 + [0] * 20)
    n = len(y)
    X = pd.DataFrame({"feat": np.random.rand(n)})

    # Geometry construction
    coords = []
    for i in range(20):
        coords.append(Point(0, i * 0.01))  # A
    for i in range(5):
        coords.append(Point(100, i * 0.01))  # B
    for i in range(5):
        coords.append(Point(101, i * 0.01))  # C
    for i in range(20):
        coords.append(Point(101.5, i * 0.01))  # D
    geometry = gpd.GeoSeries(coords)

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=False,
        search_method="interval",
        min_bandwidth=10,
        max_bandwidth=10,
        interval=1,
        criterion="log_loss",
        metrics=["log_loss"],
        verbose=False,
    )

    # This should not crash and return Inf for the log_loss metric
    search.fit(X, y, geometry)

    # Both score and log_loss metric should be Inf because y_masked is invariant
    assert (search.scores_ == np.inf).all()
    assert (search.metrics_["log_loss"] == np.inf).all()


def test_bandwidth_search_standard_data(sample_data):
    """
    Ensure log_loss calculation still works correctly on mixed data where
    local neighborhoods are also mixed.
    """
    X, y, geometry = sample_data

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=200000,
        max_bandwidth=200000,
        interval=1,
        criterion="log_loss",
        metrics=["log_loss"],
        verbose=False,
        max_iter=500,
    )

    search.fit(X, y, geometry)

    # Should calculate valid finite log_loss
    assert not np.isinf(search.metrics_["log_loss"]).all()
    assert not search.metrics_["log_loss"].isna().all()


def test_bandwidth_search_all_models_fail():
    """
    Verifies the fix for the idxmin() crash when no bandwidth produces valid models.
    """
    # Mixed y globally
    y = pd.Series([0] * 10 + [1] * 10)
    n = len(y)
    X = pd.DataFrame({"feat": np.random.rand(n)})

    # Far apart clusters
    coords = [Point(0, i * 0.01) for i in range(10)] + [
        Point(100, i * 0.01) for i in range(10)
    ]
    geometry = gpd.GeoSeries(coords)

    # Small bandwidth k=5 ensures every neighborhood is invariant
    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=False,
        search_method="interval",
        min_bandwidth=5,
        max_bandwidth=5,
        interval=1,
        criterion="log_loss",
        metrics=["log_loss", "prediction_rate"],
        verbose=False,
    )

    # This should not crash, even though all models fail
    search.fit(X, y, geometry)

    # Score should be Inf (since criterion is log_loss)
    assert (search.scores_ == np.inf).all()
    assert (search.metrics_["prediction_rate"] == 0).all()
    assert (search.metrics_["log_loss"] == np.inf).all()


def test_bandwidth_search_all_models_fail_prediction_rate_criterion():
    """
    Exercises the specific branch: if self.criterion == "prediction_rate"
    """
    y = pd.Series([0] * 10 + [1] * 10)
    X = pd.DataFrame({"feat": np.random.rand(20)})
    coords = [Point(0, i * 0.01) for i in range(10)] + [
        Point(100, i * 0.01) for i in range(10)
    ]
    geometry = gpd.GeoSeries(coords)

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=False,
        search_method="interval",
        min_bandwidth=5,
        max_bandwidth=5,
        interval=1,
        criterion="prediction_rate",
        metrics=["prediction_rate"],
        verbose=False,
    )

    search.fit(X, y, geometry)

    # Score should be 0 (the prediction rate itself)
    assert (search.scores_ == 0).all()
    assert (search.metrics_["prediction_rate"] == 0).all()
