import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import sklearn
from geodatasets import get_path
from libpysal.graph import Graph
from packaging.version import Version
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from spatialml.base import BaseClassifier, BaseRegressor, _kernel_functions


def test_init_default_parameters():
    """Test BaseClassifier initialization with default parameters."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)

    assert clf.model == LogisticRegression
    assert clf.bandwidth == 100
    assert clf.fixed is False
    assert clf.kernel == "bisquare"
    assert clf.n_jobs == -1
    assert clf.fit_global_model is True
    assert clf.strict is False
    assert clf.keep_models is False
    assert clf.temp_folder is None
    assert clf.batch_size is None
    assert clf.min_proportion == 0.2
    assert isinstance(clf._model_kwargs, dict)
    assert len(clf._model_kwargs) == 0


def test_init_custom_parameters():
    """Test BaseClassifier initialization with custom parameters."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_jobs=2,
        fit_global_model=False,
        strict=True,
        keep_models=True,
        temp_folder="/tmp",
        batch_size=10,
        min_proportion=0.3,
        max_iter=200,  # A LogisticRegression parameter
        coplanar="jitter",
    )

    assert clf.model == LogisticRegression
    assert clf.bandwidth == 50
    assert clf.fixed is True
    assert clf.kernel == "tricube"
    assert clf.n_jobs == 2
    assert clf.fit_global_model is False
    assert clf.strict is True
    assert clf.keep_models is True
    assert clf.temp_folder == "/tmp"
    assert clf.batch_size == 10
    assert clf.min_proportion == 0.3
    assert "max_iter" in clf._model_kwargs
    assert clf._model_kwargs["max_iter"] == 200
    assert clf.coplanar == "jitter"


def test_init_keep_models_path():
    """Test BaseClassifier initialization with keep_models as Path."""
    path_str = ["/tmp/models", "\\tmp\\models"]

    # Test with string
    clf = BaseClassifier(LogisticRegression, bandwidth=100, keep_models=path_str[0])
    assert isinstance(clf.keep_models, Path)
    assert str(clf.keep_models) in path_str

    # Test with Path object
    path_obj = Path(path_str[0])
    clf = BaseClassifier(LogisticRegression, bandwidth=100, keep_models=path_obj)
    assert clf.keep_models == path_obj


def test_init_kernel_assignment():
    """Test BaseClassifier initialization with various kernel options."""
    # Test with each predefined kernel
    for kernel_name in _kernel_functions:
        clf = BaseClassifier(
            LogisticRegression,
            bandwidth=100,
            kernel=kernel_name,  # ty:ignore[invalid-argument-type]
        )
        assert clf.kernel == kernel_name

    # Test with a custom kernel function
    def custom_kernel(distances, bandwidth):
        return np.exp(-distances / bandwidth) * 2

    clf = BaseClassifier(LogisticRegression, bandwidth=100, kernel=custom_kernel)
    assert clf.kernel == custom_kernel


def test_init_with_real_data():
    """Test BaseClassifier initialization with real data."""
    # Create classifier with default params
    clf = BaseClassifier(LogisticRegression, bandwidth=50000, fixed=True)

    # Just testing that initialization doesn't raise errors
    assert clf.model == LogisticRegression
    assert clf.bandwidth == 50000
    assert clf.fixed is True


@pytest.mark.parametrize("bandwidth", [0, -1, -100])
def test_fit_invalid_bandwidth_raises(sample_data, bandwidth):
    """Ensure ValueError is raised for non-positive bandwidth during fit."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=bandwidth,
        fixed=True,
    )

    with pytest.raises(ValueError, match="Bandwidth must be a positive scalar"):
        clf.fit(X, y, geometry)


def test_bandwidth_not_real(sample_data):
    """Ensure non-numeric bandwidth raises ValueError."""
    X, y, geometry = sample_data

    # Pass an invalid (non-numeric) bandwidth,
    # intentionally passing invalid types for runtime validation
    clf = BaseClassifier(LogisticRegression, bandwidth="invalid")  # type: ignore

    # Validation should fail during fit()
    with pytest.raises(ValueError, match="Bandwidth"):
        clf.fit(X, y, geometry)


def test_init_multiple_kwargs():
    """Test BaseClassifier initialization with multiple model kwargs."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        C=0.5,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
    )

    # Check that all kwargs are passed to model_kwargs
    assert clf._model_kwargs["C"] == 0.5
    assert clf._model_kwargs["penalty"] == "l2"
    assert clf._model_kwargs["solver"] == "liblinear"
    assert clf._model_kwargs["max_iter"] == 1000


def test_init_preserve_model_class():
    """Test that BaseClassifier preserves the model class without instantiating it."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)

    # The model should be stored as a class, not an instance
    assert clf.model == LogisticRegression
    assert not isinstance(clf.model, LogisticRegression)


def test_fit_basic_functionality(sample_data):
    """Test basic fitting functionality of BaseClassifier."""
    X, y, geometry = sample_data

    # Create classifier with default params
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
        max_iter=250,
    )

    # Fit the model
    fitted_clf = clf.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_clf is clf

    # Test that the global model was fitted
    assert hasattr(clf, "global_model")
    assert isinstance(clf.global_model, LogisticRegression)

    assert 0 <= clf.pred_.mean() <= 1


def test_fit_with_keep_models(sample_data):
    """Test fitting with keep_models=True to retain local models."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        random_state=42,
        max_iter=250,
        strict=False,  # To avoid warnings on invariance
        n_jobs=1,
    )

    clf.fit(X, y, geometry)

    # Check that local models were kept
    assert hasattr(clf, "_local_models")
    assert isinstance(clf._local_models, pd.Series)
    assert len(clf._local_models) > 0

    # Check that each local model is a fitted LogisticRegression
    for model in clf._local_models:
        assert isinstance(model, LogisticRegression | None)
        # Check that the model has been fitted by ensuring it has a coef_ attribute
        assert (
            hasattr(model, "coef_") if isinstance(model, LogisticRegression) else True
        )


def test_fit_with_keep_models_path(sample_data):
    """Test fitting with keep_models as a Path to save models to disk."""
    X, y, geometry = sample_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a classifier with keep_models as a path
        clf = BaseClassifier(
            LogisticRegression,
            bandwidth=10,
            fixed=False,
            keep_models=temp_dir,
            random_state=42,
            max_iter=250,
            strict=False,  # To avoid warnings on invariance
            n_jobs=1,
        )

        clf.fit(X, y, geometry)

        # Check that models were serialized to disk
        model_files = list(Path(temp_dir).glob("*"))
        assert len(model_files) > 0


@pytest.mark.parametrize("kernel", _kernel_functions)
def test_fit_different_kernels(sample_data, kernel):
    """Test fitting with different kernel functions."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        kernel=kernel,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )

    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert 0 <= clf.pred_.mean() <= 1


@pytest.mark.parametrize(
    "coplanar",
    [
        "jitter",
        # 'clique', see https://github.com/pysal/libpysal/issues/897
    ],
)
def test_fit_adaptive_bandwidth_with_coplanar(coplanar):
    """Adaptive fitting should work with duplicate points."""
    X = pd.DataFrame({"feat": np.arange(6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    geometry = gpd.GeoSeries(
        [
            Point(0, 0),
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
            Point(3, 0),
            Point(4, 0),
        ]
    )

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=2,
        fixed=False,
        coplanar=coplanar,
        strict=False,
        fit_global_model=False,
        max_iter=200,
    )

    clf.fit(X, y, geometry)

    assert hasattr(clf, "proba_")
    assert clf.coplanar == coplanar


def test_fit_fixed_bandwidth(sample_data):
    """Test fitting with adaptive bandwidth (fixed=False)."""
    X, y, geometry = sample_data

    # Use a small k for faster testing
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100_000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )

    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert 0 <= clf.pred_.mean() <= 1


def test_fit_without_global_model(sample_data):
    """Test fitting without computing a global model."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150_000,
        fixed=True,
        fit_global_model=False,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )

    clf.fit(X, y, geometry)

    # Check that global model was not fitted
    assert not hasattr(clf, "global_model")

    # But local results should still be available
    assert hasattr(clf, "proba_")


def test_fit_negative_bandwidth_raises(sample_data):
    """Negative bandwidth raises ValueError in fit()."""
    X, y, geometry = sample_data

    # Initialize with invalid negative bandwidth
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=-5,
        fixed=True,
    )

    # Validation should trigger during fit()
    with pytest.raises(ValueError, match="Bandwidth must be a positive scalar"):
        clf.fit(X, y, geometry)


def test_fit_adaptive_bandwidth_must_be_integer(sample_data):
    """Adaptive bandwidth must be an integer when fixed=False."""
    X, y, geometry = sample_data

    # Initialize with non-integer adaptive bandwidth
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=2.5,
        fixed=False,
    )

    # Fit should raise error due to invalid adaptive bandwidth type
    with pytest.raises(ValueError, match="Adaptive bandwidth"):
        clf.fit(X, y, geometry)


def test_predict_proba_rejects_nan_bandwidth(sample_data):
    """Tests that NaN bandwidth raises ValueError in predict_proba()."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=5,
        fixed=True,
        keep_models=True,
    )
    clf.fit(X, y, geometry)

    with pytest.raises(ValueError, match="Bandwidth must be a positive scalar"):
        clf.predict_proba(X, geometry, bandwidth=np.nan)


def test_predict_proba_rejects_negative_bandwidth(sample_data):
    """Tests that negative bandwidth raises ValueError in predict_proba()."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=5,
        fixed=True,
        keep_models=True,
    )
    clf.fit(X, y, geometry)

    with pytest.raises(ValueError, match="Bandwidth must be a positive scalar"):
        clf.predict_proba(X, geometry, bandwidth=-5)


def test_predict_proba_rejects_non_integer_adaptive_bandwidth(sample_data):
    """
    Tests that non-integer adaptive bandwidth raises ValueError in predict_proba().
    """
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=5,
        fixed=False,
        keep_models=True,
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    with pytest.raises(ValueError, match="must be an integer"):
        clf.predict_proba(X, geometry, bandwidth=5.5)


def test_fit_length_mismatch_raises(sample_data):
    """fit() raises ValueError when X and y have different lengths."""
    X, y, geometry = sample_data

    # Remove last observation from y
    # This creates a mismatch between X and y lengths
    y_bad = y.iloc[:-1]

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
    )

    # Fit should detect mismatch BEFORE doing any computation
    with pytest.raises(ValueError, match="X and y must have the same length"):
        clf.fit(X, y_bad, geometry)


def test_fit_requires_geometry_or_graph(sample_data):
    """fit() raises ValueError when neither geometry nor graph is provided."""
    X, y, _ = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
    )

    # Not passing geometry
    # Not providing graph in initialization
    # Should fail validation
    with pytest.raises(ValueError, match="Either geometry or graph must be provided"):
        clf.fit(X, y)


def test_fit_geometry_length_mismatch_raises(sample_data):
    """fit() raises ValueError when geometry length mismatches X."""
    X, y, geometry = sample_data

    # Remove last geometry row to create mismatch
    geometry_bad = geometry.iloc[:-1]

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
    )

    # Validation should detect geometry length issue
    with pytest.raises(ValueError, match="X and geometry must have the same length"):
        clf.fit(X, y, geometry_bad)


def test_fit_with_strict_option(sample_data):
    """Test the strict option for invariant y."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=X.shape[0] - 1,  # global bandwidth
        fixed=False,
        strict=True,  # Raise error if invariant
        random_state=42,
    )

    # The fit should complete without error because even with large bandwidth,
    # the target is likely varied enough
    clf.fit(X, y, geometry)

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=5,  # known to produce invariant subsets
        fixed=False,
        strict=True,  # Raise error if invariant
        random_state=42,
    )

    # This should raise a ValueError due to invariant y
    with pytest.raises(ValueError, match="y at locations .* is invariant"):
        clf.fit(X, y, geometry)

    # But with strict=False, it should just warn
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=5,
        fixed=False,
        strict=None,  # Just warn if invariant
        random_state=42,
    )

    # Should complete with a warning
    with pytest.warns(UserWarning, match="y at locations .* is invariant"):
        clf.fit(X, y, geometry)


def test_non_point_geometry_raises_error(sample_data):
    """Test that non-point geometries raise an error."""
    X, y, _ = sample_data

    # Get the original polygons instead of centroids
    gdf = gpd.read_file(get_path("geoda.guerry"))
    polygon_geometry = gdf.geometry

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=50000,
        fixed=True,
        strict=False,  # To avoid warnings on invariance
    )

    # This should raise a ValueError due to non-point geometries
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        clf.fit(X, y, polygon_geometry)


def test_fit_with_batch_processing(sample_data):
    """Test fitting with batch processing enabled."""
    X, y, geometry = sample_data

    # Create a classifier with a small batch size
    batch_size = 5
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        batch_size=batch_size,  # Process in small batches
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        verbose=True,
        max_iter=500,
    )

    # Capture print output to verify batch processing messages
    f = io.StringIO()
    with redirect_stdout(f):
        clf.fit(X, y, geometry)

    # Get the captured output
    output = f.getvalue()

    # Test that batch processing messages were printed
    expected_batches = ((len(X) + batch_size - 1) // batch_size) + 1  # Ceiling division
    assert f"Processing batch 1 out of {expected_batches}" in output

    # Check that the model was fit successfully
    assert hasattr(clf, "proba_")
    assert 0 <= clf.pred_.mean() <= 1

    # Compare with a model without batching to ensure results are consistent
    clf_no_batch = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        max_iter=500,
    )
    clf_no_batch.fit(X, y, geometry)

    # Results should be similar regardless of batching
    pd.testing.assert_frame_equal(
        clf.proba_, clf_no_batch.proba_, check_exact=False, rtol=1e-5
    )


def test_fit_batch_processing_non_consecutive_index(sample_data):
    """Test fitting with batch processing on non-consecutive index."""
    X, y, geometry = sample_data

    # Create non-consecutive indices
    new_index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    new_index.reverse()
    X_indexed = X.iloc[:10].copy()
    X_indexed.index = new_index
    y_indexed = y.iloc[:10].copy()
    y_indexed.index = new_index
    geometry_indexed = geometry.iloc[:10].copy()
    geometry_indexed.index = new_index

    # Create a classifier with batch processing
    batch_size = 3
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        batch_size=batch_size,
        random_state=42,
        strict=False,
        verbose=True,
        max_iter=500,
    )

    # Fit the model with non-consecutive index
    clf.fit(X_indexed, y_indexed, geometry_indexed)

    # Check that the model was fit successfully
    assert hasattr(clf, "proba_")
    assert 0 <= clf.pred_.mean() <= 1
    assert len(clf.proba_) == 10
    assert list(clf.proba_.index) == new_index


def test_fit_n_jobs_consistency(sample_data):
    """Test that parallel processing gives the same results as sequential (n_jobs=1)."""
    X, y, geometry = sample_data

    # Create a classifier with n_jobs=1 (sequential)
    clf_sequential = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf_sequential.fit(X, y, geometry)

    # Create a classifier with n_jobs=-1 (parallel)
    clf_parallel = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=-1,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf_parallel.fit(X, y, geometry)

    # Check that the results are the same regardless of parallelization
    pd.testing.assert_frame_equal(
        clf_sequential.proba_,
        clf_parallel.proba_,
        check_exact=False,
        rtol=1e-5,
    )
    # Check that global models have the same coefficients
    np.testing.assert_allclose(
        clf_sequential.global_model.coef_, clf_parallel.global_model.coef_, rtol=1e-5
    )


def test_local_class_support_exposed(sample_data):
    """Test that local_class_support_ is exposed after fitting."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        random_state=42,
        strict=False,
        n_jobs=1,
        max_iter=250,
    )

    clf.fit(X, y, geometry)

    # Check that attribute exists
    assert hasattr(clf, "local_class_support_")

    # Check the Output format
    assert isinstance(clf.local_class_support_, pd.Series)

    # Check that Length matches number of local models
    assert len(clf.local_class_support_) == len(clf._names)

    # Check that values are positive integers
    assert (clf.local_class_support_ >= 1).all()


def test_local_class_support_invariant(sample_data):
    """Test that invariant neighborhoods report support == 1 and are skipped."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=1,  # Very small to force invariance
        fixed=False,
        random_state=42,
        strict=False,
        n_jobs=1,
        max_iter=250,
    )

    clf.fit(X, y, geometry)

    # Check that at least one neighborhood should be invariant
    assert (clf.local_class_support_ == 1).any()

    # Invariant neighborhoods should correspond to skipped models
    invariant_idx = clf.local_class_support_ == 1
    assert clf.pred_[invariant_idx].isna().all()

    # some models should be skipped
    assert clf.prediction_rate_ < 1


def test_local_class_support_min_proportion(sample_data):
    """
    Test that neighborhoods failing min_proportion report 2 labels
    but are skipped.
    """
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=20,
        fixed=False,
        min_proportion=0.9,
        random_state=42,
        strict=False,
        n_jobs=1,
        max_iter=250,
    )

    clf.fit(X, y, geometry)

    # Checks that neighborhoods with two distinct labels should exist
    two_label_idx = clf.local_class_support_ == 2
    assert two_label_idx.any()

    # Checks that some of those should be skipped due to imabalance
    skipped = clf.pred_.isna()
    assert (two_label_idx & skipped).any()

    # Checks that skipping must have occured
    assert clf.prediction_rate_ < 1


def test_local_class_support_fitted_models(sample_data):
    """Test that fully valid neighborhoods report support == 2 and are fitted."""
    X, y, geometry = sample_data

    # Large bandwidth to ensure both classes present and fitted
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=len(X) - 1,
        fixed=False,
        min_proportion=0.1,
        random_state=42,
        strict=False,
        n_jobs=1,
        max_iter=250,
    )

    clf.fit(X, y, geometry)

    # Check that all neighborhoods should contain both classes
    assert (clf.local_class_support_ == 2).all()

    # All models should be fitted
    assert clf.prediction_rate_ == 1


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_predict_proba_basic(sample_data, bandwidth):
    """Test basic functionality of predict_proba method."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Predict probabilities for first 5 samples
    proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

    # Check output format
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape == (5, 2)  # Binary classification, so 2 columns
    assert all(column in proba.columns for column in [True, False])

    assert np.allclose(proba.dropna().sum(axis=1), 1.0)


@pytest.mark.parametrize("bandwidth", ["nearest", 8, None])
def test_predict_proba_adaptive(sample_data, bandwidth):
    """Test basic functionality of predict_proba method using adaptive kernel."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=7,
        fixed=False,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Predict probabilities for first 5 samples
    proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

    # Check output format
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape == (5, 2)  # Binary classification, so 2 columns
    assert all(column in proba.columns for column in [True, False])

    # Check probability values are valid
    assert np.allclose(proba.dropna().sum(axis=1), 1.0)


def test_predict_proba_global_weight(sample_data):
    """Test predict_proba with global_model_weight for classifier."""
    X, y, geometry = sample_data
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        fit_global_model=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf.fit(X, y, geometry)
    proba_local = clf.predict_proba(
        X.iloc[6:9], geometry.iloc[6:9], bandwidth="nearest", global_model_weight=0
    )
    proba_global = clf.global_model.predict_proba(X.iloc[6:9])
    proba_fused = clf.predict_proba(
        X.iloc[6:9], geometry.iloc[6:9], bandwidth="nearest", global_model_weight=1
    )
    # Fused should be average of local and global
    np.testing.assert_allclose(
        proba_fused.values, (proba_local.values + proba_global) / 2, rtol=1e-6
    )


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_predict_basic(sample_data, bandwidth):
    """Test basic functionality of predict method."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Predict classes for first 5 samples
    pred = clf.predict(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

    # Check output format
    assert isinstance(pred, pd.Series)
    assert len(pred) == 5

    if bandwidth is None:
        # Check all predicted values are either True or False
        assert pred.isin([True, False]).all()


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_predict_with_models_on_disk(sample_data, bandwidth):
    """Test prediction with models stored on disk."""
    X, y, geometry = sample_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and fit classifier with keep_models as a path
        clf = BaseClassifier(
            LogisticRegression,
            bandwidth=150000,
            fixed=True,
            keep_models=temp_dir,
            random_state=42,
            strict=False,  # To avoid warnings on invariance
            max_iter=500,
        )
        clf.fit(X, y, geometry)

        # Predict probabilities
        proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

        # Check output
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape == (5, 2)

        # Also test predict method
        pred = clf.predict(X.iloc[:5], geometry.iloc[:5])
        assert isinstance(pred, pd.Series)
        assert len(pred) == 5


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_predict_invalid_geometry(sample_data, bandwidth):
    """Test that prediction raises error with non-point geometries."""
    X, y, geometry = sample_data

    # Get the original polygons instead of centroids
    gdf = gpd.read_file(get_path("geoda.guerry"))
    polygon_geometry = gdf.geometry

    # Create and fit classifier with point geometries
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)  # Use point geometries for fitting

    # Attempt to predict with polygon geometries
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        clf.predict_proba(X.iloc[:5], polygon_geometry.iloc[:5], bandwidth=bandwidth)


@pytest.mark.parametrize("bandwidth", ["nearest", None])
def test_predict_comparison_with_focal_proba(sample_data, bandwidth):
    """Test that prediction for training data matches focal probabilities."""
    X, y, geometry = sample_data

    # Create and fit classifier
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Get predictions for the same data used for training
    predicted_proba = clf.predict_proba(X, geometry, bandwidth=bandwidth)

    # Compare with proba_ (should be very similar but not identical
    # because proba_ is calculated during training without using the focal point)
    pd.testing.assert_series_equal(
        predicted_proba.loc[2],
        clf.proba_.loc[2],
        check_exact=False,
        atol=0.05,  # Allow some tolerance because they're not identical
    )


def test_binary_target_zero_one(sample_data):
    """Test that 0/1 target values are correctly recognized as binary."""
    X, y, geometry = sample_data

    # Create a 0/1 encoded target
    y_01 = y.astype(int)

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # Should run without errors
    fitted_clf = clf.fit(X, y_01, geometry)
    assert fitted_clf is clf

    assert 0 <= clf.pred_.mean() <= 1

    # propagation to prediction
    pd.testing.assert_index_equal(clf.proba_.columns, pd.Index([0, 1]))


def test_non_binary_target_raises_error(sample_data):
    """Test that non-binary target variables raise an error."""
    X, _, geometry = sample_data

    # Create a non-binary target with values 1, 2, 3
    y_non_binary = pd.Series(np.random.choice([1, 2, 3], size=len(X)), index=X.index)

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # This should raise a ValueError due to non-binary target
    with pytest.raises(ValueError, match="Only binary dependent variable is supported"):
        clf.fit(X, y_non_binary)


def test_binary_with_string_values_raises_error(sample_data):
    """Test that binary target with string values raises an error."""
    X, _, geometry = sample_data

    # Create a binary target with string values
    y_str = pd.Series(np.random.choice(["yes", "no"], size=len(X)), index=X.index)

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # This should raise a ValueError due to string values
    with pytest.raises(ValueError, match="Only binary dependent variable is supported"):
        clf.fit(X, y_str)


def test_undersample_boolean(sample_data):
    """Test fitting with undersample=True option."""
    X, y, geometry = sample_data

    # Create a classifier with undersample enabled
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
        n_jobs=1,
    )

    # Fit should complete successfully
    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert 0 <= clf.pred_.mean() <= 1


def test_undersample_ratio(sample_data):
    """Test fitting with undersample as a float ratio."""
    X, y, geometry = sample_data

    # Create a classifier with undersample ratio
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=0.9,
        random_state=42,
        strict=False,
        max_iter=500,
        n_jobs=1,
    )

    # Fit should complete successfully
    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert 0 <= clf.pred_.mean() <= 1


def test_random_state_consistency(sample_data):
    """Test that same random_state produces consistent results."""
    X, y, geometry = sample_data

    # Create two classifiers with same random_state
    clf1 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    clf2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_frame_equal(clf1.proba_, clf2.proba_)


def test_different_random_states(sample_data):
    """Test that different random_states produce different results."""
    X, y, geometry = sample_data

    # Create two classifiers with different random_states
    clf1 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        random_state=42,
        strict=False,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        random_state=99,
        strict=False,
    )
    clf2.fit(X, y, geometry)

    # Results should be different
    assert not clf1.proba_.equals(clf2.proba_)


def test_random_state_with_undersample(sample_data):
    """Test that random_state affects undersample consistently."""
    X, y, geometry = sample_data

    # Create two classifiers with same random_state and undersample
    clf1 = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_frame_equal(clf1.proba_, clf2.proba_)


def test_repr_basic():
    """Test basic __repr__ functionality."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)
    repr_str = repr(clf)

    # Check that it contains the class name
    assert "BaseClassifier" in repr_str

    # Check that it contains the model name
    assert "LogisticRegression" in repr_str

    # Check that it contains the bandwidth
    assert "bandwidth=100" in repr_str


def test_repr_html_basic():
    """Test basic _repr_html_ functionality."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)
    html_str = clf._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)

    # Should contain HTML tags
    assert "<" in html_str and ">" in html_str

    # Should contain the class name
    assert "BaseClassifier" in html_str


@pytest.mark.skipif(
    Version(sklearn.__version__) == Version("1.7.0"),
    reason="https://github.com/scikit-learn/scikit-learn/pull/31528",
)
def test_repr_html_with_fitted_model(sample_data):
    """Test _repr_html_ with a fitted model."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    html_str = clf._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)
    assert "<" in html_str and ">" in html_str
    assert "BaseClassifier" in html_str


def test_repr_after_fitting(sample_data):
    """Test that __repr__ works correctly after fitting."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # Test repr before fitting
    repr_before = repr(clf)
    assert "BaseClassifier" in repr_before

    # Fit the model
    clf.fit(X, y, geometry)

    # Test repr after fitting (should still work)
    repr_after = repr(clf)
    assert "BaseClassifier" in repr_after

    # Should be the same representation
    assert repr_before == repr_after


def test_fit_focal_inclusion(sample_data):
    """Test basic fitting functionality of BaseClassifier."""
    X, y, geometry = sample_data

    # Create classifier with default params
    no_focal = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        include_focal=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    no_focal = no_focal.fit(X, y, geometry)

    # Create classifier with default params
    focal = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        include_focal=True,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    focal = focal.fit(X, y, geometry)

    # RF should 'remember' focal
    assert (no_focal.proba_[True] - no_focal.proba_[False]).abs().mean() < (
        focal.proba_[True] - focal.proba_[False]
    ).abs().mean()


# ------------regression tests----------------


def test_regressor_init_default_parameters():
    """Test BaseRegressor initialization with default parameters."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)

    assert reg.model == LinearRegression
    assert reg.bandwidth == 100
    assert reg.fixed is False
    assert reg.kernel == "bisquare"
    assert reg.n_jobs == -1
    assert reg.fit_global_model is True
    assert reg.strict is False
    assert reg.keep_models is False
    assert reg.temp_folder is None
    assert reg.batch_size is None
    assert isinstance(reg._model_kwargs, dict)
    assert len(reg._model_kwargs) == 0


def test_regressor_init_custom_parameters():
    """Test BaseRegressor initialization with custom parameters."""
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_jobs=2,
        fit_global_model=False,
        strict=True,
        keep_models=True,
        temp_folder="/tmp",
        batch_size=10,
        fit_intercept=False,  # A LinearRegression parameter
    )

    assert reg.model == LinearRegression
    assert reg.bandwidth == 50
    assert reg.fixed is True
    assert reg.kernel == "tricube"
    assert reg.n_jobs == 2
    assert reg.fit_global_model is False
    assert reg.strict is True
    assert reg.keep_models is True
    assert reg.temp_folder == "/tmp"
    assert reg.batch_size == 10
    assert "fit_intercept" in reg._model_kwargs
    assert reg._model_kwargs["fit_intercept"] is False


def test_regressor_fit_basic_functionality(sample_regression_data):
    """Test basic fitting functionality of BaseRegressor."""
    X, y, geometry = sample_regression_data

    # Create regressor with default params
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        random_state=42,  # For reproducibility
    )

    # Fit the model
    fitted_reg = reg.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_reg is reg

    # Test that the global model was fitted
    assert hasattr(reg, "global_model")
    assert isinstance(reg.global_model, LinearRegression)


def test_regressor_fit_with_keep_models(sample_regression_data):
    """Test fitting with keep_models=True to retain local models."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        n_jobs=1,
    )

    reg.fit(X, y, geometry)

    # Check that local models were kept
    assert hasattr(reg, "_local_models")
    assert isinstance(reg._local_models, pd.Series)
    assert len(reg._local_models) > 0

    # Check that each local model is a fitted LinearRegression
    for model in reg._local_models:
        assert isinstance(model, LinearRegression | None)
        # Check that the model has been fitted by ensuring it has a coef_ attribute
        assert hasattr(model, "coef_") if isinstance(model, LinearRegression) else True


def test_regressor_fit_with_keep_models_path(sample_regression_data):
    """Test fitting with keep_models as a Path to save models to disk."""
    X, y, geometry = sample_regression_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a regressor with keep_models as a path
        reg = BaseRegressor(
            LinearRegression,
            bandwidth=10,
            fixed=False,
            keep_models=temp_dir,
            random_state=42,
            n_jobs=1,
        )

        reg.fit(X, y, geometry)

        # Check that models were serialized to disk
        model_files = list(Path(temp_dir).glob("*"))
        assert len(model_files) > 0


@pytest.mark.parametrize("kernel", _kernel_functions)
def test_regressor_fit_different_kernels(sample_regression_data, kernel):
    """Test fitting with different kernel functions."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        kernel=kernel,
        random_state=42,
    )

    reg.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(reg, "local_r2_")


def test_regressor_fit_fixed_bandwidth(sample_regression_data):
    """Test fitting with fixed bandwidth."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=100_000,
        fixed=True,  # Fixed bandwidth
        random_state=42,
    )

    reg.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(reg, "local_r2_")


def test_regressor_fit_without_global_model(sample_regression_data):
    """Test fitting without computing a global model."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150_000,
        fixed=True,
        fit_global_model=False,
    )

    reg.fit(X, y, geometry)

    # Check that global model was not fitted
    assert not hasattr(reg, "global_model")

    # But local results should still be available
    assert hasattr(reg, "pred_")


def test_regressor_fit_without_performance_metrics(sample_regression_data):
    """Test fitting without computing performance metrics."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        strict=False,  # To avoid warnings on invariance
    )

    reg.fit(X, y, geometry)

    # Check that performance metrics were not computed
    # assert not hasattr(reg, "score_")
    assert not hasattr(reg, "mae_")

    # But focal predictions should still be available
    assert hasattr(reg, "pred_")


def test_regressor_fit_with_batch_processing(sample_regression_data):
    """Test fitting with batch processing enabled."""
    X, y, geometry = sample_regression_data

    # Create a regressor with a small batch size
    batch_size = 5
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        batch_size=batch_size,  # Process in small batches
        verbose=True,
    )

    # Capture print output to verify batch processing messages
    f = io.StringIO()
    with redirect_stdout(f):
        reg.fit(X, y, geometry)

    # Get the captured output
    output = f.getvalue()

    # Test that batch processing messages were printed
    expected_batches = ((len(X) + batch_size - 1) // batch_size) + 1  # Ceiling division
    assert f"Processing batch 1 out of {expected_batches}" in output

    # Check that the model was fit successfully
    assert hasattr(reg, "pred_")
    assert hasattr(reg, "local_r2_")


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_regressor_predict_basic(sample_regression_data, bandwidth):
    """Test basic functionality of predict method."""
    X, y, geometry = sample_regression_data

    # Create and fit regressor with keep_models=True (required for prediction)
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
    )
    reg.fit(X, y, geometry)

    # Predict values for first 5 samples
    pred = reg.predict(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

    # Check output format
    assert isinstance(pred, pd.Series)
    assert len(pred) == 5

    # Check all predicted values are numeric
    assert pd.api.types.is_numeric_dtype(pred)


@pytest.mark.parametrize("bandwidth", ["nearest", 8, None])
def test_regressor_predict_adaptive(sample_regression_data, bandwidth):
    """Test basic functionality of predict method using adaptive kernel."""
    X, y, geometry = sample_regression_data

    # Create and fit regressor with keep_models=True (required for prediction)
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=7,
        fixed=False,
        keep_models=True,
    )
    reg.fit(X, y, geometry)

    # Predict values for first 5 samples
    pred = reg.predict(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

    # Check output format
    assert isinstance(pred, pd.Series)
    assert len(pred) == 5

    # Check all predicted values are numeric
    assert pd.api.types.is_numeric_dtype(pred)


def test_regressor_predict_global_weight(sample_regression_data):
    """Test predict with global_model_weight for regressor."""
    X, y, geometry = sample_regression_data
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        fit_global_model=True,
    )
    reg.fit(X, y, geometry)
    pred_local = reg.predict(
        X.iloc[:5], geometry.iloc[:5], bandwidth="nearest", global_model_weight=0
    )
    pred_global = reg.global_model.predict(X.iloc[:5])
    pred_fused = reg.predict(
        X.iloc[:5], geometry.iloc[:5], bandwidth="nearest", global_model_weight=1
    )
    # Fused should be average of local and global
    np.testing.assert_allclose(
        pred_fused.values, (pred_local.values + pred_global) / 2, rtol=1e-6
    )


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_regressor_predict_with_models_on_disk(sample_regression_data, bandwidth):
    """Test prediction with models stored on disk."""
    X, y, geometry = sample_regression_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and fit regressor with keep_models as a path
        reg = BaseRegressor(
            LinearRegression,
            bandwidth=150000,
            fixed=True,
            keep_models=temp_dir,
        )
        reg.fit(X, y, geometry)

        # Predict values
        pred = reg.predict(X.iloc[:5], geometry.iloc[:5], bandwidth=bandwidth)

        # Check output
        assert isinstance(pred, pd.Series)
        assert len(pred) == 5


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_regressor_predict_comparison_with_focal_pred(
    sample_regression_data, bandwidth
):
    """Test that prediction for training data is close to focal predictions."""
    X, y, geometry = sample_regression_data

    # Create and fit regressor with include_focal=True for fair comparison
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        include_focal=True,
    )
    reg.fit(X, y, geometry)

    # Get predictions for the same data used for training
    predicted_values = reg.predict(X, geometry, bandwidth=bandwidth)

    if bandwidth != "nearest":
        # Compare with pred_ (should be similar since include_focal=True)
        # The values won't be identical because predict uses weighted average of
        # multiple local models while pred_ uses only the focal model
        correlation = predicted_values.corr(reg.pred_)
        assert correlation > 0.9  # Should be highly correlated
    else:
        pd.testing.assert_series_equal(predicted_values, reg.pred_)


@pytest.mark.parametrize("bandwidth", ["nearest", 100000, None])
def test_regressor_predict_invalid_geometry(sample_regression_data, bandwidth):
    """Test that prediction raises error with non-point geometries."""
    X, y, geometry = sample_regression_data

    # Get the original polygons instead of centroids
    gdf = gpd.read_file(get_path("geoda.guerry"))
    polygon_geometry = gdf.geometry

    # Create and fit regressor with point geometries
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
    )
    reg.fit(X, y, geometry)

    # Attempt to predict with polygon geometries
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        reg.predict(X.iloc[:5], polygon_geometry.iloc[:5], bandwidth=bandwidth)


def test_regressor_predict_values_ensemble(sample_regression_data):
    """Test that predicted values are reasonable and match expected results."""
    X, y, geometry = sample_regression_data

    # Create and fit regressor
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        include_focal=True,
    )
    reg.fit(X, y, geometry)

    # Predict values for first 5 samples
    pred = reg.predict(X.iloc[:5], geometry.iloc[:5], bandwidth=None)

    # Check predictions are within a reasonable range of the target variable
    assert pred.min() > y.min() - y.std() * 2
    assert pred.max() < y.max() + y.std() * 2

    # Check specific expected values (computed from the implementation)
    expected_values = pd.Series(
        [5229.26, 7171.34, 10342.14, 2779.31, 6557.50],
        index=X.iloc[:5].index,
    )
    pd.testing.assert_series_equal(pred, expected_values, check_exact=False, rtol=0.01)


def test_regressor_predict_values_nearest(sample_regression_data):
    """Test that predicted values are reasonable and match expected results."""
    X, y, geometry = sample_regression_data

    # Create and fit regressor
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        include_focal=True,
    )
    reg.fit(X, y, geometry)

    # Predict values for first 5 samples
    pred = reg.predict(X.iloc[:5], geometry.iloc[:5], bandwidth="nearest")

    # Check predictions are within a reasonable range of the target variable
    assert pred.min() > y.min() - y.std() * 2
    assert pred.max() < y.max() + y.std() * 2

    # Check specific expected values (computed from the implementation)
    expected_values = pd.Series(
        [4825.97, 6861.34, 11090.42, 2561.19, 6957.98],
        index=X.iloc[:5].index,
    )
    pd.testing.assert_series_equal(pred, expected_values, check_exact=False, rtol=0.01)


def test_regressor_random_state_consistency(sample_regression_data):
    """Test that same random_state produces consistent results."""
    X, y, geometry = sample_regression_data

    # Create two regressors with same random_state
    reg1 = BaseRegressor(
        RandomForestRegressor,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    reg1.fit(X, y, geometry)

    reg2 = BaseRegressor(
        RandomForestRegressor,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    reg2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_series_equal(reg1.pred_, reg2.pred_)


def test_regressor_n_jobs_consistency(sample_regression_data):
    """Test that parallel processing gives the same results as sequential (n_jobs=1)."""
    X, y, geometry = sample_regression_data

    # Create a regressor with n_jobs=1 (sequential)
    reg_sequential = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
    )
    reg_sequential.fit(X, y, geometry)

    # Create a regressor with n_jobs=-1 (parallel)
    reg_parallel = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=-1,
    )
    reg_parallel.fit(X, y, geometry)

    # Check that the results are the same regardless of parallelization
    pd.testing.assert_series_equal(
        reg_sequential.pred_,
        reg_parallel.pred_,
        check_exact=False,
        rtol=1e-5,
    )
    pd.testing.assert_series_equal(
        reg_sequential.local_r2_,
        reg_parallel.local_r2_,
        check_exact=False,
        rtol=1e-5,
    )

    # Check that global models have the same coefficients
    np.testing.assert_allclose(
        reg_sequential.global_model.coef_,
        reg_parallel.global_model.coef_,
        rtol=1e-5,
    )


def test_regressor_repr_basic():
    """Test basic __repr__ functionality."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)
    repr_str = repr(reg)

    # Check that it contains the class name
    assert "BaseRegressor" in repr_str

    # Check that it contains the model name
    assert "LinearRegression" in repr_str

    # Check that it contains the bandwidth
    assert "bandwidth=100" in repr_str


def test_regressor_repr_html_basic():
    """Test basic _repr_html_ functionality."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)
    html_str = reg._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)

    # Should contain HTML tags
    assert "<" in html_str and ">" in html_str

    # Should contain the class name
    assert "BaseRegressor" in html_str


def test_regressor_fit_focal_inclusion(sample_regression_data):
    """Test fitting functionality with focal inclusion parameter."""
    X, y, geometry = sample_regression_data

    # Create regressor with focal exclusion
    no_focal = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        include_focal=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    no_focal = no_focal.fit(X, y, geometry)

    # Create regressor with focal inclusion
    focal = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        include_focal=True,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    focal = focal.fit(X, y, geometry)

    # RF should 'remember' focal point when included
    assert (y - no_focal.pred_).mean() > (y - focal.pred_).mean()


def test_custom_graph_baseregressor(sample_regression_data):
    """Test BaseRegressor with a custom graph object."""
    X, y, geometry = sample_regression_data

    # Create a fixed distance weights graph
    g = Graph.build_distance_band(geometry, threshold=150000, binary=False)

    # Create regressor with custom graph
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=100,  # This should be ignored when custom graph is provided
        fixed=False,  # This should be ignored when custom graph is provided
        graph=g,
    )

    # Fit the model
    reg.fit(X, y)

    # Check that the model was fit successfully
    assert hasattr(reg, "pred_")
    assert hasattr(reg, "local_r2_")


def test_custom_graph_baseclassifier(sample_data):
    """Test BaseClassifier with a custom graph object."""
    X, y, geometry = sample_data

    # Create a fixed distance weights graph
    g = Graph.build_distance_band(geometry, threshold=150000, binary=False)
    # Create classifier with custom graph
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,  # This should be ignored when custom graph is provided
        fixed=True,  # This should be ignored when custom graph is provided
        graph=g,
        max_iter=500,
    )

    # Fit the model
    clf.fit(X, y)

    # Check that the model was fit successfully
    assert hasattr(clf, "proba_")


def test_leave_out_attributes(sample_data):
    """Test that leave_out enables out of sample log_loss_ calculation."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        leave_out=0.2,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    assert hasattr(clf, "left_out_y_")
    assert hasattr(clf, "left_out_proba_")
    assert hasattr(clf, "left_out_w_")


def test_classifier_score(sample_data):
    """Test the score method of BaseClassifier with geometry argument."""
    X, y, geometry = sample_data
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        random_state=42,
        max_iter=200,
        strict=False,
    )
    clf.fit(
        X,
        y,
        geometry,
    )
    acc = clf.score(X, y, geometry)
    assert 0.0 <= acc <= 1.0
    # Should be perfect on training data with include_focal=True, but not required
    assert isinstance(acc, float)


def test_classifier_local_metric(sample_data):
    """Test local_metric on BaseClassifier."""
    X, y, geometry = sample_data
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        random_state=42,
        max_iter=200,
        strict=False,
        n_jobs=1,
    )
    clf.fit(X, y, geometry)

    local_accuracy = clf.local_metric(accuracy_score)
    expected = np.array(
        [
            np.nan if y_local.shape[0] == 0 else accuracy_score(y_local, pred_local)
            for y_local, pred_local in zip(clf._y_local, clf._pred_local, strict=True)
        ]
    )

    np.testing.assert_allclose(local_accuracy, expected, equal_nan=True)
    assert len(local_accuracy) == len(X)


def test_regressor_score(sample_regression_data):
    """Test the score method of BaseRegressor with geometry argument."""
    X, y, geometry = sample_regression_data
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        random_state=42,
        strict=False,
    )
    reg.fit(X, y, geometry)
    r2 = reg.score(X, y, geometry)
    assert -1.0 <= r2 <= 1.0
    assert isinstance(r2, float)


def test_regressor_local_metric(sample_regression_data):
    """Test local_metric on BaseRegressor."""
    X, y, geometry = sample_regression_data
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        random_state=42,
        n_jobs=1,
    )
    reg.fit(X, y, geometry)

    local_mae = reg.local_metric(mean_absolute_error)
    expected = np.array(
        [
            np.nan
            if y_local.shape[0] == 0
            else mean_absolute_error(y_local, pred_local)
            for y_local, pred_local in zip(reg._y_local, reg._pred_local, strict=True)
        ]
    )

    np.testing.assert_allclose(local_mae, expected, equal_nan=True)
    assert len(local_mae) == len(X)


def test_metadata_routing(sample_regression_data):
    """Test compatibility with sklearn pipes"""
    sklearn.set_config(enable_metadata_routing=True)

    X, y, geometry = sample_regression_data
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        random_state=42,
        strict=False,
    )

    reg.set_fit_request(geometry=True)
    reg.set_predict_request(geometry=True)
    reg.set_score_request(geometry=True)

    gs = GridSearchCV(reg, {"include_focal": [True, False]}, cv=2)
    gs.fit(X, y, geometry=geometry)
    assert gs.best_estimator_.include_focal

    sklearn.set_config(enable_metadata_routing=False)


@pytest.mark.parametrize("kernel", ["invalid", "abc", "wrong_kernel"])
def test_fit_invalid_kernel_raises(sample_data, kernel):
    """Ensure ValueError is raised for invalid kernel during fit."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        kernel=kernel,
    )

    with pytest.raises(ValueError, match="Invalid kernel"):
        clf.fit(X, y, geometry)


def test_kernel_not_callable_or_string(sample_data):
    """Ensure invalid kernel type (not string/callable) raises ValueError."""
    X, y, geometry = sample_data

    # Kernel must be either a valid string or callable
    # intentionally passing invalid tests
    clf = BaseClassifier(LogisticRegression, bandwidth=10, kernel=123)  # type: ignore

    # Expect validation failure
    with pytest.raises(ValueError, match="kernel must"):
        clf.fit(X, y, geometry)
