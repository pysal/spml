import numpy as np
import pandas as pd
import pytest
from gwlearn.base import weighted_covariance


def test_weighted_covariance_single_focal():
    # 3 samples, 2 features
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    # Equal weights
    wt = np.array([1.0, 1.0, 1.0])

    cov = weighted_covariance(X, wt)
    expected_cov = np.cov(X.T, aweights=wt, ddof=0)

    np.testing.assert_allclose(cov, expected_cov)


def test_weighted_covariance_unequal_weights():
    X = np.array([[1.0, 1.0], [2.0, 3.0]])
    wt = np.array([2.0, 1.0])
    cov = weighted_covariance(X, wt)
    expected_cov = np.cov(X.T, aweights=wt, ddof=0)

    np.testing.assert_allclose(cov, expected_cov)


def test_weighted_covariance_against_numpy_cov_random():
    rng = np.random.default_rng(42)
    # 50 samples, 4 features
    X = rng.normal(size=(50, 4))
    # random positive weights
    wt = rng.uniform(0.1, 5.0, size=50)

    cov = weighted_covariance(X, wt)
    expected_cov = np.cov(X.T, aweights=wt, ddof=0)

    np.testing.assert_allclose(cov, expected_cov, rtol=1e-12, atol=1e-12)


def test_weighted_covariance_multi_focal():
    X = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]}, index=["loc1", "loc2", "loc3"]
    )

    # MultiIndex adjacency series
    # focal: loc1, loc2
    index = pd.MultiIndex.from_tuples(
        [("loc1", "loc1"), ("loc1", "loc2"), ("loc2", "loc2"), ("loc2", "loc3")],
        names=["focal", "neighbor"],
    )

    adjacency = pd.Series([1.0, 1.0, 1.0, 1.0], index=index, name="weight")

    covs = weighted_covariance(X, adjacency)
    assert isinstance(covs, dict)
    assert "loc1" in covs
    assert "loc2" in covs

    # For loc1, neighbors are loc1, loc2 (weights 1, 1)
    X_loc1 = X.loc[["loc1", "loc2"]].values
    cov_loc1 = weighted_covariance(X_loc1, np.array([1.0, 1.0]))
    np.testing.assert_allclose(covs["loc1"], cov_loc1)

    # For loc2, neighbors are loc2, loc3 (weights 1, 1)
    X_loc2 = X.loc[["loc2", "loc3"]].values
    cov_loc2 = weighted_covariance(X_loc2, np.array([1.0, 1.0]))
    np.testing.assert_allclose(covs["loc2"], cov_loc2)


def test_weighted_covariance_empty_or_zero_weights():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    wt = np.array([0.0, 0.0])

    cov = weighted_covariance(X, wt)
    assert np.isnan(cov).all()
    assert cov.shape == (2, 2)


def test_weighted_covariance_too_few_samples():
    X = np.array([[1.0, 2.0]])
    wt = np.array([1.0])

    cov = weighted_covariance(X, wt)
    assert np.isnan(cov).all()
    assert cov.shape == (2, 2)
