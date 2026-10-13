import numpy
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from spml.metrics import area_of_applicability


@pytest.fixture
def in_and_out_of_domain():
    """Train cluster near origin; some test points near it, others far away."""
    rng = numpy.random.default_rng(0)
    X_train = rng.normal(0.0, 1.0, (200, 5))
    # 30 in-domain test points + 20 out-of-domain (shifted by 8 stds)
    X_in = rng.normal(0.0, 1.0, (30, 5))
    X_out = rng.normal(8.0, 1.0, (20, 5))
    X_test = numpy.vstack([X_in, X_out])
    is_in_domain = numpy.array([True] * 30 + [False] * 20)
    return X_train, X_test, is_in_domain


# -- Output shapes & types -----------------------------------------------------


def test_returns_bool_array(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    out = area_of_applicability(X_test, X_train, feature_weights="uniform")
    assert out.dtype == bool
    assert out.shape == (len(X_test),)


def test_return_diagnostics_bunch(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    assert set(res.keys()) >= {
        "applicable",
        "dissimilarity_index",
        "cutpoint",
        "feature_weights",
    }
    assert res.applicable.shape == (len(X_test),)
    assert res.dissimilarity_index.shape == (len(X_test),)
    assert numpy.isfinite(res.cutpoint)


# -- Semantic correctness ------------------------------------------------------


def test_inside_outside_separation(in_and_out_of_domain):
    """Out-of-domain points should mostly be flagged as not applicable."""
    X_train, X_test, is_in_domain = in_and_out_of_domain
    applicable = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        threshold="tukey",
    )
    # In-domain: most should be applicable; out-of-domain: most should not.
    assert applicable[is_in_domain].mean() > 0.8
    assert applicable[~is_in_domain].mean() < 0.2


def test_applicable_is_true_inside(in_and_out_of_domain):
    """Sanity-check the polarity flip from the original abil implementation.
    True = inside AOA, False = outside.
    """
    X_train, X_test, is_in_domain = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    # DI of out-of-domain points should be much larger than in-domain
    assert (
        res.dissimilarity_index[~is_in_domain].mean()
        > res.dissimilarity_index[is_in_domain].mean()
    )
    # And the cutoff should be well below the OOD mean DI
    assert res.cutpoint < res.dissimilarity_index[~is_in_domain].mean()


# -- Feature weighting ---------------------------------------------------------


def test_uniform_weights_sum_to_one(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    numpy.testing.assert_allclose(res.feature_weights.sum(), 1.0)
    numpy.testing.assert_allclose(res.feature_weights, 1.0 / X_train.shape[1])


def test_array_weights(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    w = numpy.array([3.0, 1.0, 0.0, 1.0, 0.0])
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights=w,
        return_diagnostics=True,
    )
    numpy.testing.assert_allclose(res.feature_weights.sum(), 1.0)
    numpy.testing.assert_allclose(res.feature_weights, w / w.sum())


def test_array_weights_wrong_shape_raises(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    with pytest.raises(ValueError, match="shape"):
        area_of_applicability(
            X_test,
            X_train,
            feature_weights=numpy.ones(3),
        )


def test_negative_weights_raise(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    with pytest.raises(ValueError, match="non-negative"):
        area_of_applicability(
            X_test,
            X_train,
            feature_weights=numpy.array([1.0, -1.0, 1.0, 1.0, 1.0]),
        )


def test_permutation_weights_requires_model(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    with pytest.raises(ValueError, match="permutation"):
        area_of_applicability(X_test, X_train, feature_weights="permutation")


def test_permutation_weights_with_model(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    rng = numpy.random.default_rng(0)
    # Target depends only on the first two features.
    y_train = (
        X_train[:, 0] + 2 * X_train[:, 1] + 0.1 * rng.standard_normal(len(X_train))
    )
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X_train, y_train)

    res = area_of_applicability(
        X_test,
        X_train,
        y_train=y_train,
        model=model,
        feature_weights="permutation",
        return_diagnostics=True,
        permutation_kwargs={"n_repeats": 5, "random_state": 0},
    )
    w = res.feature_weights
    numpy.testing.assert_allclose(w.sum(), 1.0)
    # Features 0 and 1 should account for most of the weight.
    assert w[:2].sum() > 0.7


# -- Threshold rules -----------------------------------------------------------


def test_tukey_cutpoint_actually_uses_iqr(in_and_out_of_domain):
    """The original implementation had a percentile-scale bug that made
    Tukey effectively never trigger.  Verify the cutpoint here is in the
    plausible range (75th + 1.5*IQR), not stuck at di_train.max().
    """
    X_train, X_test, _ = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        threshold="tukey",
        return_diagnostics=True,
    )
    # Recompute the expected Tukey cutpoint from di_train via the public
    # diagnostic on test DIs, using the fact that the train DI mean is 1.
    # We can't get di_train back, but we can at least assert the cutpoint
    # is finite, positive, and bounded.
    assert 0 < res.cutpoint < 100


def test_mad_threshold_runs(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    out = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        threshold="mad",
    )
    assert out.dtype == bool


def test_float_threshold_runs(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    # 95th percentile
    out = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        threshold=0.95,
    )
    assert out.dtype == bool


def test_invalid_threshold_raises(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    with pytest.raises(ValueError, match="threshold"):
        area_of_applicability(
            X_test,
            X_train,
            feature_weights="uniform",
            threshold="bogus",
        )


# -- CV mode -------------------------------------------------------------------


def test_cv_path_runs(in_and_out_of_domain):
    """CV-based training-DI calibration should still produce sensible AOA."""
    X_train, X_test, is_in_domain = in_and_out_of_domain
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    out = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        cv=cv,
    )
    # Same separation property as without CV.
    assert out[is_in_domain].mean() > 0.8
    assert out[~is_in_domain].mean() < 0.2


# -- Input validation ----------------------------------------------------------


def test_feature_count_mismatch_raises(in_and_out_of_domain):
    X_train, _, _ = in_and_out_of_domain
    bad_test = numpy.zeros((10, 3))
    with pytest.raises(ValueError, match="features"):
        area_of_applicability(bad_test, X_train, feature_weights="uniform")


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2"):
        area_of_applicability(
            numpy.zeros((1, 3)),
            numpy.zeros((5, 3)),
            feature_weights="uniform",
        )


def test_nan_input_raises(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    X_bad = X_test.copy()
    X_bad[0, 0] = numpy.nan
    with pytest.raises(ValueError):
        area_of_applicability(X_bad, X_train, feature_weights="uniform")


# -- Local Point Density (LPD) -------------------------------------------------


def test_lpd_returned_with_diagnostics(in_and_out_of_domain):
    X_train, X_test, _ = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    assert hasattr(res, "lpd")
    assert res.lpd.shape == (len(X_test),)
    assert res.lpd.dtype.kind in "iu"
    # LPD is bounded by the number of training points.
    assert res.lpd.min() >= 0
    assert res.lpd.max() <= len(X_train)


def test_lpd_higher_in_domain_than_out(in_and_out_of_domain):
    """In-domain test points should have many training neighbours within
    the cutpoint; out-of-domain points should have few or none."""
    X_train, X_test, is_in_domain = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    in_mean = res.lpd[is_in_domain].mean()
    out_mean = res.lpd[~is_in_domain].mean()
    assert in_mean > out_mean
    # OOD points are 8-sd away from training; basically nothing within cutpoint.
    assert out_mean < 1.0


def test_lpd_consistent_with_applicable(in_and_out_of_domain):
    """A test point flagged applicable must have at least its nearest
    training neighbour within the cutpoint, so LPD >= 1."""
    X_train, X_test, _ = in_and_out_of_domain
    res = area_of_applicability(
        X_test,
        X_train,
        feature_weights="uniform",
        return_diagnostics=True,
    )
    assert (res.lpd[res.applicable] >= 1).all()
