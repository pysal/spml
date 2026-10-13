import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from sklearn.linear_model import LogisticRegression

from spml.base import BaseClassifier
from spml.ensemble import GWGradientBoostingClassifier, GWRandomForestClassifier
from spml.linear_model import GWLogisticRegression
from spml.undersample import RandomUnderSampler


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=["x", "z"])
    y = pd.Series(np.tile(["a", "b", "c"], 10))
    geometry = gpd.GeoSeries(gpd.points_from_xy(np.arange(30), np.zeros(30)))
    # Each group sees two classes, all classes, or an invariant target.
    neighbors = [[0, 1, 3, 4, 6, 7, 9, 10], list(range(12)), [2]]
    focal, neighbor = [], []
    for i in range(30):
        ids = neighbors[i % 3]
        focal.extend([i] * len(ids))
        neighbor.extend(ids)
    graph = Graph.from_arrays(focal, neighbor, np.ones(len(focal)))
    return X, y, geometry, graph


@pytest.mark.parametrize(
    "estimator",
    [
        BaseClassifier,
        GWLogisticRegression,
        GWRandomForestClassifier,
        GWGradientBoostingClassifier,
    ],
)
@pytest.mark.parametrize("storage", [True, "disk"])
def test_multiclass_predictions(multiclass_data, estimator, storage, tmp_path):
    X, y, geometry, graph = multiclass_data
    kwargs = {
        "graph": graph,
        "keep_models": tmp_path if storage == "disk" else True,
        "n_jobs": 1,
        "fit_global_model": True,
        "random_state": 0,
    }
    if estimator is BaseClassifier:
        kwargs["model"] = LogisticRegression
    if estimator in (GWRandomForestClassifier, GWGradientBoostingClassifier):
        kwargs["n_estimators"] = 10
    clf = estimator(**kwargs)
    clf.fit(X, y, geometry)
    assert list(clf.classes_) == ["a", "b", "c"]
    np.testing.assert_array_equal(
        clf.local_class_presence_.sum(axis=1), clf.local_class_support_
    )
    np.testing.assert_array_equal(clf.local_class_support_, np.tile([2, 3, 1], 10))
    assert (clf.proba_.loc[::3, "c"] == 0).all()
    assert clf.proba_.iloc[2::3].isna().all().all()
    np.testing.assert_allclose(clf.proba_.dropna().sum(axis=1), 1)
    assert clf.pred_.dropna().isin(clf.classes_).all()
    pd.testing.assert_frame_equal(clf.predict_proba(X, geometry), clf.proba_)
    assert clf.predict(X, geometry).dropna().isin(clf.classes_).all()
    for bandwidth in ["nearest", 10]:
        proba = clf.predict_proba(
            X, geometry, bandwidth=bandwidth, global_model_weight=1
        )
        np.testing.assert_allclose(proba.sum(axis=1), 1)
    if estimator is GWLogisticRegression:
        assert clf.local_coef_.shape == (30, 6)
        assert clf.local_intercept_.shape == (30, 3)
        assert clf.local_coef_.loc[0, "c"].isna().all()
        assert not hasattr(clf, "aic_")


def test_partial_coverage_warning(multiclass_data):
    X, y, geometry, graph = multiclass_data
    clf = BaseClassifier(
        LogisticRegression, graph=graph, n_jobs=1, fit_global_model=False, strict=None
    )
    with pytest.warns(UserWarning) as caught:
        clf.fit(X, y, geometry)
    assert len(caught) == 2
    assert "missing some global classes" in str(caught[0].message)
    assert "invariant" in str(caught[1].message)
    clf.set_params(strict=True)
    with pytest.raises(ValueError, match="invariant"):
        clf.fit(X, y, geometry)


@pytest.mark.parametrize("strategy", [True, 0.5])
def test_multiclass_undersampling(strategy):
    y = pd.Series(["a"] * 2 + ["b"] * 6 + ["c"] * 10)
    X = y.to_frame("label")
    X_res, y_res = RandomUnderSampler(strategy, random_state=0).fit_resample(X, y)
    assert y_res.value_counts().to_dict() == {
        "a": 2,
        "b": 2 if strategy is True else 4,
        "c": 2 if strategy is True else 4,
    }
    pd.testing.assert_series_equal(X_res.label, y_res, check_names=False)


def test_multiclass_leave_out(multiclass_data):
    X, y, geometry, graph = multiclass_data
    clf = BaseClassifier(
        LogisticRegression,
        graph=graph,
        n_jobs=1,
        leave_out=0.5,
        undersample=True,
        fit_global_model=False,
    )
    clf.fit(X, y, geometry)
    assert clf.left_out_proba_.shape == (100, 3)
    assert (clf.left_out_proba_[:, 2] == 0).sum() == 40
    np.testing.assert_allclose(clf.left_out_proba_.sum(axis=1), 1)


def test_min_proportion_uses_smallest_class(multiclass_data):
    X, y, geometry, _ = multiclass_data
    y = pd.Series(["a"] * 15 + ["b"] * 14 + ["c"])
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        fixed=True,
        include_focal=True,
        n_jobs=1,
        fit_global_model=False,
        min_proportion=0.1,
    )
    clf.fit(X, y, geometry)
    assert clf.prediction_rate_ == 0
    assert clf.local_class_presence_.all().all()


def test_continuous_target_rejected(multiclass_data):
    X, _, geometry, graph = multiclass_data
    clf = BaseClassifier(LogisticRegression, graph=graph, n_jobs=1)
    with pytest.raises(ValueError, match="Unknown label type"):
        clf.fit(X, X.x, geometry)


@pytest.mark.parametrize("labels", [[False, True], [0, 1], [2, 5], ["no", "yes"]])
def test_binary_labels_and_refit(multiclass_data, labels):
    X, _, geometry, graph = multiclass_data
    y = pd.Series(np.tile(labels, 15))
    clf = GWLogisticRegression(
        graph=graph, keep_models=True, n_jobs=1, fit_global_model=False
    )
    clf.fit(X, y, geometry)
    assert np.isfinite(clf.aic_)
    assert clf.local_coef_.shape == X.shape
    assert clf.pred_.dropna().isin(labels).all()
    assert clf.predict(X, geometry).dropna().isin(labels).all()

    clf.fit(X, pd.Series(np.tile(["a", "b", "c"], 10)), geometry)
    assert not hasattr(clf, "aic_")
    clf.fit(X, y, geometry)
    assert np.isfinite(clf.aic_)


def test_multiclass_bandwidth_search(multiclass_data):
    from spml.search import BandwidthSearch

    X, y, geometry, _ = multiclass_data
    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=40,
        max_bandwidth=50,
        interval=10,
        n_jobs=1,
    )
    search.fit(X, y, geometry)
    assert search.criterion == "log_loss"
    assert np.isfinite(search.scores_).all()
    search.criterion = "aic"
    with pytest.raises(ValueError, match="requires information criteria"):
        search.fit(X, y, geometry)
