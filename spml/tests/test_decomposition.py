"""Tests for geographically weighted PCA."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.graph import Graph
from sklearn import clone
from sklearn.decomposition import PCA

from spml.decomposition import GWPCA
from spml.search import BandwidthSearch

SMALL_BW = 30  # adaptive k for Guerry (85 observations)
N_COMP = 3
N_OBS = 85  # geoda.guerry has 85 rows
REFERENCE_FIXTURE = Path(__file__).parent / "data" / "gwpca_reference_fixture.json"

REFERENCE_DATA_VALUES = np.array(
    [
        0.3745401188473625,
        0.5986584841970366,
        0.05808361216819946,
        0.7080725777960455,
        0.8324426408004217,
        0.18340450985343382,
        0.43194501864211576,
        0.13949386065204183,
        0.45606998421703593,
        0.5142344384136116,
        0.9507143064099162,
        0.15601864044243652,
        0.8661761457749352,
        0.020584494295802447,
        0.21233911067827616,
        0.3042422429595377,
        0.2912291401980419,
        0.29214464853521815,
        0.7851759613930136,
        0.5924145688620425,
        0.7319939418114051,
        0.15599452033620265,
        0.6011150117432088,
        0.9699098521619943,
        0.18182496720710062,
        0.5247564316322378,
        0.6118528947223795,
        0.3663618432936917,
        0.19967378215835974,
        0.046450412719997725,
    ]
)


def _complete_graph(index: pd.Index, weights: np.ndarray | None = None) -> Graph:
    """Build a complete precomputed graph for deterministic local PCA tests."""
    if weights is None:
        weights = np.ones(len(index), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(index):
        raise ValueError("weights must have the same length as index.")

    index_values = index.to_numpy()
    adjacency = pd.Series(
        np.tile(weights, len(index_values)),
        index=pd.MultiIndex.from_arrays(
            [
                np.repeat(index_values, len(index_values)),
                np.tile(index_values, len(index_values)),
            ],
            names=["focal", "neighbor"],
        ),
        name="weight",
    )
    return Graph(adjacency, is_sorted=True)


def _reference_fixture_data() -> tuple[pd.DataFrame, gpd.GeoSeries]:
    """Return the standardized dataset used by the reference fixture."""
    X = pd.DataFrame(
        REFERENCE_DATA_VALUES.reshape((10, 3), order="F"),
        columns=pd.Index(["A", "B", "C"]),
    )
    X = (X - X.mean()) / X.std()
    geometry = gpd.GeoSeries.from_xy(np.arange(len(X)), np.arange(len(X)))
    return X, geometry


def _read_adaptive_reference_fixture() -> dict:
    """Read the adaptive external reference fixture."""
    with (
        Path(__file__).parent / "data" / "gwpca_adaptive_reference_fixture.json"
    ).open(encoding="utf-8") as file:
        return json.load(file)


def _read_reference_fixture() -> dict:
    """Read the external reference output fixture."""
    with REFERENCE_FIXTURE.open(encoding="utf-8") as file:
        return json.load(file)


class TestGWPCANumericalCorrectness:
    def test_matches_reference_loadings_and_variance_ratios(self):
        """Compare directly comparable outputs against the reference fixture."""
        X, geometry = _reference_fixture_data()
        baseline = _read_reference_fixture()

        model = GWPCA(
            n_components=2,
            bandwidth=5,
            fixed=True,
            kernel="bisquare",
            include_focal=True,
            fit_global_model=False,
            n_jobs=1,
        ).fit(X, geometry=geometry)

        expected_loadings = np.asarray(baseline["loadings"], dtype=float)
        expected_pv = np.asarray(
            [
                [np.nan if value == "NaN" else float(value) for value in row]
                for row in baseline["local_pv"]
            ]
        )
        valid = ~np.isnan(expected_pv).any(axis=1)

        np.testing.assert_allclose(
            np.abs(model._components[valid]),
            np.abs(expected_loadings[valid]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.explained_variance_ratio_.to_numpy()[valid] * 100,
            expected_pv[valid],
            atol=1e-5,
        )

    def test_matches_reference_loadings_and_variance_ratios_adaptive(self):
        """Compare adaptive-bandwidth outputs against the reference fixture."""
        X, geometry = _reference_fixture_data()
        baseline = _read_adaptive_reference_fixture()

        model = GWPCA(
            n_components=2,
            bandwidth=8,
            fixed=False,
            kernel="bisquare",
            include_focal=True,
            fit_global_model=False,
            n_jobs=1,
        ).fit(X, geometry=geometry)

        expected_loadings = np.asarray(baseline["loadings"], dtype=float)
        expected_pv = np.asarray(
            [
                [np.nan if value == "NaN" else float(value) for value in row]
                for row in baseline["local_pv"]
            ]
        )
        valid = ~np.isnan(expected_pv).any(axis=1)

        np.testing.assert_allclose(
            np.abs(model._components[valid]),
            np.abs(expected_loadings[valid]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.explained_variance_ratio_.to_numpy()[valid] * 100,
            expected_pv[valid],
            atol=1e-5,
        )

    def test_weighted_covariance_matches_expected_local_pca_values(self):
        """Non-uniform local weights produce known PCA values."""
        X = pd.DataFrame(
            {
                "x": [0.0, 2.0, 0.0],
                "y": [0.0, 0.0, 1.0],
            },
            index=pd.Index(["a", "b", "c"], name="id"),
        )
        model = GWPCA(
            n_components=None,
            graph=_complete_graph(X.index, weights=np.array([1.0, 3.0, 2.0])),
            fit_global_model=False,
            n_jobs=1,
        ).fit(X)

        expected_components = np.tile(
            np.array(
                [
                    [0.93788501, 0.34694625],
                    [-0.34694625, 0.93788501],
                ]
            ),
            (len(X), 1, 1),
        )
        expected_eigenvalues = np.tile([1.12330803, 0.09891420], (len(X), 1))
        expected_scores = np.array(
            [
                [-0.82223627, -0.65957459],
                [1.05353376, 0.03431791],
                [-1.16918251, 0.27831043],
            ]
        )

        np.testing.assert_allclose(
            model.local_means_.to_numpy(),
            np.tile([1.0, 1.0 / 3.0], (len(X), 1)),
        )
        np.testing.assert_allclose(
            model._components, expected_components, atol=1e-6
        )
        np.testing.assert_allclose(
            model.explained_variance_.to_numpy(),
            expected_eigenvalues,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.scores_.to_numpy(), expected_scores, atol=1e-6
        )
        np.testing.assert_allclose(
            model.explained_variance_ratio_.to_numpy(),
            np.tile([0.91907020, 0.08092980], (len(X), 1)),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.condition_number_.to_numpy(),
            np.full(len(X), 11.35638827),
            atol=1e-5,
        )
        assert model.winning_variable_.eq("x").all()

    def test_rank_deficient_covariance_has_expected_stable_values(self):
        """Rank-deficient input keeps finite components and clipped eigenvalues."""
        X = pd.DataFrame(
            {
                "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "x_duplicate": [-4.0, -2.0, 0.0, 2.0, 4.0],
            },
            index=pd.Index(["a", "b", "c", "d", "e"], name="id"),
        )
        model = GWPCA(
            n_components=None,
            graph=_complete_graph(X.index),
            fit_global_model=False,
            n_jobs=1,
        ).fit(X)

        sqrt_five = np.sqrt(5.0)
        expected_components = np.tile(
            np.array(
                [
                    [1.0 / sqrt_five, 2.0 / sqrt_five],
                    [2.0 / sqrt_five, -1.0 / sqrt_five],
                ]
            ),
            (len(X), 1, 1),
        )

        assert np.isfinite(model.components_.to_numpy()).all()
        np.testing.assert_allclose(
            model.explained_variance_.to_numpy(),
            np.tile([10.0, 0.0], (len(X), 1)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model._components,
            expected_components,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.scores_.to_numpy()[:, 1],
            0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.condition_number_.to_numpy(),
            np.full(len(X), 10.0 / 1e-10),
        )

    def test_cv_score_matches_expected_leave_one_out_error(self):
        """Leave-one-out reconstruction error is tested against a fixed value."""
        X = pd.DataFrame(
            {
                "x": [-1.0, 1.0, 0.0, 0.0],
                "y": [0.0, 0.0, -2.0, 2.0],
            },
            index=pd.Index(["a", "b", "c", "d"], name="id"),
        )
        model = GWPCA(
            n_components=1,
            graph=_complete_graph(X.index),
            fit_global_model=False,
            n_jobs=1,
        ).fit(X, cv=True)

        np.testing.assert_allclose(model.cv_score_, 32.0 / 9.0, atol=1e-12)


class TestGWPCAFit:
    def test_fit_returns_self(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        result = model.fit(X, geometry=geometry)
        assert result is model

    def test_fit_shapes(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        n_features = X.shape[1]
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert model.components_.shape == (N_OBS, n_features * N_COMP)
        assert model.explained_variance_.shape == (N_OBS, N_COMP)
        assert model.explained_variance_ratio_.shape == (N_OBS, N_COMP)
        assert model.scores_.shape == (N_OBS, N_COMP)
        assert model.local_means_.shape == (N_OBS, n_features)

    def test_fit_adaptive_bandwidth(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=SMALL_BW, fixed=False).fit(
            X, geometry=geometry
        )
        assert model.components_.shape[0] == N_OBS

    def test_fit_fixed_bandwidth(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=200_000, fixed=True).fit(
            X, geometry=geometry
        )
        assert model.components_.shape[0] == N_OBS

    def test_fit_all_components(self, sample_decomposition_data):
        """n_components=None retains all components."""
        X, geometry = sample_decomposition_data
        n_features = X.shape[1]
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert model.components_.shape == (N_OBS, n_features * n_features)
        assert model.explained_variance_.shape == (N_OBS, n_features)

    def test_names_match_index(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert len(model._names) == N_OBS

    def test_feature_names_stored(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        np.testing.assert_array_equal(model.feature_names_in_, X.columns.to_numpy())

    def test_n_features_stored(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert model.n_features_in_ == X.shape[1]


class TestGWPCAInvariants:
    def test_explained_variance_ratio_sums_to_one(self, sample_decomposition_data):
        """For n_components=None, ratios should sum to one at each location."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        row_sums = model.explained_variance_ratio_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_explained_variance_ratio_nonneg(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert np.all(model.explained_variance_ratio_ >= -1e-10)

    def test_components_orthogonality(self, sample_decomposition_data):
        """Local component loadings must be orthonormal."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        for i in range(len(model._names)):
            L = model._components[i]  # (n_features, n_features)
            gram = L.T @ L
            np.testing.assert_allclose(gram, np.eye(L.shape[1]), atol=1e-6)

    def test_eigenvalues_descending(self, sample_decomposition_data):
        """Eigenvalues must be in descending order at every location."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        for i in range(len(model._names)):
            eigs = model.explained_variance_.iloc[i].values
            assert np.all(np.diff(eigs) <= 1e-10), (
                f"Eigenvalues not descending at loc {i}"
            )

    def test_eigenvalues_nonneg(self, sample_decomposition_data):
        """Eigenvalues of a covariance matrix must be non-negative."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert np.all(model.explained_variance_ >= -1e-10)

    def test_global_limit_convergence(self, sample_decomposition_data):
        """Full adaptive bandwidth should match global PCA variance ratios."""
        X, geometry = sample_decomposition_data
        n = len(X)

        # Global PCA (sklearn)
        global_pca = PCA(n_components=None).fit(X)
        global_evr = np.sort(global_pca.explained_variance_ratio_)[::-1]

        # GWPCA at full bandwidth (all observations equally weighted)
        model = GWPCA(n_components=None, bandwidth=n, fixed=False).fit(
            X, geometry=geometry
        )
        # Take one location; median across all for robustness
        local_evr = np.median(model.explained_variance_ratio_, axis=0)
        np.testing.assert_allclose(local_evr, global_evr, atol=0.05)


class TestGWPCADerivedAttributes:
    def test_winning_variable_dtype(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        wv = model.winning_variable_
        assert isinstance(wv, pd.Series)
        assert len(wv) == N_OBS
        assert set(wv.unique()).issubset(set(X.columns))

    def test_condition_number_positive(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        cond = model.condition_number_
        assert isinstance(cond, pd.Series)
        assert (cond > 0).all()

    def test_global_model_fitted(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(
            n_components=N_COMP, bandwidth=SMALL_BW, fit_global_model=True
        ).fit(X, geometry=geometry)
        assert hasattr(model, "global_model")
        assert hasattr(model.global_model, "explained_variance_ratio_")

    def test_no_global_model_when_disabled(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(
            n_components=N_COMP, bandwidth=SMALL_BW, fit_global_model=False
        ).fit(X, geometry=geometry)
        assert not hasattr(model, "global_model")


class TestGWPCATransform:
    def test_transform_shape(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        scores = model.transform(X, geometry=geometry)
        assert scores.shape == (N_OBS, N_COMP)

    def test_fit_transform_matches_scores(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        # fit_transform returns the same in-sample scores exposed by scores_.
        ft = model.fit_transform(X, geometry=geometry)
        np.testing.assert_array_equal(ft, model.scores_)

    def test_transform_requires_geometry(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        with pytest.raises(ValueError):
            model.transform(X, geometry=None)


class TestGWPCACVScore:
    def test_cv_score_is_positive(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(
            X, geometry=geometry, cv=True
        )
        assert model.cv_score_ is not None
        assert model.cv_score_ > 0

    def test_cv_score_none_when_cv_false(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(
            X, geometry=geometry, cv=False
        )
        assert model.cv_score_ is None

    def test_cv_scores_are_finite_across_bandwidths(self, sample_decomposition_data):
        """Different bandwidths should produce finite reconstruction errors."""
        X, geometry = sample_decomposition_data
        model_small = GWPCA(n_components=N_COMP, bandwidth=15).fit(
            X, geometry=geometry, cv=True
        )
        model_large = GWPCA(n_components=N_COMP, bandwidth=60).fit(
            X, geometry=geometry, cv=True
        )
        assert np.isfinite(model_small.cv_score_)
        assert np.isfinite(model_large.cv_score_)


class TestGWPCAEstimatorInterface:
    def test_get_params(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        params = model.get_params()
        assert params["n_components"] == N_COMP
        assert params["bandwidth"] == SMALL_BW

    def test_set_params(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        model.set_params(n_components=2)
        assert model.n_components == 2

    def test_clone(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        cloned = clone(model)
        assert cloned.n_components == model.n_components
        assert cloned.bandwidth == model.bandwidth

    def test_y_none_accepted(self, sample_decomposition_data):
        """fit accepts y=None for transformer-style pipeline usage."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        model.fit(X, y=None, geometry=geometry)
        assert model.components_ is not None

    def test_custom_graph(self, sample_decomposition_data):
        """Users can pass a precomputed libpysal Graph."""
        from libpysal import graph as libgraph

        X, geometry = sample_decomposition_data
        g = libgraph.Graph.build_kernel(geometry, kernel="bisquare", bandwidth=N_OBS)
        model = GWPCA(n_components=N_COMP, graph=g).fit(X, geometry=geometry)
        assert model.components_.shape[0] == N_OBS

    def test_parallel_matches_serial(self, sample_decomposition_data):
        """n_jobs=1 and n_jobs=-1 must give identical results."""
        X, geometry = sample_decomposition_data
        m1 = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW, n_jobs=1).fit(
            X, geometry=geometry
        )
        m2 = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW, n_jobs=-1).fit(
            X, geometry=geometry
        )
        np.testing.assert_allclose(
            np.abs(m1.components_), np.abs(m2.components_), atol=1e-10
        )


class TestBandwidthSearchUnsupervised:
    def test_interval_search_no_y(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=15,
            max_bandwidth=30,
            interval=5,
            n_components=N_COMP,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert search.optimal_bandwidth_ in [15, 20, 25, 30]
        assert "cv_score" in search.metrics_.columns

    def test_golden_section_no_y(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="golden_section",
            criterion="cv_score",
            min_bandwidth=10,
            max_bandwidth=50,
            n_components=N_COMP,
        ).fit(X, y=None, geometry=geometry)

        assert hasattr(search, "optimal_bandwidth_")
        assert 10 <= search.optimal_bandwidth_ <= 50

    def test_scores_series_indexed_by_bandwidth(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        search = BandwidthSearch(
            GWPCA,
            fixed=False,
            search_method="interval",
            criterion="cv_score",
            min_bandwidth=15,
            max_bandwidth=25,
            interval=5,
            n_components=2,
        ).fit(X, y=None, geometry=geometry)

        assert isinstance(search.scores_, pd.Series)
        assert len(search.scores_) >= 3


class TestGWPCAEdgeCases:
    def test_n_components_greater_than_n_features_raises(
        self, sample_decomposition_data
    ):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=10, bandwidth=SMALL_BW)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(X, geometry=geometry)

    def test_zero_weight_neighborhood(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data

        n = len(X)
        focal_ids = X.index
        adjacency = pd.Series(
            [0.0] * n,
            index=pd.MultiIndex.from_arrays(
                [focal_ids, focal_ids], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        g = Graph(adjacency, is_sorted=True)

        model = GWPCA(n_components=2, graph=g)
        model.fit(X, geometry=geometry)

        assert model.components_.isna().all().all()
        assert model.scores_.isna().all().all()

    def test_single_observation_in_neighborhood(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=1, include_focal=False)
        model.fit(X, geometry=geometry)

        assert model.components_.isna().all().all()
        assert model.scores_.isna().all().all()
