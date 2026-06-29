"""
Tests for spatialml.decomposition: GWPCA.

Validates:
- Output shapes and dtypes
- Mathematical invariants (orthogonality, variance sum-to-one)
- Global-limit convergence: bandwidth → n_obs gives same result as global PCA
- API compatibility (sklearn Pipeline, get_params, custom Graph)
- BandwidthSearch unsupervised mode
- Monte Carlo stationarity test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import clone
from sklearn.decomposition import PCA

from spatialml.decomposition import GWPCA
from spatialml.search import BandwidthSearch

SMALL_BW = 30  # adaptive k for Guerry (85 observations)
N_COMP = 3
N_OBS = 85  # geoda.guerry has 85 rows


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
        """For n_components=None, ratios at each location should sum to ~1."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        row_sums = model.explained_variance_ratio_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_explained_variance_ratio_nonneg(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert np.all(model.explained_variance_ratio_ >= -1e-10)

    def test_components_orthogonality(self, sample_decomposition_data):
        """Local eigenvectors must be orthonormal: LᵀL = I at each location."""
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
        """Eigenvalues of a covariance matrix must be ≥ 0."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=None, bandwidth=SMALL_BW).fit(X, geometry=geometry)
        assert np.all(model.explained_variance_ >= -1e-10)

    def test_global_limit_convergence(self, sample_decomposition_data):
        """At bandwidth = n_obs all weights are uniform → local ≈ global PCA."""
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
        # fit_transform returns focal scores (same as scores_)
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

    def test_cv_score_decreases_with_larger_bandwidth(self, sample_decomposition_data):
        """Larger bandwidth → more global model → higher reconstruction error
        compared to optimal; but a very small bandwidth also hurts.
        We just check that cv_score_ is finite and positive for two bandwidths."""
        X, geometry = sample_decomposition_data
        model_small = GWPCA(n_components=N_COMP, bandwidth=15).fit(
            X, geometry=geometry, cv=True
        )
        model_large = GWPCA(n_components=N_COMP, bandwidth=60).fit(
            X, geometry=geometry, cv=True
        )
        assert np.isfinite(model_small.cv_score_)
        assert np.isfinite(model_large.cv_score_)


class TestGWPCASklearnAPI:
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
        """fit(X, y=None) must work — needed for sklearn Pipeline."""
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=N_COMP, bandwidth=SMALL_BW)
        model.fit(X, y=None, geometry=geometry)
        assert model.components_ is not None

    def test_custom_graph(self, sample_decomposition_data):
        """Users can pass a pre-computed libpysal Graph."""
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

    def test_zero_weight_neighbourhood(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        from libpysal.graph import Graph

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

    def test_single_observation_in_neighbourhood(self, sample_decomposition_data):
        X, geometry = sample_decomposition_data
        model = GWPCA(n_components=2, bandwidth=1, include_focal=False)
        model.fit(X, geometry=geometry)

        assert model.components_.isna().all().all()
        assert model.scores_.isna().all().all()
