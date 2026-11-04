from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal import graph
from shapely import Point
from sklearn.decomposition import PCA

from spml.decomposition import GWPCA, BaseGWDecomposition


@pytest.fixture
def toy_data():
    """Small synthetic point pattern with a spatial gradient in correlation."""
    rng = np.random.default_rng(42)
    n = 60
    coords = rng.uniform(0, 10, (n, 2))
    geometry = gpd.GeoSeries([Point(*c) for c in coords])
    g = coords[:, 0] / 10
    f1 = rng.normal(size=n)
    f2 = g * f1 + (1 - g) * rng.normal(size=n)
    f3 = rng.normal(size=n) + 0.5 * f1
    f4 = rng.normal(size=n)
    X = pd.DataFrame({"a": f1, "b": f2, "c": f3, "d": f4})
    return X, geometry


# --------------------------------------------------------------------- init ---
def test_gwpca_init():
    model = GWPCA(n_components=3, bandwidth=40)

    assert model.n_components == 3
    assert model.bandwidth == 40
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model.standardize is True
    assert model.include_focal is True
    assert model._model_type == "gwpca"
    # IC machinery must stay switched off for an unsupervised model
    assert model._supports_ic is False

    model = GWPCA(n_components=2, bandwidth=25, fixed=True, standardize=False)
    assert model.fixed is True
    assert model.standardize is False


# ---------------------------------------------------------------------- fit ---
def test_gwpca_fit_basic(toy_data):
    X, geometry = toy_data
    model = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1)
    fitted = model.fit(X, geometry=geometry)

    assert fitted is model
    n, p, k = len(X), X.shape[1], 2

    assert model.components_.shape == (n, k, p)
    assert model.local_eigenvalues_.shape == (n, p)
    assert model.explained_variance_.shape == (n, k)
    assert model.explained_variance_ratio_.shape == (n, k)
    # local_scores_ is GWmodel-faithful: one row per (location, neighbor),
    # not one row per location - ragged, so only column count/index names
    # are checked here (see test_gwpca_local_scores_match_neighbor_formula).
    assert model.local_scores_.shape[1] == k
    assert list(model.local_scores_.index.names) == ["location", "neighbor"]
    assert model.proportion_total_var_.shape == (n,)
    assert model.local_loadings_.shape == (n * k, p)
    assert list(model.local_loadings_.index.names) == ["location", "component"]
    assert isinstance(model.winning_variable_, pd.Series)
    assert set(model.winning_variable_).issubset(set(X.columns))
    assert isinstance(model.global_model_, PCA)


def test_gwpca_attribute_consistency(toy_data):
    X, geometry = toy_data
    model = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1).fit(
        X, geometry=geometry
    )

    ev = model.local_eigenvalues_.to_numpy()
    # eigenvalues are returned in descending order and non-negative
    assert np.all(np.diff(ev, axis=1) <= 1e-9)
    assert np.all(ev > -1e-9)

    # explained variance ratio rows sum to the reported PTV, PTV in (0, 1]
    np.testing.assert_allclose(
        model.explained_variance_ratio_.sum(axis=1).to_numpy(),
        model.proportion_total_var_.to_numpy(),
        rtol=1e-12,
    )
    assert model.proportion_total_var_.between(0, 1 + 1e-9).all()

    # component vectors are orthonormal
    for i in range(len(X)):
        comp = model.components_[i]
        np.testing.assert_allclose(comp @ comp.T, np.eye(2), atol=1e-8)

    # local_loadings_ is just a tidy view of components_
    for pos, name in enumerate(model._names):
        np.testing.assert_allclose(
            model.local_loadings_.loc[name].to_numpy(), model.components_[pos]
        )

    # winning variable is the largest |loading| on PC1
    pc1 = np.abs(model.components_[:, 0, :])
    expected = X.columns.to_numpy()[pc1.argmax(axis=1)]
    np.testing.assert_array_equal(model.winning_variable_.to_numpy(), expected)


# -------------------------------------------------------------- correctness ---
def _hand_rolled_local_pca(x_local, wv):
    """Weighted mean-center + eigh, returning the UNNORMALIZED scatter's
    eigenvalues/eigenvectors (i.e. not divided by wv.sum()) - matching what
    GWmodel's `wpca` computes before gwpca()'s (deliberately replicated,
    shared-divisor) eigenvalue rescaling. See the decomposition.py module
    docstring's "Deliberate parity with GWmodel::gwpca" note.
    """
    mean = (wv[:, None] * x_local).sum(axis=0) / wv.sum()
    xc = x_local - mean
    scatter = (xc * wv[:, None]).T @ xc
    evals, evecs = np.linalg.eigh(scatter)
    return evals[::-1], evecs[:, ::-1]


def test_gwpca_matches_manual_weighted_pca(toy_data):
    """The local decomposition equals a hand-rolled weighted covariance + eigh.

    Eigenvalues are checked against the GWmodel-faithful shared divisor (the
    LAST location's own weight-sum, not the focal location's) - see the
    module docstring's "Deliberate parity with GWmodel::gwpca" note.
    """
    X, geometry = toy_data
    g = graph.Graph.build_kernel(
        geometry, kernel="bisquare", bandwidth=6.0, decay=True
    ).assign_self_weight(1)

    model = GWPCA(
        n_components=X.shape[1],
        standardize=False,
        graph=g,
        fit_global_model=False,
        n_jobs=1,
    ).fit(X, geometry=geometry)

    focal = model._names[0]
    w = g._adjacency.loc[focal]
    xn = X.loc[w.index].to_numpy()
    evals_raw, evecs = _hand_rolled_local_pca(xn, w.to_numpy())

    last = model._names[-1]
    w_last = g._adjacency.loc[last].to_numpy()
    wsum_last = w_last.sum()
    evals = evals_raw / wsum_last

    np.testing.assert_allclose(
        model.local_eigenvalues_.loc[focal].to_numpy(), evals, rtol=1e-9
    )
    for kk in range(X.shape[1]):
        got, exp = model.components_[0, kk], evecs[:, kk]
        assert np.allclose(got, exp, atol=1e-8) or np.allclose(got, -exp, atol=1e-8)


def test_gwpca_standardize_true_matches_global_then_local(toy_data):
    """standardize=True == global z-score once, then local weighted PCA.

    Matches the R workflow of manually calling `scale()` before `gwpca()`: a
    single global standardization followed by local mean-centering, *not* a
    true local correlation matrix (see the `standardize` docstring).
    Eigenvalues use the GWmodel-faithful shared (last-location) divisor.
    """
    X, geometry = toy_data
    g = graph.Graph.build_kernel(
        geometry, kernel="bisquare", bandwidth=6.0, decay=True
    ).assign_self_weight(1)

    model = GWPCA(
        n_components=X.shape[1],
        standardize=True,
        graph=g,
        fit_global_model=False,
        n_jobs=1,
    ).fit(X, geometry=geometry)

    Xs = BaseGWDecomposition._standardize(X.to_numpy())
    Xs = pd.DataFrame(Xs, columns=X.columns, index=X.index)

    focal = model._names[0]
    w = g._adjacency.loc[focal]
    xn = Xs.loc[w.index].to_numpy()
    evals_raw, evecs = _hand_rolled_local_pca(xn, w.to_numpy())

    last = model._names[-1]
    wsum_last = g._adjacency.loc[last].to_numpy().sum()
    evals = evals_raw / wsum_last

    np.testing.assert_allclose(
        model.local_eigenvalues_.loc[focal].to_numpy(), evals, rtol=1e-9
    )
    for kk in range(X.shape[1]):
        got, exp = model.components_[0, kk], evecs[:, kk]
        assert np.allclose(got, exp, atol=1e-8) or np.allclose(got, -exp, atol=1e-8)


def test_gwpca_local_scores_match_neighbor_formula(toy_data):
    """local_scores_ matches GWmodel's gwpca.scores: raw (uncentered)
    neighbourhood values dotted with the retained loadings, one row per
    neighbour of each location (not one mean-centered row per focal point).
    """
    X, geometry = toy_data
    g = graph.Graph.build_kernel(
        geometry, kernel="bisquare", bandwidth=6.0, decay=True
    ).assign_self_weight(1)

    model = GWPCA(
        n_components=2,
        standardize=False,
        graph=g,
        fit_global_model=False,
        n_jobs=1,
    ).fit(X, geometry=geometry)

    focal = model._names[0]
    neighbors = g._adjacency.loc[focal].index
    expected = X.loc[neighbors].to_numpy() @ model.components_[0].T

    got = model.local_scores_.loc[focal].loc[neighbors].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_gwpca_reduces_to_global_pca(toy_data):
    """boxcar kernel + all-encompassing bandwidth == global PCA."""
    X, geometry = toy_data
    gw = GWPCA(
        n_components=X.shape[1],
        bandwidth=1e6,
        fixed=True,
        kernel="boxcar",
        standardize=True,
        include_focal=True,
        fit_global_model=False,
        n_jobs=1,
    ).fit(X, geometry=geometry)

    glob = PCA(n_components=X.shape[1]).fit(
        BaseGWDecomposition._standardize(X.to_numpy())
    )

    # every location sees the same neighbourhood, so all local fits are identical
    np.testing.assert_allclose(
        np.abs(gw.components_[0]), np.abs(glob.components_), atol=1e-8
    )
    np.testing.assert_allclose(
        gw.explained_variance_ratio_.iloc[0].to_numpy(),
        glob.explained_variance_ratio_,
        atol=1e-8,
    )
    # identical across locations
    assert np.allclose(gw.components_ - gw.components_[0], 0, atol=1e-8)


# --------------------------------------------------------------- parameters ---
def test_gwpca_standardize_flag_changes_result(toy_data):
    X, geometry = toy_data
    common: dict[str, Any] = {
        "n_components": 2,
        "bandwidth": 4.0,
        "fixed": True,
        "n_jobs": 1,
    }
    corr = GWPCA(standardize=True, **common).fit(X, geometry=geometry)
    cov = GWPCA(standardize=False, **common).fit(X, geometry=geometry)
    assert not np.allclose(corr.components_, cov.components_)


def test_gwpca_graph_matches_geometry(toy_data):
    X, geometry = toy_data
    g = graph.Graph.build_kernel(
        geometry, kernel="bisquare", bandwidth=5.0, decay=True
    ).assign_self_weight(1)

    from_geom = GWPCA(
        n_components=2, bandwidth=5.0, fixed=True, include_focal=True, n_jobs=1
    ).fit(X, geometry=geometry)
    from_graph = GWPCA(n_components=2, graph=g, n_jobs=1).fit(X, geometry=geometry)

    np.testing.assert_allclose(
        from_geom.local_eigenvalues_.to_numpy(),
        from_graph.local_eigenvalues_.to_numpy(),
        rtol=1e-9,
    )


def test_gwpca_adaptive_bandwidth(toy_data):
    X, geometry = toy_data
    model = GWPCA(n_components=2, bandwidth=15, fixed=False, n_jobs=1).fit(
        X, geometry=geometry
    )
    assert model.components_.shape == (len(X), 2, X.shape[1])
    assert np.isfinite(model.local_eigenvalues_.to_numpy()).all()


def test_gwpca_loadings_and_ratio_are_index_order_invariant(toy_data):
    """Loadings and explained_variance_ratio_/proportion_total_var_ do not
    depend on X's row order at fit time - only local_eigenvalues_ /
    explained_variance_ (raw magnitude) do, by design; see
    test_gwpca_eigenvalue_scale_depends_on_row_order.
    """
    X, geometry = toy_data
    base = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1).fit(
        X, geometry=geometry
    )

    order = np.random.default_rng(0).permutation(len(X))
    shuffled = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1).fit(
        X.iloc[order].reset_index(drop=True),
        geometry=geometry.iloc[order].reset_index(drop=True),
    )

    got_ratio = shuffled.explained_variance_ratio_.copy()
    got_ratio.index = np.asarray(order)[got_ratio.index.to_numpy()]
    np.testing.assert_allclose(
        base.explained_variance_ratio_.sort_index().to_numpy(),
        got_ratio.sort_index().to_numpy(),
        rtol=1e-9,
    )

    got_comp = shuffled.components_[np.argsort(order)]
    for a, b in zip(base.components_, got_comp, strict=True):
        for kk in range(a.shape[0]):
            assert np.allclose(a[kk], b[kk], atol=1e-8) or np.allclose(
                a[kk], -b[kk], atol=1e-8
            )


def test_gwpca_eigenvalue_scale_depends_on_row_order(toy_data):
    """Documents a deliberate GWmodel::gwpca parity choice (gh#79):
    local_eigenvalues_ / explained_variance_ are normalized by the LAST
    location processed (in X's row order at fit time), not each location's
    own neighbourhood weight-sum - exactly replicating an upstream GWmodel
    bug (see the decomposition.py module docstring). Reordering X therefore
    rescales every reported eigenvalue by one shared constant factor.
    """
    X, geometry = toy_data
    base = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1).fit(
        X, geometry=geometry
    )

    order = np.random.default_rng(0).permutation(len(X))
    shuffled = GWPCA(n_components=2, bandwidth=4.0, fixed=True, n_jobs=1).fit(
        X.iloc[order].reset_index(drop=True),
        geometry=geometry.iloc[order].reset_index(drop=True),
    )

    got_eig = shuffled.local_eigenvalues_.copy()
    got_eig.index = np.asarray(order)[got_eig.index.to_numpy()]

    scale = (
        base.local_eigenvalues_.sort_index().to_numpy()
        / got_eig.sort_index().to_numpy()
    )
    np.testing.assert_allclose(scale, scale.flat[0], rtol=1e-9)
    assert not np.isclose(scale.flat[0], 1.0)


def test_gwpca_global_model_optional(toy_data):
    X, geometry = toy_data
    model = GWPCA(
        n_components=2, bandwidth=4.0, fixed=True, fit_global_model=False, n_jobs=1
    ).fit(X, geometry=geometry)
    assert model.global_model_ is None


# -------------------------------------------------------------- keep_models ---
def test_gwpca_keep_models_payload(toy_data):
    X, geometry = toy_data
    model = GWPCA(
        n_components=2, bandwidth=4.0, fixed=True, keep_models=True, n_jobs=1
    ).fit(X, geometry=geometry)

    assert len(model._local_models) == len(X)
    payload = model._local_models.iloc[0]
    assert sorted(payload) == ["components", "mean", "sd"]
    np.testing.assert_allclose(payload["components"], model.components_[0])


# ------------------------------------------------------------------- errors ---
def test_gwpca_transform_not_implemented(toy_data):
    X, geometry = toy_data
    model = GWPCA(
        n_components=2, bandwidth=4.0, fixed=True, keep_models=True, n_jobs=1
    ).fit(X, geometry=geometry)
    with pytest.raises(NotImplementedError, match="gh#79"):
        model.transform(X.iloc[:3], geometry=geometry.iloc[:3])


def test_gwpca_n_components_out_of_range(toy_data):
    X, geometry = toy_data
    with pytest.raises(ValueError, match="n_components"):
        GWPCA(n_components=X.shape[1] + 1, bandwidth=4.0, fixed=True).fit(
            X, geometry=geometry
        )


def test_gwpca_requires_geometry_or_graph(toy_data):
    X, _ = toy_data
    with pytest.raises(ValueError, match="geometry or graph"):
        GWPCA(n_components=2, bandwidth=4.0, fixed=True).fit(X)


# ------------------------------------------------------------ R parity (gh#79) ---
_R_DATA_DIR = Path(__file__).parent / "data"
_R_BANDWIDTH = 3.5
_R_VARS = ["a", "b", "c", "d"]


def _load_r_reference(suffix):
    eig = pd.read_csv(_R_DATA_DIR / f"gwpca_reference_eigenvalues_{suffix}.csv")
    loadings = pd.read_csv(_R_DATA_DIR / f"gwpca_reference_loadings_{suffix}.csv")
    scores = pd.read_csv(_R_DATA_DIR / f"gwpca_reference_scores_{suffix}.csv")
    return eig, loadings, scores


@pytest.fixture
def r_reference_input():
    df = pd.read_csv(_R_DATA_DIR / "gwpca_reference_input.csv")
    X = df[_R_VARS]
    geometry = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(df["x"], df["y"], strict=True)]
    )
    return X, geometry


@pytest.mark.parametrize(("standardize", "suffix"), [(False, "cov"), (True, "std")])
def test_gwpca_matches_r_gwmodel_reference(r_reference_input, standardize, suffix):
    """spml's GWPCA matches GWmodel::gwpca (R) on eigenvalues, loadings, and
    per-neighbour scores, for both the raw-covariance and manually-pre-scaled
    ("standardize=True") cases.

    Reference values were generated once by
    spml/tests/data/generate_gwpca_reference.R (GWmodel 2.3.1 / R 4.3.3) and
    are checked in as CSVs - this test needs no R installation. Eigenvalues
    and scores are only comparable because spml deliberately replicates
    GWmodel's shared-last-location eigenvalue divisor and its per-neighbour,
    uncentered score definition; see the "Deliberate parity with
    GWmodel::gwpca" note in spml/decomposition.py.
    """
    X, geometry = r_reference_input
    model = GWPCA(
        n_components=len(_R_VARS),
        bandwidth=_R_BANDWIDTH,
        fixed=True,
        kernel="boxcar",
        standardize=standardize,
        include_focal=True,
        fit_global_model=False,
        n_jobs=1,
    ).fit(X, geometry=geometry)

    r_eig, r_loadings, r_scores = _load_r_reference(suffix)

    # --- eigenvalues: no sign ambiguity, compare directly ---
    got_eig = model.local_eigenvalues_.sort_index().to_numpy()
    exp_eig = r_eig.sort_values("location")[[f"PC{j + 1}" for j in range(4)]].to_numpy()
    np.testing.assert_allclose(got_eig, exp_eig, rtol=1e-6)

    # --- loadings + scores: align R's arbitrary eigenvector sign to spml's
    # own canonical convention (largest |loading| positive) before comparing,
    # applying the same flip to that component's scores.
    r_loadings = r_loadings.set_index(["location", "component"])
    r_scores = r_scores.set_index(["location", "neighbor"])

    for loc in range(len(X)):
        for comp in range(4):
            r_load = r_loadings.loc[(loc, comp + 1), _R_VARS].to_numpy(dtype=float)
            j = np.argmax(np.abs(r_load))
            sign = 1.0 if r_load[j] > 0 else -1.0
            r_load = r_load * sign

            got_load = model.components_[loc, comp]
            np.testing.assert_allclose(got_load, r_load, atol=1e-6)

            col = f"PC{comp + 1}"
            r_score_loc = r_scores.loc[loc, col] * sign
            got_score_loc = model.local_scores_.loc[loc, col]
            np.testing.assert_allclose(
                got_score_loc.sort_index().to_numpy(),
                r_score_loc.sort_index().to_numpy(),
                atol=1e-6,
            )
