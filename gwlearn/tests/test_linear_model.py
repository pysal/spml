import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geodatasets import get_path
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pandas.testing import assert_series_equal
from sklearn.linear_model import LinearRegression, LogisticRegression

from gwlearn.linear_model import GWLinearRegression, GWLogisticRegression

try:
    from mgwr.gwr import GWR

    HAS_MGWR = True
except ImportError:
    HAS_MGWR = False


def test_gwlogistic_init():
    """Test GWLogisticRegression initialization."""
    model = GWLogisticRegression(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "logistic"
    assert model.model == LogisticRegression

    # Check custom parameters
    model = GWLogisticRegression(
        bandwidth=50, fixed=True, kernel="tricube", C=0.5, max_iter=200
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["C"] == 0.5
    assert model._model_kwargs["max_iter"] == 200


def test_gwlogistic_fit_basic(sample_data):  # noqa: F811
    """Test that GWLogisticRegression fit method works and as expected."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
        n_jobs=1,
        include_focal=False,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWLogisticRegression
    assert hasattr(model, "local_coef_")
    assert hasattr(model, "local_intercept_")

    # Check structure of coefficients
    assert isinstance(model.local_coef_, pd.DataFrame)
    assert model.local_coef_.shape[0] == len(X)
    assert model.local_coef_.shape[1] == X.shape[1]

    pd.testing.assert_series_equal(
        model.local_coef_.mean(),
        pd.Series(
            [-0.0004301675501645129, -0.0620546230731815, 0.06715275989171457],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
        atol=0.001,
    )

    # Check structure of intercepts
    assert isinstance(model.local_intercept_, pd.Series)
    assert len(model.local_intercept_) == len(X)
    assert pytest.approx(7.8, abs=0.1) == model.local_intercept_.mean()


def test_gwlogistic_coefficients_structure(sample_data):  # noqa: F811
    """Test the structure and consistency of the coefficients."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    model.fit(X, y, geometry)

    # Check that coefficient names match feature names
    assert all(col in model.local_coef_.columns for col in X.columns)

    # Pick a sample location and check consistency between local_coef_
    # and the stored model
    sample_loc = model._local_models.index[0]
    local_model = model._local_models[sample_loc]

    if local_model is not None:  # Some models might be None due to invariance
        # Compare coefficients from stored model with the ones in local_coef_
        np.testing.assert_allclose(
            local_model.coef_.flatten(),
            model.local_coef_.loc[sample_loc].values,
            rtol=1e-5,
        )

        # Compare intercept
        assert local_model.intercept_[0] == pytest.approx(
            model.local_intercept_[sample_loc]
        )


def test_gwlinear_init():
    """Test GWLinearRegression initialization."""
    model = GWLinearRegression(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "linear"
    assert model.model == LinearRegression

    # Check custom parameters
    model = GWLinearRegression(
        bandwidth=50, fixed=True, kernel="tricube", fit_intercept=False
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["fit_intercept"] is False


def test_gwlinear_fit_basic(sample_regression_data):
    """Test that GWLinearRegression fit method works as expected."""
    X, y, geometry = sample_regression_data

    model = GWLinearRegression(
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        include_focal=False,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWLinearRegression
    assert hasattr(model, "local_coef_")
    assert hasattr(model, "local_intercept_")

    # Check structure of coefficients
    assert isinstance(model.local_coef_, pd.DataFrame)
    assert model.local_coef_.shape[0] == len(X)
    assert model.local_coef_.shape[1] == X.shape[1]

    # Check structure of intercepts
    assert isinstance(model.local_intercept_, pd.Series)
    assert len(model.local_intercept_) == len(X)


def test_index_order_influence(sample_regression_data):
    X, y, geometry = sample_regression_data

    model = GWLinearRegression(
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        include_focal=False,
    )
    model.fit(X, y, geometry)
    pred_expected = model.pred_.sort_index()

    rng = np.random.default_rng()
    order = np.arange(len(y))
    rng.shuffle(order)
    X = X.iloc[order]
    y = y.iloc[order]
    geometry = geometry.iloc[order]

    model = GWLinearRegression(
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        include_focal=False,
    )
    model.fit(X, y, geometry)
    pred_re_ordered = model.pred_.sort_index()

    assert_series_equal(pred_expected, pred_re_ordered)


# def test_gwlinear_coefficients_structure(sample_regression_data):
#     """Test the structure and consistency of the coefficients."""
#     X, y, geometry = sample_regression_data

#     model = GWLinearRegression(
#         bandwidth=150000,
#         fixed=True,
#         keep_models=True,
#     )

#     model.fit(X, y, geometry)

#     # Check that coefficient names match feature names
#     assert all(col in model.local_coef_.columns for col in X.columns)

#     # Pick a sample location and check consistency between local_coef_
#     # and the stored model
#     sample_loc = model.local_models.index[0]
#     local_model = model.local_models[sample_loc]

#     if local_model is not None:  # Some models might be None due to invariance
#         # Compare coefficients from stored model with the ones in local_coef_
#         np.testing.assert_allclose(
#             local_model.coef_.flatten(),
#             model.local_coef_.loc[sample_loc].values,
#             rtol=1e-5,
#         )

#         # Compare intercept
#         assert local_model.intercept_ == pytest.approx(
#             model.local_intercept_[sample_loc]
#         )


@pytest.mark.skipif(not HAS_MGWR, reason="needs mgwr")
def test_against_mgwr():
    gdf = gpd.read_file(get_path("geoda.ncovr"))
    gdf = gdf.set_geometry(gdf.representative_point()).to_crs(5070)
    y = gdf["FH90"]

    gwlr = GWLinearRegression(
        bandwidth=250,
        fixed=False,
        n_jobs=1,
        keep_models=False,
        kernel="bisquare",
    )
    gwlr.fit(
        gdf.iloc[:, 9:15],
        y,
        geometry=gdf.geometry,
    )

    gwr = GWR(
        coords=gdf.geometry.get_coordinates(),
        y=y.values.reshape(-1, 1),
        X=gdf.iloc[:, 9:15].values,
        bw=250,
        n_jobs=1,
        fixed=False,
        kernel="bisquare",
    )
    res = gwr.fit()

    assert_array_almost_equal(gwlr.local_r2_, res.localR2.flatten())
    assert_array_almost_equal(gwlr.pred_, res.predy.flatten())
    assert_array_almost_equal(gwlr.TSS_, res.TSS.flatten())
    assert_array_almost_equal(gwlr.RSS_, res.RSS.flatten())
    # assert_almost_equal(gwlr.focal_r2_, res.R2)
    # assert_almost_equal(gwlr.focal_adj_r2_, res.adj_R2)
    assert_array_almost_equal(gwlr.local_intercept_, res.params[:, 0])
    assert_array_almost_equal(gwlr.local_coef_, res.params[:, 1:])
    assert_almost_equal(gwlr.aic_, res.aic)
    assert_almost_equal(gwlr.bic_, res.bic)
    assert_almost_equal(gwlr.aicc_, res.aicc, decimal=0)
    assert_almost_equal(gwlr.effective_df_, res.ENP)
    assert_almost_equal(gwlr.log_likelihood_, res.llf)


def test_gwlogistic_predict(sample_data):
    # Unpack sample dataset (features, target, spatial geometry)
    X, y, geometry = sample_data

    # Initialize model with keep_models=True (required for prediction)
    model = GWLogisticRegression(
        bandwidth=10,
        fixed=False,
        keep_models=True,
        max_iter=1000,
    )

    # Fit geographically weighted logistic model
    model.fit(X, y, geometry=geometry)

    # Generate predictions using fitted local models
    preds = model.predict(X, geometry=geometry)

    # Assert: number of predictions matches number of input samples
    assert len(preds) == len(X)

    # Assert: predictions are valid binary outputs (0/1 or True/False)
    assert set(np.unique(preds.dropna())).issubset({0, 1})

    # Ensure predictions are not all NaN
    assert not preds.isna().all()


def test_gwlogistic_predict_proba(sample_data):
    # Unpack sample dataset
    X, y, geometry = sample_data

    # Initialize model with keep_models=True for probability prediction
    model = GWLogisticRegression(
        bandwidth=10,
        fixed=False,
        keep_models=True,
        max_iter=1000,
    )

    # Fit model
    model.fit(X, y, geometry=geometry)

    # Get class probability predictions
    proba = model.predict_proba(X, geometry=geometry)

    # Assert: number of rows equals number of samples
    assert proba.shape[0] == len(X)

    # Assert: binary classification → exactly 2 probability columns
    assert proba.shape[1] == 2


def test_predict_requires_keep_models(sample_data):
    # Unpack sample dataset
    X, y, geometry = sample_data

    # Initialize model WITHOUT storing local models
    model = GWLogisticRegression(
        bandwidth=10, fixed=False, keep_models=False, max_iter=1000
    )
    # Fit model
    model.fit(X, y, geometry=geometry)

    # prediction requires stored local models → should fail
    with pytest.raises(AttributeError, match="_local_models"):
        model.predict(X, geometry=geometry)
