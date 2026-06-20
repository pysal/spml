import pandas as pd

from spatialml.ensemble import (
    GWGradientBoostingClassifier,
    GWGradientBoostingRegressor,
    GWRandomForestClassifier,
    GWRandomForestRegressor,
)


def test_gwrf_init():
    """Test GWRandomForestClassifier initialization."""
    model = GWRandomForestClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "random_forest"

    # Check custom parameters
    model = GWRandomForestClassifier(
        bandwidth=50, fixed=True, kernel="tricube", n_estimators=100, max_depth=5
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["max_depth"] == 5


def test_gwrf_fit_basic(sample_data):  # noqa: F811
    """Test that GWRandomForestClassifier fit method works and runs as expected."""
    X, y, geometry = sample_data

    model = GWRandomForestClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWRandomForestClassifier
    assert hasattr(model, "feature_importances_")
    assert hasattr(model, "oob_y_pooled_")
    assert hasattr(model, "oob_pred_pooled_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)
    pd.testing.assert_series_equal(
        model.feature_importances_.mean(),
        pd.Series(
            [0.33556985859855293, 0.36600269841232724, 0.2984274429891198],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
        atol=0.01,
    )


def test_gwgb_init():
    """Test GWGradientBoostingClassifier initialization."""
    model = GWGradientBoostingClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "gradient_boosting"

    # Check custom parameters
    model = GWGradientBoostingClassifier(
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["learning_rate"] == 0.1
    assert model._model_kwargs["subsample"] == 0.8


def test_gwgb_fit_basic(sample_data):  # noqa: F811
    """Test that GWGradientBoostingClassifier fit method works as expected."""
    X, y, geometry = sample_data

    model = GWGradientBoostingClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWGradientBoostingClassifier
    assert hasattr(model, "feature_importances_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)


def test_gwrf_regressor_init():
    """Test GWRandomForestRegressor initialization."""
    model = GWRandomForestRegressor(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "random_forest"

    # Check custom parameters
    model = GWRandomForestRegressor(
        bandwidth=50, fixed=True, kernel="tricube", n_estimators=100, max_depth=5
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["max_depth"] == 5


def test_gwrf_regressor_fit_basic(sample_regression_data):  # noqa: F811
    """Test that GWRandomForestRegressor fit method works and runs as expected."""
    X, y, geometry = sample_regression_data

    model = GWRandomForestRegressor(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWRandomForestRegressor
    assert hasattr(model, "feature_importances_")
    assert hasattr(model, "oob_y_pooled_")
    assert hasattr(model, "oob_pred_pooled_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)

    # Check regressor-specific attributes
    assert hasattr(model, "pred_")
    assert hasattr(model, "resid_")
    assert hasattr(model, "RSS_")
    assert hasattr(model, "TSS_")
    assert hasattr(model, "y_bar_")
    assert hasattr(model, "local_r2_")
    # IC-related attributes are not defined for non-linear models
    assert not hasattr(model, "hat_values_")
    assert not hasattr(model, "effective_df_")
    assert not hasattr(model, "log_likelihood_")
    assert not hasattr(model, "aic_")
    assert not hasattr(model, "aicc_")
    assert not hasattr(model, "bic_")


def test_gwgb_regressor_init():
    """Test GWGradientBoostingRegressor initialization."""
    model = GWGradientBoostingRegressor(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "gradient_boosting"

    # Check custom parameters
    model = GWGradientBoostingRegressor(
        bandwidth=50,
        fixed=True,
        kernel="tricube",
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "tricube"
    assert model._model_kwargs["n_estimators"] == 100
    assert model._model_kwargs["learning_rate"] == 0.1
    assert model._model_kwargs["subsample"] == 0.8


def test_gwgb_regressor_fit_basic(sample_regression_data):  # noqa: F811
    """Test that GWGradientBoostingRegressor fit method works as expected."""
    X, y, geometry = sample_regression_data

    model = GWGradientBoostingRegressor(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWGradientBoostingRegressor
    assert hasattr(model, "feature_importances_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)

    # Check regressor-specific attributes
    assert hasattr(model, "pred_")
    assert hasattr(model, "resid_")
    assert hasattr(model, "RSS_")
    assert hasattr(model, "TSS_")
    assert hasattr(model, "y_bar_")
    assert hasattr(model, "local_r2_")
    # IC-related attributes are not defined for non-linear models
    assert not hasattr(model, "hat_values_")
    assert not hasattr(model, "effective_df_")
    assert not hasattr(model, "log_likelihood_")
    assert not hasattr(model, "aic_")
    assert not hasattr(model, "aicc_")
    assert not hasattr(model, "bic_")
