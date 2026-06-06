import geopandas as gpd
import numpy as np
import pytest
from geodatasets import get_path


@pytest.fixture(scope="session")
def sample_decomposition_data():
    """Return standardised multivariate data for unsupervised decomposition tests."""
    gdf = gpd.read_file(get_path("geoda.guerry"))
    gdf = gdf.set_geometry(gdf.centroid)
    cols = ["Crm_prs", "Litercy", "Wealth", "Donatns", "Infants"]
    X = gdf[cols].astype(float)
    X = (X - X.mean()) / X.std()  # standardise as per Harris et al. (2011)
    geometry = gdf.geometry
    return X, geometry


@pytest.fixture(scope="session")
def sample_data():
    """Return sample data from geoda.guerry dataset."""
    gdf = gpd.read_file(get_path("geoda.guerry"))
    # Create point geometries from polygon centroids
    gdf = gdf.set_geometry(gdf.centroid)
    # Create binary target variable
    gdf["binary_target"] = gdf["Donatns"] > gdf["Donatns"].median()

    # Select features
    X = gdf[["Crm_prs", "Litercy", "Wealth"]]
    y = gdf["binary_target"]
    geometry = gdf.geometry

    return X, y, geometry


@pytest.fixture(scope="session")
def sample_regression_data():
    """Return sample regression data from geoda.guerry dataset."""
    gdf = gpd.read_file(get_path("geoda.guerry"))
    # Create point geometries from polygon centroids
    gdf = gdf.set_geometry(gdf.centroid)

    # Select features and continuous target
    X = gdf[["Crm_prs", "Litercy", "Wealth"]]
    y = gdf["Donatns"]  # Continuous target for regression
    geometry = gdf.geometry

    return X, y, geometry
