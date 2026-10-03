import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from spml.validation import HilbertKFold


@pytest.fixture
def grid_gdf():
    """5 * 5 regular grid of points."""
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x in range(5) for y in range(5)])


# -- Basic split behaviour -----------------------------------------------------


def test_correct_number_of_folds(grid_gdf):
    splits = list(HilbertKFold(n_splits=5, random_state=0).split(grid_gdf))
    assert len(splits) == 5


def test_train_test_partition(grid_gdf):
    n = len(grid_gdf)
    for train, test in HilbertKFold(n_splits=5, random_state=0).split(grid_gdf):
        assert len(set(train) & set(test)) == 0  # disjoint
        assert len(set(train) | set(test)) == n  # exhaustive


def test_non_empty_folds(grid_gdf):
    for train, test in HilbertKFold(n_splits=5, random_state=0).split(grid_gdf):
        assert len(train) > 0
        assert len(test) > 0


def test_get_n_splits():
    assert HilbertKFold(n_splits=4).get_n_splits() == 4


# -- Input types ---------------------------------------------------------------


def test_array_input():
    coords = np.random.default_rng(0).uniform(0, 10, (30, 2))
    splits = list(HilbertKFold(n_splits=3, random_state=0).split(coords))
    assert len(splits) == 3


def test_geoseries_input(grid_gdf):
    splits = list(HilbertKFold(n_splits=3, random_state=0).split(grid_gdf.geometry))
    assert len(splits) == 3


# -- Spatial balance -----------------------------------------------------------


def test_folds_are_spatially_balanced(grid_gdf):
    """Each fold should cover the full spatial extent (well-mixed, not clustered).

    On a 5*5 grid the x-centroid of each fold should be near the overall mean
    (2.0). Balanced folds have centroids CLOSER together than a random split,
    and far closer than the old cluster-based folds which spanned > 1 unit.
    """
    skf = HilbertKFold(n_splits=5, random_state=0)
    fold_x_means = []
    for _, test in skf.split(grid_gdf):
        fold_x_means.append(grid_gdf.geometry.iloc[test].x.mean())
    # All fold x-centroids should be close to the global mean (2.0)
    assert max(fold_x_means) - min(fold_x_means) < 1.0


def test_within_fold_spread_greater_than_random(grid_gdf):
    """Mean within-fold nearest-neighbour distance should exceed random k-fold."""
    from scipy.spatial.distance import cdist
    from sklearn.model_selection import KFold

    coords = np.column_stack([grid_gdf.geometry.x, grid_gdf.geometry.y])

    def mean_min_dist(splits):
        dists = []
        for _, test in splits:
            if len(test) < 2:
                continue
            D = cdist(coords[test], coords[test])
            np.fill_diagonal(D, np.inf)
            dists.append(D.min(axis=1).mean())
        return np.mean(dists)

    skf = HilbertKFold(n_splits=5, random_state=0)
    rkf = KFold(n_splits=5, shuffle=True, random_state=0)

    spatial_d = mean_min_dist(skf.split(grid_gdf))
    random_d = mean_min_dist(rkf.split(coords))
    assert spatial_d > random_d


# -- Edge cases ----------------------------------------------------------------


def test_too_many_splits_raises(grid_gdf):
    with pytest.raises(ValueError, match="n_splits"):
        list(HilbertKFold(n_splits=100).split(grid_gdf))


def test_reproducible(grid_gdf):
    s1 = list(HilbertKFold(n_splits=5, random_state=7).split(grid_gdf))
    s2 = list(HilbertKFold(n_splits=5, random_state=7).split(grid_gdf))
    for (tr1, te1), (tr2, te2) in zip(s1, s2, strict=False):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


# -- sklearn compat ------------------------------------------------------------


def test_works_with_cross_val_score(grid_gdf):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score

    X = np.column_stack([grid_gdf.geometry.x, grid_gdf.geometry.y])
    y = np.zeros(len(grid_gdf), dtype=int)

    skf = HilbertKFold(n_splits=5, random_state=0)
    scores = cross_val_score(DummyClassifier(), X, y, cv=skf)
    assert len(scores) == 5
