import numpy
import pytest
import geopandas
from shapely.geometry import Point
from sklearn.cluster import KMeans, HDBSCAN

from spml.validation import ClusterStratifiedKFold


@pytest.fixture
def three_clusters():
    """60 points in 3 well-separated blobs of 20 each."""
    rng = numpy.random.default_rng(0)
    A = rng.normal([0, 0], 0.2, (20, 2))
    B = rng.normal([5, 0], 0.2, (20, 2))
    C = rng.normal([2.5, 4], 0.2, (20, 2))
    coords = numpy.vstack([A, B, C])
    gdf = geopandas.GeoDataFrame(geometry=[Point(x, y) for x, y in coords])
    return gdf, coords


@pytest.fixture
def clusters_with_noise():
    """60 cluster points + 12 noise points scattered far away."""
    rng = numpy.random.default_rng(1)
    A = rng.normal([0, 0], 0.2, (20, 2))
    B = rng.normal([5, 0], 0.2, (20, 2))
    C = rng.normal([2.5, 4], 0.2, (20, 2))
    noise = rng.uniform(-10, 10, (12, 2))   # far-away outliers
    coords = numpy.vstack([A, B, C, noise])
    gdf = geopandas.GeoDataFrame(geometry=[Point(x, y) for x, y in coords])
    return gdf, coords


# -- Basic split behaviour -----------------------------------------------------

def test_correct_number_of_folds(three_clusters):
    gdf, _ = three_clusters
    splits = list(ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=5, random_state=0,
    ).split(gdf))
    assert len(splits) == 5


def test_train_test_partition(three_clusters):
    gdf, _ = three_clusters
    n = len(gdf)
    for train, test in ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=5, random_state=0,
    ).split(gdf):
        assert len(set(train) & set(test)) == 0
        assert len(set(train) | set(test)) == n


def test_get_n_splits():
    assert ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=4,
    ).get_n_splits() == 4


# -- Stratification ------------------------------------------------------------

def test_each_fold_spans_all_clusters(three_clusters):
    """With balanced clusters and n_splits dividing the cluster size,
    every test fold should contain points from every cluster."""
    gdf, _ = three_clusters
    ckf = ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=5, random_state=0,
    )
    for _, test in ckf.split(gdf):
        labels_in_test = numpy.unique(ckf.labels_[test])
        assert len(labels_in_test) == 3


def test_per_cluster_counts_balanced(three_clusters):
    """Each cluster's points should be split as evenly as possible."""
    gdf, _ = three_clusters
    ckf = ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=5, random_state=0,
    )
    # 20 points / 5 folds = exactly 4 per fold per cluster
    for _, test in ckf.split(gdf):
        per_cluster = numpy.bincount(ckf.labels_[test], minlength=3)
        assert per_cluster.tolist() == [4, 4, 4]


# -- Clusterer plumbing --------------------------------------------------------

def test_clusterer_is_cloned_not_mutated(three_clusters):
    """The user's clusterer instance should not be fitted in place."""
    gdf, _ = three_clusters
    km = KMeans(n_clusters=3, random_state=0, n_init=10)
    ckf = ClusterStratifiedKFold(km, n_splits=3, random_state=0)
    list(ckf.split(gdf))
    assert not hasattr(km, "labels_")
    assert hasattr(ckf.clusterer_, "labels_")


def test_hdbscan_runs(three_clusters):
    gdf, _ = three_clusters
    ckf = ClusterStratifiedKFold(HDBSCAN(min_cluster_size=5, copy=True), n_splits=4, random_state=0)
    splits = list(ckf.split(gdf))
    assert len(splits) == 4
    # at least 1 real cluster discovered
    assert (ckf.labels_ >= 0).any()


# -- Input types ---------------------------------------------------------------

def test_array_input(three_clusters):
    _, coords = three_clusters
    splits = list(ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=3, random_state=0,
    ).split(coords))
    assert len(splits) == 3


def test_geoseries_input(three_clusters):
    gdf, _ = three_clusters
    splits = list(ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=3, random_state=0,
    ).split(gdf.geometry))
    assert len(splits) == 3


# -- Edge cases ----------------------------------------------------------------

def test_too_many_splits_raises(three_clusters):
    gdf, _ = three_clusters
    with pytest.raises(ValueError, match="n_splits"):
        list(ClusterStratifiedKFold(
            KMeans(n_clusters=3, random_state=0, n_init=10),
            n_splits=1000,
        ).split(gdf))


def test_reproducible(three_clusters):
    gdf, _ = three_clusters
    s1 = list(ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=7, n_init=10),
        n_splits=5, random_state=7,
    ).split(gdf))
    s2 = list(ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=7, n_init=10),
        n_splits=5, random_state=7,
    ).split(gdf))
    for (tr1, te1), (tr2, te2) in zip(s1, s2):
        numpy.testing.assert_array_equal(tr1, tr2)
        numpy.testing.assert_array_equal(te1, te2)


# -- sklearn compat ------------------------------------------------------------

def test_works_with_cross_val_score(three_clusters):
    from sklearn.model_selection import cross_val_score
    from sklearn.dummy import DummyClassifier

    gdf, coords = three_clusters
    y = numpy.zeros(len(gdf), dtype=int)
    ckf = ClusterStratifiedKFold(
        KMeans(n_clusters=3, random_state=0, n_init=10),
        n_splits=5, random_state=0,
    )
    scores = cross_val_score(DummyClassifier(), coords, y, cv=ckf)
    assert len(scores) == 5


# -- Noise handling ------------------------------------------------------------

def test_noise_stratify_default(clusters_with_noise):
    """Default mode: noise points are distributed across folds."""
    gdf, _ = clusters_with_noise
    ckf = ClusterStratifiedKFold(HDBSCAN(min_cluster_size=5, copy=True), n_splits=4, random_state=0)
    n = len(gdf)
    seen = set()
    for tr, te in ckf.split(gdf):
        seen.update(tr.tolist())
        seen.update(te.tolist())
    # every index participates in either train or test
    assert seen == set(range(n))


def test_noise_drop_excludes_noise(clusters_with_noise):
    """noise='drop': noise indices never appear in train or test."""
    gdf, _ = clusters_with_noise
    ckf = ClusterStratifiedKFold(
        HDBSCAN(min_cluster_size=5, copy=True), n_splits=4,
        noise="drop", random_state=0,
    )
    all_idx = set()
    for tr, te in ckf.split(gdf):
        all_idx.update(tr.tolist())
        all_idx.update(te.tolist())
    noise_idx = set(numpy.flatnonzero(ckf.labels_ == -1).tolist())
    assert noise_idx  # the fixture actually produces noise
    assert noise_idx.isdisjoint(all_idx)


def test_noise_train_only_keeps_noise_out_of_test(clusters_with_noise):
    """noise='train_only': noise indices appear in every train, never in test."""
    gdf, _ = clusters_with_noise
    ckf = ClusterStratifiedKFold(
        HDBSCAN(min_cluster_size=5, copy=True), n_splits=4,
        noise="train_only", random_state=0,
    )
    splits = list(ckf.split(gdf))
    noise_idx = set(numpy.flatnonzero(ckf.labels_ == -1).tolist())
    assert noise_idx
    for tr, te in splits:
        assert noise_idx.isdisjoint(set(te.tolist()))
        assert noise_idx.issubset(set(tr.tolist()))


def test_noise_nearest_no_noise_in_labels(clusters_with_noise):
    """noise='nearest': labels_ still records original -1 noise labels."""
    gdf, _ = clusters_with_noise
    ckf = ClusterStratifiedKFold(
        HDBSCAN(min_cluster_size=5, copy=True), n_splits=4,
        noise="nearest", random_state=0,
    )
    list(ckf.split(gdf))
    assert (ckf.labels_ == -1).any()


def test_noise_nearest_all_indices_covered(clusters_with_noise):
    """noise='nearest': every index (including original noise) appears in train or test."""
    gdf, _ = clusters_with_noise
    n = len(gdf)
    ckf = ClusterStratifiedKFold(
        HDBSCAN(min_cluster_size=5, copy=True), n_splits=4,
        noise="nearest", random_state=0,
    )
    seen = set()
    for tr, te in ckf.split(gdf):
        seen.update(tr.tolist())
        seen.update(te.tolist())
    assert seen == set(range(n))


def test_noise_nearest_no_train_test_overlap(clusters_with_noise):
    """noise='nearest': train and test are disjoint on every fold."""
    gdf, _ = clusters_with_noise
    ckf = ClusterStratifiedKFold(
        HDBSCAN(min_cluster_size=5, copy=True), n_splits=4,
        noise="nearest", random_state=0,
    )
    for tr, te in ckf.split(gdf):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_invalid_noise_mode_raises(three_clusters):
    gdf, _ = three_clusters
    with pytest.raises(ValueError, match="noise="):
        list(ClusterStratifiedKFold(
            KMeans(n_clusters=3, random_state=0, n_init=10),
            n_splits=3, noise="bogus",
        ).split(gdf))


# -- Pre-fitted clusterer ------------------------------------------------------

def test_prefitted_clusterer_used_as_is(three_clusters):
    """Passing a fitted clusterer should skip the internal fit."""
    gdf, coords = three_clusters
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(coords)
    original_labels = km.labels_.copy()

    ckf = ClusterStratifiedKFold(km, n_splits=3, random_state=0)
    list(ckf.split(gdf))

    # clusterer_ is the same object (not cloned), and labels match the pre-fit
    assert ckf.clusterer_ is km
    numpy.testing.assert_array_equal(ckf.labels_, original_labels)


def test_prefitted_clusterer_length_mismatch_raises(three_clusters):
    """Pre-fitted clusterer with wrong number of labels should raise."""
    gdf, coords = three_clusters
    # Fit on only half the data, then try to split the full set.
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(coords[:30])

    ckf = ClusterStratifiedKFold(km, n_splits=3, random_state=0)
    with pytest.raises(ValueError, match="Pre-fitted"):
        list(ckf.split(gdf))
