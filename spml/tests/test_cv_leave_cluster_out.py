import numpy
import pytest
import geopandas
from shapely.geometry import Point
from sklearn.cluster import KMeans, HDBSCAN

from spml.validation import LeaveClusterOut


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
    noise = rng.uniform(-10, 10, (12, 2))
    coords = numpy.vstack([A, B, C, noise])
    gdf = geopandas.GeoDataFrame(geometry=[Point(x, y) for x, y in coords])
    return gdf, coords


# -- Basic split behaviour -----------------------------------------------------

def test_correct_number_of_folds(three_clusters):
    gdf, _ = three_clusters
    lco = LeaveClusterOut(KMeans(n_clusters=3, random_state=0, n_init=10))
    splits = list(lco.split(gdf))
    assert len(splits) == 3
    assert lco.n_clusters_ == 3


def test_train_test_disjoint(three_clusters):
    gdf, _ = three_clusters
    for tr, te in LeaveClusterOut(
        KMeans(n_clusters=3, random_state=0, n_init=10)
    ).split(gdf):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_test_is_single_cluster(three_clusters):
    """Each test fold should contain exactly the members of one cluster."""
    gdf, _ = three_clusters
    lco = LeaveClusterOut(KMeans(n_clusters=3, random_state=0, n_init=10))
    for tr, te in lco.split(gdf):
        assert len(numpy.unique(lco.labels_[te])) == 1


def test_get_n_splits_before_split_raises(three_clusters):
    lco = LeaveClusterOut(KMeans(n_clusters=3, random_state=0, n_init=10))
    with pytest.raises(ValueError):
        lco.get_n_splits()


def test_get_n_splits_after_split(three_clusters):
    gdf, _ = three_clusters
    lco = LeaveClusterOut(KMeans(n_clusters=3, random_state=0, n_init=10))
    list(lco.split(gdf))
    assert lco.get_n_splits() == 3


# -- Input types ---------------------------------------------------------------

def test_array_input(three_clusters):
    _, coords = three_clusters
    splits = list(LeaveClusterOut(
        KMeans(n_clusters=3, random_state=0, n_init=10)
    ).split(coords))
    assert len(splits) == 3


def test_geoseries_input(three_clusters):
    gdf, _ = three_clusters
    splits = list(LeaveClusterOut(
        KMeans(n_clusters=3, random_state=0, n_init=10)
    ).split(gdf.geometry))
    assert len(splits) == 3


# -- Clusterer plumbing --------------------------------------------------------

def test_clusterer_is_cloned_not_mutated(three_clusters):
    gdf, _ = three_clusters
    km = KMeans(n_clusters=3, random_state=0, n_init=10)
    lco = LeaveClusterOut(km)
    list(lco.split(gdf))
    assert not hasattr(km, "labels_")
    assert hasattr(lco.clusterer_, "labels_")


def test_prefitted_clusterer_used_as_is(three_clusters):
    gdf, coords = three_clusters
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(coords)
    original_labels = km.labels_.copy()
    lco = LeaveClusterOut(km)
    list(lco.split(gdf))
    assert lco.clusterer_ is km
    numpy.testing.assert_array_equal(lco.labels_, original_labels)


def test_prefitted_length_mismatch_raises(three_clusters):
    gdf, coords = three_clusters
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(coords[:30])
    with pytest.raises(ValueError, match="Pre-fitted"):
        list(LeaveClusterOut(km).split(gdf))


# -- Noise handling ------------------------------------------------------------

def test_noise_train_only_all_noise_in_every_train(clusters_with_noise):
    gdf, _ = clusters_with_noise
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="train_only")
    splits = list(lco.split(gdf))
    noise_idx = set(numpy.flatnonzero(lco.labels_ == -1).tolist())
    assert noise_idx
    for tr, te in splits:
        assert noise_idx.isdisjoint(set(te.tolist()))
        assert noise_idx.issubset(set(tr.tolist()))


def test_noise_drop_excludes_noise(clusters_with_noise):
    gdf, _ = clusters_with_noise
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="drop")
    all_idx = set()
    for tr, te in lco.split(gdf):
        all_idx.update(tr.tolist())
        all_idx.update(te.tolist())
    noise_idx = set(numpy.flatnonzero(lco.labels_ == -1).tolist())
    assert noise_idx
    assert noise_idx.isdisjoint(all_idx)


def test_noise_nearest_all_indices_covered(clusters_with_noise):
    """noise='nearest': every index including original noise appears in train or test."""
    gdf, _ = clusters_with_noise
    n = len(gdf)
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="nearest")
    seen = set()
    for tr, te in lco.split(gdf):
        seen.update(tr.tolist())
        seen.update(te.tolist())
    assert seen == set(range(n))


def test_noise_nearest_no_train_test_overlap(clusters_with_noise):
    gdf, _ = clusters_with_noise
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="nearest")
    for tr, te in lco.split(gdf):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_noise_nearest_labels_preserves_original(clusters_with_noise):
    """noise='nearest': labels_ still records original -1 labels."""
    gdf, _ = clusters_with_noise
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="nearest")
    list(lco.split(gdf))
    assert (lco.labels_ == -1).any()


def test_noise_nearest_noise_appears_in_correct_test_fold(clusters_with_noise):
    """Each original noise point appears in exactly one test fold (its nearest cluster's)."""
    gdf, _ = clusters_with_noise
    lco = LeaveClusterOut(HDBSCAN(min_cluster_size=5, copy=True), noise="nearest")
    splits = list(lco.split(gdf))
    noise_idx = set(numpy.flatnonzero(lco.labels_ == -1).tolist())
    test_appearances = {i: 0 for i in noise_idx}
    for _, te in splits:
        for i in te:
            if i in test_appearances:
                test_appearances[i] += 1
    assert all(v == 1 for v in test_appearances.values())


def test_noise_nearest_haversine_metric(clusters_with_noise):
    """noise='nearest' with a haversine-metric clusterer assigns correctly.

    HDBSCAN with metric='haversine' is fitted on (lat, lon) in radians.
    _assign_noise_to_nearest must convert _get_coords output (lon, lat in
    degrees) to (lat, lon) in radians before querying NearestNeighbors.
    """
    _, coords = clusters_with_noise
    # HDBSCAN haversine wants (lat, lon) in radians
    coords_rad = numpy.radians(coords[:, [1, 0]])
    hdb = HDBSCAN(min_cluster_size=5, metric="haversine", copy=True).fit(coords_rad)

    # split() receives the GeoDataFrame / array in degrees (lon, lat)
    lco = LeaveClusterOut(hdb, noise="nearest")
    folds = list(lco.split(coords))

    n = len(coords)
    seen = set()
    for tr, te in folds:
        assert not set(tr.tolist()) & set(te.tolist())
        seen.update(tr.tolist())
        seen.update(te.tolist())
    assert seen == set(range(n))


def test_invalid_noise_mode_raises(three_clusters):
    gdf, _ = three_clusters
    with pytest.raises(ValueError, match="noise="):
        list(LeaveClusterOut(
            KMeans(n_clusters=3, random_state=0, n_init=10), noise="bogus"
        ).split(gdf))
