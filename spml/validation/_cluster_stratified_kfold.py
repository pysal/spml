"""Cluster-Stratified K-fold Crossvalidation

References
----------
.. [1] Ploton et al. "Spatial validation reveals poor
        predictive performance of large-scale ecological
        mapping models. (2020). *Nature Communications*
        11:45450. https://doi.org/10.1038/s41467-020-18321-y
"""

import numpy
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state

from ._utils import _assign_noise_to_nearest, _get_coords


# Sentinel values stored in the fold_id array.
_EXCLUDE = -1  # point participates in neither train nor test
_TRAIN_POOL = -2  # point is always in train, never in test


class ClusterStratifiedKFold(BaseEstimator):
    """Cluster-stratified k-fold cross-validator.

    Fits a user-supplied clustering estimator (e.g.
    ``sklearn.cluster.HDBSCAN``, ``sklearn.cluster.OPTICS``,
    ``sklearn.cluster.KMeans``) to the input locations, then partitions each
    cluster's members into *n_splits* equally sized parts.  Every test fold
    is a stratified sample drawn from every cluster -- so each fold spans
    all clusters rather than holding out a single region.

    This is the spatial analogue of ``StratifiedKFold`` where the strata are
    spatial clusters discovered at fit time.

    Parameters
    ----------
    clusterer : sklearn-compatible clustering estimator
        Anything with a ``fit(coords)`` method that sets ``labels_`` after
        fitting.  HDBSCAN and OPTICS are the intended targets; KMeans also
        works.  If the estimator already has a ``labels_`` attribute (i.e.
        was fitted ahead of time) it is used as-is; otherwise it is cloned
        and refit on the input coordinates.  The fitted estimator is stored
        as ``clusterer_``.
    n_splits : int, default 5
        Number of folds.  Each cluster's members are split into this many
        parts; clusters smaller than *n_splits* contribute to only the
        first ``len(cluster)`` folds.
    noise : {'stratify', 'drop', 'train_only', 'nearest'}, default 'stratify'
        How to handle noise points (those whose label equals *noise_label*).

        - 'stratify'  : distribute noise across folds like a normal cluster.
        - 'drop'      : exclude noise points from both train and test.
        - 'train_only': noise points appear in every train set, never test.
        - 'nearest'   : each noise point is reassigned to the cluster of its
          nearest non-noise point and then participates in that cluster's
          fold-assignment pool as a regular member.
    noise_label : int, default -1
        Cluster label used by density-based clusterers (HDBSCAN, OPTICS)
        to mark noise points.
    random_state : int, RandomState instance, or None
        Controls within-cluster shuffling before splitting.

    Attributes
    ----------
    clusterer_ : fitted clustering estimator
        The cloned (or passed-in) clusterer after fitting.
    labels_ : ndarray of int, shape (n,)
        Cluster labels assigned by ``clusterer_``.
    n_clusters_ : int
        Number of distinct cluster labels (including noise if present).

    Examples
    --------
    >>> from sklearn.cluster import HDBSCAN
    >>> ckf = ClusterStratifiedKFold(HDBSCAN(min_cluster_size=5), n_splits=5, random_state=0)
    >>> for train_idx, test_idx in ckf.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    _VALID_NOISE = ("stratify", "drop", "train_only", "nearest")

    def __init__(
        self,
        clusterer,
        n_splits: int = 5,
        noise: str = "stratify",
        noise_label: int = -1,
        random_state=None,
    ):
        self.clusterer = clusterer
        self.n_splits = n_splits
        self.noise = noise
        self.noise_label = noise_label
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each cluster-stratified fold.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations to cluster.
        y, groups : ignored, present for sklearn API compatibility.

        Yields
        ------
        train : ndarray of int
        test  : ndarray of int
        """
        if self.noise not in self._VALID_NOISE:
            raise ValueError(f"noise={self.noise!r} not in {self._VALID_NOISE}.")

        coords = _get_coords(X)
        n = len(coords)
        k = self.n_splits
        if n < k:
            raise ValueError(f"n_splits={k} cannot exceed n_samples={n}.")

        rng = check_random_state(self.random_state)

        # Use a pre-fitted clusterer as-is; otherwise clone and fit so the
        # caller's instance is never mutated.
        if hasattr(self.clusterer, "labels_"):
            self.clusterer_ = self.clusterer
            labels = numpy.asarray(self.clusterer.labels_, dtype=int)
            if len(labels) != n:
                raise ValueError(
                    f"Pre-fitted clusterer has {len(labels)} labels but X has {n} rows."
                )
        else:
            self.clusterer_ = clone(self.clusterer).fit(coords)
            labels = numpy.asarray(self.clusterer_.labels_, dtype=int)

        self.labels_ = labels
        self.n_clusters_ = len(numpy.unique(labels))

        if self.noise == "nearest":
            labels = _assign_noise_to_nearest(coords, labels, self.noise_label, clusterer=self.clusterer_)

        # Assign each observation a fold id.  Within each cluster, shuffle
        # the members and deal them out round-robin to folds.  Noise points
        # are flagged with sentinels and handled per the noise mode.
        fold_ids = numpy.empty(n, dtype=int)
        for label in numpy.unique(labels):
            members = numpy.flatnonzero(labels == label)
            if label == self.noise_label and self.noise != "stratify":
                fold_ids[members] = _EXCLUDE if self.noise == "drop" else _TRAIN_POOL
            else:
                rng.shuffle(members)
                fold_ids[members] = numpy.arange(len(members)) % k

        for fold in range(k):
            test = numpy.flatnonzero(fold_ids == fold)
            train_mask = (fold_ids != fold) & (fold_ids != _EXCLUDE)
            train = numpy.flatnonzero(train_mask)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
