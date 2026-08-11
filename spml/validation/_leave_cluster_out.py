"""Spatially blocked leave-one-cluster-out cross-validation.

References
----------
.. [1] Roberts et al. "Cross-validation strategies for data with temporal,
   spatial, hierarchical, or phylogenetic structure." (2017). *Ecography*
   40(8): 913-929. https://doi.org/10.1111/ecog.02881
"""

import numpy

from sklearn.base import BaseEstimator, clone

from ._utils import _assign_noise_to_nearest, _get_coords


class LeaveClusterOut(BaseEstimator):
    """Leave-one-cluster-out cross-validator.

    Fits a user-supplied clustering estimator to the input locations, then
    yields one fold per cluster: the test set is all observations in that
    cluster, and the training set is all observations in every other cluster.

    This is the spatial analogue of sklearn's ``LeavePGroupsOut(P=1)`` where
    the groups are spatial clusters discovered at fit time. It tests whether a
    model generalises to held-out geographic regions rather than held-out
    individual points, making it appropriate for assessing spatial
    transferability.

    Parameters
    ----------
    clusterer : sklearn-compatible clustering estimator
        Anything with a ``fit(coords)`` method that sets ``labels_`` after
        fitting. HDBSCAN, OPTICS, KMeans, and AgglomerativeClustering all
        work. If the estimator already has a ``labels_`` attribute it is used
        as-is; otherwise it is cloned and refit on the input coordinates. The
        fitted estimator is stored as ``clusterer_``.
    noise : {'train_only', 'drop', 'nearest'}, default 'train_only'
        How to handle noise points (those whose label equals *noise_label*).
        Noise points are never used as a test fold regardless of this setting.

        - ``'train_only'``: noise points appear in every training set, never test.
        - ``'drop'``: exclude noise points from both train and test.
        - ``'nearest'``: each noise point is reassigned to the cluster of its
          nearest non-noise point; it then participates in that cluster's fold
          as a regular member (appears in the test fold when that cluster is
          held out, and in the training fold otherwise).
    noise_label : int, default -1
        Cluster label used by density-based clusterers (HDBSCAN, OPTICS) to
        mark noise points.

    Attributes
    ----------
    clusterer_ : fitted clustering estimator
        The cloned (or passed-in) clusterer after fitting.
    labels_ : ndarray of int, shape (n,)
        Cluster labels assigned by ``clusterer_``.
    n_clusters_ : int
        Number of non-noise clusters (i.e. the number of folds yielded).

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> lco = LeaveClusterOut(KMeans(n_clusters=10, random_state=0))
    >>> for train_idx, test_idx in lco.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])

    With HDBSCAN and noise points excluded from training:

    >>> from sklearn.cluster import HDBSCAN
    >>> lco = LeaveClusterOut(HDBSCAN(min_cluster_size=20), noise="drop")
    >>> for train_idx, test_idx in lco.split(gdf):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])
    """

    _VALID_NOISE = ("train_only", "drop", "nearest")

    def __init__(self, clusterer, noise: str = "train_only", noise_label: int = -1):
        self.clusterer = clusterer
        self.noise = noise
        self.noise_label = noise_label

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each cluster.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray
            Locations to cluster.
        y, groups : ignored, present for sklearn API compatibility.

        Yields
        ------
        train : ndarray of int
        test : ndarray of int
        """
        if self.noise not in self._VALID_NOISE:
            raise ValueError(f"noise={self.noise!r} not in {self._VALID_NOISE}.")

        coords = _get_coords(X)
        n = len(coords)

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

        if self.noise == "nearest":
            labels = _assign_noise_to_nearest(coords, labels, self.noise_label, clusterer=self.clusterer_)

        all_labels = numpy.unique(labels)
        non_noise = all_labels[all_labels != self.noise_label]
        self.n_clusters_ = len(non_noise)

        indices = numpy.arange(n)

        for label in non_noise:
            test = indices[labels == label]
            if self.noise == "drop":
                train = indices[(labels != label) & (labels != self.noise_label)]
            else:  # 'train_only' or 'nearest' (nearest: no noise remains after reassignment)
                train = indices[labels != label]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if hasattr(self, "n_clusters_"):
            return self.n_clusters_
        raise ValueError(
            "n_splits is not known until split() has been called, because the "
            "number of clusters depends on the fitted clusterer."
        )
