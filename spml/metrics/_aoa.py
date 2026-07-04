"""Area of Applicability (Meyer & Pebesma 2021).

References
----------
.. [1] Meyer, H., & Pebesma, E. (2021). Predicting into unknown space?
   Estimating the area of applicability of spatial prediction models.
   *Methods in Ecology and Evolution*, 12(9), 1620-1633.
   https://doi.org/10.1111/2041-210X.13650
"""

import numpy
from sklearn.inspection import permutation_importance
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils import Bunch, check_array


def area_of_applicability(
    X_test,
    X_train,
    *,
    y_train=None,
    model=None,
    cv=None,
    feature_weights="permutation",
    metric="euclidean",
    threshold="tukey",
    permutation_kwargs=None,
    return_diagnostics=False,
):
    """Estimate the Area of Applicability (AOA) for ``X_test`` relative to ``X_train``.

    For each test sample, computes the feature-weighted distance to its
    nearest training sample, normalises by the mean within-training
    distance to obtain a Dissimilarity Index (DI), and flags the sample
    as "applicable" if its DI does not exceed a cut-off calibrated from
    the distribution of training DIs.  Method introduced by
    :cite:t:`meyer2021`.

    Parameters
    ----------
    X_test : (n_test, n_features) array
        Feature matrix for points where the model would be applied.
    X_train : (n_train, n_features) array
        Feature matrix used to train the model.
    y_train : (n_train,) array, optional
        Training targets.  Required when
        ``feature_weights='permutation'``.
    model : fitted sklearn estimator, optional
        Required when ``feature_weights='permutation'``; passed to
        ``sklearn.inspection.permutation_importance``.
    cv : sklearn CV splitter, optional
        If provided, the training-DI distribution is computed by holding
        out each fold in turn -- the held-out points get their distance
        to the in-fold training points.  If None, each training point's
        DI is its distance to its nearest OTHER training point.
    feature_weights : {'permutation', 'uniform'} or (n_features,) array
        How to weight features when computing distances.

        - 'permutation' : sklearn permutation importance, normalised to
          sum to 1.  Negative importances are clipped to zero.
        - 'uniform' / False / None : all features weighted equally.
        - array : pre-computed non-negative weights; normalised to sum
          to 1 internally.
    metric : str, default 'euclidean'
        Passed to ``sklearn.metrics.pairwise_distances``.
    threshold : {'tukey', 'mad'} or float in (0, 1)
        Rule for converting the training-DI distribution into a single
        cut-off.

        - 'tukey' : 75th percentile + 1.5 * IQR.
        - 'mad'   : median + 3 * median absolute deviation.
        - float p : the p-quantile of the training DIs.
    permutation_kwargs : dict, optional
        Extra arguments forwarded to
        ``sklearn.inspection.permutation_importance``.  Defaults set
        ``n_repeats=10`` and ``n_jobs=-1`` if not overridden.
    return_diagnostics : bool, default False
        If False, returns the boolean ``applicable`` mask (cheap path:
        only the nearest training point per test sample is computed).
        If True, returns a ``sklearn.utils.Bunch`` with ``applicable``,
        ``dissimilarity_index``, ``cutpoint``, ``feature_weights``, and
        ``lpd`` (Local Point Density).  The diagnostics path computes
        the full (n_test, n_train) distance matrix, so memory cost is
        O(n_test * n_train).

    Returns
    -------
    applicable : (n_test,) bool ndarray
        True where the model is considered applicable.
    OR a Bunch with attributes ``applicable``, ``dissimilarity_index``,
    ``cutpoint``, ``feature_weights``, and ``lpd`` when
    ``return_diagnostics=True``.

    References
    ----------
    .. [meyer2021] Meyer, H. & Pebesma, E. (2021). Predicting into unknown
       space? Estimating the area of applicability of spatial prediction
       models. *Methods in Ecology and Evolution*, 12(9), 1620-1633.
       https://doi.org/10.1111/2041-210X.13650
    """
    X_test = check_array(X_test, ensure_all_finite=True)
    X_train = check_array(X_train, ensure_all_finite=True)

    n_test, n_features = X_test.shape
    n_train, n_features_train = X_train.shape

    if n_features != n_features_train:
        raise ValueError(
            f"X_test has {n_features} features but X_train has {n_features_train}."
        )
    if n_test < 2 or n_train < 2:
        raise ValueError("Both X_train and X_test need at least 2 samples.")

    weights = _resolve_weights(
        feature_weights,
        X_train,
        y_train,
        model,
        n_features,
        permutation_kwargs,
    )

    Xw_train = X_train * weights[None, :]
    Xw_test = X_test * weights[None, :]

    # Training-DI distribution
    train_dists = pairwise_distances(Xw_train, metric=metric)
    if cv is None:
        # Each training point's DI = distance to its nearest OTHER training
        # point, normalised by the mean of all pairwise training distances.
        numpy.fill_diagonal(train_dists, numpy.inf)
        d_mins = train_dists.min(axis=1)
        numpy.fill_diagonal(train_dists, 0.0)
        # Upper triangle without diagonal for the mean (avoids the
        # double-count + zero-diagonal bias).
        iu = numpy.triu_indices_from(train_dists, k=1)
        d_mean = train_dists[iu].mean()
    else:
        # Held-out points get their distance to in-fold training points.
        # NOTE: sklearn yields (train_idx, test_idx) in that order.
        d_mins = numpy.empty(n_train, dtype=float)
        s = 0.0
        c = 0
        for train_idx, test_idx in cv.split(Xw_train):
            held = train_dists[test_idx[:, None], train_idx]
            d_mins[test_idx] = held.min(axis=1)
            s += held.sum()
            c += held.size
        if c == 0:
            raise ValueError("CV produced no held-out samples.")
        d_mean = s / c

    di_train = d_mins / d_mean
    cutpoint = _compute_cutpoint(di_train, threshold)

    # Test-DI: nearest training point per test sample.  When diagnostics
    # are requested we need the full (n_test, n_train) distance matrix so
    # we can also compute Local Point Density per test sample; otherwise
    # we take the cheap argmin_min path.
    if return_diagnostics:
        test_dists = pairwise_distances(Xw_test, Xw_train, metric=metric)
        test_min = test_dists.min(axis=1)
        # LPD: for each test point, count training points whose DI is at
        # or below the cutpoint.
        lpd = ((test_dists / d_mean) <= cutpoint).sum(axis=1).astype(int)
    else:
        _, test_min = pairwise_distances_argmin_min(
            Xw_test,
            Xw_train,
            metric=metric,
        )
    di_test = test_min / d_mean

    # Convention here: True = INSIDE AOA (applicable).
    applicable = di_test <= cutpoint

    if return_diagnostics:
        return Bunch(
            applicable=applicable,
            dissimilarity_index=di_test,
            cutpoint=float(cutpoint),
            feature_weights=weights,
            lpd=lpd,
        )
    return applicable


def _resolve_weights(
    feature_weights, X_train, y_train, model, n_features, permutation_kwargs
):
    """Return non-negative (n_features,) weight vector summing to 1."""
    if feature_weights is None or feature_weights is False:
        return numpy.full(n_features, 1.0 / n_features)
    if isinstance(feature_weights, str) and feature_weights == "uniform":
        return numpy.full(n_features, 1.0 / n_features)

    if isinstance(feature_weights, str) and feature_weights == "permutation":
        if model is None or y_train is None:
            raise ValueError(
                "feature_weights='permutation' requires both 'model' and 'y_train'."
            )
        kwargs = dict(permutation_kwargs or {})
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("n_repeats", 10)
        importances = permutation_importance(
            model,
            X_train,
            y_train,
            **kwargs,
        ).importances_mean
        importances = numpy.clip(importances, 0.0, None)
        total = importances.sum()
        if total == 0:
            raise ValueError(
                "All permutation importances are zero or negative; "
                "cannot derive feature weights."
            )
        return importances / total

    arr = numpy.asarray(feature_weights, dtype=float)
    if arr.shape != (n_features,):
        raise ValueError(
            f"feature_weights must have shape ({n_features},), got {arr.shape}."
        )
    if (arr < 0).any():
        raise ValueError("feature_weights must be non-negative.")
    total = arr.sum()
    if total == 0:
        raise ValueError("feature_weights sum to zero.")
    return arr / total


def _compute_cutpoint(di_train, threshold):
    """Translate the training-DI distribution into an applicability cut-off."""
    if threshold == "tukey":
        # FIXED relative to nanophyto/abil: percentile expects 0-100,
        # not 0-1.  Original passed (0.25, 0.75) -> always ~0.
        lo, hi = numpy.percentile(di_train, [25, 75])
        return float(hi + 1.5 * (hi - lo))
    if threshold == "mad":
        med = numpy.median(di_train)
        mad = numpy.median(numpy.abs(di_train - med))
        return float(med + 3.0 * mad)
    if isinstance(threshold, (int, float)) and 0.0 < threshold < 1.0:
        # FIXED: scale 0-1 quantile to 0-100 for numpy.percentile.
        return float(numpy.percentile(di_train, threshold * 100.0))
    raise ValueError(
        f"threshold must be 'tukey', 'mad', or a float in (0, 1); got {threshold!r}."
    )
