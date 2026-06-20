import numpy as np
import pandas as pd


class BinaryRandomUnderSampler:
    """Random undersampling for binary targets.

    This helper implements a minimal subset of the imbalanced-learn API
    (``fit_resample``) used internally by geographically weighted classifiers.

    Parameters
    ----------
    sampling_strategy : bool | float, default=True
        If ``True``, undersample the majority class to match the minority class
        (i.e., minority/majority ratio = 1.0).

        If a float ``alpha > 0``, target a minority/majority ratio of ``alpha`` after
        resampling, i.e. ``alpha = N_min / N_resampled_majority``.
    random_state : int | numpy.random.Generator | None, default=None
        Random seed (or RNG) used to subsample the majority class.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from spatialml.undersample import BinaryRandomUnderSampler
    >>> X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    >>> y = pd.Series([0, 0, 0, 0, 1, 1])
    >>> rus = BinaryRandomUnderSampler(random_state=0)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> y_res.value_counts().loc[0] == y_res.value_counts().loc[1]
    np.True_
    """

    def __init__(
        self, sampling_strategy: bool | float = True, random_state: int | None = None
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Resample ``X`` and ``y`` by undersampling the majority class.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Binary target.

        Returns
        -------
        X_resampled : array-like
            Resampled feature matrix.
        y_resampled : array-like
            Resampled target.
        """
        # convert y to numpy for processing but remember original types
        y_arr = np.asarray(y).ravel()

        # identify minority / majority labels
        uniques, counts = np.unique(y_arr, return_counts=True)
        order = np.argsort(counts)
        min_label, maj_label = uniques[order[0]], uniques[order[1]]
        n_min, n_maj = counts[order[0]], counts[order[1]]

        # interpret sampling_strategy as minority/majority ratio alpha
        if self.sampling_strategy is True:
            alpha = 1.0
        elif isinstance(self.sampling_strategy, float):
            alpha = float(self.sampling_strategy)
            if alpha <= 0:
                raise ValueError("sampling_strategy float must be > 0.")
        else:
            raise ValueError("sampling_strategy must be True or a float.")

        # compute target majority count (undersample majority only)
        # alpha = N_min / N_resampled_majority  => N_resampled_majority = N_min / alpha
        target_maj = int(np.floor(n_min / alpha))

        # if no undersampling required, return originals (preserve types)
        if target_maj >= n_maj:
            return X, y

        # get indices
        all_idx = np.arange(len(y_arr))
        maj_idx = all_idx[y_arr == maj_label]
        min_idx = all_idx[y_arr == min_label]

        if isinstance(self.random_state, np.random.Generator):
            rng = self.random_state
        else:
            rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(maj_idx))
        selected_maj_idx = maj_idx[perm[:target_maj]]

        keep_idx = np.concatenate([min_idx, selected_maj_idx])
        # keep original order (optional)
        keep_idx.sort()

        # index X and y preserving types
        if isinstance(X, pd.DataFrame | pd.Series):
            X_res = X.iloc[keep_idx].copy()
        else:
            X_res = np.asarray(X)[keep_idx]

        y_res = y.iloc[keep_idx].copy() if isinstance(y, pd.Series) else y_arr[keep_idx]

        return X_res, y_res
