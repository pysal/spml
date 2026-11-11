import numpy as np
import pandas as pd


class RandomUnderSampler:
    """Random undersampling for classification targets.

    This helper implements a minimal subset of the imbalanced-learn API
    (``fit_resample``) used internally by geographically weighted classifiers.

    Parameters
    ----------
    sampling_strategy : bool | float, default=True
        If ``True``, undersample all larger classes to match the smallest class
        (i.e., minority/majority ratio = 1.0).

        If a float ``alpha > 0``, target a minority/majority ratio of ``alpha`` after
        resampling, i.e. ``alpha = N_min / N_resampled_majority``. For multiclass
        targets this caps each larger class at ``floor(N_min / alpha)``.
    random_state : int | numpy.random.Generator | None, default=None
        Random seed (or RNG) used to subsample larger classes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from spml.undersample import RandomUnderSampler
    >>> X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    >>> y = pd.Series([0, 0, 0, 0, 1, 1])
    >>> rus = RandomUnderSampler(random_state=0)
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
        """Resample ``X`` and ``y`` by undersampling classes larger than the smallest.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Class labels.

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
        n_min = counts.min()

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

        if target_maj >= counts.max():
            return X, y
        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        selected = []
        for label, count in zip(uniques, counts, strict=True):
            indices = np.flatnonzero(y_arr == label)
            if count > n_min:
                indices = rng.permutation(indices)[:target_maj]
            selected.append(indices)
        keep_idx = np.sort(np.concatenate(selected))

        # index X and y preserving types
        if isinstance(X, pd.DataFrame | pd.Series):
            X_res = X.iloc[keep_idx].copy()
        else:
            X_res = np.asarray(X)[keep_idx]

        y_res = y.iloc[keep_idx].copy() if isinstance(y, pd.Series) else y_arr[keep_idx]

        return X_res, y_res
