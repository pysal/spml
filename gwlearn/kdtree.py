from quadtree_regression import _Cell as _Cell, QuadtreeRegressor, QuadtreeClassifier
import numpy, shapely, geopandas
from collections import defaultdict
import pandas
from scipy import sparse, stats, optimize
from sklearn.base import clone
from sklearn import linear_model, metrics,
import warnings

__all__ = ['KDTreeRegressor', 'KDTreeClassifier', 'KDTreeEnsembleRegressor', 'KDTreeEnsembleClassifier', 'KDTreeBoostingRegressor']

class _Node(_Cell):
    def _split_by_plane(self, plane, boost=False):
        min_x, min_y, max_x, max_y = self.bounds
        if plane[0][0] == plane[1][0]: # vertical, left-right split
            labels = self.coordinates[:,0] > plane[0][0]
        else: # horizontal, up/down split
            labels = self.coordinates[:,1] > plane[0][1]
        bounds = [
            [min_x, min_y, *plane[1]],
            [*plane[0], max_x, max_y]
        ]

        children = []
        if boost:
            y = self.residuals_
        else:
            y = self.y
        for i, bbox in enumerate(bounds):
            mask = labels == i
            ids = self.ids[mask]
            cell = type(self)(
                # binary index by depth
                index=self.index * 10 + (i + 1),
                parent=self.index,
                ids=ids,
                parent_mask=mask,
                coordinates=self.coordinates[mask,],
                bounds=bbox,
                X=self.X[mask],
                y=y[mask],
                score_function=self._score_function,
                model=clone(self._model),
            )
            if boost:
                # self.y is always reset to the actual data after boosting
                # which is necessary to propagate downwards. But, the preceeding
                # boost check before splitting ensures that the residuals are
                # used to train when boosting, and the y/preds are constructed
                # correctly afterwards.
                cell.y = self.y[mask]
                cell.prediction_ += self.prediction_[mask]
                cell.boosted = True
            children.append(cell)
        return children

    def split(self, boost=False):
        min_x, min_y, max_x, max_y = self.bounds
        if abs(max_x - min_x)>abs(max_y-min_y):
            mid_x = numpy.median(self.coordinates[:,0])
            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            mid_y = numpy.median(self.coordinates[:,1])
            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

class _RandomSplitNode(_Node):
    def split(self, boost=False, frac=1):
        min_x, min_y, max_x, max_y = self.bounds
        n_in_node = len(self.y)
        take = numpy.ceil(frac*n_in_node).astype(int)
        # allow replacement iff the take is bigger than the leaf
        ixs = numpy.random.choice(numpy.arange(n_in_node), take, replace=take>n_in_node)
        if abs(max_x - min_x)>abs(max_y-min_y):
            mid_x = numpy.median(self.coordinates[ixs,0])
            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            mid_y = numpy.median(self.coordinates[ixs,1])
            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

class _WeightedMedianSplitNode(_Node):
    def split(self, boost=False):
            min_x, min_y, max_x, max_y = self.bounds
            if abs(max_x - min_x)>abs(max_y-min_y):
                # in classification, assigns even prediction misses to each half
                # in regression, assigns even residuals to each half
                mid_x = _weighted_median(self.coordinates[:,0], weights=numpy.abs(self.residuals_))
                plane = [(mid_x, min_y), (mid_x, max_y)]
            else:
                mid_y = _weighted_median(self.coordinates[:,1], weights=numpy.abs(self.residuals_))
                plane = [(min_x, mid_y), (max_x, mid_y)]
            return self._split_by_plane(plane, boost=boost)

class _OptimalSplitNode(_Node):
    def split(self, boost=False, min_leaf_size=10):
        min_x, min_y, max_x, max_y = self.bounds
        if abs(max_x - min_x)>abs(max_y-min_y):
            # in classification, assigns even prediction misses to each half
            # in regression, assigns even residuals to each half
            def objective(t):
                plane = [(t, min_y), (t, max_y)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.score_+right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_x = type(min_x)(optimize.minimize(
                    objective,
                    bounds=[(min_x, max_x)],
                    x0=[_weighted_median(self.coordinates[:,0], weights=numpy.abs(self.residuals_))]
                    ).x.item()
                    )

            plane = [(mid_x, min_y), (mid_x, max_y)]
        else:
            def objective(t):
                plane = [(min_x, t), (max_x, t)]
                left, right = self._split_by_plane(plane, boost=boost)
                if (left.size < min_leaf_size) | (right.size < min_leaf_size):
                    return numpy.inf
                return -(left.score_ + right.score_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mid_y = type(min_x)(optimize.minimize(
                    objective,
                    bounds=[(min_y, max_y)],
                    x0=[_weighted_median(self.coordinates[:,1], weights=numpy.abs(self.residuals_))]
                    ).x.item()
                    )

            plane = [(min_x, mid_y), (max_x, mid_y)]
        return self._split_by_plane(plane, boost=boost)

class KDTreeRegressor(QuadtreeRegressor):
    def __init__(self,
        model=None,
        score=None,
        split_method='weighted',
        split_test='eps',
        split_tolerance=None,
        min_leaf_size=10,
        max_depth=10,
        pbar = True,
        prune = False,
        prune_tolerance=None,
        prune_repeats = None,
        n_jobs=None,
        random_state = None,
        boost = False,
        bounds=None
    ):
        """
        Initialize a KDTreeRegressor object

        Parameters
        ----------
        model : class instance
            instance of a scikit-learn model supporting both fit and
            predict methods. Should not generally cache global data,
            since a model is instantiated at each node in the tree.
        score : callable or None (default: None)
            function to use to score the model, set to the negative
            mean squared error by default.
        split_method: str or int/float (default: median)
            method to use to define the splitting plane in each node.
            one of:
            1. median (default): split the page into two equal halves along its longest dimension
            2. int/float: split the page into two equal halves along its longest dimension
                by sampling this many points/this percentage of the page and calculating the
                median. If the sample is smaller than min_leaf_size, then min_leaf_size will
                be used instead.
            3. weighted: split the page into two equal halves along its longest dimension
                the loss-weighted median along the longest dimension.
            4. optimal: split the page into two halves along its longest dimension in
                a way that minimizes the loss for the split. This tends to result in the
                same splits as the weighted outcome for many score functions.
        split_test : either 'eps' or 'lrt' (default: 'eps')
            splitting rule to use to grow the tree. If 'eps', then
            a node is split when the improvement from introducing
            a one-hot-encoded variable (and any attendant interaction terms
            in X) that expresses the split improves `score` by more than
            `split_tolerance`. If `lrt`, then a likelihood ratio test
            compares the model with the split to a model without the split.
            The `lrt` option is only applicable when the model being fit
            is a linear model. GLMs are not yet supported.
        split_tolerance : float (default: None)
            the value to use as a cutoff for the split rule. For `eps` split
            rule, this measures the minimum change in score that is necessary
            for a split to be "worth" making. For `lrt` split rule, this
            expresses the required p-value the likelihood ratio test
            must attain for a split to be "worth" making. By default,
            this is set to .05 under 'lrt' splitting, and 1e-5 under 'eps' splitting
        min_leaf_size : int (default: 10)
            The smallest allowed size of a leaf in the tree. If a split has fewer
            observations than `min_leaf_size`, it is discarded, and further splitting
            below the split is prevented.
        max_depth : int (default 10)
            The maximum depth of the Quadtree allowed. If a split is deeper than
            `max_depth`, then it is discarded, and further splitting below the split
            is prevented.
        pbar : bool (default: True)
            Whether to indicate progress during splitting and pruning stages.
        prune : bool or str (default: False)
            Whether to prune the quadtree according to a relevant pruning rule. If True,
            then the pruning strategy is picked based on the `split_test`. If `split_test='eps'`,
            then permutation importance pruning (`prune='perm'`) is used. This removes
            feature : cell interaction terms if they are not in the top `prune_tolerance` percentile
            of the permutation feature importance distribution for the un-pruned model.
            If `split_test='lrt'`, then Wald pruning (`prune='wald'`) is used by default instead.
            This pruning rule recursively removes feature : cell interaction terms if their
            Wald statistics are not statistically significant according to the `prune_tolerance`
            p-value. In ether case, pruning "folds" leaves upwards merging interacation effects
            into their parent node until significance is achieved or the feature
            is removed entirely from the model. Further, it is fullly valid to use
            `split_test='lrt'` and `prune='perm'` if you want to use permutation
            feature importance testing of likelihood ratio-derived splits. If False, no
            pruning is used, so the final feature matrix will be the full set of cell : feature
            interaction pairs.
        prune_tolerance : float (default: None)
            the value to use as a cutoff for the pruning procedure. For `perm` pruning,
            this reflects a percentile of permutation feature importances from the full
            un-pruned model. For example, `prune_tolerance=.01` indicates that a feature : cell
            term must be in the top 1% of feature importance replications over the full model
            to be retained. For `wald` pruning, this reflects the p-value of the Wald t-test for
            the regression coefficient. For more information, consult
            sklearn.inspection.permutation_importance and scipy.stats.t.sf, respecitvely.
        prune_repeats : int (default: None)
            number of repeats to use when computing permutation importance. Ignored when
            prune != 'perm'. For the default value, consult sklearn.inspection.permutation_importance
        n_jobs : int (default: None)
            number of jobs to use when computing permutation importance. Ignored when `prune`='perm'.
            For the default value, consult sklearn.inspection.permutation_imporatnce
        random_state : int or numpy.random.RandomState instance
            value (or random number generator) to use to initialize the permutation importance.
            Ignored hwen `prune` != 'perm'.
        boost : bool (default: False)
            Whether to fit the model using boosting. If False, cells at depth *d* are fit directly
            on data in that cell, without reference to other depth values. If True, cells at depth *d* are fit
            to the model residuals from predictions at *d-1*, and predictions are constructed
            from the sum of predictions at all levels. For example, the global
            model at depth 0 is fit to y, and the cell models at depth 1 are fit to the
            residuals of depth 0, cell models at depth 2 are fit to the residuals of *this* model,
            and so-on. The prediction from a cell at depth 2 is the prediction at depth 0 plus
            the prediction at depth 1 plus the prediction at depth 2.
        bounds : tuple or numpy.ndarray (default: None)
            total bounds to use for the root of the tree. This is particularly useful when the
            problem frame is well-defined, and you intend to keep the model fitting over the
            same area during cross-validation. If not provided, the bounds are derived from
            the input coordinates.

        Returns
        -------
        """
        if score is None:
            def score(y_true, y_pred):
                return -metrics.mean_squared_error(y_true, y_pred)
        self._score_function = score
        if model is None:
            model = linear_model.LinearRegression()
        self.model = model
        self.max_depth = max_depth
        self.split_method = split_method
        if split_tolerance is None:
            split_tolerance = .05 if split_test == 'lrt' else 1e-5
        self.split_tolerance = split_tolerance
        self.split_test = split_test
        self.min_leaf_size = min_leaf_size
        self.pbar = pbar
        self.n_jobs = n_jobs
        self.random_state = random_state

        if (prune is True):
            prune = "perm" if split_test == 'eps' else "wald"
        if prune == "perm":
            prune_repeats = 10 if (prune_repeats is None) else prune_repeats
            prune_tolerance = .5 if prune_tolerance is None else prune_tolerance
        if prune == "wald":
            if not isinstance(self.model, linear_model.LinearRegression):
                raise NotImplementedError(
                    "Wald test pruning is only implemented for ordinary least squares models!"
                )
            if prune_repeats is not None:
                warnings.warn(f"prune_repeats = {prune_repeats}, but is ignored when prune={prune}")
            prune_tolerance = split_tolerance if prune_tolerance is None else prune_tolerance

        self.prune = prune
        self.prune_tolerance = prune_tolerance
        self.prune_repeats = prune_repeats

        self.bounds = bounds

        if boost:
            if split_test != 'eps':
                raise NotImplementedError('Boosting is not implemented for "lrt" split_test.')
            if prune == "wald":
                raise NotImplementedError('Boosting is not implemented for "wald" prune_rule.')
        self.boost = boost

    def _build_node(self, *args, **kwargs):
        if self.split_method == 'median':
            return _Node(*args, **kwargs)
        elif self.split_method == 'weighted':
            return _WeightedMedianSplitNode(*args, **kwargs)
        elif isinstance(self.split_method, (float,int)):
            return _RandomSplitNode(*args, **kwargs)
        elif self.split_method == "optimal":
            return _OptimalSplitNode(*args, **kwargs)
        else:
            raise ValueError("split_method must be either 'median', 'gini', or a float/int value")

    def fit(self, X, y, coordinates=None):
        """
        Fit a KDTreeRegressor on input coordinates, with features X and outcome y.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame
            array of shape (n_samples, n_features) features to be fit, or
            (n_rows, n_cols, n_features) if a raster is input.
            An intercept should not be included.
        y : numpy.ndarray, pandas.Series, pandas.DataFrame
            array of outcome shaped (n_samples,) or (n_rows, n_cols) to be predicted. In nearly all cases, this should
            be centered on zero for splits to be found successfully.
        coordinates: numpy.ndarray
            array of shape (n_samples, 2) where observations occur.

        Returns
        -------
        KDTreeRegressor() object
        """
        if isinstance(X, pandas.DataFrame):
            self._x_names = numpy.asarray(X.columns)
            X = X.values
        else:
            self._x_names = numpy.asarray([f"x{i}" for i in range(X.shape[0])])
        if isinstance(y, (pandas.Series, pandas.DataFrame)):
            y = y.values.squeeze()
        else:
            assert y.ndim == 1, "y must be 1 dimensional"
            assert len(y) == len(X), "y must match X length"
        if self.bounds is None:
            self.bounds = [*numpy.min(coordinates, axis=0), *numpy.max(coordinates, axis=0)]
        root = self._build_node(
                index=0,
                parent=None,
                parent_mask=numpy.ones_like(y).astype(bool),
                coordinates=coordinates,
                bounds=self.bounds,
                ids=numpy.arange(X.shape[0]),
                X=X,
                y=y,
                score_function=self._score_function,
                model=clone(self.model),
            )

        self.levels_ = defaultdict(list)
        self.levels_[0].append(root)


        self.prediction_ = root.prediction_

        self.labels_ = numpy.zeros_like(y).squeeze()
        self.scores_ = numpy.empty_like(y).squeeze()
        self.depth_ = numpy.empty_like(y).squeeze()

        n_samples, n_features = X.shape
        queue = [root]
        self.leaves_ = dict()
        if (self.pbar is not False):
            from tqdm import tqdm
            self.pbar = tqdm(unit=' leafs', desc="Evaluating splits...")

        while len(queue) > 0:
            cell = queue.pop()
            splits = self._evaluate_splits(cell, X, y)
            queue.extend(splits)

        if (self.pbar is not False):
            self.pbar.close()
            self.pbar = True


        if self.prune == "perm":
            self.features_ = self._prune(X, y, score=self._permutation_importance)
        elif self.prune == "wald":
            self.features_ = self._prune(X, y, score=self._sparse_wald_feature_importance)
        elif (self.prune is False) or (self.prune is None):
            self.features_ = self._prune(X, y, score=None)
        else:
            raise ValueError(f'prune option {self.prune} not understood.')


        ## FINALIZE MODEL
        self.model_ = self.model.fit(self.features_, y)

        # TODO: why did i build this? was it needed and i stopped needing it?
        remaining_leaves = self.features_.columns.get_level_values("leaf").unique()
        all_leaves = [leaf for level in self.levels_.values() for leaf in level]
        self.labels_ = self.query(coordinates)
        tmp_geoms = []

        for leaf in all_leaves:
            leaf.pruned = leaf.index not in self.leaves_.keys()
            self.depth_[self.labels_ == leaf.index] = leaf.depth

            geom = dict(
                index=leaf.index,
                parent=leaf.parent,
                depth=leaf.depth,
                score=leaf.score_,
                pruned=leaf.pruned,
                model=leaf.model_,
                geometry=shapely.box(*leaf.bounds),
            )
            tmp_geoms.append(geom)
        self.geoms_ = geopandas.GeoDataFrame(pandas.DataFrame.from_records(tmp_geoms)).sort_values("depth")

        return self

    def _evaluate_splits(self, parent, X_global, y_global):
        # NOTE: parent.y refers to the y within parent. y_global refers to *all y*.
        # same with X_global
        if isinstance(self.split_method, (float,int)):
            # check node size limits first
            if self.split_method < 0:
                raise ValueError("numeric split_method must be positive!")
            if 0 < self.split_method < 1:
                n_in_parent = len(parent.y)
                f = numpy.maximum(
                    self.split_method,
                    self.min_leaf_size/n_in_parent
                )
            else:
                n_in_parent = len(parent.y)
                f = numpy.maximum(
                    # if the int is bigger than the leaf,
                    # just sample from the leaf itself
                    numpy.minimum(
                        self.split_method/n_in_parent,
                        1
                    ),
                    # if the int is smaller than the min_leaf_size, sample the min_leaf_size
                    self.min_leaf_size/n_in_parent
                )
            left, right = parent.split(boost=self.boost, frac=f)
        elif self.split_method == 'optimal':
            left, right = parent.split(boost=self.boost, min_leaf_size=self.min_leaf_size)
        else:
            left, right = parent.split(boost=self.boost)

        self.levels_[parent.depth+1].extend([left, right])
        retain_split = numpy.ones((2,)).astype(bool)
        candidate_labels = self.labels_.copy()

        for i,split in enumerate([left, right]):
            too_deep = split.depth > self.max_depth
            if too_deep:
                #print(f"Cell {split.index} is too high-res {split.depth}")
                retain_split[i] = False
                continue

            big_enough = split.size > self.min_leaf_size
            if not big_enough:
                #print(f"Cell {split.index} is too small {split.size}, {cell.size}")
                retain_split[i] = False
                continue
            if self.split_test == 'eps':
                parent_subpixel_score = parent._score_function(
                    parent.y[split.parent_mask], parent.prediction_[split.parent_mask]
                )
                improvement = split.score_ - parent_subpixel_score
                is_improvement = improvement >= self.split_tolerance
                if not is_improvement:
                    retain_split[i] = False
            else:
                # LRT is a global check, so we update labels and then
                # run the LRT checks
                candidate_labels[split.ids] = split.index

        if self.split_test == 'lrt':
            ### Now that we have all the labels, run the test on label : feature terms
            # we use get dummies here because the one hot encoder isn't built yet!
            cell_dummies = pandas.get_dummies(candidate_labels, sparse=True).sparse.to_coo().tocsc()
            Xc = numpy.column_stack((numpy.ones_like(y_global), X_global))
            features = sparse.hstack([col.T.multiply(Xc) for col in cell_dummies.T]).tocsr()

            n_samples, n_features = Xc.shape
            n_samples, n_new_features = features.shape
            n_old_features = len(numpy.unique(self.labels_))*n_features

            # while this results in different t-tests, this gives
            # the same f-stat.
            beta, *_ = sparse.linalg.lsqr(features, y_global[:,None])
            new_prediction = features @ beta


            ssq_new = ((y_global - new_prediction)**2).sum()
            ssq_current = ((y_global - self.prediction_)**2).sum()
            # each additional leaf introduces n_features effects, at worst case
            # and intercept is assumed throughout
            sigma2_new = ssq_new/(n_samples - n_new_features)
            sigma2_old = ssq_current/(n_samples - n_old_features)
            l_new = -n_samples*numpy.log(numpy.sqrt(sigma2_new)) - .5*(1/sigma2_new) * ssq_new
            l_old = -n_samples*numpy.log(numpy.sqrt(sigma2_old)) - .5*(1/sigma2_old) * ssq_current
            lr_stat = -2 * (l_old - l_new)
            # dof is the number of constrained features. Here, we're constraining
            # n_features interaction effects to be zero in the "restricted" model,
            # or adding n_features to the model to capture this specific leaf.
            p = stats.chi2.sf(lr_stat, n_new_features - n_old_features)
            is_improvement = p < self.split_tolerance

            retain_split &= is_improvement

        retained_splits = []
        for split, retain in zip([left,right], retain_split):
            if retain:
                #print(f"Split {split.index} is retained")
                self._update_map(split, leaf=False)
                retained_splits.append(split)
            else:
                #print(f"Split {split.index} is not retained")
                self._update_map(parent, leaf=True)
        return retained_splits

class KDTreeBoostingRegressor(KDTreeRegressor):
    """
    Boosting version of the KDTreeRegressor, setting defaults
    boost = True, split_test='eps', and prune=True.
    Consult KDTreeRegressor for specifics.
    """
    def __init__(self, *, boost=True, **kwargs):
        kwargs['split_test'] = 'eps'
        kwargs.setdefault("prune", True)
        super().__init__(**kwargs)

class KDTreeClassifier(QuadtreeRegressor):
    """
    Classifier version of the quadtree regression, setting defaults
    to model=LogisticRegression(), split_test='eps', and split_test='eps'.
    Consult KDTreeRegressor for specifics.
    """
    def __init__(self, *, model=None, score_function=metrics.accuracy_score, split_test='eps', prune=False, **kwargs):
        if model is None:
            model = linear_model.LogisticRegression()
        if split_test != "eps":
            raise NotImplementedError("only 'eps' splitting is supported with classifiers.")
        if prune not in ("perm", True, False):
            raise NotImplementedError("only 'perm' splitting is supported with classifiers.")
        super().__init__(model=model, score_function=score_function, split_test=split_test, prune=prune)


def _weighted_median(x, weights):
    df = pandas.DataFrame.from_dict(dict(data=x, weight=weights))
    df.sort_values("data", inplace=True)
    cutpoint = df.weight.sum() / 2
    wsums = df.weight.cumsum()
    return df.data[wsums >= cutpoint].iloc[0]

KDTreeEnsembleRegressor = KDTreeRegressor
KDTreeEnsembleRegressor.predict = KDTreeEnsembleRegressor._predict_local
KDTreeEnsembleClassifier = KDTreeClassifier
KDTreeEnsembleClassifier.predict = KDTreeEnsembleClassifier._predict_local
