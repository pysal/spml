import numpy, pandas, geopandas, shapely, warnings
from collections import defaultdict
from pandas import Series, get_dummies
from scipy import sparse, stats
from sklearn import linear_model, metrics, preprocessing
from sklearn.base import clone
from sklearn.inspection import permutation_importance

__all__ = ['QuadtreeRegressor', 'QuadtreeClassifier', 'QuadtreeEnsembleRegressor', 'QuadtreeEnsembleClassifier', 'QuadtreeBoostingRegressor']

class _Cell:

    def __init__(
        self,
        index,
        parent,
        parent_mask,
        coordinates,
        bounds,
        X,
        y,
        ids,
        score_function,
        model,
        pruned=None,
    ):
        """
        Class to contain cells at each level of a QuadtreeRegressor object.
        This class should not be instantiated directly by users.

        index : int
            index of the cell, recording parent relations from left to right.
        parent : int
            index of the cell's parent; matches `index` up to the last digit.
        parent_mask : numpy.ndarray
            boolean array that can be used to slice this cell's data from
            parent.X, parent.y, and parents.coordinates
        coordinates : numpy.ndarray
            locations of coordinates used in the cell. Must be two-dimensional,
            with the second dimension being "2"
        bounds : numpy.ndarray
            the cell's bounding box
        X : numpy.ndarray
            feature matrix containing observations that fall within the cell
        y : numpy.ndarray
            outcome array containing observations that fall within the cell
        ids : numpy.ndarray
            indices of points in the cell. These indices relate to the *full data*,
            not the indices of the parent's data.
        score_function : callable
            function used to calculate the score of the cell's model. negative mean
            squared error used by default.
        model : class instance
            instance of a scikit-learn model supporting both fit and
            predict methods. Should not generally cache global data,
            since a model is instantiated at each node in the tree.
        pruned : bool (default: None)
            whether the leaf has been pruned in the tree. If None, the pruning
            for the cell has not been evaluated. If True, the cell is
            considered "pruned", and should not be used for prediction
            or fitting.
        """
        self.index = index
        self.parent = parent
        self.parent_mask = parent_mask
        self.depth = len(str(int(index)).split(".")[0]) if index>0 else 0
        self.ids = ids
        self.coordinates = coordinates
        self.bounds = bounds
        self.size = X.shape[0]
        self.X = X
        self.y = y
        self.pruned = pruned
        try:
            self._score_function = score_function
            self._model = model
            self.model_ = self._model.fit(self.X, self.y)
            self.prediction_ = model.predict(self.X)
        except ValueError:
            self._score_function = score_function
            self._model = model
            self.model_ = None
            self.prediction_ = numpy.ones_like(y) * numpy.nan

    @property
    def residuals_(self):
        """Construct on demand: self.prediction_ or self.y get overridden in boosting"""
        return self.y - self.prediction_

    @property
    def score_(self):
        """Construct on demand: self.prediction_ or self.y get overridden in boosting"""
        if (len(self.y) == 0) or (len(self.prediction_) == 0):
            return -numpy.inf
        return self._score_function(self.y, self.prediction_)

    def split(self, boost=False):
        """
        Split a cell into its four constituent cells
        """
        min_x, min_y, max_x, max_y = self.bounds
        # mid_x,mid_y = numpy.median(self.coordinates, axis=0) #not quite quadtree, since splitting based on median location
        mid_x, mid_y = min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2

        # Z-shaped! 0: bigx & bigy, 1: smallx & bigy, 2: bigx & smally, 3: smallx & smally
        labels = (self.coordinates[:, 0] <= mid_x) + (
            self.coordinates[:, 1] <= mid_y
        ) * 2

        bounds = [
            [mid_x, mid_y, max_x, max_y],
            [min_x, mid_y, mid_x, max_y],
            [mid_x, min_y, max_x, mid_y],
            [min_x, min_y, mid_x, mid_y],
        ]
        children = []
        if boost:
            y = self.residuals_
        else:
            y = self.y
        for i in range(4):
            mask = labels == i
            ids = self.ids[mask]
            cell = _Cell(
                index=self.index * 10 + (i + 1),
                parent=self.index,
                ids=ids,
                parent_mask=mask,
                coordinates=self.coordinates[mask,],
                bounds=bounds[i],
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


    def query(self, coordinates):
        """
        Query a cell to determine if coordinates in the input
        lie within the cell's geographical boundaries.
        """
        x,y = coordinates.T
        min_x, min_y, max_x, max_y = self.bounds
        in_xrange = (min_x <= x) & (x <= max_x)
        in_yrange = (min_y <= y) & (y <= max_y)
        return in_xrange*in_yrange

class QuadtreeRegressor:
    def __init__(
        self,
        model=None,
        score=None,
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
        Initialize a QuadtreeRegressor.

        Parameters
        ----------
        model : class instance
            instance of a scikit-learn model supporting both fit and
            predict methods. Should not generally cache global data,
            since a model is instantiated at each node in the tree.
        score : callable or None (default: None)
            function to use to score the model, set to the negative
            mean squared error by default.
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

    @staticmethod
    def _build_node(*args, **kwargs):
        return _Cell(*args, **kwargs)

    def _evaluate_splits(self, cell, X, y):
        if self.split_test == "eps":
            return self._evaluate_splits_abs(cell)
        elif self.split_test == "lrt":
            return self._evaluate_splits_lrt(cell, X, y)
        else:
            raise ValueError(f"Divide rule {self.split_test} not understood. Must be either 'lrt', or 'eps'")

    def _evaluate_splits_abs(self, cell):
        """
        Accept splits that improve upon the cell's prediction
        score by a minimum amount.
        """
        splits = cell.split(boost=self.boost)
        self.levels_[cell.depth+1].extend(splits)

        retained_splits = []
        for i, split in enumerate(splits):
            too_deep = split.depth > self.max_depth
            if too_deep:
                #print(f"Cell {split.index} is too high-res {split.depth}")
                self._update_map(cell, leaf=True)
                continue

            big_enough = split.size > self.min_leaf_size
            if not big_enough:
                #print(f"Cell {split.index} is too small {split.size}, {cell.size}")
                self._update_map(cell, leaf=True)
                continue

            parent_subpixel_score = cell._score_function(
                cell.y[split.parent_mask], cell.prediction_[split.parent_mask]
            )
            improvement = split.score_ - parent_subpixel_score
            is_improvement = improvement >= self.split_tolerance
            if is_improvement:
                self._update_map(split, leaf=False)
                retained_splits.append(split)
            else:
                self._update_map(cell, leaf=True)

        return retained_splits

    # TODO: this needs to change for QuadtreeClassifier
    def _evaluate_splits_lrt(self, cell, X, y):
        """
        Accept splits that are statistically significant
        using a likelihood ratio test to compare the split model
        nested within the unsplit model.
        """
        splits = cell.split(boost=self.boost)
        self.levels_[cell.depth+1].extend(splits)
        retain_split = numpy.ones_like(splits).astype(bool)
        candidate_labels = self.labels_.copy()
        for i, split in enumerate(splits):
            too_deep = split.depth > (self.max_depth + 1)
            if too_deep:
                retain_split[i] = False
                continue

            big_enough = split.size > self.min_leaf_size
            if not big_enough:
                retain_split[i] = False
                continue
            candidate_labels[split.ids] = split.index

        # we use get dummies here because the one hot encoder isn't built yet!
        cell_dummies = get_dummies(candidate_labels, sparse=True).sparse.to_coo().tocsc()
        Xc = numpy.column_stack((numpy.ones_like(y), X))
        features = sparse.hstack([col.T.multiply(Xc) for col in cell_dummies.T]).tocsr()

        n_samples, n_features = Xc.shape
        n_samples, n_new_features = features.shape
        n_old_features = len(numpy.unique(self.labels_))*n_features

        # while this results in different t-tests, this gives
        # the same f-stat.
        beta, *_ = sparse.linalg.lsqr(features, y[:,None])
        new_prediction = features @ beta


        ssq_new = ((y - new_prediction)**2).sum()
        ssq_current = ((y - self.prediction_)**2).sum()
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
        for split, retain in zip(splits, retain_split):
            if retain:
                #print(f"Split {split.index} is retained")
                self._update_map(split, leaf=False)
                retained_splits.append(split)
            else:
                #print(f"Split {split.index} is not retained")
                self._update_map(cell, leaf=True)
        return retained_splits

    def fit(self, X, y, coordinates=None):
        """
        Fit a Quadtree Regression on input coordinates, with features X and outcome y.

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
        QuadtreeRegressor() object
        """
        if coordinates is None:
            if X.ndim != 3:
                raise ValueError(
                    "If no coordinates are provided, X must be a raster"
                    " of shape (n_rows, n_cols, n_features)"
                    )
            # assume input is a raster
            rows,cols,bands = X.shape
            rows_,cols_ = y.shape
            if not (rows == rows_)&(cols == cols_):
                raise ValueError("if input is raster, then X and y shapes must align")
            X_c, Y_c = numpy.meshgrid(
                numpy.arange(rows),
                numpy.arange(cols)
            )
            coordinates = numpy.column_stack((X_c.flatten(), Y_c.flatten()))
            X = X.reshape(rows*cols, bands)
            y = y.reshape(rows*cols, 1)
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
        root =  self._build_node(
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
            good_splits = self._evaluate_splits(cell, X, y)
            queue.extend(good_splits)

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

    def _update_map(self, cell, leaf=True):
        """
        Update object state given a cell and leaf flag
        """
        self.labels_[cell.ids] = cell.index
        self.prediction_[cell.ids] = cell.prediction_
        self.scores_[cell.ids] = cell.score_
        self.depth_[cell.ids] = cell.depth
        if leaf:
            self.leaves_[cell.index] = cell
        if (self.pbar is not False):
            self.pbar.update(int(leaf))

    def _predict_local(self, X, coordinates, extrapolate=False):
        """
        makes aggregate predictions on new data by finding the
        lowest cell at which the data exists, and then making
        predictions from that low-cell model. Not currently used,
        implemented in case the local models are boosted.
        """
        lowest_cell = self.query(coordinates)
        n_samples, n_features = X
        output = numpy.empty((n_samples,), dtype=X.dtype)
        if extrapolate: # extrapolate from the root
            lowest_cell[lowest_cell == -1] = 0
        else: # give no predictions outside of area
            output[lowest_cell == -1] = numpy.nan
        for leaf_name, leaf in self.leaves_:
            mask = lowest_cell == leaf_name
            output[mask] = leaf.predict(X)
        return output

    def predict(self, X, coordinates, extrapolate=False):
        """
        Predict values given an input dataset and coordinates.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame
            array of shape (n_samples, n_features) features to be fit.
            An intercept should not be included.
        coordinates: numpy.ndarray
            array of shape (n_samples, 2) where observations occur
        extrapolate: bool (default: False)
            whether or not to extrapolate outside of the tree area. If False,
            then any observations that fall outside of the tree bounds are
            predicted as numpy.nan. If True, all observations outside of the
            tree bounds are predicted using a generic model fit with leaf 0.

        Returns
        -------
        predictions of length (n_samples,) containing the predictions made
        at each of the input coordinates using X features.
        """
        if not isinstance(X, pandas.DataFrame):
            Xc = numpy.column_stack((numpy.ones((X.shape[0],)), X))
            X = pandas.DataFrame(X, columns=self._x_names)
        else:
            X = X.copy()
            X['constant'] = 1
        lowest_cell = self.query(coordinates)
        # extrapolate from the root, censor later
        cell_features = lowest_cell.copy()
        cell_features[cell_features == -1] = 0
        n_samples, n_features = X.shape

        leaf_dummies = self._ohe.transform(cell_features[:,None])
        leaf_name_list = list(self._ohe.categories_[0])
        features = sparse.hstack([
            leaf_dummies[:,leaf_name_list.index(leaf_name)].multiply(X[x_name].values[:,None])
            for leaf_name, x_name in self.features_.columns
            ])

        output = self.model_.predict(features)

        if not extrapolate: # give no predictions outside of area
            output[lowest_cell == -1] = numpy.nan
        else: # predict at the base, even if the base isn't a leaf
            highest_model = self.levels[0][0].model
            output[lowest_cell == -1] = highest_model.predict(X[lowest_cell])

        return output

    def score(self, X=None, y=None, coordinates=None):
        """
        Score of the model after being fit.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame
            array of shape (n_samples, n_features) features to be fit. An intercept should not be included.
        y : numpy.ndarray, pandas.Series, pandas.DataFrame
            array of outcome shaped (n_samples,) to be predicted. In nearly all cases, this should
            be centered on zero for splits to be found successfully.
        coordinates: numpy.ndarray
            array of shape (n_samples, 2) where observations occur

        Returns
        -------
        score (according to self._score_function) for the model.
        """
        if not hasattr(self, "leaves_"):
            raise ValueError("Model must be fit in order to be scored. call the .fit() method first.")
        provided = [X is not None, y is not None, coordinates is not None]
        if any(provided):
            assert all(provided), (
                "either cached values or a full new input X, y, coordinates is needed to be scored."
            )
            prediction = self.predict(X, coordinates)
        else:
            prediction = self.prediction_
        return self._score_function(self.leaves_[0].y, prediction)

    @property
    def n_leaves(self):
        """Dynamically compute the current number of leaves from the leaf dict"""
        return len(self.leaves_)

    def _prune(self, X, y, score=None):
        """
        Prune the feature set using wald or permutation tests for each value interaction term.
        If a term is not significant at the lowest depth, it is "folded" up into
        its parent leaf for that feature.

        That is,
        0. estimate the feature X leaf interaction model
        1. start at the deepest leaf with features.
        2. For each feature in 1, is the feature X leaf interaction term significant?
        2a. if so, keep the feature
        2b. If not, "pool" the values upwards into the parent leaf's feature X leaf
            interaction term and remove the feature from consideration. If the leaf is
            the root of the tree, remove the feature from consideration.
        3. move to the next leaf at the same depth as 1. If no other leafs are available
            at the depth from step 1, go to step 0

        eventually, this either folds all leaves upwards into the root node (and then removes
        them from consideration) or will find some subset of feature x leaf interaction terms
        that are statistically significant. This is like spatial reverse stepwise selection.
        At worst, the test will be evaluated n_leaves * n_features times.
        """
        Xc = numpy.column_stack((numpy.ones_like(y), X))
        n_samples, n_features = Xc.shape
        # levels_ contains *everyone*, while leaves_ only contains the leaf nodes used
        # in a prediction somewhere. so, we *must* use levels_, not leaves_ to prune,
        # since any intermediate node may not be present in leaves_
        all_nodes = {node.index:node for level in self.levels_.values() for node in level}

        self._ohe = preprocessing.OneHotEncoder().fit(
            numpy.asarray(list(all_nodes.keys())).reshape(-1,1)
            )
        node_name_list = list(self._ohe.categories_[0])

        cell_dummies = self._ohe.transform(self.labels_[:,None])
        # must do the leafs, then the variates
        features = sparse.hstack([col.T.multiply(Xc) for col in cell_dummies.T]).tocsr()
        _, n_map_features = features.shape
        # while we can end up folding into non-node nodes...
        node_ixs = numpy.repeat(node_name_list, n_features)
        effect_ixs = numpy.tile(numpy.arange(n_features), len(node_name_list))
        effect_names =  numpy.asarray(['constant'] + list(self._x_names))
        if score is None:
            _, cols = features.nonzero()
            used_cols = numpy.unique(cols)
            used_features = numpy.zeros((n_map_features)).astype(bool)
            used_features[used_cols] = True
            return _df_from_sparse_features(
                features[:,used_features],
                node_ixs[used_features],
                effect_ixs[used_features],
                effect_names
                )

        pruning = True
        reference_dist = None

        if (self.pbar is not False):
            from tqdm import tqdm
            if self.prune == "wald":
                test_name = "Wald"
            else:
                test_name = "Permutation"
            prune_pbar = tqdm(
                total= n_map_features,
                unit = " pruning rounds",
                desc=f"Pruning leaves using {test_name} tests..."
            )
        last_prune = numpy.ones((n_map_features,))*-1
        while pruning:
            is_significant, sig_scores, reference_dist = score(features, y, reference_dist=reference_dist)
            # if a column does not have data, then we ignore it's non-significance!
            # need to set all those outside of these indices to true,
            # or just care about these indices.
            pruned = numpy.isnan(sig_scores)

            if is_significant[~ pruned].all():
                break

            prune_leaf_effects = ((~is_significant) & (~pruned)).nonzero()[0]
            current_depth = -numpy.inf
            for effect_loc in reversed(prune_leaf_effects):
                leaf_ix = node_ixs[effect_loc]
                effect_ix = effect_ixs[effect_loc]
                leaf = all_nodes[leaf_ix]
                if leaf.depth >= current_depth:
                    current_depth = leaf.depth
                else:
                    #print(f"skipping leaf {leaf.index} at depth {leaf.depth}")
                    continue
                if (self.pbar is not False):
                    prune_pbar.update(1)
                if leaf.parent is not None:
                    #print(f"merging leaf {leaf.index} for effect {effect_ix} at position {effect_loc} into {leaf.parent}")
                    parent_iloc = node_name_list.index(leaf.parent)
                    new_effect_loc = (parent_iloc * n_features + effect_ix)


                    #assert node_ixs[new_effect_loc] == leaf.parent
                    #assert effect_ixs[new_effect_loc] == effect_ix

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=sparse.SparseEfficiencyWarning)
                        features[leaf.ids, new_effect_loc] = features[leaf.ids, effect_loc].todense()
                        features[leaf.ids, effect_loc] = 0
                else:
                    # since we've rolled up all the leaves for this feature, we can
                    # remove it from the remaining tree entirely.
                    #print(f"removing effect {effect_ix} from tree entirely...")
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=sparse.SparseEfficiencyWarning)
                        features[:, effect_loc] = 0
            #last_prune = prune_leaf_effects.copy()
            #features.eliminate_zeros()

        keep_feature = is_significant & (~ pruned)
        features_ = features[:,keep_feature]
        features_.eliminate_zeros()
        leaf_ixs_ = node_ixs[keep_feature]
        effect_ixs_ = effect_ixs[keep_feature]
        self._ohe = preprocessing.OneHotEncoder().fit(node_ixs[:,None])
        return _df_from_sparse_features(features_, leaf_ixs_, effect_ixs_, effect_names)

    def _permutation_importance(self, features, y, *, reference_dist=None):
        """
        Compute the permutation feature importance for an input sparse
        design matrix. This returns the binary significance decision at a nonparametric
        two-tailed significance level of self.tolerance, the significance score (i.e.
        the percentile of the full model significance replications at which the
        sub-model's significance score falls), and the reference distribution used
        for the test.

        This densifies the input, and so may not be suitable for very large datasets.
        """
        top_down_model = self.model.fit(features, y)
        imp = permutation_importance(
            top_down_model,
            features.toarray(),
            y,
            n_repeats=self.prune_repeats,
            scoring=metrics.make_scorer(self._score_function),
            n_jobs = self.n_jobs,
            random_state=self.random_state
        )
        if reference_dist is None:
            reference_dist = numpy.abs(imp.importances).flatten()
        percentile = stats.percentileofscore(
            reference_dist,
            # median before absolute value, because we want
            # to measure the net importance, relative to other
            # observed importances
            numpy.abs(numpy.median(imp.importances, axis=1))
        )/100
        sig = percentile > (1 - self.prune_tolerance/2)
        # zero importance deviation means the feature has been pruned
        percentile[imp.importances_std == 0] = numpy.nan
        return sig, percentile, reference_dist

    def _sparse_wald_feature_importance(self, features, y, **__):
        """
        Compute the Wald test feature importance for an input sparse
        feature matrix. This does not densify, and is appropriate for larger data.
        This computes the classic two-tailed wald t-test for significant regression
        coefficients, and is only applicable for linear regression.
        """
        n_samples, n_map_features = features.shape
        beta, *_, var = sparse.linalg.lsqr(features, y, calc_var=True)
        n_used_features = (var > 0).sum() # pruned features will have var == 0
        #print(n_used_features)
        self._coefs_ = beta
        resids = y - features @ beta
        sigma2 = (resids @ resids.T)/(n_samples - n_used_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t = numpy.divide(beta, numpy.sqrt(var*sigma2))
        p_t = stats.t.sf(numpy.abs(t), n_samples - n_used_features)
        sig = p_t < (self.prune_tolerance/2)
        return sig, p_t, t

    def query(self, coordinates, return_all=False):
        """
        Query to identify the lowest cell that contains the coordinate input.

        Parameters
        ----------
        coordinates: numpy.ndarray
            array of shape (n_samples, 2) where observations occur
        return_all: bool
            whether to return all leaves that intersect with a given
            input coordinate, or only the *lowest* cell. If return_all=True,
            then all cells that contain the coordinate returned. Otherwise,
            only the *lowest* cell (i.e. biggest depth) of cell is returned.

        Returns
        -------
        numpy.ndarray of shape (n_samples,) containing the index
        of the lowest cell containing each point, using -1 to indicate
        that a point is outside of the bounds of the tree.

        If return_all is True, the full table of (n_samples, n_leaves) will
        be provided.
        """

        leaf_names = list(self.leaves_.keys())
        depthsort = numpy.argsort([leaf.depth for leaf in self.leaves_.values()])[::-1]
        leaves_by_depth = {leaf_names[ix]:self.leaves_[leaf_names[ix]] for ix in depthsort}
        query_result = pandas.concat([
        pandas.Series(
            pandas.arrays.SparseArray(
                leaf.query(coordinates)),
            name=leaf.index
            )
            for leaf in leaves_by_depth.values()], axis=1)
        query_result.loc[:,-1] = True # put an out-of-bounds marker at the end
        if not return_all:
            # TODO: this used to work, but now does not. figure out
            # when this broke and swap back when pandas works.
            #query_result = query_result.idxmax(axis=1)
            query_result = query_result.columns[query_result.values.argmax(axis=1)]
        return query_result.values

    @property
    def local_coefs_(self):
        return pandas.Series(self.model_.coef_, index=self.features_.columns)

class QuadtreeBoostingRegressor(QuadtreeRegressor):
    """
    Boosting version of the QuadtreeRegressor, setting defaults
    boost = True, split_test='eps', and prune=True.
    Consult QuadtreeRegressor for specifics.
    """
    def __init__(self, *, boost=True, **kwargs):
        kwargs['split_test'] = 'eps'
        kwargs.setdefault("prune", True)
        super().__init__(**kwargs)

class QuadtreeClassifier(QuadtreeRegressor):
    """
    Classifier version of the quadtree regression, setting defaults
    to model=LogisticRegression(), split_test='eps', and split_test='eps'.
    Consult QuadtreeRegressor for specifics.
    """
    def __init__(self, *, model=None, score_function=metrics.accuracy_score, split_test='eps', prune=False, **kwargs):
        if model is None:
            model = linear_model.LogisticRegression()
        if split_test != "eps":
            raise NotImplementedError("only 'eps' splitting is supported with classifiers.")
        if prune not in ("perm", True, False):
            raise NotImplementedError("only 'perm' splitting is supported with classifiers.")
        super().__init__(model=model, score_function=score_function, split_test=split_test, prune=prune)

def _df_from_sparse_features(features, leaf_ixs, effect_ixs, effect_names):
    """
    This builds a sparse dataframe from a set of input indices and a
    sparse matrix.
    """
    df = pandas.concat([
        pandas.Series(
    pandas.arrays.SparseArray.from_spmatrix(col.T)
        )
    for col in features.T
    ], axis=1)
    df.columns = pandas.MultiIndex.from_arrays(
        (leaf_ixs.astype(int), effect_names[effect_ixs.astype(int)]),
        names = ("leaf", "effect")
    )
    return df

QuadtreeEnsembleRegressor = QuadtreeRegressor
QuadtreeEnsembleRegressor.predict = QuadtreeEnsembleRegressor._predict_local
QuadtreeEnsembleClassifier = QuadtreeClassifier
QuadtreeEnsembleClassifier.predict = QuadtreeEnsembleClassifier._predict_local
