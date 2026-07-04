.. _reference:

API reference
=============

The API reference provides an overview of all public functions in ``spml``.

Base classes
------------

Base classes allow creation of geographically weighted versions of scikit-learn
estimators.

.. currentmodule:: spml.base
.. autosummary::
   :toctree: generated/

   BaseClassifier
   BaseRegressor

Linear models
-------------

Implementation of linear models with access to relevant attributes (e.g. local
coefficients).

.. currentmodule:: spml.linear_model
.. autosummary::
   :toctree: generated/

   GWLinearRegression
   GWLogisticRegression

Ensemble models
---------------

Implementation of linear models with access to relevant attributes (e.g. local
feature importance).


.. currentmodule:: spml.ensemble
.. autosummary::
   :toctree: generated/

   GWGradientBoostingClassifier
   GWGradientBoostingRegressor
   GWRandomForestClassifier
   GWRandomForestRegressor


Bandwidth search
----------------

Tooling to determine the optimal bandwidths of geographically weighted models.

.. currentmodule:: spml.search
.. autosummary::
   :toctree: generated/

   BandwidthSearch

Validation
----------

Spatial cross-validation splitters, point samplers, and range-finding
utilities for validating spatial models.

.. currentmodule:: spml.validation
.. autosummary::
   :toctree: generated/

   BallKFold
   CellStratifiedKFold
   ClusterStratifiedKFold
   HilbertKFold
   LeaveBallOut
   LeaveCellOut
   LeaveClusterOut
   LocalBootstrap
   LocalPermutation
   correlogram_range
   knn_range
   PointSampler
   ConstantClassSampler
   StratifiedClassSampler
   MultinomialSampler
   PoissonSampler

Metrics
-------

Metrics for evaluating spatial sampler and cross-validation outputs.
Regionalization and autocorrelation metrics from ``esda`` (e.g.
``completeness``, ``boundary_silhouette``, ``correlogram``) are also
re-exported here.

.. currentmodule:: spml.metrics
.. autosummary::
   :toctree: generated/

   area_of_applicability
   gearygram

Preprocessing
-------------

Spatial feature engineering transformers for scikit-learn pipelines.

.. currentmodule:: spml.preprocessing
.. autosummary::
   :toctree: generated/

   KNeighborsFeatures
   RadiusNeighborsFeatures