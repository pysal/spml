.. _reference:

API reference
=============

The API reference provides an overview of all public functions in ``spatialml``.

Base classes
------------

Base classes allow creation of geographically weighted versions of scikit-learn
estimators.

.. currentmodule:: spatialml.base
.. autosummary::
   :toctree: generated/

   BaseClassifier
   BaseRegressor

Linear models
-------------

Implementation of linear models with access to relevant attributes (e.g. local
coefficients).

.. currentmodule:: spatialml.linear_model
.. autosummary::
   :toctree: generated/

   GWLinearRegression
   GWLogisticRegression

Ensemble models
---------------

Implementation of linear models with access to relevant attributes (e.g. local
feature importance).


.. currentmodule:: spatialml.ensemble
.. autosummary::
   :toctree: generated/

   GWGradientBoostingClassifier
   GWGradientBoostingRegressor
   GWRandomForestClassifier
   GWRandomForestRegressor


Bandwidth search
----------------

Tooling to determine the optimal bandwidths of geographically weighted models.

.. currentmodule:: spatialml.search
.. autosummary::
   :toctree: generated/

   BandwidthSearch