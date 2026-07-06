from abc import abstractmethod

import geopandas
from sklearn.base import BaseEstimator


class BasePointSampler(BaseEstimator):
    """Abstract base for all point samplers.

    Inheriting from ``sklearn.base.BaseEstimator`` provides ``get_params`` /
    ``set_params`` and a clean ``__repr__`` for free.  Concrete subclasses must
    implement :meth:`sample`.
    """

    @abstractmethod
    def sample(self, geometry, **kwargs) -> geopandas.GeoDataFrame:
        """Generate sampled point locations.

        Parameters
        ----------
        geometry
            The spatial source to sample from.  Accepted types depend on the
            concrete sampler (Shapely geometry, GeoDataFrame, rasterio dataset,
            or a file path).

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame whose ``geometry`` column contains the sampled
            ``Point`` geometries.  Additional columns (e.g. ``class_label``,
            ``intensity``) are added by subclasses.
        """
