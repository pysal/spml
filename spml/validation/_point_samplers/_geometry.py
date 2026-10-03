import geopandas as gpd
from sklearn.utils import check_random_state

from ._base import BasePointSampler
from ._utils import _collect_geometries, _sample_geometry


class PointSampler(BasePointSampler):
    """Sample points uniformly at random inside a Shapely geometry.

    Mimics the sklearn model-selection estimator API: hyperparameters are set
    in ``__init__`` and :meth:`sample` acts as the primary callable.

    Parameters
    ----------
    n_samples : int, default 100
        Number of points to generate.
    quasi_random: str or None
        whether to use a fully random sequence (default), or choose between
        three supported quasi-random spatial sequences:
        the ``sobol``, ``halton``, or ``r2`` sequences. Quasi-random sequences are
        more dispersed/space-filling than truly random sequences.
    random_state : int, RandomState instance, or None, default None
        Seed / random state passed to ``sklearn.utils.check_random_state``.
        TODO: all random_state docstrings across spml shoud say the same

    Examples
    --------
    >>> from shapely.geometry import box
    >>> from spml.validation import PointSampler
    >>> pts = PointSampler(n_samples=200, random_state=0).sample(box(0, 0, 1, 1))
    >>> len(pts)
    200
    """

    def __init__(
        self, n_samples: int = 100, quasi_random: str | None = None, random_state=None
    ):
        self.n_samples = n_samples
        self.quasi_random = quasi_random
        self.random_state = random_state

    def sample(
        self,
        geometry,
        n_samples: int | None = None,
    ) -> gpd.GeoDataFrame:  # ty:ignore[invalid-method-override]
        """Sample ``n_samples`` points uniformly inside ``geometry``.

        Parameters
        ----------
        geometry : shapely.Geometry | geopandas.GeoSeries | geopandas.GeoDataFrame
            Region to sample from.  A GeoSeries / GeoDataFrame is treated as
            a single union for sampling. CRS is inferred automatically.
        n_samples : int, optional
            Overrides ``self.n_samples`` for this call.

        Returns
        -------
        geopandas.GeoDataFrame
            Single-column ``geometry`` GeoDataFrame of sampled Points.
        """
        rng = check_random_state(self.random_state)
        n = n_samples if n_samples is not None else self.n_samples
        crs = None

        if isinstance(geometry, gpd.GeoDataFrame | gpd.GeoSeries):
            crs = geometry.crs
            geometry = _collect_geometries(geometry.geometry.to_numpy())

        pts = _sample_geometry(geometry, n, rng, self.quasi_random)
        return gpd.GeoDataFrame(geometry=pts, crs=crs)
