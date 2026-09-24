"""Shared internal helpers used across sampler modules."""

from contextlib import contextmanager

import numpy
import shapely


# ---------------------------------------------------------------------------
# Public dispatch entry-point
# ---------------------------------------------------------------------------

def _sample_geometry(geometry, n_samples: int, rng: numpy.random.RandomState,
                     quasi_random: str | None = None) -> list:
    """Return *n_samples* points drawn uniformly at random inside/along *geometry*.

    Dispatch table
    --------------
    Point / MultiPoint                          -> return the point(s) directly
    LineString / LinearRing / MultiLineString   -> uniform arc-length sampling
    Polygon / MultiPolygon                      -> rejection sampling

    Parameters
    ----------
    quasi_random : str or None
        One of ``"sobol"``, ``"halton"``, ``"r2"``, or ``None`` (fully random).
        Applies to continuous coordinate generation only; ignored for
        Point / MultiPoint geometry types.
    """
    geom_type = geometry.geom_type

    if geom_type == "Point":
        return [geometry] * n_samples

    if geom_type == "MultiPoint":
        geoms = list(geometry.geoms)
        idx = rng.choice(len(geoms), n_samples, replace=True)
        return [geoms[i] for i in idx]

    if geom_type in ("LineString", "LinearRing", "MultiLineString"):
        return _sample_line(geometry, n_samples, rng, quasi_random)

    if shapely.area(geometry) == 0:
        raise ValueError(
            f"Cannot rejection-sample from a zero-area {geom_type!r} geometry."
        )

    if quasi_random is not None:
        return _sample_polygon_qrn(geometry, n_samples, rng, quasi_random)
    return _sample_polygon_random(geometry, n_samples, rng)


# ---------------------------------------------------------------------------
# Polygon samplers
# ---------------------------------------------------------------------------

def _sample_polygon_random(geometry, n_samples: int,
                            rng: numpy.random.RandomState) -> list:
    """Adaptive batch rejection sampling (Shapely 2 vectorised)."""
    shapely.prepare(geometry)
    minx, miny, maxx, maxy = geometry.bounds

    pts: list = []
    remaining = n_samples
    batch = max(remaining * 4, 256)

    while remaining > 0:
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        candidates = shapely.points(xs, ys)
        inside = candidates[shapely.contains(geometry, candidates)]

        take = min(inside.size, remaining)
        pts.extend(inside[:take].tolist())
        remaining -= take

        if inside.size > 0:
            rate = inside.size / batch
            batch = max(int(remaining / rate * 1.5), 256)
        else:
            batch *= 2

    return pts


def _sample_polygon_qrn(geometry, n_samples: int,
                         rng: numpy.random.RandomState, sequence: str) -> list:
    """Quasi-random rejection sampling for polygons.

    Generates a single batch of QRN candidates sized for the expected
    acceptance rate.  Any shortfall (e.g. very thin geometries) is topped
    up with ordinary random sampling so the count is always exact.
    """
    from ._quasi import qrn_2d

    shapely.prepare(geometry)
    minx, miny, maxx, maxy = geometry.bounds
    width, height = maxx - minx, maxy - miny

    coverage = shapely.area(geometry) / (width * height)
    n_cands = max(int(numpy.ceil(n_samples / max(coverage, 0.01) * 2)), n_samples + 256)

    seed = int(rng.randint(0, 2 ** 31))
    xy = qrn_2d(n_cands, sequence, seed)
    xs = minx + xy[:, 0] * width
    ys = miny + xy[:, 1] * height
    candidates = shapely.points(xs, ys)
    inside = candidates[shapely.contains(geometry, candidates)]

    if inside.size >= n_samples:
        return inside[:n_samples].tolist()

    # Coverage lower than estimated -- top up with random for the remainder.
    extra = _sample_polygon_random(geometry, n_samples - inside.size, rng)
    return inside.tolist() + extra


# ---------------------------------------------------------------------------
# Line sampler
# ---------------------------------------------------------------------------

def _sample_line(geometry, n_samples: int, rng: numpy.random.RandomState,
                 quasi_random: str | None = None) -> list:
    """Sample *n_samples* points uniformly by arc length along a line geometry.

    Works on LineString, LinearRing, and MultiLineString.  For MultiLineString
    the distances are cumulative across segments so gaps are skipped and the
    sampling remains uniform in arc length.

    When *quasi_random* is set the arc-length distances are drawn from the
    requested 1-D quasi-random sequence instead of a uniform RNG.
    """
    total_length = shapely.length(geometry)
    if total_length == 0:
        raise ValueError(
            f"Cannot sample from a zero-length {geometry.geom_type!r} geometry."
        )
    if quasi_random is not None:
        from ._quasi import qrn_1d
        seed = int(rng.randint(0, 2 ** 31))
        distances = qrn_1d(n_samples, quasi_random, seed) * total_length
    else:
        distances = rng.uniform(0, total_length, n_samples)
    return shapely.line_interpolate_point(geometry, distances).tolist()


# ---------------------------------------------------------------------------
# Integer allocation (largest-remainder / Hamilton method)
# ---------------------------------------------------------------------------

def _allocate_proportionally(proportions: numpy.ndarray, n_total: int) -> numpy.ndarray:
    """Allocate *n_total* items across bins according to *proportions*.

    Uses the largest-remainder method so the sum equals *n_total* exactly.
    """
    proportions = numpy.asarray(proportions, dtype=float)
    raw = proportions * n_total
    floor = numpy.floor(raw).astype(int)
    deficit = n_total - floor.sum()
    remainder = raw - floor
    top = numpy.argsort(remainder)[::-1][:deficit]
    floor[top] += 1
    return floor


# ---------------------------------------------------------------------------
# Rasterio context manager
# ---------------------------------------------------------------------------

@contextmanager
def _open_raster(source):
    """Yield a rasterio ``DatasetReader``, closing it only if *we* opened it."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for raster-based sampling. "
            "Install it with:  pip install rasterio"
        ) from None

    if isinstance(source, rasterio.DatasetReader):
        yield source
    else:
        with rasterio.open(source) as ds:
            yield ds
