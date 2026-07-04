"""Quasi-random (low-discrepancy) sequence generators.

Used as drop-in replacements for uniform random draws in spatial sampling.
All generators return values in [0, 1]^d and accept a plain integer ``seed``
so the parent sampler's ``random_state`` controls reproducibility.

Sequences
---------
sobol   scipy.stats.qmc.Sobol (scrambled)         -- base-2 digital net
halton  scipy.stats.qmc.Halton (scrambled)        -- mixed-radix Van der Corput
r2      Roberts' R2 / golden-ratio sequence        -- pure NumPy, no scipy

R2 implementation notes
-----------------------
Follows the formula in Roberts (2018)
"The Unreasonable Effectiveness of Quasirandom Sequences"
https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    phi_d  = unique positive root of  x^(d+1) = x + 1
    alpha  = [1/phi_d, 1/phi_d^2, ..., 1/phi_d^d]
    t[n]   = (seed + n * alpha) mod 1,   n = 1, 2, 3, ...

The scalar ``seed`` (0.5 by default, as recommended by Roberts) is shared
across all d dimensions.  For reproducible scrambling we replace the fixed
0.5 with a single random scalar derived from ``random_state``.
"""

import warnings
from scipy import stats
import numpy

VALID_SEQUENCES = frozenset({"sobol", "halton", "r2"})


def _check(sequence: str) -> None:
    if sequence not in VALID_SEQUENCES:
        raise ValueError(
            f"Unknown quasi_random sequence {sequence!r}. "
            f"Choose from {sorted(VALID_SEQUENCES)}."
        )


def _phi(d: int) -> float:
    """Compute phi_d: the unique positive root of x^(d+1) = x + 1.

    Uses the iterative fixed-point method from Roberts (2018):
        x = (1 + x)^(1/(d+1))   repeated until convergence.

    d=1 -> golden ratio ≈ 1.6180339887
    d=2 -> plastic constant ≈ 1.3247179572

    Convergence rate is |1/(d+1) * phi_d^(-d)| per step.  d=2 converges in
    ~20 iterations; d=1 (square-root iteration) is slower and needs ~50 to
    reach float64 precision.  We run 60 iterations to be safe for all d.
    """
    x = 2.0
    for _ in range(60):
        x = (1 + x) ** (1 / (d + 1))
    return x


# ---------------------------------------------------------------------------
# 2-D generators -- return (n, 2) arrays in [0, 1]^2
# ---------------------------------------------------------------------------


def _sobol_2d(n: int, seed: int) -> numpy.ndarray:
    """Scrambled Sobol sequence.  Rounds up to next power of 2 internally."""
    m = int(2 ** numpy.ceil(numpy.log2(max(n, 1))))
    sampler = stats.qmc.Sobol(d=2, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts = sampler.random(m)
    return pts[:n]


def _halton_2d(n: int, seed: int) -> numpy.ndarray:
    return stats.qmc.Halton(d=2, scramble=True, seed=seed).random(n)


def _r2_2d(n: int, seed: int) -> numpy.ndarray:
    """Roberts R2 sequence (d=2).

    A single scalar offset is shared across both dimensions, matching the
    formula  t[n] = (seed + n * alpha) mod 1  from Roberts (2018).
    The sequence starts at n=1 (not n=0) to avoid placing the first point
    on the diagonal at (seed, seed).
    """
    phi2 = _phi(2)
    alpha = numpy.array([1.0 / phi2, 1.0 / phi2**2])
    # Single shared scalar seed -- same offset for x and y
    offset = float(numpy.random.default_rng(seed).random())
    return (offset + numpy.outer(numpy.arange(1, n + 1), alpha)) % 1.0


def qrn_2d(n: int, sequence: str, seed: int) -> numpy.ndarray:
    """Dispatch to a 2-D quasi-random generator. Returns shape (n, 2)."""
    _check(sequence)
    if sequence == "sobol":
        return _sobol_2d(n, seed)
    if sequence == "halton":
        return _halton_2d(n, seed)
    return _r2_2d(n, seed)


# ---------------------------------------------------------------------------
# 1-D generators -- return (n,) arrays in [0, 1]
# ---------------------------------------------------------------------------


def _sobol_1d(n: int, seed: int) -> numpy.ndarray:
    m = int(2 ** numpy.ceil(numpy.log2(max(n, 1))))
    sampler = stats.qmc.Sobol(d=1, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sampler.random(m)[:n, 0]


def _halton_1d(n: int, seed: int) -> numpy.ndarray:
    return stats.qmc.Halton(d=1, scramble=True, seed=seed).random(n)[:, 0]


def _r2_1d(n: int, seed: int) -> numpy.ndarray:
    """Roberts R1 sequence -- golden-ratio additive recurrence (d=1).

    t[n] = (seed + n / phi_1) mod 1,   n = 1, 2, 3, ...
    """
    phi1 = _phi(1)  # golden ratio ≈ 1.6180339887498948482
    offset = float(numpy.random.default_rng(seed).random())
    return (offset + numpy.arange(1, n + 1) / phi1) % 1.0


def qrn_1d(n: int, sequence: str, seed: int) -> numpy.ndarray:
    """Dispatch to a 1-D quasi-random generator. Returns shape (n,)."""
    _check(sequence)
    if sequence == "sobol":
        return _sobol_1d(n, seed)
    if sequence == "halton":
        return _halton_1d(n, seed)
    return _r2_1d(n, seed)
