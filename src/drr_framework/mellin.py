"""Finite-scale arithmetic and Mellin diagnostics; not a zero-free certificate."""

import numpy as np
from scipy.special import gammainc, gammaincc

from .arithmetic_spectrum import ZERO_ORDINATES, chebyshev_table

EULER_GAMMA = 0.5772156649015329


def _grid(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size or np.any(~np.isfinite(x)) or np.any(x < 1):
        raise ValueError("x must be a nonempty finite vector with x >= 1")
    return x


def smoothed_prime_sum(x, character=None, log_derivative=None):
    """Evaluate G or G_chi (paper equations 2.1/2.2) on a bounded grid.

    For G_chi supply a periodic character table indexed by n mod q, and
    L'(1, chi)/L(1, chi). Character validity and normalization are the
    caller's responsibility. No estimate of that constant is hidden here.
    """
    x = _grid(x)
    if x.max() > 1_000_000:
        raise ValueError("sieve grid limited to 1,000,000")
    limit = max(2, int(x.max()))
    n = np.arange(1, limit + 1)
    weights = np.diff(chebyshev_table(limit))
    if character is None:
        if log_derivative is not None:
            raise ValueError("log_derivative requires a character")
        trend = -x * np.log(x) + (1 + EULER_GAMMA) * x
    else:
        chi = np.asarray(character, dtype=complex)
        if chi.ndim != 1 or chi.size < 2 or not np.isfinite(chi).all():
            raise ValueError("provide a finite periodic character table")
        if log_derivative is None or not np.isfinite(log_derivative):
            raise ValueError("provide finite L'(1, chi)/L(1, chi)")
        weights = weights * chi[n % chi.size].real
        trend = float(np.real(log_derivative)) * x
    index = np.floor(x).astype(int) - 1
    return x * np.cumsum(weights / n)[index] - np.cumsum(weights)[index] + trend


def smoothed_zero_sum(x, count, trivial_terms=20):
    """Truncated residue expansion for G; first known zeros, not estimated ones."""
    x = _grid(x)
    if not isinstance(count, (int, np.integer)) or not 0 <= count <= len(ZERO_ORDINATES):
        raise ValueError("invalid zero count")
    if not isinstance(trivial_terms, (int, np.integer)) or not 0 <= trivial_terms <= 1000:
        raise ValueError("invalid trivial zero count")
    rho = 0.5 + 1j * ZERO_ORDINATES[:count]
    terms = np.exp(np.log(x)[:, None] * rho) / (rho * (rho - 1))
    k = np.arange(1, trivial_terms + 1, dtype=float)
    trivial = np.exp(-2 * np.log(x)[:, None] * k) / (2 * k * (2 * k + 1))
    return np.log(2 * np.pi) - 2 * terms.sum(axis=1).real - trivial.sum(axis=1)


def log_spectrum(x, y, beta=0.5):
    """One-sided amplitude spectrum of x**(-beta) y on a uniform log grid."""
    x = _grid(x)
    y = np.asarray(y, dtype=float)
    if y.shape != x.shape or x.size < 4 or not np.isfinite(y).all():
        raise ValueError("provide matching finite observations (at least four)")
    u = np.log(x)
    du = np.diff(u)
    if du[0] <= 0 or not np.allclose(du, du[0], rtol=1e-7, atol=0):
        raise ValueError("x must be strictly increasing and uniformly log-spaced")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    z = y * np.exp(-beta * u)
    if not np.isfinite(z).all():
        raise ValueError("weighted signal is not finite")
    amplitude = np.abs(np.fft.rfft(z)) / x.size
    amplitude[1:] *= 2
    if x.size % 2 == 0:
        amplitude[-1] /= 2
    return 2 * np.pi * np.fft.rfftfreq(x.size, du[0]), amplitude


def window_excess(x, y, theta, bound):
    """Sampled one-sided inequality diagnostic; no interpolation or proof."""
    x = _grid(x)
    y = np.asarray(y, dtype=float)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError("provide matching finite observations")
    if not np.isfinite(theta) or not np.isfinite(bound) or bound < 0:
        raise ValueError("theta must be finite and bound finite and nonnegative")
    normalized = y * np.exp(-theta * np.log(x))
    if not np.isfinite(normalized).all():
        raise ValueError("normalized signal is not finite")
    return {
        "samples": x.size,
        "max_normalized": float(normalized.max()),
        "violations": int(np.count_nonzero(normalized > bound)),
        "sampled_only": True,
    }


def gamma_window_mass(k, sigma, lower, upper):
    """Mass in a log-scale window for the normalized gamma kernel of Section 2."""
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if not np.isfinite([sigma, lower, upper]).all() or sigma <= 1 or not 0 <= lower <= upper:
        raise ValueError("require sigma > 1 and finite 0 <= lower <= upper")
    a = k + 1
    lo, hi = (sigma - 1) * lower, (sigma - 1) * upper
    if lo > a:
        return float(gammaincc(a, lo) - gammaincc(a, hi))
    return float(gammainc(a, hi) - gammainc(a, lo))


def benchmark():
    """Fixed arithmetic reconstruction grid; descriptive, not a forecast."""
    x = np.arange(10, 1001) + 0.5
    target = smoothed_prime_sum(x)
    rows = []
    for count in (0, 1, 4, 10):
        error = (smoothed_zero_sum(x, count) - target) / np.sqrt(x)
        rows.append({"zeros": count, "normalized_rmse": float(np.sqrt(np.mean(error**2)))})
    return {"samples": len(x), "cases": rows}


def _mode_inputs(x, y, betas, omegas):
    """Validate a single-mode scan and return centred log scale and scaled target."""
    x = _grid(x)
    y = np.asarray(y, dtype=float)
    if y.shape != x.shape or x.size < 5 or not np.isfinite(y).all():
        raise ValueError("provide at least five finite matching observations")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    b = np.asarray(betas, dtype=float)
    w = np.asarray(omegas, dtype=float)
    if (
        b.ndim != 1
        or w.ndim != 1
        or not b.size
        or not w.size
        or b.size * w.size > 10_000
        or not np.isfinite(b).all()
        or not np.isfinite(w).all()
        or np.any(w <= 0)
    ):
        raise ValueError("provide finite beta and positive omega grids (<= 10,000 pairs)")
    u = np.log(x)
    u -= u.mean()
    scale = np.max(np.abs(y))
    if scale == 0:
        raise ValueError("zero signal has no identifiable mode")
    return u, y / scale, b, w


def log_mode_profile(x, y, betas, omegas):
    """Relative residual energy of every candidate that ``fit_log_mode`` scans.

    Same model, centring, scaling and candidate rules as ``fit_log_mode``,
    evaluated in closed form from 2x2 normal equations instead of one
    least-squares solve per candidate. Rows follow ``betas`` and columns
    ``omegas``; a candidate ``fit_log_mode`` would skip is NaN. The row-major
    ``nanargmin`` selects the candidate ``fit_log_mode`` returns, up to
    floating-point ties; near zero residual the closed form is accurate only
    to about 1e-15 relative energy. Descriptive, like ``fit_log_mode``.
    """
    u, target, b, w = _mode_inputs(x, y, betas, omegas)
    energy = float(target @ target)
    angle = np.outer(u, w)
    usable = np.isfinite(angle).all(axis=0)
    cos, sin = np.cos(angle), np.sin(angle)
    cc, ss, cs = cos * cos, sin * sin, cos * sin
    surface = np.full((b.size, w.size), np.nan)
    for row, beta in enumerate(b):
        exponent = beta * u
        if not np.isfinite(exponent).all() or np.ptp(exponent) > 600:
            continue
        envelope = np.exp(exponent - exponent.max())
        weight = envelope * envelope
        a, d, off = weight @ cc, weight @ ss, weight @ cs
        weighted = envelope * target
        p, q = weighted @ cos, weighted @ sin
        largest = 0.5 * (a + d + np.sqrt((a - d) ** 2 + 4 * off * off))
        determinant = a * d - off * off
        with np.errstate(divide="ignore", invalid="ignore"):
            # fit_log_mode skips designs whose singular-value ratio is <= 1e-10.
            keep = usable & (largest > 0) & (determinant / largest > 1e-20 * largest)
            explained = (d * p * p - 2 * off * p * q + a * q * q) / determinant
        surface[row, keep] = np.maximum(energy - explained[keep], 0.0) / energy
    return surface


def fit_log_mode(x, y, betas, omegas):
    """Least-squares scan for one real conjugate mode; descriptive, not inference.

    Returns the best supplied candidate and relative residual energy. A
    finite grid cannot exclude off-grid modes or certify a spectral pole.
    """
    u, target, b, w = _mode_inputs(x, y, betas, omegas)
    energy = float(target @ target)
    best = None
    best_error = float("inf")
    for beta in b:
        exponent = beta * u
        if not np.isfinite(exponent).all() or np.ptp(exponent) > 600:
            continue
        envelope = np.exp(exponent - exponent.max())
        for omega in w:
            angle = omega * u
            if not np.isfinite(angle).all():
                continue
            design = envelope[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
            coef, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
            if rank < 2 or singular[-1] <= 1e-10 * singular[0]:
                continue
            residual = target - design @ coef
            error = float(residual @ residual) / energy
            if error < best_error:
                best_error = error
                best = {"beta": float(beta), "omega": float(omega), "relative_sse": error}
    if best is None:
        raise ValueError("no numerically identifiable candidate")
    return best
