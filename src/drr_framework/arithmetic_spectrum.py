"""Offline numerical illustration of the Chebyshev explicit formula."""

import numpy as np

# First ten positive zeta-zero ordinates, rounded to ten decimals.
# Source: https://www.lmfdb.org/zeros/zeta/
ZERO_ORDINATES = np.array(
    [
        14.1347251417,
        21.0220396388,
        25.0108575801,
        30.4248761259,
        32.9350615877,
        37.5861781588,
        40.9187190121,
        43.3270732809,
        48.0051508812,
        49.7738324777,
    ]
)


def chebyshev_table(limit: int) -> np.ndarray:
    """Return psi(n) for integer n from 0 through limit, including prime powers."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    prime = np.ones(limit + 1, dtype=bool)
    prime[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if prime[p]:
            start = p * p
            prime[start::p] = False
    weights = np.zeros(limit + 1)
    for p in np.flatnonzero(prime):
        q = int(p)
        while q <= limit:
            weights[q] = np.log(p)
            q *= int(p)
    return np.cumsum(weights)


def reconstruct(x: np.ndarray, count: int) -> np.ndarray:
    """Truncated symmetric zero sum away from prime-power discontinuities."""
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 1):
        raise ValueError("x must be finite and greater than one")
    if not isinstance(count, (int, np.integer)) or not 0 <= count <= len(ZERO_ORDINATES):
        raise ValueError("invalid zero count")
    rho = 0.5 + 1j * ZERO_ORDINATES[:count]
    terms = np.exp(np.log(x)[..., None] * rho) / rho
    inverse_square = np.power(x, -2)
    correction = 0.5 * np.log1p(-inverse_square)
    return x - np.log(2 * np.pi) - correction - 2 * np.real(terms.sum(axis=-1))


def benchmark() -> dict:
    """Fixed-grid exploratory reconstruction; not an out-of-sample forecast."""
    x = np.arange(10, 1001) + 0.5
    truth = chebyshev_table(1000)[10:]
    rows = []
    for count in (0, 1, 4, 10):
        residual = (reconstruct(x, count) - truth) / np.sqrt(x)
        rows.append({"zeros": count, "normalized_rmse": float(np.sqrt(np.mean(residual**2)))})
    return {"grid": "10.5 to 1000.5, step 1", "samples": len(x), "cases": rows}
