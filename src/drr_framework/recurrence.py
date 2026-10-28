"""Recurrence quantification and second-order linear recurrences.

Two classical ways to describe repeating structure in one series.

**Recurrence plots.** Delay coordinates ``v_t = (x_t, x_{t + tau}, ...,
x_{t + (d - 1) tau})`` reconstruct a trajectory, and ``R_ij = 1`` marks a
recurrence when ``||v_i - v_j|| <= epsilon`` (Eckmann, Kamphorst and Ruelle 1987).
Pairs closer in time than a Theiler window are excluded, so that a smooth
trajectory sampled finely is not counted as returning to itself. ``epsilon`` is
chosen to fix the recurrence rate, which keeps the measures comparable across
series (Marwan et al. 2007, *Phys. Rep.* 438, 237):

* ``determinism``: share of recurrent points on diagonal lines of at least
  ``min_line`` points. A diagonal line means two stretches of the trajectory
  evolve alike for that many steps.
* ``laminarity``: share of recurrent points on vertical lines of at least
  ``min_line`` points, states that persist.
* ``mean_diagonal_length``: average length of those diagonal lines.
* ``recurrence_period``: the first time offset beyond the Theiler window whose
  recurrence density is within 90% of the highest (multiples of a period recur
  almost as densely), reported only when that density is at least twice the
  median over offsets. For an oscillation it estimates the period without a
  Fourier transform.

These measures depend on distances in the reconstructed space, so unlike the
spectrum they are not fixed by second-order statistics. They are also sensitive
to noise, to the embedding, and to record length;
``drr_framework.resonance_math_benchmark`` measures how far that sensitivity goes.

**Linear recurrences.** A second-order linear recurrence ``x_t = a1 x_{t-1} +
a2 x_{t-2}`` (Fibonacci is ``a1 = a2 = 1``) has characteristic polynomial
``z^2 - a1 z - a2``. Complex roots ``rho exp(+-i omega)`` make it oscillate at
``omega / 2 pi`` cycles per sample while shrinking by ``rho`` per sample.
Driven by noise it is the AR(2) process, the simplest stochastic resonator.
Fitting it by least squares gives a frequency, a damping time ``-1 / (fs ln
rho)`` and a quality factor ``pi f`` times the damping time. Additive
measurement noise biases ``rho`` toward zero (an errors-in-variables effect), so
a noisy tone looks more damped than it is: for a unit tone at 0.05 cycles per
sample, noise of standard deviation 0.1 already moves the fitted frequency by
about 20%, and from 0.2 the fitted roots are no longer complex
(``tests/test_recurrence_geometry.py``). Use it only for high signal-to-noise
records.

The motivation for treating recursive sequences and recurrence together draws
on the *Patterns* volume of Discovering the Art of Mathematics
(https://www.artofmathematics.org/books); the definitions are the cited ones.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: Identifier of this module's definitions.
RECURRENCE_METHOD_VERSION = "drr_recurrence_v1"


def _as_series(values: Any, name: str = "data") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def delay_embedding(data: Any, dimension: int, delay: int) -> np.ndarray:
    """Delay coordinates ``(x_t, x_{t + delay}, ..., x_{t + (dimension - 1) delay})``.

    This is the reconstruction behind ``DynamicResonanceRooting.time_delay_embedding``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input data must be 1-dimensional for embedding")
    if dimension < 1:
        raise ValueError("embedding_dim must be at least 1")
    if delay < 1:
        raise ValueError("tau must be at least 1")
    span = (dimension - 1) * delay
    if x.size <= span:
        raise ValueError("Data too short for specified embedding parameters")
    rows = x.size - span
    return np.column_stack([x[i * delay : i * delay + rows] for i in range(dimension)])


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    changes = np.diff(padded)
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def _first_zero_crossing(x: np.ndarray, limit: int) -> int:
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered, 2 * x.size)
    acf = np.fft.irfft(np.abs(spectrum) ** 2)[: limit + 1]
    below = np.flatnonzero(acf[1:] <= 0)
    return int(below[0] + 1) if below.size else 1


def recurrence_quantification(
    data: Any,
    dimension: int = 3,
    delay: Optional[int] = None,
    recurrence_rate: float = 0.05,
    theiler_window: Optional[int] = None,
    min_line: int = 2,
    sampling_rate: float = 1.0,
    frequency: Optional[float] = None,
    max_points: int = 2000,
) -> Dict[str, Any]:
    """Recurrence quantification of one series at a fixed recurrence rate.

    The series is z-scored first. ``delay`` defaults to a quarter period of
    ``frequency`` when given and otherwise to the first zero crossing of the
    autocorrelation, and never to less than two samples: with a delay of one,
    consecutive delay vectors share coordinates, which draws diagonal lines
    even in white noise (determinism 0.57 instead of about 0.1 at a 5%
    recurrence rate). The Theiler window defaults to ``(dimension - 1) * delay + 1``
    samples. Memory grows with the square of the number of embedded points,
    which is capped by ``max_points``.
    """
    x = _as_series(data)
    if not 0 < recurrence_rate < 1:
        raise ValueError("recurrence_rate must be in the interval (0, 1)")
    if min_line < 2:
        raise ValueError("min_line must be at least 2")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    spread = float(np.std(x))
    if spread == 0:
        raise ValueError("data must not be constant")
    z = (x - x.mean()) / spread
    if delay is None:
        if frequency is not None and frequency > 0:
            delay = int(round(sampling_rate / (4.0 * frequency)))
        else:
            delay = _first_zero_crossing(z, max(1, z.size // 10))
        delay = max(2, delay)
    vectors = delay_embedding(z, dimension, delay)
    m = vectors.shape[0]
    if m > max_points:
        raise ValueError(
            f"{m} embedded points exceed max_points={max_points}; analyse a shorter window"
        )
    window = (dimension - 1) * delay + 1 if theiler_window is None else int(theiler_window)
    if window < 0 or m - window - 1 < 2 * min_line:
        raise ValueError("the Theiler window leaves too few pairs for recurrence quantification")

    squared = np.sum(vectors**2, axis=1)
    distance = np.sqrt(
        np.maximum(squared[:, None] + squared[None, :] - 2.0 * vectors @ vectors.T, 0.0)
    )
    rows, cols = np.triu_indices(m, k=window + 1)
    epsilon = float(np.quantile(distance[rows, cols], recurrence_rate))
    recurrent = distance <= epsilon
    band = np.abs(np.subtract.outer(np.arange(m), np.arange(m))) <= window
    recurrent[band] = False

    diagonal_lengths = []
    density = np.zeros(m)
    for offset in range(window + 1, m):
        diagonal = np.diagonal(recurrent, offset=offset)
        density[offset] = diagonal.mean()
        diagonal_lengths.append(_run_lengths(diagonal))
    lengths = np.concatenate(diagonal_lengths) if diagonal_lengths else np.array([], dtype=int)
    recurrent_points = float(lengths.sum())
    long_lines = lengths[lengths >= min_line]
    determinism = float(long_lines.sum() / recurrent_points) if recurrent_points else 0.0
    mean_diagonal = float(long_lines.mean()) if long_lines.size else 0.0

    padded = np.vstack(
        (np.zeros((1, m), np.int8), recurrent.astype(np.int8), np.zeros((1, m), np.int8))
    )
    steps = np.diff(padded, axis=0).T
    vertical = np.nonzero(steps == -1)[1] - np.nonzero(steps == 1)[1]
    vertical_points = float(vertical.sum())
    laminarity = (
        float(vertical[vertical >= min_line].sum() / vertical_points) if vertical_points else 0.0
    )

    upper_half = m // 2
    period = None
    if upper_half > window + 1:
        candidates = np.arange(window + 1, upper_half + 1)
        values = density[candidates]
        # A period is reported only when some offset recurs at least twice as densely
        # as the median offset; noise recurs about equally at every offset.
        if values.max() > 0 and values.max() >= 2.0 * float(np.median(values)):
            # Multiples of the period recur almost as densely; take the first offset
            # within 90% of the densest one so that the fundamental wins.
            best = int(candidates[np.flatnonzero(values >= 0.9 * values.max())[0]])
            period = float(best / sampling_rate)
    return {
        "method": RECURRENCE_METHOD_VERSION,
        "dimension": int(dimension),
        "delay": int(delay),
        "theiler_window": int(window),
        "epsilon": epsilon,
        "recurrence_rate": float(recurrent[rows, cols].mean()),
        "determinism": determinism,
        "laminarity": laminarity,
        "mean_diagonal_length": mean_diagonal,
        "recurrence_period_seconds": period,
        "embedded_points": int(m),
    }


def second_order_recurrence(data: Any, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """Least-squares fit of ``x_t = a1 x_{t-1} + a2 x_{t-2} + e_t`` and its roots.

    ``oscillatory`` is true when the characteristic roots are complex. Then
    ``frequency_hz`` is ``omega fs / 2 pi``, ``damping_time_seconds`` is
    ``-1 / (fs ln rho)`` and ``quality_factor`` is ``pi f`` times the damping
    time; both are ``None`` when ``rho >= 1`` (no decay). ``stationary`` is true
    when every root lies inside the unit circle. ``explained_variance`` is the
    fit's coefficient of determination.
    """
    x = _as_series(data)
    if x.size < 8:
        raise ValueError("at least eight samples are required")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    centered = x - x.mean()
    design = np.column_stack((centered[1:-1], centered[:-2]))
    target = centered[2:]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    a1, a2 = float(coefficients[0]), float(coefficients[1])
    residual = target - design @ coefficients
    total = float(np.sum(target**2))
    roots = np.roots([1.0, -a1, -a2])
    modulus = float(np.max(np.abs(roots)))
    result: Dict[str, Any] = {
        "method": RECURRENCE_METHOD_VERSION,
        "coefficients": [a1, a2],
        "root_modulus": modulus,
        "stationary": bool(modulus < 1.0),
        "oscillatory": bool(a1**2 + 4.0 * a2 < 0.0),
        "frequency_hz": None,
        "damping_time_seconds": None,
        "quality_factor": None,
        "explained_variance": float(1.0 - np.sum(residual**2) / total) if total > 0 else 0.0,
    }
    if result["oscillatory"]:
        rho = float(np.sqrt(-a2))
        omega = float(np.arccos(np.clip(a1 / (2.0 * rho), -1.0, 1.0)))
        frequency = omega * sampling_rate / (2.0 * np.pi)
        result["frequency_hz"] = frequency
        if 0.0 < rho < 1.0:
            damping = -1.0 / (sampling_rate * np.log(rho))
            result["damping_time_seconds"] = float(damping)
            result["quality_factor"] = float(np.pi * frequency * damping)
    return result


__all__ = [
    "RECURRENCE_METHOD_VERSION",
    "delay_embedding",
    "recurrence_quantification",
    "second_order_recurrence",
]
