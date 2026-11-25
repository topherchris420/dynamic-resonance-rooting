"""Checks that decide whether DRR's relationship tests can be trusted on a dataset.

Every surrogate null in DRR assumes at least stationarity. The dependent-noise
study measured what happens when that fails: on independent random walks each
null reported a false edge in 27-89% of systems at a nominal 5%. Phase analysis
adds its own conditions: a phase needs an oscillation, enough cycles, and a
frequency that does not drift during the window. This module runs those checks
before inference and records why it would abstain.

**Scaling (DFA).** Detrended fluctuation analysis integrates the centred series
into a profile ``Y(k) = sum_{t <= k} (x_t - mean x)``, removes a least-squares
polynomial from non-overlapping windows of ``s`` samples, and reports the root
mean square residual ``F(s)``. The exponent ``alpha`` is the slope of
``log F(s)`` against ``log s``. For fractional Gaussian noise with Hurst
exponent ``H``, which is stationary, ``alpha = H < 1``. For its cumulative sum,
fractional Brownian motion, which is not stationary, ``alpha = H + 1 > 1``; a
random walk has ``alpha = 1.5`` (Peng et al. 1994, *Phys. Rev. E* 49, 1685).
``alpha`` above a threshold therefore flags random-walk-like nonstationarity.
The threshold allows for finite-sample estimation error, and scales start
above two periods of the dominant oscillation, where an oscillation's bounded
profile no longer inflates the slope.

**Frequency drift.** A linear frequency drift makes the unwrapped phase
quadratic, ``phi(t) = a + 2 pi f t + pi beta t^2``. The band-limited phase is
fitted with a quadratic, and the drift across the record, ``beta T``, is
compared with what the channel's own phase diffusion produces. Under the null
of a stationary oscillator the phase is a Brownian motion with drift and
diffusion ``D``; projecting it onto the second shifted Legendre polynomial gives
a standard deviation of ``(6 / pi) sqrt(50 D / (210 T))`` hertz for the fitted
drift (``1 / 210`` is the integral of ``s^2 (s - 1)^2 (2 s - 1)^2`` over
``[0, 1]``). ``D`` is estimated from phase increments at a lag of four filter
time constants. A drift more than ``drift_z_max`` standard deviations from
zero, and larger than ``min_relative_drift`` of the frequency, means the
oscillation is not stationary. Two independent oscillators that share such a
drift look phase-aligned, and shift-based nulls would call that a
relationship. A drift smaller than the oscillator's own wandering cannot be
told from it, and is within what the stationary null already allows.

**Sampling.** Gaps and irregular timestamps are resolved by linear
interpolation onto a regular grid. The record keeps the fraction interpolated
and the longest gap; a gap longer than one cycle of the oscillation means the
phase across it is invented by the interpolation.

**Aliasing.** A frequency ``f`` and ``fs - f`` produce identical samples, so
aliasing cannot be detected from the samples. A peak near Nyquist only earns a
warning that the assumption cannot be checked.

The approach to multiscale structure draws on the *Patterns* volume of
Discovering the Art of Mathematics (https://www.artofmathematics.org/books);
the definitions above are the standard ones from the cited literature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phase_sync import (
    DEFAULT_RELATIVE_BANDWIDTH,
    band_limited_analytic_signal,
    dominant_frequency,
)

#: Identifier of the checks and their default thresholds.
DATA_QUALITY_VERSION = "drr_data_quality_v1"

#: Default thresholds. They were set on the development conditions of
#: ``drr_framework.resonance_math_benchmark`` before its held-out run.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_samples": 128,
    "dfa_alpha_max": 1.2,
    "min_cycles": 10.0,
    "drift_z_max": 3.0,
    "min_relative_drift": 0.01,
    "max_missing_fraction": 0.2,
    "max_gap_cycles": 1.0,
    "nyquist_fraction": 0.4,
    "min_band_power_fraction": 0.2,
}


def _as_series(values: Any, name: str = "data") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def detrended_fluctuation(
    data: Any,
    scales: Optional[Sequence[int]] = None,
    order: int = 1,
    min_scale: int = 16,
    max_scale: Optional[int] = None,
    n_scales: int = 12,
) -> Dict[str, Any]:
    """DFA fluctuation function ``F(s)`` and scaling exponent ``alpha``.

    Windows are taken from both ends of the profile so that every sample is
    used. ``order`` is the degree of the polynomial removed in each window
    (DFA-1 removes lines). The default scales are ``n_scales`` geometrically
    spaced window lengths from ``min_scale`` to ``N // 4``. ``r_squared`` is the
    coefficient of determination of the log-log fit, a check on whether a single
    exponent describes the range.
    """
    x = _as_series(data)
    n = x.size
    if order < 0:
        raise ValueError("order must be non-negative")
    if scales is None:
        upper = n // 4 if max_scale is None else int(max_scale)
        lower = max(int(min_scale), order + 3)
        if upper <= lower:
            raise ValueError("series too short for the requested DFA scale range")
        grid = np.unique(np.round(np.geomspace(lower, upper, n_scales)).astype(int))
    else:
        grid = np.unique(np.asarray(scales, dtype=int))
    if grid.size < 3:
        raise ValueError("at least three distinct scales are required")
    if grid.min() < order + 2 or grid.max() > n // 2:
        raise ValueError("scales must exceed order + 1 and leave at least two windows")

    profile = np.cumsum(x - x.mean())
    fluctuation = np.empty(grid.size)
    for index, scale in enumerate(grid):
        windows = n // scale
        forward = profile[: windows * scale].reshape(windows, scale)
        backward = profile[n - windows * scale :].reshape(windows, scale)
        segments = np.vstack((forward, backward)).T
        design = np.vander(np.linspace(-1.0, 1.0, scale), order + 1)
        coefficients = np.linalg.lstsq(design, segments, rcond=None)[0]
        residual = segments - design @ coefficients
        fluctuation[index] = np.sqrt(np.mean(residual**2))
    if np.any(fluctuation <= 0):
        raise ValueError("the profile is exactly polynomial at some scale (constant series?)")
    log_s, log_f = np.log(grid), np.log(fluctuation)
    slope, intercept = np.polyfit(log_s, log_f, 1)
    fitted = slope * log_s + intercept
    total = float(np.sum((log_f - log_f.mean()) ** 2))
    r_squared = float(1.0 - np.sum((log_f - fitted) ** 2) / total) if total > 0 else 1.0
    return {
        "alpha": float(slope),
        "scales": [int(s) for s in grid],
        "fluctuation": [float(f) for f in fluctuation],
        "r_squared": r_squared,
        "order": int(order),
    }


def dfa_scale_range(
    n_samples: int, sampling_rate: float, frequency: float, base: int = 16
) -> Tuple[int, int]:
    """Scales from two periods of ``frequency`` (at least ``base``) to ``N // 4``.

    When two periods do not fit below ``N // 8`` (a slow or absent oscillation),
    the range starts at ``base``.
    """
    upper = n_samples // 4
    lower = base
    if frequency > 0:
        period = sampling_rate / frequency
        candidate = int(np.ceil(2.0 * period))
        if candidate <= n_samples // 8:
            lower = max(base, candidate)
    return lower, upper


#: Standard deviation of the fitted drift ``beta T`` for a Brownian phase is
#: ``_DRIFT_SD_FACTOR * sqrt(D / T)`` hertz: ``(6 / pi) sqrt(50 / 210)``.
_DRIFT_SD_FACTOR = (6.0 / np.pi) * np.sqrt(50.0 / 210.0)


def frequency_drift(
    data: Any,
    sampling_rate: float,
    relative_bandwidth: float = DEFAULT_RELATIVE_BANDWIDTH,
) -> Dict[str, Any]:
    """Test the dominant oscillation for a linear frequency drift.

    The band-limited phase around the dominant frequency is unwrapped from its
    per-sample increments, trimmed of filter edges, and fitted with a quadratic.
    ``total_drift_hz`` is the fitted change of frequency across the record and
    ``z`` its ratio to the standard deviation a stationary, phase-diffusing
    oscillator would produce (see the module docstring). ``assessed`` is false
    when there is no dominant peak, the pass band reaches zero or Nyquist, or
    too little of the record remains after trimming.
    """
    x = _as_series(data)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    frequency = dominant_frequency(x, sampling_rate)
    record: Dict[str, Any] = {"assessed": False, "frequency_hz": frequency}
    nyquist = 0.5 * sampling_rate
    if not 0.0 < frequency < nyquist:
        record["reason"] = "no positive spectral peak below Nyquist"
        return record
    band = band_limited_analytic_signal(x, sampling_rate, frequency, relative_bandwidth)
    if frequency - 3.0 * band.bandwidth_hz <= 0.0 or frequency + 3.0 * band.bandwidth_hz >= nyquist:
        record["reason"] = "pass band reaches zero or the Nyquist frequency"
        return record
    edge = band.edge_samples
    z = band.signal[edge : x.size - edge]
    lag = int(np.ceil(4.0 * sampling_rate / (2.0 * np.pi * band.bandwidth_hz)))
    if z.size < 64 or lag >= z.size // 4:
        record["reason"] = "too few samples after trimming filter edges"
        return record
    phase = np.concatenate(([0.0], np.cumsum(np.angle(z[1:] * np.conj(z[:-1])))))
    times = np.arange(z.size) / sampling_rate
    duration = float(z.size / sampling_rate)
    coefficients = np.polyfit(times, phase, 2)
    residual = phase - np.polyval(coefficients, times)
    increments = residual[lag:] - residual[:-lag]
    diffusion = float(np.var(increments) / (2.0 * lag / sampling_rate))
    drift_rate = float(coefficients[0] / np.pi)
    total = drift_rate * duration
    # The floor keeps z finite for a noiseless tone, whose phase does not diffuse.
    error = max(float(_DRIFT_SD_FACTOR * np.sqrt(diffusion / duration)), 1e-12 * frequency)
    record.update(
        {
            "assessed": True,
            "drift_hz_per_s": drift_rate,
            "total_drift_hz": total,
            "relative_drift": abs(total) / frequency,
            "standard_error_hz": error,
            "z": float(total / error),
            "phase_diffusion_rad2_per_s": diffusion,
            "analysed_seconds": duration,
        }
    )
    return record


def regularize_series(
    values: Any,
    timestamps: Optional[Any] = None,
    sampling_rate: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Linearly interpolate missing values and irregular timestamps onto a regular grid.

    ``values`` has shape ``(time,)`` or ``(time, channels)`` and may contain
    ``NaN`` for missing observations. Without ``timestamps`` the rows are taken
    to be regularly spaced. With ``timestamps`` (strictly increasing seconds)
    the output grid starts at the first time every channel is observed and
    steps by ``1 / sampling_rate``. Rows before the first or after the last
    observation common to all channels are dropped rather than extrapolated.
    The report records what was done so that a reader can judge it.
    """
    array = np.asarray(values, dtype=float)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("values must have shape (time,) or (time, channels) with time >= 2")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinities; mark missing values with NaN")
    observed = np.isfinite(array)
    if not np.all(observed.any(axis=0)):
        raise ValueError("every channel needs at least one observation")
    missing_fraction = float(1.0 - observed.mean())

    if timestamps is None:
        times = np.arange(array.shape[0], dtype=float)
        step = 1.0
        interval_cv = 0.0
    else:
        if sampling_rate is None or not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate is required with timestamps and must be positive")
        times = np.asarray(timestamps, dtype=float)
        if times.shape != (array.shape[0],) or not np.all(np.isfinite(times)):
            raise ValueError("timestamps must be finite with one entry per row")
        intervals = np.diff(times)
        if np.any(intervals <= 0):
            raise ValueError("timestamps must be strictly increasing")
        step = 1.0 / float(sampling_rate)
        interval_cv = float(np.std(intervals) / np.mean(intervals))

    start = max(float(times[observed[:, c]][0]) for c in range(array.shape[1]))
    stop = min(float(times[observed[:, c]][-1]) for c in range(array.shape[1]))
    if stop <= start:
        raise ValueError("the channels share no observed time span")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    grid = start + step * np.arange(count)
    output = np.empty((count, array.shape[1]))
    longest_gap = 0.0
    for c in range(array.shape[1]):
        t_obs, v_obs = times[observed[:, c]], array[observed[:, c], c]
        output[:, c] = np.interp(grid, t_obs, v_obs)
        inside = (t_obs >= start - 1e-12) & (t_obs <= stop + 1e-12)
        if np.count_nonzero(inside) > 1:
            longest_gap = max(longest_gap, float(np.max(np.diff(t_obs[inside]))))
    gap_seconds = longest_gap if timestamps is not None else None
    report = {
        "method": "linear interpolation onto a regular grid",
        "input_rows": int(array.shape[0]),
        "output_rows": int(count),
        "missing_fraction": missing_fraction,
        "interval_cv": interval_cv,
        "longest_gap_samples": float(longest_gap / step),
        "longest_gap_seconds": gap_seconds,
        "dropped_edge_rows": int(
            array.shape[0] - np.count_nonzero((times >= start) & (times <= stop))
        ),
    }
    return (output[:, 0] if squeeze else output), report


def _channel_checks(
    series: np.ndarray, sampling_rate: float, thresholds: Dict[str, float]
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """Checks for one channel: (record, hard failures, phase-only limits, warnings)."""
    hard: List[str] = []
    phase_limits: List[str] = []
    warnings: List[str] = []
    n = series.size
    record: Dict[str, Any] = {"n_samples": int(n)}
    if np.ptp(series) == 0:
        hard.append("constant series")
        return record, hard, phase_limits, warnings
    if n < thresholds["min_samples"]:
        hard.append(f"{n} samples (< {thresholds['min_samples']:g})")
        return record, hard, phase_limits, warnings

    frequency = dominant_frequency(series, sampling_rate)
    record["dominant_frequency_hz"] = frequency
    cycles = frequency * n / sampling_rate
    record["cycles"] = float(cycles)

    lower, upper = dfa_scale_range(n, sampling_rate, frequency)
    if upper > lower + 2:
        dfa = detrended_fluctuation(series, min_scale=lower, max_scale=upper)
        record["dfa"] = dfa
        if dfa["alpha"] > thresholds["dfa_alpha_max"]:
            hard.append(
                f"DFA exponent {dfa['alpha']:.2f} > {thresholds['dfa_alpha_max']:g}: "
                "random-walk-like nonstationarity; difference or detrend first"
            )
    else:
        record["dfa"] = None
        warnings.append("record too short to estimate a DFA exponent above two periods")

    if cycles < thresholds["min_cycles"]:
        phase_limits.append(f"{cycles:.1f} cycles of the dominant oscillation")
    if frequency > thresholds["nyquist_fraction"] * sampling_rate:
        warnings.append(
            f"dominant frequency {frequency:.4g} Hz is near Nyquist; aliasing cannot be ruled out"
        )
    oscillatory = False
    if 0.0 < frequency < 0.5 * sampling_rate:
        band = band_limited_analytic_signal(
            series, sampling_rate, frequency, DEFAULT_RELATIVE_BANDWIDTH
        )
        record["band_power_fraction"] = band.band_power_fraction
        oscillatory = band.band_power_fraction >= thresholds["min_band_power_fraction"]
        if not oscillatory:
            warnings.append(
                f"only {band.band_power_fraction:.0%} of the power lies in the analysed band; "
                "a phase relationship there describes that band, not the whole signal"
            )

    # A drift presupposes one oscillation that can be followed from segment to
    # segment; without a dominant band, segment peaks scatter at random.
    drift = (
        frequency_drift(series, sampling_rate)
        if oscillatory
        else {
            "assessed": False,
            "reason": "no dominant narrowband oscillation to follow",
        }
    )
    record["frequency_drift"] = drift
    if (
        drift["assessed"]
        and abs(drift["z"]) > thresholds["drift_z_max"]
        and drift["relative_drift"] > thresholds["min_relative_drift"]
    ):
        hard.append(
            f"dominant frequency drifts {drift['total_drift_hz']:+.3g} Hz across the record "
            f"({drift['z']:+.1f} standard errors of a stationary oscillator's wandering): the "
            "oscillation is not stationary"
        )
    return record, hard, phase_limits, warnings


def assess_series_quality(
    data: Any,
    sampling_rate: float,
    *,
    column_names: Optional[Sequence[str]] = None,
    regularization: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run every check and decide whether relationship inference should abstain.

    ``abstain`` is true when any channel fails a check that invalidates the
    stationarity every DRR null assumes (a random-walk-like scaling exponent, a
    drifting frequency), when the data are unusable (constant, too short), or
    when ``regularization`` (the report of :func:`regularize_series`) shows too
    much interpolation. ``phase_inconclusive`` additionally lists channels
    whose oscillation is too short-lived for phase analysis. ``warnings`` are
    assumptions that hold or fail without the data being able to say.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("data must have shape (time,) or (time, channels)")
    if not np.all(np.isfinite(values)):
        raise ValueError("data contains non-finite values; regularize missing samples first")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"unknown thresholds: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in thresholds.items()})
    names = (
        [str(name) for name in column_names]
        if column_names is not None
        else [f"dim_{index}" for index in range(values.shape[1])]
    )
    if len(names) != values.shape[1]:
        raise ValueError("column_names must match the number of channels")

    reasons: List[str] = []
    phase_inconclusive: List[str] = []
    warnings: List[str] = []
    channels: Dict[str, Any] = {}
    for name, column in zip(names, values.T):
        record, hard, phase_limits, notes = _channel_checks(column, float(sampling_rate), limits)
        channels[name] = record
        reasons.extend(f"{name}: {reason}" for reason in hard)
        phase_inconclusive.extend(f"{name}: {reason}" for reason in phase_limits)
        warnings.extend(f"{name}: {note}" for note in notes)

    if regularization is not None:
        if regularization.get("missing_fraction", 0.0) > limits["max_missing_fraction"]:
            reasons.append(
                f"{regularization['missing_fraction']:.0%} of values were interpolated "
                f"(> {limits['max_missing_fraction']:.0%})"
            )
        gap_samples = float(regularization.get("longest_gap_samples", 0.0))
        for name, record in channels.items():
            frequency = record.get("dominant_frequency_hz", 0.0) or 0.0
            gap_cycles = gap_samples * frequency / float(sampling_rate)
            if gap_cycles > limits["max_gap_cycles"]:
                reasons.append(
                    f"{name}: longest gap spans {gap_cycles:.2f} cycles; the phase across it "
                    "is invented by interpolation"
                )
        if regularization.get("missing_fraction", 0.0) > 0 or regularization.get(
            "interval_cv", 0.0
        ):
            warnings.append(
                "values were interpolated onto a regular grid; interpolation smooths the "
                "highest frequencies and can bias phase"
            )

    return {
        "method": DATA_QUALITY_VERSION,
        "thresholds": limits,
        "abstain": bool(reasons),
        "reasons": reasons,
        "phase_inconclusive": phase_inconclusive,
        "warnings": warnings,
        "channels": channels,
        "regularization": regularization,
    }


__all__ = [
    "DATA_QUALITY_VERSION",
    "DEFAULT_THRESHOLDS",
    "assess_series_quality",
    "detrended_fluctuation",
    "dfa_scale_range",
    "frequency_drift",
    "regularize_series",
]
