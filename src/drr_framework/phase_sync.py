"""Phase synchronization and small-integer frequency ratios between channels.

DRR's rooting test asks whether one series' recent values track another's.
Two oscillators can be related in a way that test cannot see: their phases can
stay aligned while their waveforms are uncorrelated, as when one runs at twice
the frequency of the other. This module asks the phase question directly.

**Phase.** Each channel is filtered with a Gaussian pass band centred on its
dominant frequency ``f0`` with standard deviation ``f0 / 6``, keeping positive
frequencies only. The result is the analytic signal ``z(t) = A(t) exp(i phi(t))``
of the band. That pass band is the frequency response of the Morlet wavelet
with ``omega0 = 6`` used by ``ResonanceDetector(method="wavelet")``, so
``phi`` is that wavelet's phase at the dominant frequency.

**Ratio.** When ``f_y / f_x`` lies within a tolerance of ``n / m`` for coprime
integers ``n, m <= max_order`` (the Farey fractions of that order and their
reciprocals), the generalized phase difference ``psi(t) = n phi_x(t) - m phi_y(t)``
is formed. Locked oscillators keep ``psi`` bounded; unrelated ones let it
wander. The ratio is chosen from each channel's own spectrum, never from the
alignment between channels. Had it been chosen from each channel's mean
frequency, a symmetric function of its phase increments, the test below would
stay exact; the Welch peak used here approximates that, and its effect on size
is measured on independent oscillators at exact 1:2 and 2:3 ratios.

**Statistic.** ``PLV_{n:m} = |mean_t exp(i psi(t))|``, the phase-locking value.
It is 1 for a constant ``psi`` and near 0 for a ``psi`` spread evenly around the
circle. It ignores amplitude and is unchanged by a constant phase offset, so a
delay shorter than the phase-coherence time appears as an offset.

**Test.** Under the null hypothesis the two channels are independent and the
second channel's phasor ``exp(i m phi_y)`` is invariant in law under circular
shifts. The p-value counts the circular shifts of that phasor whose PLV is at
least the observed one. Over the full cyclic group this is an exact
randomization test (Lehmann and Romano, *Testing Statistical Hypotheses*,
3rd ed., Section 15.2); no shift is drawn at random, so the result is
deterministic.
Two corrections make the invariance assumption hold for real oscillators.
Edge samples distorted by the filter are trimmed. The phasor is then multiplied
by a linear phase ramp of at most ``pi / T`` radians per sample so that it
returns to its starting phase at the wrap. Without the ramp a circular shift
introduces a phase jump at the seam, so shifted copies look less locked than
the original and the test rejects too often on unrelated, highly coherent
oscillators. With independent, identically distributed phase increments the
ramp-corrected phasor is exactly shift invariant; band-pass filtering makes the
increments dependent, so for filtered data the invariance is approximate and is
measured in ``results/expected/resonance_math_benchmark.json``.

**Identifiability.** Two pure tones at the same frequency are perfectly locked
whether or not they interact. Coupling can only be told from coincidence when
phases wander during the window. The effective number of independent phase
samples, Bartlett's variance formula applied to the two phasors, measures how
much they wander. A pair that is not significant with fewer than
``min_effective_samples`` is reported as *inconclusive*, never as evidence of
no relationship.

The ``n:m`` phase-locking value follows Tass et al. (1998, *Phys. Rev. Lett.*
81, 3291) and Lachaux et al. (1999, *Hum. Brain Mapp.* 8, 194). The framing of
frequency ratios and rates of change draws on the *Music* and *Ideas of
Calculus* volumes of Discovering the Art of Mathematics
(https://www.artofmathematics.org/books); no text or code is taken from them.
A significant result is a statistical phase relationship. It is undirected,
and it is not evidence of causation or of a physical mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .modules import DepthCalculator

#: Identifier of this module's definitions. Any change to a reported number for
#: a fixed input bumps it.
PHASE_SYNC_METHOD_VERSION = "drr_phase_sync_v1"

#: Gaussian pass-band standard deviation as a fraction of the centre frequency.
#: ``1 / 6`` matches the Morlet wavelet (``omega0 = 6``) of ``ResonanceDetector``.
DEFAULT_RELATIVE_BANDWIDTH = 1.0 / 6.0

TWO_PI = 2.0 * np.pi
_TIE_TOLERANCE = 1e-12

PHASE_ASSUMPTIONS = (
    "Each channel is a band-limited oscillation around its dominant frequency; "
    "outside that band the phase is not defined.",
    "The sampling rate exceeds twice the highest frequency present (no aliasing); "
    "a sampled series cannot reveal aliasing by itself.",
    "Under the null hypothesis the channels are independent and each phase process "
    "is stationary; frequency drift or trends violate this.",
    "The phase must wander during the window for coupling to be identifiable; "
    "near-periodic channels yield inconclusive results.",
)


@dataclass(frozen=True)
class AnalyticBand:
    """Band-limited analytic signal of one channel."""

    signal: np.ndarray
    sampling_rate: float
    center_frequency_hz: float
    bandwidth_hz: float
    edge_samples: int
    band_power_fraction: float

    @property
    def phase(self) -> np.ndarray:
        """Wrapped phase in radians."""
        return np.angle(self.signal)

    @property
    def amplitude(self) -> np.ndarray:
        """Envelope ``|z(t)|`` in the units of the input."""
        return np.abs(self.signal)


def _as_series(values: Any, name: str = "data") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_phase_pair(phase_x: Any, phase_y: Any) -> Tuple[np.ndarray, np.ndarray]:
    x = _as_series(phase_x, "phase_x")
    y = _as_series(phase_y, "phase_y")
    if x.shape != y.shape:
        raise ValueError("phase_x and phase_y must have the same length")
    return x, y


def dominant_frequency(data: Any, sampling_rate: float) -> float:
    """Welch peak refined to sub-bin precision, or ``0.0`` without a positive peak.

    This is the estimate ``DepthCalculator`` uses for its target frequency, so
    phase features and resonance depth describe the same spectral peak.
    """
    values = _as_series(data)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    return float(DepthCalculator()._select_target_frequency(values, float(sampling_rate), None))


def band_limited_analytic_signal(
    data: Any,
    sampling_rate: float,
    center_frequency: float,
    relative_bandwidth: float = DEFAULT_RELATIVE_BANDWIDTH,
    min_bandwidth_bins: float = 2.0,
) -> AnalyticBand:
    """Analytic signal of a Gaussian pass band around ``center_frequency``.

    The band's standard deviation is ``max(relative_bandwidth * f0,
    min_bandwidth_bins * fs / N)`` hertz, so it never resolves finer than the
    record allows. Filtering is zero-phase and done in the frequency domain on
    a reflect-padded copy; samples within two impulse-response widths of
    either end (``edge_samples``) still carry edge effects and should be
    trimmed before inference. ``band_power_fraction`` is the share of the
    series' power that passes the band, a measure of how narrowband it is.
    """
    values = _as_series(data)
    n = values.size
    if n < 16:
        raise ValueError("at least 16 samples are required")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    if not 0.0 < center_frequency < 0.5 * sampling_rate:
        raise ValueError("center_frequency must lie strictly between 0 and the Nyquist frequency")
    if relative_bandwidth <= 0 or min_bandwidth_bins <= 0:
        raise ValueError("relative_bandwidth and min_bandwidth_bins must be positive")

    bandwidth = max(relative_bandwidth * center_frequency, min_bandwidth_bins * sampling_rate / n)
    # Standard deviation of the filter's impulse-response envelope, in samples.
    sigma_samples = sampling_rate / (TWO_PI * bandwidth)
    pad = int(min(n - 1, np.ceil(4.0 * sigma_samples)))
    edge = int(np.ceil(2.0 * sigma_samples))

    centered = values - values.mean()
    padded = np.pad(centered, pad, mode="reflect") if pad else centered
    freqs = np.fft.fftfreq(padded.size, d=1.0 / sampling_rate)
    gain = np.where(
        freqs > 0, 2.0 * np.exp(-0.5 * ((freqs - center_frequency) / bandwidth) ** 2), 0.0
    )
    analytic = np.fft.ifft(np.fft.fft(padded) * gain)[pad : pad + n]

    power = np.abs(np.fft.rfft(centered)) ** 2
    rfreqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    positive = rfreqs > 0
    total = float(power[positive].sum())
    response = np.exp(-0.5 * ((rfreqs[positive] - center_frequency) / bandwidth) ** 2)
    fraction = float(np.sum(response**2 * power[positive]) / total) if total > 0 else 0.0

    return AnalyticBand(
        signal=analytic,
        sampling_rate=float(sampling_rate),
        center_frequency_hz=float(center_frequency),
        bandwidth_hz=float(bandwidth),
        edge_samples=edge,
        band_power_fraction=fraction,
    )


def instantaneous_frequency(band: AnalyticBand, amplitude_floor: float = 0.1) -> np.ndarray:
    """Instantaneous frequency ``(1 / 2 pi) d phi / dt`` in hertz.

    Each per-sample phase increment is the argument of ``z[k + 1] conj(z[k])``,
    which needs no unwrapping and is exact while the true increment is below
    ``pi`` (a frequency below Nyquist). Adjacent increments are averaged, which
    is the second-order central difference. The phase of a vanishing amplitude
    is undefined, so samples whose amplitude is below ``amplitude_floor`` times
    the median amplitude are returned as ``NaN``.
    """
    z = np.asarray(band.signal, dtype=complex)
    if z.size < 2:
        raise ValueError("at least two samples are required")
    increments = np.angle(z[1:] * np.conj(z[:-1]))
    centered = np.empty(z.size)
    centered[0] = increments[0]
    centered[-1] = increments[-1]
    centered[1:-1] = 0.5 * (increments[:-1] + increments[1:])
    frequency = centered * band.sampling_rate / TWO_PI
    amplitude = np.abs(z)
    frequency[amplitude < amplitude_floor * float(np.median(amplitude))] = np.nan
    return frequency


def phase_locking_value(phase_x: Any, phase_y: Any, n: int = 1, m: int = 1) -> Tuple[float, float]:
    """``(PLV, mean phase difference)`` of ``psi = n phi_x - m phi_y``.

    PLV is ``|mean exp(i psi)|`` in ``[0, 1]``; the mean phase difference is its
    argument in radians.
    """
    x, y = _as_phase_pair(phase_x, phase_y)
    if x.size == 0:
        raise ValueError("phases must not be empty")
    mean = np.mean(np.exp(1j * (n * x - m * y)))
    return float(np.abs(mean)), float(np.angle(mean))


def farey_ratio_candidates(
    frequency_x: float,
    frequency_y: float,
    max_order: int = 3,
    relative_tolerance: float = 0.08,
) -> List[Tuple[int, int]]:
    """Coprime ``(n, m)`` with ``n, m <= max_order`` and ``n / m`` near ``f_y / f_x``.

    A candidate is kept when ``|n / m - f_y / f_x| <= relative_tolerance * f_y / f_x``.
    Candidates are sorted by relative error, then by ``n + m`` so that simpler
    ratios win ties. An empty list means no small-integer ratio is plausible.
    """
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    if not (np.isfinite(frequency_x) and np.isfinite(frequency_y)):
        return []
    if frequency_x <= 0 or frequency_y <= 0:
        return []
    ratio = frequency_y / frequency_x
    kept = []
    for n in range(1, max_order + 1):
        for m in range(1, max_order + 1):
            if gcd(n, m) != 1:
                continue
            error = abs(n / m - ratio) / ratio
            if error <= relative_tolerance:
                kept.append((error, n + m, n, m))
    kept.sort()
    return [(n, m) for _, _, n, m in kept]


def _autocovariance(phasor: np.ndarray, max_lag: int) -> np.ndarray:
    """``g(k) = (1/N) sum_t a(t + k) conj(a(t))`` for ``k = 0..max_lag``."""
    n = phasor.size
    spectrum = np.fft.fft(phasor, 2 * n)
    return np.fft.ifft(np.conj(spectrum) * spectrum)[: max_lag + 1] / n


def effective_phase_samples(phasor_a: Any, phasor_b: Any, max_lag: Optional[int] = None) -> float:
    """Effective number of independent samples of ``mean(a conj(b))`` under independence.

    For independent stationary phasors, ``E|mean(a conj b)|^2 = (1/N) sum_k
    (1 - |k|/N) g_a(k) conj(g_b(k))`` (Bartlett's formula for a sample cross
    product). The return value is ``N`` divided by that sum, estimated with
    sample autocovariances and a Bartlett lag window to ``max_lag``
    (default ``N // 4``). It is ``N`` for serially independent phases and about
    ``T D`` for two phase-diffusing oscillators with diffusion ``D`` observed
    for ``T`` seconds. The sum is floored at one, so the result never exceeds ``N``.
    """
    a = np.asarray(phasor_a, dtype=complex)
    b = np.asarray(phasor_b, dtype=complex)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("phasors must be one-dimensional, equal length, and longer than one")
    n = a.size
    lag = n // 4 if max_lag is None else int(max_lag)
    lag = int(min(max(lag, 0), n - 1))
    product = (_autocovariance(a, lag) * np.conj(_autocovariance(b, lag))).real
    k = np.arange(lag + 1)
    terms = (1.0 - k / (lag + 1.0)) * product / (1.0 - k / n)
    total = float(terms[0] + 2.0 * np.sum(terms[1:]))
    return float(n / max(total, 1.0))


def _shift_profiles(
    phase_x: np.ndarray,
    phase_y: np.ndarray,
    ratios: Sequence[Tuple[int, int]],
    seam_correction: bool,
) -> Tuple[np.ndarray, List[float]]:
    """PLV of every circular shift of the second phasor, one row per ratio."""
    length = phase_x.size - 1
    times = np.arange(length)
    profiles = np.empty((len(ratios), length))
    seams = []
    for row, (n, m) in enumerate(ratios):
        a = np.exp(1j * n * phase_x[:length])
        b_full = np.exp(1j * m * phase_y)
        # The phasor that would follow the last analysed sample is b_full[length];
        # the wrap replaces it with b_full[0]. The ramp removes that jump.
        seam = float(np.angle(b_full[length] * np.conj(b_full[0])))
        b = b_full[:length]
        if seam_correction:
            b = b * np.exp(-1j * seam * times / length)
        seams.append(seam)
        cross = np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b))
        profiles[row] = np.abs(cross) / length
    return profiles, seams


def _rank_p_values(profile: np.ndarray) -> np.ndarray:
    """For each shift, the fraction of shifts whose statistic is at least as large."""
    ordered = np.sort(profile)
    at_least = profile.size - np.searchsorted(ordered, profile - _TIE_TOLERANCE, side="left")
    return at_least / profile.size


def phase_shift_test(
    phase_x: Any,
    phase_y: Any,
    ratios: Sequence[Tuple[int, int]],
    seam_correction: bool = True,
) -> Dict[str, Any]:
    """Exact randomization test of ``n:m`` locking over every circular shift.

    The statistic for each candidate ratio is the PLV between ``exp(i n phi_x)``
    and a circularly shifted, seam-corrected ``exp(i m phi_y)``. With several
    ratios the single-step min-p rule of Westfall and Young combines them: each
    shift's per-ratio p-values are ranks among all shifts, the shift's score is
    their minimum, and the reported p-value is the rank of the observed score.
    The family-wise error over ratios is therefore controlled. The smallest
    attainable p-value is ``1 / n_shifts``. Phases should already be trimmed of
    filter edge effects.
    """
    x, y = _as_phase_pair(phase_x, phase_y)
    if x.size < 9:
        raise ValueError("at least nine phase samples are required")
    if not ratios:
        raise ValueError("at least one candidate ratio is required")
    profiles, seams = _shift_profiles(x, y, ratios, seam_correction)
    marginal = np.vstack([_rank_p_values(row) for row in profiles])
    combined = marginal.min(axis=0)
    p_value = float(np.mean(combined <= combined[0] + _TIE_TOLERANCE))
    observed = marginal[:, 0]
    # Smallest marginal p-value; ties go to the larger statistic, then the earlier ratio.
    order = sorted(range(len(ratios)), key=lambda r: (observed[r], -profiles[r, 0], r))
    best = order[0]
    return {
        "p_value": p_value,
        "selected_ratio": [int(ratios[best][0]), int(ratios[best][1])],
        "candidate_ratios": [[int(n), int(m)] for n, m in ratios],
        "test_statistic": [float(value) for value in profiles[:, 0]],
        "marginal_p_values": [float(value) for value in observed],
        "n_shifts": int(profiles.shape[1]),
        "minimum_attainable_p_value": 1.0 / profiles.shape[1],
        "seam_correction": bool(seam_correction),
        "seam_phase_jump_rad": seams,
    }


def phase_difference_dynamics(
    phase_x: Any,
    phase_y: Any,
    n: int,
    m: int,
    sampling_rate: float,
    lag_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Rate of change and accumulated spread of ``psi = n phi_x - m phi_y``.

    ``psi`` is unwrapped by summing each channel's per-sample phase increment,
    which is exact below Nyquist. The least-squares slope of ``psi`` against time
    is ``2 pi`` times the frequency mismatch ``n f_x - m f_y``. After removing
    that trend, ``Var[psi(t + tau) - psi(t)] / (2 tau)`` is the phase diffusion
    of the difference at lag ``tau`` (default an eighth of the record). Net slips
    is the number of whole turns ``psi`` gains or loses over the record. Locked
    pairs have mismatch, diffusion, and slips near zero.
    """
    x, y = _as_phase_pair(phase_x, phase_y)
    if x.size < 4:
        raise ValueError("at least four phase samples are required")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    increments = n * np.angle(np.exp(1j * np.diff(x))) - m * np.angle(np.exp(1j * np.diff(y)))
    psi = (n * x[0] - m * y[0]) + np.concatenate(([0.0], np.cumsum(increments)))
    times = np.arange(psi.size) / sampling_rate
    slope, intercept = np.polyfit(times, psi, 1)
    residual = psi - (slope * times + intercept)
    lag = max(1, psi.size // 8) if lag_samples is None else int(lag_samples)
    lag = int(min(max(lag, 1), psi.size - 1))
    spread = residual[lag:] - residual[:-lag]
    diffusion = float(np.var(spread) / (2.0 * lag / sampling_rate))
    return {
        "frequency_mismatch_hz": float(slope / TWO_PI),
        "phase_diffusion_rad2_per_s": diffusion,
        "diffusion_lag_seconds": float(lag / sampling_rate),
        "net_phase_slips": float(np.round((psi[-1] - psi[0]) / TWO_PI)),
    }


def windowed_phase_locking(
    phase_x: Any,
    phase_y: Any,
    n: int,
    m: int,
    window: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """PLV of ``psi`` in sliding windows: ``(window centre indices, PLV)``.

    This is descriptive. Windowed values are not tested and overlapping windows
    are strongly dependent.
    """
    x, y = _as_phase_pair(phase_x, phase_y)
    if window < 2 or window > x.size:
        raise ValueError("window must be between 2 and the number of samples")
    if step < 1:
        raise ValueError("step must be at least 1")
    phasor = np.exp(1j * (n * x - m * y))
    cumulative = np.concatenate(([0.0 + 0.0j], np.cumsum(phasor)))
    starts = np.arange(0, x.size - window + 1, step)
    values = np.abs(cumulative[starts + window] - cumulative[starts]) / window
    return starts + 0.5 * (window - 1), values


def single_change_point(values: Any, min_segment: int = 2) -> Dict[str, float]:
    """Least-squares split of a sequence into two constant segments.

    Returns the first index of the second segment, both segment means, and the
    fraction of the total sum of squares the split removes. Descriptive only:
    the location of the best split is reported even when no change exists.
    """
    series = _as_series(values, "values")
    if min_segment < 1 or series.size < 2 * min_segment:
        raise ValueError("values must hold at least two segments of min_segment points")
    n = series.size
    cumulative = np.concatenate(([0.0], np.cumsum(series)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(series**2)))
    splits = np.arange(min_segment, n - min_segment + 1)
    left_n, right_n = splits, n - splits
    left_sum, right_sum = cumulative[splits], cumulative[-1] - cumulative[splits]
    left_sq, right_sq = cumulative_sq[splits], cumulative_sq[-1] - cumulative_sq[splits]
    sse = (left_sq - left_sum**2 / left_n) + (right_sq - right_sum**2 / right_n)
    best = int(np.argmin(sse))
    index = int(splits[best])
    total = float(cumulative_sq[-1] - cumulative[-1] ** 2 / n)
    return {
        "index": float(index),
        "mean_before": float(series[:index].mean()),
        "mean_after": float(series[index:].mean()),
        "explained_fraction": float(1.0 - sse[best] / total) if total > 0 else 0.0,
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise control, any dependence)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (p.size - rank) * float(p[index])))
        adjusted[index] = running
    return adjusted


class PhaseSynchronyAnalyzer:
    """Test every channel pair for ``n:m`` phase synchronization.

    Each pair receives one of four statuses:

    * ``synchronized``: the Holm-adjusted p-value is at most ``alpha``.
    * ``not_detected``: not significant although the phases wandered enough
      (``effective_samples >= min_effective_samples``) to give the test power.
    * ``inconclusive``: not significant and too little phase variability to
      tell, or a channel has no usable oscillation (no spectral peak, fewer than
      ``min_cycles`` cycles, or a pass band touching zero or Nyquist).
    * ``not_tested``: no ratio ``n / m`` with ``n, m <= max_ratio_order`` lies
      within ``ratio_tolerance`` of the observed frequency ratio.

    ``trim_edges`` and ``seam_correction`` exist for ablation studies; turning
    either off removes a correction the test's calibration depends on.
    """

    def __init__(
        self,
        sampling_rate: float,
        *,
        relative_bandwidth: float = DEFAULT_RELATIVE_BANDWIDTH,
        max_ratio_order: int = 3,
        ratio_tolerance: float = 0.08,
        alpha: float = 0.05,
        min_effective_samples: float = 20.0,
        min_cycles: float = 10.0,
        trim_edges: bool = True,
        seam_correction: bool = True,
        ratios: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in the interval (0, 1)")
        if max_ratio_order < 1:
            raise ValueError("max_ratio_order must be at least 1")
        self.sampling_rate = float(sampling_rate)
        self.relative_bandwidth = float(relative_bandwidth)
        self.max_ratio_order = int(max_ratio_order)
        self.ratio_tolerance = float(ratio_tolerance)
        self.alpha = float(alpha)
        self.min_effective_samples = float(min_effective_samples)
        self.min_cycles = float(min_cycles)
        self.trim_edges = bool(trim_edges)
        self.seam_correction = bool(seam_correction)
        self.ratios = None if ratios is None else [(int(n), int(m)) for n, m in ratios]

    def parameters(self) -> Dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "relative_bandwidth": self.relative_bandwidth,
            "max_ratio_order": self.max_ratio_order,
            "ratio_tolerance": self.ratio_tolerance,
            "alpha": self.alpha,
            "min_effective_samples": self.min_effective_samples,
            "min_cycles": self.min_cycles,
            "trim_edges": self.trim_edges,
            "seam_correction": self.seam_correction,
            "ratios": None if self.ratios is None else [list(r) for r in self.ratios],
            "correction": "holm",
        }

    def _channel(self, series: np.ndarray) -> Dict[str, Any]:
        frequency = dominant_frequency(series, self.sampling_rate)
        nyquist = 0.5 * self.sampling_rate
        record: Dict[str, Any] = {
            "dominant_frequency_hz": frequency,
            "usable": False,
            "reason": None,
            "band": None,
        }
        if np.ptp(series) == 0:
            record["reason"] = "constant channel"
            return record
        if not 0.0 < frequency < nyquist:
            record["reason"] = "no positive spectral peak below Nyquist"
            return record
        band = band_limited_analytic_signal(
            series, self.sampling_rate, frequency, self.relative_bandwidth
        )
        record["band"] = band
        record["bandwidth_hz"] = band.bandwidth_hz
        record["edge_samples"] = band.edge_samples
        record["band_power_fraction"] = band.band_power_fraction
        inst = instantaneous_frequency(band)
        finite = inst[np.isfinite(inst)]
        if finite.size:
            q25, q50, q75 = np.percentile(finite, [25, 50, 75])
            record["instantaneous_frequency_hz"] = {
                "median": float(q50),
                "interquartile_range": float(q75 - q25),
            }
        if frequency - 3.0 * band.bandwidth_hz <= 0.0:
            record["reason"] = "pass band reaches zero frequency"
        elif frequency + 3.0 * band.bandwidth_hz >= nyquist:
            record["reason"] = "pass band reaches the Nyquist frequency"
        else:
            record["usable"] = True
        return record

    def analyze(self, data: Any, column_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("data must have shape (time, channels >= 2)")
        if values.shape[0] < 32:
            raise ValueError("at least 32 samples are required")
        if not np.all(np.isfinite(values)):
            raise ValueError("data contains non-finite values")
        names = (
            [str(name) for name in column_names]
            if column_names is not None
            else [f"dim_{index}" for index in range(values.shape[1])]
        )
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("column_names must be unique and match the number of channels")

        n = values.shape[0]
        channels = [self._channel(values[:, index]) for index in range(values.shape[1])]
        pairs: List[Dict[str, Any]] = []
        tested: List[int] = []
        for i in range(values.shape[1]):
            for j in range(i + 1, values.shape[1]):
                pairs.append(self._pair(channels[i], channels[j], names[i], names[j], n))
                if pairs[-1]["p_value"] is not None:
                    tested.append(len(pairs) - 1)

        adjusted = holm_adjust([pairs[k]["p_value"] for k in tested])
        for k, value in zip(tested, adjusted):
            pair = pairs[k]
            pair["adjusted_p_value"] = float(value)
            if value <= self.alpha:
                pair["status"] = "synchronized"
                pair["reason"] = None
            elif pair["effective_samples"] < self.min_effective_samples:
                pair["status"] = "inconclusive"
                pair["reason"] = (
                    f"not significant with {pair['effective_samples']:.1f} effective phase "
                    f"samples (< {self.min_effective_samples:g}); the phases varied too little "
                    "to tell coupling from coincidence"
                )
            else:
                pair["status"] = "not_detected"
                pair["reason"] = None

        channel_summary = {}
        for name, record in zip(names, channels):
            channel_summary[name] = {
                key: value for key, value in record.items() if key not in ("band",)
            }
        return {
            "method": PHASE_SYNC_METHOD_VERSION,
            "parameters": self.parameters(),
            "channels": channel_summary,
            "pairs": pairs,
            "assumptions": list(PHASE_ASSUMPTIONS),
            "interpretation": (
                "A synchronized pair keeps an n:m phase relationship more consistently than "
                "independent channels with the same phase dynamics would. It is undirected and "
                "does not establish causation or a physical mechanism."
            ),
        }

    def _pair(
        self,
        first: Dict[str, Any],
        second: Dict[str, Any],
        name_x: str,
        name_y: str,
        n: int,
    ) -> Dict[str, Any]:
        pair: Dict[str, Any] = {
            "channels": [name_x, name_y],
            "status": "inconclusive",
            "reason": None,
            "candidate_ratios": [],
            "selected_ratio": None,
            "plv": None,
            "mean_phase_difference_rad": None,
            "p_value": None,
            "adjusted_p_value": None,
            "effective_samples": None,
            "n_shifts": None,
            "minimum_attainable_p_value": None,
            "phase_dynamics": None,
        }
        for name, record in ((name_x, first), (name_y, second)):
            if not record["usable"]:
                pair["reason"] = f"{name}: {record['reason']}"
                return pair
        if self.ratios is not None:
            ratios = list(self.ratios)
        else:
            ratios = farey_ratio_candidates(
                first["dominant_frequency_hz"],
                second["dominant_frequency_hz"],
                self.max_ratio_order,
                self.ratio_tolerance,
            )
        pair["candidate_ratios"] = [[n_x, m_y] for n_x, m_y in ratios]
        if not ratios:
            pair["status"] = "not_tested"
            ratio = second["dominant_frequency_hz"] / first["dominant_frequency_hz"]
            pair["reason"] = (
                f"frequency ratio {ratio:.3f} is not within {self.ratio_tolerance:g} of any "
                f"n/m with n, m <= {self.max_ratio_order}"
            )
            return pair

        edge = max(first["edge_samples"], second["edge_samples"]) if self.trim_edges else 0
        phase_x = first["band"].phase[edge : n - edge]
        phase_y = second["band"].phase[edge : n - edge]
        if phase_x.size < 32:
            pair["reason"] = "fewer than 32 samples remain after trimming filter edges"
            return pair
        duration = phase_x.size / self.sampling_rate
        cycles = min(first["dominant_frequency_hz"], second["dominant_frequency_hz"]) * duration
        if cycles < self.min_cycles:
            pair["reason"] = (
                f"{cycles:.1f} cycles of the slower channel in the analysed window "
                f"(< {self.min_cycles:g})"
            )
            return pair

        test = phase_shift_test(phase_x, phase_y, ratios, seam_correction=self.seam_correction)
        n_x, m_y = test["selected_ratio"]
        plv, offset = phase_locking_value(phase_x, phase_y, n_x, m_y)
        pair.update(
            {
                "selected_ratio": [n_x, m_y],
                "plv": plv,
                "mean_phase_difference_rad": offset,
                "p_value": test["p_value"],
                "effective_samples": effective_phase_samples(
                    np.exp(1j * n_x * phase_x), np.exp(1j * m_y * phase_y)
                ),
                "n_shifts": test["n_shifts"],
                "minimum_attainable_p_value": test["minimum_attainable_p_value"],
                "cycles_in_window": float(cycles),
                "analysed_samples": int(phase_x.size),
                "phase_dynamics": phase_difference_dynamics(
                    phase_x, phase_y, n_x, m_y, self.sampling_rate
                ),
            }
        )
        return pair


__all__ = [
    "AnalyticBand",
    "DEFAULT_RELATIVE_BANDWIDTH",
    "PHASE_ASSUMPTIONS",
    "PHASE_SYNC_METHOD_VERSION",
    "PhaseSynchronyAnalyzer",
    "band_limited_analytic_signal",
    "dominant_frequency",
    "effective_phase_samples",
    "farey_ratio_candidates",
    "holm_adjust",
    "instantaneous_frequency",
    "phase_difference_dynamics",
    "phase_locking_value",
    "phase_shift_test",
    "single_change_point",
    "windowed_phase_locking",
]
