"""Do phase and harmonic mathematics add measurable value to DRR's relationship tests?

The benchmark simulates pairs of channels whose relationship is known and asks
each method to declare whether they are related. *Related* means the second
channel's dynamics depend on the first: a coupled phase oscillator, at a 1:1,
1:2 or 2:3 frequency ratio, with or without delay, amplitude modulation,
missing samples, irregular timestamps, or a frequency sweep. *Unrelated*
pairs are built to mislead: independent oscillators at similar, identical or
harmonically related frequencies, red noise, phase-randomized copies of a
coupled pair (the classic negative control: same spectra, no relationship),
random walks, and independent oscillators that share a frequency sweep or an
amplitude trend.

Every method sees the same series. Missing values and irregular timestamps are
first interpolated linearly onto the regular grid, and the interpolation is
recorded. The methods are existing DRR rooting (lagged correlation and transfer
entropy with DRR's circular-shift surrogates), conventional baselines with
their textbook thresholds (Pearson correlation, Welch coherence, the
phase-locking value with a Rayleigh threshold), Welch coherence with a
circular-shift null, the proposed phase-synchrony test of
:mod:`drr_framework.phase_sync`, four ablations of it, an oriented-area test of
:mod:`drr_framework.trajectory_geometry`, the union of DRR rooting and the
phase test, and both of those behind the data-quality gate of
:mod:`drr_framework.data_quality`.

Conditions are split. **Development** conditions (100 Hz, 1024 samples) were
used while the methods and the gate's thresholds were set. **Held-out**
conditions use other sampling rates, lengths, frequencies, coupling and noise,
other seeds, and seven condition types never run in development. The
hypotheses and their decision rule were fixed in ``DEFAULT_CONFIG`` before the
held-out conditions were first run, and ``preregistration_sha256`` fingerprints
them. Hypotheses are evaluated on held-out rows only. The artifact stores each
trial's outcome as one character, so every rate, interval and outcome can be
recomputed from the rows; ``tests/test_resonance_math_benchmark.py`` does so
and reruns a prefix of committed rows exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._spectral import hilbert
from .calibration import binomial_rate, simulate_ar1
from .data_quality import DEFAULT_THRESHOLDS, assess_series_quality, regularize_series
from .dependent_noise import simulate_random_walk
from .modules import PYINFORM_AVAILABLE, RootingAnalyzer, phase_randomized_surrogate
from .phase_sync import (
    DEFAULT_RELATIVE_BANDWIDTH,
    PhaseSynchronyAnalyzer,
    band_limited_analytic_signal,
    dominant_frequency,
    farey_ratio_candidates,
    phase_shift_test,
)
from .trajectory_geometry import lag_one_antisymmetric_covariance, oriented_area_matrix

SCHEMA_VERSION = "drr-resonance-math-benchmark-v1"
TWO_PI = 2.0 * np.pi

#: Every method, the family it belongs to, and what it does.
METHODS: Dict[str, Dict[str, str]] = {
    "drr_rooting": {
        "family": "existing DRR",
        "description": "lagged correlation, lags 1-4, 99 circular-shift surrogates, max-statistic correction",
    },
    "drr_rooting_te": {
        "family": "existing DRR",
        "description": "the same test with the transfer-entropy backend (pyinform)",
    },
    "pearson_analytic": {
        "family": "conventional",
        "description": "Pearson correlation at lag 0, Fisher z test assuming independent samples",
    },
    "coherence_analytic": {
        "family": "conventional",
        "description": "Welch magnitude-squared coherence at the first channel's peak, threshold (1 - C)^(K - 1) for K segments",
    },
    "coherence_shift": {
        "family": "conventional statistic, shift null",
        "description": "the same coherence against every circular shift of the second channel",
    },
    "plv_rayleigh": {
        "family": "conventional",
        "description": "1:1 phase-locking value with the Rayleigh threshold exp(-N PLV^2) assuming independent samples",
    },
    "phase_sync": {
        "family": "proposed",
        "description": "n:m phase-locking value, Farey ratio search, exact seam-corrected shift test",
    },
    "phase_sync_1to1": {
        "family": "ablation",
        "description": "phase_sync with the ratio fixed at 1:1",
    },
    "phase_sync_no_seam": {
        "family": "ablation",
        "description": "phase_sync without the seam-correcting phase ramp",
    },
    "phase_sync_no_trim": {
        "family": "ablation",
        "description": "phase_sync without trimming filter edges",
    },
    "phase_sync_broadband": {
        "family": "ablation",
        "description": "phase_sync on the Hilbert phase of the unfiltered series",
    },
    "oriented_area": {
        "family": "geometry",
        "description": "normalized oriented area against every circular shift, two-sided",
    },
    "drr_plus_phase": {
        "family": "proposed, combined",
        "description": "related if drr_rooting or phase_sync is significant at alpha / 2 (Bonferroni)",
    },
    "phase_sync_gated": {
        "family": "proposed, gated",
        "description": "phase_sync, abstaining when the data-quality gate abstains",
    },
    "drr_plus_phase_gated": {
        "family": "proposed, combined, gated",
        "description": "drr_plus_phase, abstaining when the data-quality gate abstains",
    },
}

DIRECTIONAL_METHODS = ("drr_rooting", "drr_rooting_te", "oriented_area")

#: Methods that report a :class:`PhaseSynchronyAnalyzer` status, coded one
#: character per trial.
STATUS_METHODS = (
    "phase_sync",
    "phase_sync_1to1",
    "phase_sync_no_seam",
    "phase_sync_no_trim",
    "phase_sync_gated",
)
STATUS_CODES = {"synchronized": "S", "not_detected": "N", "inconclusive": "I", "not_tested": "X"}
STATUS_NAMES = {
    "S": "status_synchronized",
    "N": "status_not_detected",
    "I": "status_inconclusive",
    "X": "status_not_tested",
    "A": "status_abstained",
}


def _oscillators(**fields: Any) -> Dict[str, Any]:
    spec = {
        "family": "phase_oscillators",
        "ratio": [1, 1],
        "coupling": 0.0,
        "delay_seconds": 0.0,
        "noise": 0.5,
        "am_depth": 0.0,
        "am_rate_hz": 0.0,
        "amplitude_ramp": False,
        "chirp_sweep_hz": 0.0,
        "common_chirp": False,
        "missing_fraction": 0.0,
        "timing_jitter": 0.0,
        "burn_in_seconds": 10.0,
    }
    spec.update(fields)
    return spec


_DEV = {"sampling_rate": 100.0, "n_samples": 1024}
_HOLD = {"sampling_rate": 250.0, "n_samples": 2048}
_SHORT = {"sampling_rate": 50.0, "n_samples": 512}

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20261028,
    "alpha": 0.05,
    "rooting": {"max_lag": 4, "n_surrogates": 99},
    "transfer_entropy": True,
    "phase": {
        "relative_bandwidth": DEFAULT_RELATIVE_BANDWIDTH,
        "max_ratio_order": 3,
        "ratio_tolerance": 0.08,
        "min_effective_samples": 20.0,
        "min_cycles": 10.0,
    },
    "coherence": {"max_segment": 256},
    "quality_thresholds": dict(DEFAULT_THRESHOLDS),
    "development": {
        "n_trials": {"related": 100, "unrelated": 200},
        "conditions": [
            dict(
                id="dev_coupled_1to1",
                label="Coupled 1:1",
                related=True,
                stationary=True,
                seed_offset=101,
                **_DEV,
                **_oscillators(fx=10.0, fy=10.25, coupling=6.0, phase_diffusion=2.0),
            ),
            dict(
                id="dev_harmonic_1to2",
                label="Coupled 1:2 (harmonic)",
                related=True,
                stationary=True,
                seed_offset=102,
                **_DEV,
                **_oscillators(fx=10.0, fy=20.4, coupling=10.0, ratio=[2, 1], phase_diffusion=2.0),
            ),
            dict(
                id="dev_delayed_1to1",
                label="Coupled 1:1, 0.1 s delay",
                related=True,
                stationary=True,
                seed_offset=103,
                **_DEV,
                **_oscillators(
                    fx=10.0, fy=10.25, coupling=6.0, phase_diffusion=2.0, delay_seconds=0.1
                ),
            ),
            dict(
                id="dev_am_coupled",
                label="Coupled 1:1, amplitude modulated",
                related=True,
                stationary=True,
                seed_offset=104,
                **_DEV,
                **_oscillators(
                    fx=10.0,
                    fy=10.25,
                    coupling=6.0,
                    phase_diffusion=2.0,
                    am_depth=0.8,
                    am_rate_hz=0.3,
                ),
            ),
            dict(
                id="dev_coupled_missing",
                label="Coupled 1:1, 10% missing",
                related=True,
                stationary=True,
                seed_offset=105,
                **_DEV,
                **_oscillators(
                    fx=10.0, fy=10.25, coupling=6.0, phase_diffusion=2.0, missing_fraction=0.1
                ),
            ),
            dict(
                id="dev_coupled_chirp",
                label="Coupled 1:1, swept 8-12 Hz",
                related=True,
                stationary=False,
                seed_offset=106,
                **_DEV,
                **_oscillators(
                    fx=10.0, fy=10.0, coupling=20.0, phase_diffusion=2.0, chirp_sweep_hz=4.0
                ),
            ),
            dict(
                id="dev_independent_similar",
                label="Independent, similar frequencies",
                related=False,
                stationary=True,
                seed_offset=201,
                **_DEV,
                **_oscillators(fx=10.0, fy=10.25, phase_diffusion=2.0),
            ),
            dict(
                id="dev_independent_coherent",
                label="Independent, identical frequency, coherent",
                related=False,
                stationary=True,
                seed_offset=202,
                **_DEV,
                **_oscillators(fx=10.0, fy=10.0, phase_diffusion=0.5),
            ),
            dict(
                id="dev_independent_harmonic",
                label="Independent, exact 1:2 frequencies",
                related=False,
                stationary=True,
                seed_offset=203,
                **_DEV,
                **_oscillators(fx=10.0, fy=20.0, phase_diffusion=2.0),
            ),
            dict(
                id="dev_ar1",
                label="Independent AR(1), coefficient 0.9",
                related=False,
                stationary=True,
                seed_offset=204,
                family="ar1",
                ar_coefficient=0.9,
                **_DEV,
            ),
            dict(
                id="dev_phase_randomized",
                label="Phase-randomized coupled 1:1 (negative control)",
                related=False,
                stationary=True,
                seed_offset=205,
                family="phase_randomized",
                **_DEV,
                base=_oscillators(fx=10.0, fy=10.25, coupling=6.0, phase_diffusion=2.0),
            ),
            dict(
                id="dev_independent_irregular",
                label="Independent, irregular timestamps",
                related=False,
                stationary=True,
                seed_offset=206,
                **_DEV,
                **_oscillators(fx=10.0, fy=10.25, phase_diffusion=2.0, timing_jitter=0.4),
            ),
            dict(
                id="dev_random_walk",
                label="Independent random walks",
                related=False,
                stationary=False,
                seed_offset=207,
                family="random_walk",
                **_DEV,
            ),
            dict(
                id="dev_common_chirp",
                label="Independent, shared 8-12 Hz sweep",
                related=False,
                stationary=False,
                seed_offset=208,
                **_DEV,
                **_oscillators(
                    fx=10.0, fy=10.0, phase_diffusion=2.0, chirp_sweep_hz=4.0, common_chirp=True
                ),
            ),
        ],
    },
    "holdout": {
        "n_trials": {"related": 200, "unrelated": 400},
        "conditions": [
            dict(
                id="coupled_1to1",
                label="Coupled 1:1",
                related=True,
                stationary=True,
                seed_offset=301,
                **_HOLD,
                **_oscillators(fx=7.0, fy=7.15, coupling=5.0, phase_diffusion=3.0, noise=0.6),
            ),
            dict(
                id="harmonic_1to2",
                label="Coupled 1:2 (harmonic)",
                related=True,
                stationary=True,
                seed_offset=302,
                **_HOLD,
                **_oscillators(
                    fx=6.0, fy=12.3, coupling=8.0, ratio=[2, 1], phase_diffusion=3.0, noise=0.6
                ),
            ),
            dict(
                id="harmonic_2to3",
                label="Coupled 2:3 (harmonic)",
                related=True,
                stationary=True,
                seed_offset=303,
                **_HOLD,
                **_oscillators(
                    fx=6.0, fy=9.1, coupling=8.0, ratio=[3, 2], phase_diffusion=3.0, noise=0.6
                ),
            ),
            dict(
                id="delayed_1to1",
                label="Coupled 1:1, 0.15 s delay",
                related=True,
                stationary=True,
                seed_offset=304,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.15,
                    coupling=5.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    delay_seconds=0.15,
                ),
            ),
            dict(
                id="am_coupled",
                label="Coupled 1:1, amplitude modulated",
                related=True,
                stationary=True,
                seed_offset=305,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.15,
                    coupling=5.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    am_depth=0.9,
                    am_rate_hz=0.2,
                ),
            ),
            dict(
                id="coupled_missing",
                label="Coupled 1:1, 15% missing",
                related=True,
                stationary=True,
                seed_offset=306,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.15,
                    coupling=5.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    missing_fraction=0.15,
                ),
            ),
            dict(
                id="coupled_irregular",
                label="Coupled 1:1, irregular timestamps",
                related=True,
                stationary=True,
                seed_offset=307,
                **_HOLD,
                **_oscillators(
                    fx=7.0, fy=7.15, coupling=5.0, phase_diffusion=3.0, noise=0.6, timing_jitter=0.3
                ),
            ),
            dict(
                id="short_coupled_1to1",
                label="Coupled 1:1, 50 Hz, 512 samples",
                related=True,
                stationary=True,
                seed_offset=308,
                **_SHORT,
                **_oscillators(fx=4.0, fy=4.1, coupling=4.0, phase_diffusion=2.0),
            ),
            dict(
                id="amplitude_ramp_coupled",
                label="Coupled 1:1, shared amplitude trend",
                related=True,
                stationary=False,
                seed_offset=309,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.15,
                    coupling=5.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    amplitude_ramp=True,
                ),
            ),
            dict(
                id="coupled_chirp",
                label="Coupled 1:1, swept 6-8 Hz",
                related=True,
                stationary=False,
                seed_offset=310,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.0,
                    coupling=15.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    chirp_sweep_hz=2.0,
                ),
            ),
            dict(
                id="independent_similar",
                label="Independent, similar frequencies",
                related=False,
                stationary=True,
                seed_offset=401,
                **_HOLD,
                **_oscillators(fx=7.0, fy=7.15, phase_diffusion=3.0, noise=0.6),
            ),
            dict(
                id="independent_coherent",
                label="Independent, identical frequency, coherent",
                related=False,
                stationary=True,
                seed_offset=402,
                **_HOLD,
                **_oscillators(fx=7.0, fy=7.0, phase_diffusion=0.4, noise=0.6),
            ),
            dict(
                id="independent_harmonic_1to2",
                label="Independent, exact 1:2 frequencies",
                related=False,
                stationary=True,
                seed_offset=403,
                **_HOLD,
                **_oscillators(fx=6.0, fy=12.0, phase_diffusion=3.0, noise=0.6),
            ),
            dict(
                id="independent_harmonic_2to3",
                label="Independent, exact 2:3 frequencies",
                related=False,
                stationary=True,
                seed_offset=404,
                **_HOLD,
                **_oscillators(fx=6.0, fy=9.0, phase_diffusion=3.0, noise=0.6),
            ),
            dict(
                id="ar1",
                label="Independent AR(1), coefficient 0.95",
                related=False,
                stationary=True,
                seed_offset=405,
                family="ar1",
                ar_coefficient=0.95,
                **_HOLD,
            ),
            dict(
                id="phase_randomized",
                label="Phase-randomized coupled 1:1 (negative control)",
                related=False,
                stationary=True,
                seed_offset=406,
                family="phase_randomized",
                **_HOLD,
                base=_oscillators(fx=7.0, fy=7.15, coupling=5.0, phase_diffusion=3.0, noise=0.6),
            ),
            dict(
                id="independent_missing",
                label="Independent, 15% missing",
                related=False,
                stationary=True,
                seed_offset=407,
                **_HOLD,
                **_oscillators(
                    fx=7.0, fy=7.15, phase_diffusion=3.0, noise=0.6, missing_fraction=0.15
                ),
            ),
            dict(
                id="independent_irregular",
                label="Independent, irregular timestamps",
                related=False,
                stationary=True,
                seed_offset=408,
                **_HOLD,
                **_oscillators(fx=7.0, fy=7.15, phase_diffusion=3.0, noise=0.6, timing_jitter=0.3),
            ),
            dict(
                id="short_independent_similar",
                label="Independent, 50 Hz, 512 samples",
                related=False,
                stationary=True,
                seed_offset=409,
                **_SHORT,
                **_oscillators(fx=4.0, fy=4.1, phase_diffusion=2.0),
            ),
            dict(
                id="amplitude_ramp_independent",
                label="Independent, shared amplitude trend",
                related=False,
                stationary=False,
                seed_offset=410,
                **_HOLD,
                **_oscillators(
                    fx=7.0, fy=7.15, phase_diffusion=3.0, noise=0.6, amplitude_ramp=True
                ),
            ),
            dict(
                id="random_walk",
                label="Independent random walks",
                related=False,
                stationary=False,
                seed_offset=411,
                family="random_walk",
                **_HOLD,
            ),
            dict(
                id="common_chirp",
                label="Independent, shared 6-8 Hz sweep",
                related=False,
                stationary=False,
                seed_offset=412,
                **_HOLD,
                **_oscillators(
                    fx=7.0,
                    fy=7.0,
                    phase_diffusion=3.0,
                    noise=0.6,
                    chirp_sweep_hz=2.0,
                    common_chirp=True,
                ),
            ),
        ],
    },
    "preregistration": {
        "decision_rule": (
            "A method is controlled on an unrelated condition when the Wilson 95% lower bound of "
            "its rate of declared relationships is at most alpha, and inflated when that bound "
            "exceeds alpha. A detection rate is the rate of declared relationships on a related "
            "condition. Abstentions are never counted as declared relationships. Hypotheses are "
            "evaluated on held-out conditions only. The rule, conditions, seeds, trial counts, "
            "methods, thresholds, and hypotheses were fixed before the held-out conditions were "
            "first run and are not changed after the rows are seen."
        ),
        "hypotheses": [
            {
                "id": "H1",
                "statement": (
                    "The phase-synchrony test is controlled on every stationary unrelated held-out "
                    "condition, including the phase-randomized negative control and independent "
                    "oscillators at identical or harmonically related frequencies."
                ),
            },
            {
                "id": "H2",
                "statement": (
                    "Each conventional baseline with a textbook threshold (Pearson correlation, Welch "
                    "coherence, Rayleigh phase-locking value) is inflated on at least one stationary "
                    "unrelated held-out condition."
                ),
            },
            {
                "id": "H3",
                "statement": (
                    "On every held-out harmonic coupling (1:2 and 2:3), the phase-synchrony test detects "
                    "the relationship at least 0.25 more often than existing DRR rooting."
                ),
            },
            {
                "id": "H4",
                "statement": (
                    "The ratio search carries H3: with the ratio fixed at 1:1 the phase test detects each "
                    "held-out harmonic coupling in at most 15% of trials."
                ),
            },
            {
                "id": "H5",
                "statement": (
                    "The seam correction is load-bearing: without it the phase test is inflated on the "
                    "held-out coherent identical-frequency condition, where the corrected test is "
                    "controlled."
                ),
            },
            {
                "id": "H6",
                "statement": (
                    "Existing DRR rooting is inflated on the held-out coherent identical-frequency "
                    "condition."
                ),
            },
            {
                "id": "H7",
                "statement": (
                    "Without the gate, the phase-synchrony test is inflated on the held-out shared-sweep "
                    "condition, because its stationarity assumption fails there."
                ),
            },
            {
                "id": "H8",
                "statement": (
                    "The data-quality gate abstains in at least 80% of trials on the held-out random "
                    "walks and shared sweep, and in at most 10% of trials on every stationary held-out "
                    "condition."
                ),
            },
            {
                "id": "H9",
                "statement": (
                    "On the stationary held-out 1:1 couplings, the phase-synchrony test's mean detection "
                    "rate is no more than 0.10 below existing DRR rooting's."
                ),
            },
        ],
    },
}

HARMONIC_CONDITIONS = ("harmonic_1to2", "harmonic_2to3")
ONE_TO_ONE_CONDITIONS = (
    "coupled_1to1",
    "delayed_1to1",
    "am_coupled",
    "coupled_missing",
    "coupled_irregular",
    "short_coupled_1to1",
)
COHERENT_CONDITION = "independent_coherent"
GATE_TARGETS = ("random_walk", "common_chirp")


# --------------------------------------------------------------------------- simulators


def _simulate_oscillators(
    spec: Mapping[str, Any], n: int, fs: float, rng: np.random.Generator
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Two stochastic phase oscillators; the second may be driven by the first.

    ``theta_x`` diffuses around its natural frequency (Wiener phase noise of
    intensity ``phase_diffusion`` rad^2/s). ``theta_y`` adds
    ``coupling * sin(n theta_x(t - delay) - m theta_y)`` to its own drift, the
    ``n:m`` Kuramoto coupling, integrated by Euler-Maruyama at ``1 / fs``. A
    burn-in lets the coupled phase settle before observation. Observed values
    are ``A(t) cos(theta) + noise``. Returns values (with ``NaN`` for missing
    samples) and jittered timestamps when ``timing_jitter`` is set.
    """
    dt = 1.0 / fs
    burn = int(round(float(spec["burn_in_seconds"]) * fs))
    delay = int(round(float(spec["delay_seconds"]) * fs))
    total = burn + n + 1
    diffusion = float(spec["phase_diffusion"])
    coupling = float(spec["coupling"])
    n_ratio, m_ratio = (int(v) for v in spec["ratio"])
    sweep = float(spec["chirp_sweep_hz"])
    offset = np.full(total, -0.5 * sweep)
    offset[burn:] = -0.5 * sweep + sweep * np.arange(total - burn) / max(n - 1, 1)
    noise_x = np.sqrt(2.0 * diffusion * dt) * rng.standard_normal(total)
    noise_y = np.sqrt(2.0 * diffusion * dt) * rng.standard_normal(total)
    theta_x = np.empty(total)
    theta_x[0] = rng.uniform(0.0, TWO_PI)
    theta_x[1:] = theta_x[0] + np.cumsum(
        TWO_PI * (float(spec["fx"]) + offset[:-1]) * dt + noise_x[:-1]
    )
    drift_y = TWO_PI * (float(spec["fy"]) + (offset if spec["common_chirp"] else 0.0)) * dt
    drift_y = np.broadcast_to(drift_y, (total,))
    theta_y = np.empty(total)
    theta_y[0] = rng.uniform(0.0, TWO_PI)
    step = coupling * dt
    for k in range(total - 1):
        drive = step * math.sin(
            n_ratio * theta_x[k - delay if k >= delay else 0] - m_ratio * theta_y[k]
        )
        theta_y[k + 1] = theta_y[k] + drift_y[k] + drive + noise_y[k]

    phases_x: np.ndarray = theta_x[burn:]
    phases_y: np.ndarray = theta_y[burn:]
    times: np.ndarray = np.arange(n) * dt
    timestamps: Optional[np.ndarray] = None
    observed_times: np.ndarray
    jitter = float(spec["timing_jitter"])
    if jitter > 0:
        timestamps = times + rng.uniform(-0.5, 0.5, n) * jitter * dt
        grid = np.arange(n + 1) * dt
        phases_x = np.interp(timestamps, grid, phases_x)
        phases_y = np.interp(timestamps, grid, phases_y)
        observed_times = timestamps
    else:
        phases_x, phases_y = phases_x[:n], phases_y[:n]
        observed_times = times
    amp_x: np.ndarray = np.ones(n)
    amp_y: np.ndarray = np.ones(n)
    depth = float(spec["am_depth"])
    if depth > 0:
        rate = float(spec["am_rate_hz"])
        amp_x = 1.0 + depth * np.sin(TWO_PI * rate * observed_times + rng.uniform(0.0, TWO_PI))
        amp_y = 1.0 + depth * np.sin(TWO_PI * rate * observed_times + rng.uniform(0.0, TWO_PI))
    if spec["amplitude_ramp"]:
        ramp = 0.25 + 1.5 * observed_times / (n * dt)
        amp_x, amp_y = amp_x * ramp, amp_y * ramp
    noise = float(spec["noise"])
    data = np.column_stack(
        (
            amp_x * np.cos(phases_x) + noise * rng.standard_normal(n),
            amp_y * np.cos(phases_y) + noise * rng.standard_normal(n),
        )
    )
    missing = float(spec["missing_fraction"])
    if missing > 0:
        data[rng.random(data.shape) < missing] = np.nan
    return data, timestamps


def simulate_condition(
    spec: Mapping[str, Any], rng: np.random.Generator
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Draw one pair of channels for a condition: ``(values, timestamps or None)``."""
    family = spec["family"]
    n, fs = int(spec["n_samples"]), float(spec["sampling_rate"])
    if family == "phase_oscillators":
        return _simulate_oscillators(spec, n, fs, rng)
    if family == "ar1":
        return simulate_ar1(n, 2, float(spec["ar_coefficient"]), rng), None
    if family == "random_walk":
        return simulate_random_walk(n, 2, rng), None
    if family == "phase_randomized":
        data, _ = _simulate_oscillators(spec["base"], n, fs, rng)
        return phase_randomized_surrogate(data, rng), None
    raise ValueError(f"unknown condition family {family!r}")


# --------------------------------------------------------------------------- methods

Outcome = Dict[str, Any]


def _rooting(data: np.ndarray, cfg: Mapping[str, Any], seed: int, backend: str) -> Outcome:
    result = RootingAnalyzer().analyze(
        data,
        max_lag=int(cfg["rooting"]["max_lag"]),
        n_surrogates=int(cfg["rooting"]["n_surrogates"]),
        random_state=seed,
        alpha=float(cfg["alpha"]),
        method=backend,
        surrogate_method="circular_shift",
        correction="max_statistic",
    )
    adjusted = result["adjusted_p_values"]
    candidates = [
        float(adjusted[int(e["source"][4:]), int(e["target"][4:])])
        for e in result["candidate_edges"]
    ]
    edges = {(e["source"], e["target"]) for e in result["significant_edges"]}
    forward, backward = ("dim_0", "dim_1") in edges, ("dim_1", "dim_0") in edges
    direction = "=" if forward and backward else ">" if forward else "<" if backward else "."
    return {
        "p": min(candidates) if candidates else None,
        "direction": direction,
        "method_used": result["method"],
    }


def _pearson(data: np.ndarray) -> Outcome:
    r = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    z = math.atanh(min(max(r, -0.999999999), 0.999999999)) * math.sqrt(data.shape[0] - 3)
    return {"p": math.erfc(abs(z) / math.sqrt(2.0))}


def _coherence_profile(
    data: np.ndarray, fs: float, frequency: float, max_segment: int
) -> Tuple[np.ndarray, int]:
    """Welch coherence at ``frequency`` for every circular shift of the second channel.

    Hann windows of ``min(max_segment, N // 4)`` samples overlap by half. Each
    segment's Fourier coefficient at ``frequency`` is a windowed sum; for the
    shifted channel those sums are one circular cross-correlation, so every
    shift costs one FFT. Shift 0 is the ordinary estimate.
    """
    x = data[:, 0] - data[:, 0].mean()
    y = data[:, 1] - data[:, 1].mean()
    n = x.size
    length = int(min(max_segment, n // 4))
    hop = length // 2
    starts = np.arange(0, n - length + 1, hop)
    window = np.hanning(length)
    kernel = np.zeros(n, dtype=complex)
    kernel[:length] = window * np.exp(-1j * TWO_PI * frequency * np.arange(length) / fs)
    x_coeff = np.array([np.sum(kernel[:length] * x[s : s + length]) for s in starts])
    correlation = np.fft.ifft(np.conj(np.fft.fft(np.conj(kernel))) * np.fft.fft(y))
    shifts = np.arange(n)
    y_coeff = correlation[(starts[:, None] + shifts[None, :]) % n]
    cross = np.sum(np.conj(x_coeff)[:, None] * y_coeff, axis=0)
    power_x = float(np.sum(np.abs(x_coeff) ** 2))
    power_y = np.sum(np.abs(y_coeff) ** 2, axis=0)
    denominator = power_x * power_y
    coherence = np.divide(np.abs(cross) ** 2, denominator, out=np.zeros(n), where=denominator > 0)
    return coherence, int(starts.size)


def _coherence(data: np.ndarray, fs: float, cfg: Mapping[str, Any]) -> Tuple[Outcome, Outcome]:
    frequency = dominant_frequency(data[:, 0], fs)
    if not 0 < frequency < 0.5 * fs:
        return {"p": None}, {"p": None}
    profile, segments = _coherence_profile(
        data, fs, frequency, int(cfg["coherence"]["max_segment"])
    )
    analytic = float((1.0 - min(profile[0], 1.0)) ** (segments - 1))
    shifted = float(np.mean(profile >= profile[0] - 1e-12))
    return {"p": analytic}, {"p": shifted}


def _plv_rayleigh(data: np.ndarray, fs: float, cfg: Mapping[str, Any]) -> Outcome:
    phases = []
    edge = 0
    for column in data.T:
        frequency = dominant_frequency(column, fs)
        if not 0 < frequency < 0.5 * fs:
            return {"p": None}
        band = band_limited_analytic_signal(
            column, fs, frequency, float(cfg["phase"]["relative_bandwidth"])
        )
        phases.append(band.phase)
        edge = max(edge, band.edge_samples)
    px, py = phases[0][edge : data.shape[0] - edge], phases[1][edge : data.shape[0] - edge]
    if px.size < 32:
        return {"p": None}
    plv = float(np.abs(np.mean(np.exp(1j * (px - py)))))
    return {"p": float(math.exp(-px.size * plv**2))}


def _phase(data: np.ndarray, fs: float, cfg: Mapping[str, Any], **options: Any) -> Outcome:
    analyzer = PhaseSynchronyAnalyzer(
        fs, alpha=float(cfg["alpha"]), **dict(cfg["phase"]), **options
    )
    pair = analyzer.analyze(data)["pairs"][0]
    return {
        "p": pair["p_value"],
        "status": pair["status"],
        "inconclusive": pair["status"] == "inconclusive",
    }


def _phase_broadband(data: np.ndarray, fs: float, cfg: Mapping[str, Any]) -> Outcome:
    frequencies = [dominant_frequency(column, fs) for column in data.T]
    ratios = farey_ratio_candidates(
        frequencies[0],
        frequencies[1],
        int(cfg["phase"]["max_ratio_order"]),
        float(cfg["phase"]["ratio_tolerance"]),
    )
    if not ratios:
        return {"p": None}
    phases = [np.angle(hilbert(column - column.mean())) for column in data.T]
    return {"p": phase_shift_test(phases[0], phases[1], ratios)["p_value"]}


def _oriented_area(data: np.ndarray) -> Outcome:
    x = data[:, 0] - data[:, 0].mean()
    y = data[:, 1] - data[:, 1].mean()
    n = x.size
    scale = n * float(x.std()) * float(y.std())
    if scale <= 0:
        return {"p": None, "direction": "."}
    correlation = np.real(np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y)))
    shifts = np.arange(n)
    area = 0.5 * (correlation[(shifts + 1) % n] - correlation[(shifts - 1) % n]) / scale
    p = float(np.mean(np.abs(area) >= abs(area[0]) - 1e-12))
    return {"p": p, "direction": ">" if area[0] > 0 else "<"}


def _gate(
    data: np.ndarray, fs: float, regularization: Optional[Dict[str, Any]], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    quality = assess_series_quality(
        data, fs, regularization=regularization, thresholds=dict(cfg["quality_thresholds"])
    )
    categories = set()
    for reason in quality["reasons"]:
        if "DFA exponent" in reason:
            categories.add("scaling")
        elif "drifts" in reason:
            categories.add("drift")
        elif "interpolated" in reason:
            categories.add("missing")
        elif "gap" in reason:
            categories.add("gap")
        else:
            categories.add("other")
    return {"abstain": bool(quality["abstain"]), "categories": sorted(categories)}


def _union_p(rooting: Outcome, phase: Outcome) -> Optional[float]:
    values = [p for p in (rooting.get("p"), phase.get("p")) if p is not None]
    return min(1.0, 2.0 * min(values)) if values else None


def run_trial(spec: Mapping[str, Any], cfg: Mapping[str, Any], trial: int) -> Dict[str, Any]:
    """Draw one trial of a condition and apply every method to it.

    Each trial has its own seed stream, ``SeedSequence([seed, seed_offset,
    trial])``, so any trial reproduces on its own and in any order.
    """
    stream = np.random.SeedSequence([int(cfg["seed"]), int(spec["seed_offset"]), int(trial)])
    data_seq, rooting_seq, te_seq = stream.spawn(3)
    rng = np.random.default_rng(data_seq)
    fs = float(spec["sampling_rate"])
    raw, timestamps = simulate_condition(spec, rng)
    regularization = None
    data = raw
    if timestamps is not None or not np.all(np.isfinite(raw)):
        data, regularization = regularize_series(raw, timestamps, fs)

    outcomes: Dict[str, Outcome] = {}
    timings: Dict[str, float] = {}

    def timed(name: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = function()
        timings[name] = time.perf_counter() - started
        return value

    rooting_seed = int(rooting_seq.generate_state(1)[0])
    te_seed = int(te_seq.generate_state(1)[0])
    outcomes["drr_rooting"] = timed(
        "drr_rooting", lambda: _rooting(data, cfg, rooting_seed, "lagged_correlation")
    )
    if cfg.get("transfer_entropy", True):
        outcomes["drr_rooting_te"] = timed(
            "drr_rooting_te", lambda: _rooting(data, cfg, te_seed, "transfer_entropy")
        )
    outcomes["pearson_analytic"] = timed("pearson_analytic", lambda: _pearson(data))
    coherence = timed("coherence", lambda: _coherence(data, fs, cfg))
    outcomes["coherence_analytic"], outcomes["coherence_shift"] = coherence
    timings["coherence_analytic"] = timings["coherence_shift"] = timings.pop("coherence")
    outcomes["plv_rayleigh"] = timed("plv_rayleigh", lambda: _plv_rayleigh(data, fs, cfg))
    outcomes["phase_sync"] = timed("phase_sync", lambda: _phase(data, fs, cfg))
    outcomes["phase_sync_1to1"] = timed(
        "phase_sync_1to1", lambda: _phase(data, fs, cfg, ratios=[(1, 1)])
    )
    outcomes["phase_sync_no_seam"] = timed(
        "phase_sync_no_seam", lambda: _phase(data, fs, cfg, seam_correction=False)
    )
    outcomes["phase_sync_no_trim"] = timed(
        "phase_sync_no_trim", lambda: _phase(data, fs, cfg, trim_edges=False)
    )
    outcomes["phase_sync_broadband"] = timed(
        "phase_sync_broadband", lambda: _phase_broadband(data, fs, cfg)
    )
    outcomes["oriented_area"] = timed("oriented_area", lambda: _oriented_area(data))
    gate = timed("gate", lambda: _gate(data, fs, regularization, cfg))

    union = _union_p(outcomes["drr_rooting"], outcomes["phase_sync"])
    outcomes["drr_plus_phase"] = {"p": union}
    outcomes["phase_sync_gated"] = dict(outcomes["phase_sync"], abstain=gate["abstain"])
    outcomes["drr_plus_phase_gated"] = {"p": union, "abstain": gate["abstain"]}

    identity = oriented_area_matrix(data, normalize=False) - lag_one_antisymmetric_covariance(data)
    scale = max(1.0, float(np.abs(lag_one_antisymmetric_covariance(data)).max()))
    return {
        "outcomes": outcomes,
        "gate": gate,
        "timings": timings,
        "identity_error": float(np.abs(identity).max() / scale),
    }


# --------------------------------------------------------------------------- rows


def level_code(p: Optional[float], abstain: bool = False) -> str:
    """One character per trial: ``1`` p <= 0.01, ``5`` p <= 0.05, ``T`` p <= 0.10,
    ``.`` larger, ``-`` not tested, ``A`` abstained."""
    if abstain:
        return "A"
    if p is None:
        return "-"
    if p <= 0.01:
        return "1"
    if p <= 0.05:
        return "5"
    if p <= 0.10:
        return "T"
    return "."


_DECLARED = {0.01: "1", 0.05: "15", 0.10: "15T"}


def run_condition(
    spec: Mapping[str, Any], cfg: Mapping[str, Any], n_trials: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """All trials of one condition: ``(row, runtime)``."""
    methods = [m for m in METHODS if cfg.get("transfer_entropy", True) or m != "drr_rooting_te"]
    levels: Dict[str, List[str]] = {m: [] for m in methods}
    directions: Dict[str, List[str]] = {m: [] for m in DIRECTIONAL_METHODS if m in methods}
    statuses: Dict[str, List[str]] = {m: [] for m in methods if m in STATUS_METHODS}
    gate_codes: List[str] = []
    gate_reasons: Dict[str, int] = {}
    timings: Dict[str, List[float]] = {}
    identity = 0.0
    backends: Dict[str, str] = {}
    for trial in range(n_trials):
        result = run_trial(spec, cfg, trial)
        for method in methods:
            outcome = result["outcomes"][method]
            levels[method].append(level_code(outcome.get("p"), bool(outcome.get("abstain"))))
            if method in directions:
                declared = levels[method][-1] in _DECLARED[float(cfg["alpha"])]
                directions[method].append(outcome.get("direction", ".") if declared else ".")
            if method in statuses:
                code = "A" if outcome.get("abstain") else STATUS_CODES[outcome["status"]]
                statuses[method].append(code)
            if "method_used" in outcome:
                backends[method] = outcome["method_used"]
        gate_codes.append("A" if result["gate"]["abstain"] else ".")
        for category in result["gate"]["categories"]:
            gate_reasons[category] = gate_reasons.get(category, 0) + 1
        for name, seconds in result["timings"].items():
            timings.setdefault(name, []).append(seconds)
        identity = max(identity, result["identity_error"])
    row = {
        "condition": spec["id"],
        "label": spec["label"],
        "related": bool(spec["related"]),
        "stationary": bool(spec["stationary"]),
        "n_trials": int(n_trials),
        "levels": {m: "".join(codes) for m, codes in levels.items()},
        "directions": {m: "".join(codes) for m, codes in directions.items()},
        "statuses": {m: "".join(codes) for m, codes in statuses.items()},
        "gate": {"abstain": "".join(gate_codes), "reasons": dict(sorted(gate_reasons.items()))},
        "backends": dict(sorted(backends.items())),
        "max_identity_error": identity,
    }
    runtime = {
        name: {"median_seconds": float(np.median(values))}
        for name, values in sorted(timings.items())
    }
    return row, runtime


def declared_rate(codes: str, alpha: float = 0.05) -> Dict[str, Any]:
    """Rate of declared relationships among all trials, with its Wilson interval."""
    hits = sum(code in _DECLARED[alpha] for code in codes)
    return binomial_rate(hits, len(codes))


def summarize_row(row: Mapping[str, Any], alpha: float) -> Dict[str, Any]:
    """Every rate the report and the hypotheses use, recomputed from one row's codes."""
    summary: Dict[str, Any] = {}
    for method, codes in row["levels"].items():
        entry: Dict[str, Any] = {
            "declared": declared_rate(codes, alpha),
            "declared_at_0.01": declared_rate(codes, 0.01),
            "declared_at_0.10": declared_rate(codes, 0.10),
            "abstained": binomial_rate(codes.count("A"), len(codes)),
            "not_tested": binomial_rate(codes.count("-"), len(codes)),
        }
        if method in row["statuses"]:
            marks = row["statuses"][method]
            for code, name in STATUS_NAMES.items():
                entry[name] = binomial_rate(marks.count(code), len(marks))
        if method in row["directions"] and row["related"]:
            marks = row["directions"][method]
            declared = sum(mark != "." for mark in marks)
            correct = sum(mark == ">" for mark in marks)
            entry["direction_correct_among_declared"] = (
                binomial_rate(correct, declared) if declared else None
            )
        summary[method] = entry
    summary["gate"] = {
        "abstained": binomial_rate(row["gate"]["abstain"].count("A"), row["n_trials"])
    }
    return summary


def is_controlled(rate: Mapping[str, Any], alpha: float) -> bool:
    """The preregistered rule: the Wilson 95% lower bound is at most ``alpha``."""
    return bool(rate["ci95"][0] <= alpha)


# --------------------------------------------------------------------------- decisions


def evaluate_hypotheses(
    rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Apply each preregistered rule to the held-out rows and record the outcome beside it."""
    alpha = float(cfg["alpha"])
    by_id = {row["condition"]: row for row in rows}
    summaries = {cid: summarize_row(row, alpha) for cid, row in by_id.items()}

    def rate(condition: str, method: str, key: str = "declared") -> float:
        return float(summaries[condition][method][key]["rate"])

    def controlled(condition: str, method: str) -> bool:
        return is_controlled(summaries[condition][method]["declared"], alpha)

    stationary_nulls = [r["condition"] for r in rows if not r["related"] and r["stationary"]]
    stationary_all = [r["condition"] for r in rows if r["stationary"]]
    harmonic = [c for c in HARMONIC_CONDITIONS if c in by_id]
    one_to_one = [c for c in ONE_TO_ONE_CONDITIONS if c in by_id]
    targets = [c for c in GATE_TARGETS if c in by_id]
    conventional = ("pearson_analytic", "coherence_analytic", "plv_rayleigh")
    coherent = COHERENT_CONDITION in by_id

    def gate_rate(condition: str) -> float:
        return float(summaries[condition]["gate"]["abstained"]["rate"])

    mean_gap = None
    if one_to_one:
        mean_gap = float(
            np.mean([rate(c, "phase_sync") for c in one_to_one])
            - np.mean([rate(c, "drr_rooting") for c in one_to_one])
        )
    rules = {
        "H1": (
            "phase_sync is controlled on every stationary unrelated condition",
            bool(stationary_nulls) and all(controlled(c, "phase_sync") for c in stationary_nulls),
            stationary_nulls,
        ),
        "H2": (
            "each of pearson_analytic, coherence_analytic, plv_rayleigh is inflated on at least one "
            "stationary unrelated condition",
            bool(stationary_nulls)
            and all(
                any(not controlled(c, method) for c in stationary_nulls) for method in conventional
            ),
            stationary_nulls,
        ),
        "H3": (
            "phase_sync detection rate minus drr_rooting detection rate >= 0.25 on every harmonic "
            "condition",
            bool(harmonic)
            and all(rate(c, "phase_sync") - rate(c, "drr_rooting") >= 0.25 for c in harmonic),
            harmonic,
        ),
        "H4": (
            "phase_sync_1to1 detection rate <= 0.15 on every harmonic condition",
            bool(harmonic) and all(rate(c, "phase_sync_1to1") <= 0.15 for c in harmonic),
            harmonic,
        ),
        "H5": (
            "phase_sync_no_seam is inflated and phase_sync controlled on the coherent condition",
            coherent
            and not controlled(COHERENT_CONDITION, "phase_sync_no_seam")
            and controlled(COHERENT_CONDITION, "phase_sync"),
            [COHERENT_CONDITION] if coherent else [],
        ),
        "H6": (
            "drr_rooting is inflated on the coherent condition",
            coherent and not controlled(COHERENT_CONDITION, "drr_rooting"),
            [COHERENT_CONDITION] if coherent else [],
        ),
        "H7": (
            "phase_sync is inflated on common_chirp",
            "common_chirp" in by_id and not controlled("common_chirp", "phase_sync"),
            ["common_chirp"] if "common_chirp" in by_id else [],
        ),
        "H8": (
            "gate abstention >= 0.8 on random_walk and common_chirp and <= 0.1 on every stationary "
            "condition",
            bool(targets)
            and all(gate_rate(c) >= 0.8 for c in targets)
            and all(gate_rate(c) <= 0.1 for c in stationary_all),
            targets + stationary_all,
        ),
        "H9": (
            "mean phase_sync detection rate on the stationary 1:1 couplings is at least the mean "
            "drr_rooting rate minus 0.10",
            mean_gap is not None and mean_gap >= -0.10,
            one_to_one,
        ),
    }
    outcomes = []
    for hypothesis in cfg["preregistration"]["hypotheses"]:
        rule, holds, evidence = rules[hypothesis["id"]]
        outcomes.append(
            {
                "id": hypothesis["id"],
                "statement": hypothesis["statement"],
                "rule": rule,
                "evidence": list(evidence),
                "outcome": "supported" if holds else "not_supported",
            }
        )
    return outcomes


def integrity_checks(
    development: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> Dict[str, bool]:
    """Harness checks that must hold for the artifact to be trusted at all."""
    expected = {
        "development": {c["id"] for c in cfg["development"]["conditions"]},
        "holdout": {c["id"] for c in cfg["holdout"]["conditions"]},
    }

    def complete(rows: Sequence[Mapping[str, Any]], split: str) -> bool:
        trials = cfg[split]["n_trials"]
        return {r["condition"] for r in rows} == expected[split] and all(
            r["n_trials"] == int(trials["related" if r["related"] else "unrelated"])
            and all(len(codes) == r["n_trials"] for codes in r["levels"].values())
            for r in rows
        )

    rows = list(development) + list(holdout)
    return {
        "every_development_row_complete": complete(development, "development"),
        "every_holdout_row_complete": complete(holdout, "holdout"),
        "oriented_area_identity_holds_on_every_trial": all(
            r["max_identity_error"] < 1e-9 for r in rows
        ),
        "holdout_seeds_disjoint_from_development": not (
            {c["seed_offset"] for c in cfg["development"]["conditions"]}
            & {c["seed_offset"] for c in cfg["holdout"]["conditions"]}
        ),
    }


def preregistration_sha256(cfg: Mapping[str, Any]) -> str:
    """Fingerprint of everything fixed before the held-out run."""
    frozen = {
        key: cfg[key]
        for key in (
            "seed",
            "alpha",
            "rooting",
            "transfer_entropy",
            "phase",
            "coherence",
            "quality_thresholds",
            "holdout",
            "preregistration",
        )
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()


#: Fingerprint of ``DEFAULT_CONFIG`` taken when it was frozen, before the held-out
#: conditions were first run. ``run_benchmark`` records whether the configuration
#: it ran still matches.
FROZEN_PREREGISTRATION_SHA256 = "d7e5fd7a0396ac03b5225bff0a01fee4740036218e17a3ebe286cdb09f126d8f"


# --------------------------------------------------------------------------- the study


def _run_split(
    cfg: Mapping[str, Any], split: str, workers: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    conditions = cfg[split]["conditions"]
    trials = cfg[split]["n_trials"]
    jobs = [
        (spec, int(trials["related" if spec["related"] else "unrelated"])) for spec in conditions
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_condition, spec, cfg, n) for spec, n in jobs]
            results = [future.result() for future in futures]
    else:
        results = [run_condition(spec, cfg, n) for spec, n in jobs]
    rows = [row for row, _ in results]
    runtime = {row["condition"]: timing for row, (_, timing) in zip(rows, results)}
    return rows, runtime


def run_benchmark(
    config: Optional[Mapping[str, Any]] = None,
    workers: int = 1,
    splits: Sequence[str] = ("development", "holdout"),
) -> Dict[str, Any]:
    """Run both splits and return a JSON-ready artifact.

    Rows are identical for any ``workers``. ``runtime`` holds wall-clock medians,
    which depend on the machine and on ``workers`` and are kept apart from the
    rows.
    """
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    started = time.perf_counter()
    development, dev_runtime = (
        _run_split(cfg, "development", workers) if "development" in splits else ([], {})
    )
    holdout, hold_runtime = _run_split(cfg, "holdout", workers) if "holdout" in splits else ([], {})
    elapsed = time.perf_counter() - started
    fingerprint = preregistration_sha256(cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "preregistration_sha256": fingerprint,
        "matches_frozen_preregistration": fingerprint == FROZEN_PREREGISTRATION_SHA256,
        "methods": METHODS,
        "pyinform_available": bool(PYINFORM_AVAILABLE),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "development": development,
        "holdout": holdout,
        "hypotheses": evaluate_hypotheses(holdout, cfg) if holdout else [],
        "checks": integrity_checks(development, holdout, cfg),
        "runtime": {
            "note": "Wall-clock seconds per trial on the generating machine; not reproducible.",
            "workers": int(workers),
            "total_seconds": float(elapsed),
            "machine": platform.machine(),
            "development": dev_runtime,
            "holdout": hold_runtime,
        },
    }


# --------------------------------------------------------------------------- report


def _cell(entry: Mapping[str, Any], alpha: float, mark_inflated: bool) -> str:
    rate = entry["declared"]
    text = f"{rate['rate']:.3f}"
    return f"**{text}**" if mark_inflated and not is_controlled(rate, alpha) else text


def _table(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    alpha: float,
    mark_inflated: bool,
) -> List[str]:
    lines = [
        "| Condition | " + " | ".join(f"`{m}`" for m in methods) + " |",
        "| --- |" + " ---: |" * len(methods),
    ]
    for row in rows:
        summary = row["summary"]
        cells = [
            _cell(summary[m], alpha, mark_inflated) if m in summary else "n/a" for m in methods
        ]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    return lines


def render_benchmark_report(artifact: Mapping[str, Any]) -> str:
    """Render the Markdown companion of a benchmark artifact."""
    cfg = artifact["config"]
    alpha = float(cfg["alpha"])
    # Rates are never stored; they are recomputed here from the per-trial codes.
    holdout = [dict(row, summary=summarize_row(row, alpha)) for row in artifact["holdout"]]
    development = [dict(row, summary=summarize_row(row, alpha)) for row in artifact["development"]]
    stationary_nulls = [r for r in holdout if not r["related"] and r["stationary"]]
    related = [r for r in holdout if r["related"] and r["stationary"]]
    nonstationary = [r for r in holdout if not r["stationary"]]
    phase = cfg["phase"]
    frozen = "yes" if artifact["matches_frozen_preregistration"] else "no"
    lines = [
        "# Phase and harmonic mathematics for DRR relationship tests",
        "",
        "Generated by `python scripts/run_resonance_math_benchmark.py --output-dir results/expected`",
        f"from `{artifact['schema_version']}` (config SHA-256 `{artifact['config_sha256'][:12]}…`,",
        f"preregistration SHA-256 `{artifact['preregistration_sha256'][:12]}…`, matches the frozen",
        f"preregistration: {frozen}). Every value is a Monte Carlo estimate on simulated channel",
        "pairs with a known truth. The JSON beside this file stores one character per trial, and",
        "every rate below is recomputed from those characters.",
        "",
        f"Shared settings: alpha = {alpha}. DRR rooting searches lags 1–{cfg['rooting']['max_lag']}"
        f" with {cfg['rooting']['n_surrogates']} circular-shift surrogates and max-statistic"
        f" correction. The phase test searches ratios up to order {phase['max_ratio_order']}"
        f" within {phase['ratio_tolerance']} of the observed frequency ratio, filters with a"
        f" Gaussian band of standard deviation f0 × {phase['relative_bandwidth']:.4f}, and calls a"
        f" non-significant pair inconclusive below {phase['min_effective_samples']:g} effective"
        " phase samples. Missing values and irregular timestamps are interpolated linearly onto"
        " the regular grid before any method runs.",
        "",
        "Bold marks an inflated rate on an unrelated condition: the Wilson 95% lower bound of the"
        f" rate of declared relationships exceeds {alpha}.",
        "",
        "## Methods",
        "",
        "| Method | Family | What it does |",
        "| --- | --- | --- |",
    ]
    # METHODS fixes the order; the artifact's own mapping is sorted by JSON.
    for name in [m for m in METHODS if m in artifact["methods"]]:
        method = artifact["methods"][name]
        lines.append(f"| `{name}` | {method['family']} | {method['description']} |")

    null_methods = [
        "drr_rooting",
        "drr_rooting_te",
        "pearson_analytic",
        "coherence_analytic",
        "coherence_shift",
        "plv_rayleigh",
        "phase_sync",
        "oriented_area",
    ]
    lines += [
        "",
        "## Held-out: false relationships on stationary unrelated pairs",
        "",
        f"{cfg['holdout']['n_trials']['unrelated']} pairs per row. Any declared relationship is"
        " false.",
        "",
        *_table(stationary_nulls, null_methods, alpha, True),
    ]
    detect_methods = [
        "drr_rooting",
        "drr_rooting_te",
        "coherence_shift",
        "phase_sync",
        "phase_sync_1to1",
        "oriented_area",
        "drr_plus_phase",
    ]
    lines += [
        "",
        "## Held-out: detection on related pairs",
        "",
        f"{cfg['holdout']['n_trials']['related']} pairs per row. The second channel is driven by"
        " the first. Rates for methods inflated on the unrelated rows above are not comparable"
        " detection rates.",
        "",
        *_table(related, detect_methods, alpha, False),
    ]
    lines += [
        "",
        "## Held-out: nonstationary pairs and the data-quality gate",
        "",
        "Rows marked related are coupled; the others are independent pairs that share a"
        " nonstationarity. The gate's abstentions are never counted as declared relationships.",
        "",
        "| Condition | Related | `drr_rooting` | `phase_sync` | `drr_plus_phase` | Gate abstains |"
        " `phase_sync_gated` | `drr_plus_phase_gated` |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in nonstationary:
        s = row["summary"]
        mark = not row["related"]
        lines.append(
            f"| {row['label']} | {'yes' if row['related'] else 'no'} |"
            f" {_cell(s['drr_rooting'], alpha, mark)} | {_cell(s['phase_sync'], alpha, mark)} |"
            f" {_cell(s['drr_plus_phase'], alpha, mark)} | {s['gate']['abstained']['rate']:.3f} |"
            f" {_cell(s['phase_sync_gated'], alpha, mark)} |"
            f" {_cell(s['drr_plus_phase_gated'], alpha, mark)} |"
        )
    ablations = [
        "phase_sync",
        "phase_sync_1to1",
        "phase_sync_no_seam",
        "phase_sync_no_trim",
        "phase_sync_broadband",
    ]
    lines += [
        "",
        "## Held-out: ablations of the phase test",
        "",
        "Rate of declared relationships for every held-out condition. Bold marks inflation on an"
        " unrelated condition.",
        "",
        "| Condition | Related | " + " | ".join(f"`{m}`" for m in ablations) + " |",
        "| --- | --- |" + " ---: |" * len(ablations),
    ]
    for row in holdout:
        cells = [_cell(row["summary"][m], alpha, not row["related"]) for m in ablations]
        lines.append(
            f"| {row['label']} | {'yes' if row['related'] else 'no'} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "## Held-out: calibration at three levels",
        "",
        "Rate of declared relationships at alpha = 0.01, 0.05 and 0.10 on the stationary"
        " unrelated pairs. A calibrated test sits near each level.",
        "",
        "| Condition | Method | 0.01 | 0.05 | 0.10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in stationary_nulls:
        for method in ("drr_rooting", "phase_sync"):
            s = row["summary"][method]
            lines.append(
                f"| {row['label']} | `{method}` | {s['declared_at_0.01']['rate']:.3f} |"
                f" {s['declared']['rate']:.3f} | {s['declared_at_0.10']['rate']:.3f} |"
            )
    lines += [
        "",
        "## Held-out: what the phase test says when it does not detect",
        "",
        "Share of trials with each status of `phase_sync`. Not detected: not significant although"
        " the phases varied enough to give the test power. Inconclusive: not significant with too"
        " few effective phase samples, or no usable oscillation. Not tested: no small-integer"
        " ratio near the frequency ratio.",
        "",
        "| Condition | Related | Synchronized | Not detected | Inconclusive | Not tested |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in holdout:
        s = row["summary"]["phase_sync"]
        lines.append(
            f"| {row['label']} | {'yes' if row['related'] else 'no'} |"
            f" {s['status_synchronized']['rate']:.3f} | {s['status_not_detected']['rate']:.3f} |"
            f" {s['status_inconclusive']['rate']:.3f} | {s['status_not_tested']['rate']:.3f} |"
        )
    lines += [
        "",
        "## Held-out: direction among detections",
        "",
        "Share of declared relationships that name the true direction (first channel to second)."
        " The phase test is undirected and is not listed.",
        "",
        "| Condition | `drr_rooting` | `drr_rooting_te` | `oriented_area` |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in [r for r in holdout if r["related"]]:
        cells = []
        for method in DIRECTIONAL_METHODS:
            entry = row["summary"].get(method, {}).get("direction_correct_among_declared")
            cells.append("n/a" if entry is None else f"{entry['rate']:.3f} ({entry['trials']})")
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    runtime = artifact.get("runtime", {}).get("holdout", {})
    if runtime:
        names = sorted({name for timing in runtime.values() for name in timing})
        lines += [
            "",
            "## Runtime",
            "",
            f"Median wall-clock seconds per trial on the generating machine"
            f" ({artifact['runtime']['machine']}, {artifact['runtime']['workers']} worker"
            f"{'s' if artifact['runtime']['workers'] != 1 else ''}), the median over held-out"
            " conditions. These numbers are not reproducible and are kept out of the rows.",
            "",
            "| Step | Median seconds per trial |",
            "| --- | ---: |",
        ]
        for name in names:
            values = [
                timing[name]["median_seconds"] for timing in runtime.values() if name in timing
            ]
            lines.append(f"| `{name}` | {float(np.median(values)):.4f} |")
    lines += [
        "",
        "## Preregistered hypotheses",
        "",
        cfg["preregistration"]["decision_rule"],
        "",
        "| Id | Statement | Rule | Outcome |",
        "| --- | --- | --- | --- |",
    ]
    for h in artifact["hypotheses"]:
        lines.append(f"| {h['id']} | {h['statement']} | {h['rule']} | **{h['outcome']}** |")
    lines += ["", "## Integrity checks", "", "| Statement | Holds |", "| --- | --- |"]
    for name, holds in sorted(artifact["checks"].items()):
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if holds else 'no'} |")
    if development:
        dev_methods = [
            "drr_rooting",
            "coherence_analytic",
            "plv_rayleigh",
            "phase_sync",
            "oriented_area",
        ]
        lines += [
            "",
            "## Development conditions",
            "",
            "These conditions were used to set the methods and the gate's thresholds, so they are"
            " not evidence. Bold marks inflation on an unrelated condition.",
            "",
            "| Condition | Related | "
            + " | ".join(f"`{m}`" for m in dev_methods)
            + " | Gate abstains |",
            "| --- | --- |" + " ---: |" * (len(dev_methods) + 1),
        ]
        for row in development:
            s = row["summary"]
            cells = [_cell(s[m], alpha, not row["related"]) for m in dev_methods]
            lines.append(
                f"| {row['label']} | {'yes' if row['related'] else 'no'} | "
                + " | ".join(cells)
                + f" | {s['gate']['abstained']['rate']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIG",
    "FROZEN_PREREGISTRATION_SHA256",
    "METHODS",
    "SCHEMA_VERSION",
    "declared_rate",
    "evaluate_hypotheses",
    "integrity_checks",
    "is_controlled",
    "level_code",
    "preregistration_sha256",
    "render_benchmark_report",
    "run_benchmark",
    "run_condition",
    "run_trial",
    "simulate_condition",
    "summarize_row",
]
