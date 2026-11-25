"""Does DRR-Mellin recover a hidden log-periodic mode more reliably than established methods?

Each simulated record holds one real Mellin mode, a power-law envelope times
an oscillation in log scale,

    s(x) = x**beta * cos(omega * ln(x) + phi),

buried in noise. The question is how often a method returns the hidden
log-frequency ``omega`` to within half a Rayleigh width,
``|omega_hat - omega| <= pi / ln(x_max / x_min)``, and how often it does so
after declaring a detection at a threshold calibrated to a 5% false-alarm
rate on noise-only records. ``omega`` and ``phi`` are drawn continuously, so
the true frequency is almost never on any method's grid.

Methods (every method sees the same samples):

* ``drr_mellin_scan``: the DRR-Mellin candidate surface
  :func:`drr_framework.mellin.log_mode_profile` (the closed form of
  :func:`~drr_framework.mellin.fit_log_mode`), a joint growth-frequency scan.
* ``drr_log_spectrum``: :func:`~drr_framework.mellin.log_spectrum` with its
  documented ``beta = 0.5`` weighting.
* ``lomb_scargle``: :func:`scipy.signal.lombscargle` against ``ln x``, the
  standard periodogram for log-periodicity.
* ``nls``: nonlinear least squares started at the Lomb-Scargle peak.
* ``esprit``: least-squares ESPRIT with model order two.

Records come from two sampling designs (uniform in ``ln x``, and the integers
as with arithmetic or time-to-event data) and three noise models (white
Gaussian, scale-proportional Gaussian whose standard deviation follows the
envelope, and Student-t with three degrees of freedom).

Conditions are split. **Development** conditions (256 samples, white noise,
``beta`` in {-0.5, 0, 0.5}) were run while the benchmark was built and its
signal-to-noise levels chosen. **Held-out** conditions use 512 samples, other
sampling ranges, other ``beta`` values, two noise models never run in
development, and disjoint seeds. The hypotheses and decision rules in
``DEFAULT_CONFIG["preregistration"]`` were fixed before the held-out
conditions were first run; ``preregistration_sha256`` fingerprints them.
Hypotheses are evaluated on held-out rows only. The artifact stores one
character per trial and method, so every rate and test can be recomputed.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla
from scipy import optimize, signal, stats

from .mellin import fit_log_mode, log_mode_profile, log_spectrum

SCHEMA_VERSION = "drr-mellin-recovery-benchmark-v1"

PRIMARY = "drr_mellin_scan"
ESTABLISHED = ("lomb_scargle", "nls", "esprit")

#: Every method, its family, and what it computes.
METHODS: Dict[str, Dict[str, str]] = {
    "drr_mellin_scan": {
        "family": "DRR-Mellin (primary)",
        "estimates": "omega, beta",
        "description": (
            "drr_framework.mellin.log_mode_profile on the raw samples: joint least-squares "
            "scan over 17 growth exponents and 581 log-frequencies; statistic 1 - relative SSE"
        ),
    },
    "drr_log_spectrum": {
        "family": "DRR-Mellin (secondary)",
        "estimates": "omega",
        "description": (
            "drr_framework.mellin.log_spectrum with its documented beta = 0.5 weighting on a "
            "uniform ln-x grid (linear interpolation when samples are not log-uniform); "
            "statistic peak / mean power in the search band"
        ),
    },
    "lomb_scargle": {
        "family": "established",
        "estimates": "omega",
        "description": (
            "scipy.signal.lombscargle of the mean-removed samples against ln x on the same "
            "frequency grid; statistic 2 x peak power / total sum of squares"
        ),
    },
    "nls": {
        "family": "established",
        "estimates": "omega, beta",
        "description": (
            "scipy.optimize.least_squares fit of exp(beta u)(a cos omega u + b sin omega u), "
            "u = centred ln x, started at the Lomb-Scargle peak with beta = 0; "
            "statistic 1 - relative SSE"
        ),
    },
    "esprit": {
        "family": "established",
        "estimates": "omega, beta",
        "description": (
            "least-squares ESPRIT, model order 2, Hankel depth n // 3, on a uniform ln-x grid "
            "(linear interpolation when samples are not log-uniform); statistic energy "
            "fraction of the two leading singular values"
        ),
    },
}

REFERENCES = [
    "Lomb (1976), Astrophys. Space Sci. 39, 447-462, https://doi.org/10.1007/BF00648343",
    "Scargle (1982), Astrophys. J. 263, 835-853, https://doi.org/10.1086/160554",
    "Roy & Kailath (1989), IEEE Trans. ASSP 37, 984-995, https://doi.org/10.1109/29.32276",
    "Zhou & Sornette (2002), Int. J. Mod. Phys. C 13, 137, "
    "https://doi.org/10.1142/S0129183102003024",
    "Stoica & Moses (2005), Spectral Analysis of Signals, Prentice Hall",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20261125,
    "alpha": 0.05,
    "omega_grid": {"start": 2.0, "stop": 60.0, "step": 0.1},
    "beta_grid": {"start": -1.0, "stop": 1.0, "count": 17},
    "omega_draw": [8.0, 40.0],
    "methods": {
        "log_spectrum_beta": 0.5,
        "esprit_depth_divisor": 3,
        "nls_beta_bounds": [-1.5, 1.5],
        "nls_max_nfev": 200,
    },
    "development": {
        "seed_offset": 10000,
        "designs": [
            {"id": "log_uniform_256", "kind": "log_uniform", "n": 256, "x_min": 1.0, "span": 6.0},
            {"id": "integers_256", "kind": "integers", "n": 256, "x_min": 15.0},
        ],
        "noises": ["white"],
        "betas": [-0.5, 0.0, 0.5],
        "integrated_snr_db": [6.0, 9.0, 12.0, 15.0, 18.0],
        "n_signal": 60,
        "n_null_calibration": 300,
        "n_null_fresh": 300,
    },
    "holdout": {
        "seed_offset": 110000,
        "designs": [
            {
                "id": "log_uniform_512",
                "kind": "log_uniform",
                "n": 512,
                "x_min": 1.0,
                "span": 6.907755278982137,
            },
            {"id": "integers_512", "kind": "integers", "n": 512, "x_min": 20.0},
        ],
        "noises": ["white", "scale", "student_t3"],
        "betas": [-0.6, -0.3, 0.0, 0.3, 0.6],
        "integrated_snr_db": [8.0, 11.0, 14.0, 17.0],
        "n_signal": 100,
        "n_null_calibration": 500,
        "n_null_fresh": 500,
    },
    "preregistration": {
        "frozen_before_holdout": True,
        "frozen_on": "2026-11-25",
        "primary_endpoint": "recovered: |omega_hat - omega| <= pi / ln(x_max / x_min)",
        "secondary_endpoint": (
            "detected_and_recovered: statistic above the calibrated 5% false-alarm threshold "
            "of its design and noise model, and recovered"
        ),
        "primary_method": PRIMARY,
        "comparators": list(ESTABLISHED),
        "familywise_alpha": 0.05,
        "noninferiority_margin": 0.03,
        "hypotheses": [
            {
                "id": "H1",
                "statement": (
                    "Across all held-out signal trials, the DRR-Mellin scan recovers the hidden "
                    "log-frequency more often than each of Lomb-Scargle, periodogram-initialized "
                    "NLS and ESPRIT."
                ),
                "test": "superiority",
                "endpoint": "recovered",
                "subset": {},
                "decision": (
                    "one-sided exact McNemar test per comparator, Holm-adjusted at familywise "
                    "0.05; supported if all three adjusted p < 0.05, partially supported if "
                    "some, not supported if none"
                ),
            },
            {
                "id": "H2",
                "statement": (
                    "Across all held-out signal trials, the DRR-Mellin scan both detects (at a "
                    "calibrated 5% false-alarm rate) and recovers the mode more often than each "
                    "established comparator."
                ),
                "test": "superiority",
                "endpoint": "detected_and_recovered",
                "subset": {},
                "decision": "as H1",
            },
            {
                "id": "H3",
                "statement": (
                    "Under additive noise (white and Student-t), the DRR-Mellin recovery "
                    "advantage over Lomb-Scargle is larger for strongly growing or decaying "
                    "modes (|beta| = 0.6) than for constant-amplitude modes (beta = 0)."
                ),
                "test": "difference_in_differences",
                "endpoint": "recovered",
                "comparator": "lomb_scargle",
                "subset": {"noise": ["white", "student_t3"], "abs_beta": [0.6]},
                "reference_subset": {"noise": ["white", "student_t3"], "abs_beta": [0.0]},
                "decision": (
                    "one-sided z test of the difference between the two mean paired "
                    "differences (independent trials); supported if p < 0.05"
                ),
            },
            {
                "id": "H4",
                "statement": (
                    "At beta = 0, where Lomb-Scargle is the matched statistic, searching over "
                    "growth costs the DRR-Mellin scan at most 3 percentage points of recovery."
                ),
                "test": "noninferiority",
                "endpoint": "recovered",
                "comparator": "lomb_scargle",
                "subset": {"abs_beta": [0.0]},
                "decision": (
                    "supported if the one-sided 95% lower confidence bound of the mean paired "
                    "difference (DRR minus Lomb-Scargle) exceeds -0.03"
                ),
            },
            {
                "id": "H5",
                "statement": (
                    "Predicted limitation: under scale-proportional noise with |beta| = 0.6, an "
                    "unweighted envelope fit over-weights the noisiest samples, so Lomb-Scargle "
                    "recovers the mode more often than the DRR-Mellin scan."
                ),
                "test": "comparator_superiority",
                "endpoint": "recovered",
                "comparator": "lomb_scargle",
                "subset": {"noise": ["scale"], "abs_beta": [0.6]},
                "decision": "one-sided exact McNemar test favouring Lomb-Scargle; supported if p < 0.05",
            },
        ],
    },
}


# --------------------------------------------------------------------------- simulation


def design_grid(design: Mapping[str, Any]) -> np.ndarray:
    """Sample locations of a design: uniform in ``ln x`` or consecutive integers."""
    n = int(design["n"])
    if design["kind"] == "log_uniform":
        return float(design["x_min"]) * np.exp(np.linspace(0.0, float(design["span"]), n))
    if design["kind"] == "integers":
        return float(design["x_min"]) + np.arange(n, dtype=float)
    raise ValueError(f"unknown design kind {design['kind']!r}")


def recovery_tolerance(x: np.ndarray) -> float:
    """Half the Rayleigh resolution in log-frequency: ``pi / ln(x_max / x_min)``."""
    return float(np.pi / np.log(x[-1] / x[0]))


def noise_shape(x: np.ndarray, noise: str, beta: float) -> np.ndarray:
    """Per-sample noise standard deviation profile with unit mean square."""
    if noise == "scale":
        u = np.log(x)
        shape = np.exp(beta * (u - u.mean()))
        return shape / np.sqrt(np.mean(shape**2))
    if noise in ("white", "student_t3"):
        return np.ones_like(x)
    raise ValueError(f"unknown noise model {noise!r}")


def standard_noise(rng: np.random.Generator, noise: str, n: int) -> np.ndarray:
    """Unit-variance innovations of a noise model."""
    if noise == "student_t3":
        return rng.standard_t(3, size=n) / np.sqrt(3.0)
    return rng.standard_normal(n)


def simulate_signal_trial(
    x: np.ndarray,
    noise: str,
    beta: float,
    integrated_snr_db: float,
    omega_range: Sequence[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """One record with a hidden mode at a continuous random ``omega`` and phase.

    The integrated signal-to-noise ratio is ``n * mean(s**2) / mean(sigma**2)``.
    """
    omega = float(rng.uniform(omega_range[0], omega_range[1]))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    innovations = standard_noise(rng, noise, x.size)
    u = np.log(x)
    clean = np.exp(beta * (u - u.mean())) * np.cos(omega * u + phase)
    per_sample_snr = 10.0 ** (integrated_snr_db / 10.0) / x.size
    sigma = np.sqrt(np.mean(clean**2) / per_sample_snr)
    y = clean + sigma * noise_shape(x, noise, beta) * innovations
    return y, {"omega": omega, "beta": float(beta), "phase": phase}


def simulate_null_trial(
    x: np.ndarray, noise: str, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """A noise-only record with the noise model (and envelope shape) of a group."""
    return noise_shape(x, noise, beta) * standard_noise(rng, noise, x.size)


# --------------------------------------------------------------------------- methods


def search_grids(cfg: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """The growth and log-frequency candidates shared by the scanning methods."""
    g = cfg["omega_grid"]
    count = int(round((g["stop"] - g["start"]) / g["step"])) + 1
    omegas = np.round(g["start"] + g["step"] * np.arange(count), 10)
    b = cfg["beta_grid"]
    betas = np.linspace(b["start"], b["stop"], int(b["count"]))
    return betas, omegas


def _log_uniform_view(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Samples on a uniform ln-x grid: exact when already uniform, else linear interpolation."""
    u = np.log(x)
    du = np.diff(u)
    if np.allclose(du, du[0], rtol=1e-7, atol=0):
        return x, y
    uniform = np.linspace(u[0], u[-1], u.size)
    return np.exp(uniform), np.interp(uniform, u, y)


def method_drr_scan(x, y, betas, omegas) -> Tuple[float, float, float]:
    surface = log_mode_profile(x, y, betas, omegas)
    row, col = np.unravel_index(int(np.nanargmin(surface)), surface.shape)
    return float(omegas[col]), float(betas[row]), float(1.0 - surface[row, col])


def method_log_spectrum(x, y, omegas, beta_weight) -> Tuple[float, float, float]:
    xu, yu = _log_uniform_view(x, y)
    frequency, amplitude = log_spectrum(xu, yu, beta=beta_weight)
    band = (frequency >= omegas[0]) & (frequency <= omegas[-1])
    power = amplitude[band] ** 2
    peak = int(np.argmax(power))
    return float(frequency[band][peak]), float("nan"), float(power[peak] / power.mean())


def method_lomb_scargle(x, y, omegas) -> Tuple[float, float, float]:
    u = np.log(x)
    centred = y - y.mean()
    power = signal.lombscargle(u, centred, omegas)
    peak = int(np.argmax(power))
    return float(omegas[peak]), float("nan"), float(2.0 * power[peak] / (centred @ centred))


def method_nls(x, y, omegas, beta_bounds, max_nfev) -> Tuple[float, float, float]:
    u = np.log(x)
    u = u - u.mean()
    target = y / np.max(np.abs(y))
    omega0 = method_lomb_scargle(x, y, omegas)[0]
    design = np.column_stack((np.cos(omega0 * u), np.sin(omega0 * u)))
    a0, b0 = np.linalg.lstsq(design, target, rcond=None)[0]

    def residual(p):
        a, b, beta, omega = p
        envelope = np.exp(beta * u)
        return envelope * (a * np.cos(omega * u) + b * np.sin(omega * u)) - target

    def jacobian(p):
        a, b, beta, omega = p
        envelope = np.exp(beta * u)
        c, s = np.cos(omega * u), np.sin(omega * u)
        mode = a * c + b * s
        return np.column_stack(
            (envelope * c, envelope * s, u * envelope * mode, u * envelope * (b * c - a * s))
        )

    step = float(omegas[1] - omegas[0])
    lower = [-np.inf, -np.inf, beta_bounds[0], float(omegas[0]) - step]
    upper = [np.inf, np.inf, beta_bounds[1], float(omegas[-1]) + step]
    fit = optimize.least_squares(
        residual,
        [a0, b0, 0.0, omega0],
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        max_nfev=int(max_nfev),
    )
    explained = 1.0 - float(fit.fun @ fit.fun) / float(target @ target)
    return float(fit.x[3]), float(fit.x[2]), explained


def method_esprit(x, y, depth_divisor) -> Tuple[float, float, float]:
    xu, yu = _log_uniform_view(x, y)
    du = float(np.log(xu[1] / xu[0]))
    centred = yu - yu.mean()
    depth = centred.size // int(depth_divisor)
    hankel = sla.hankel(centred[:depth], centred[depth - 1 :])
    left, singular, _ = np.linalg.svd(hankel, full_matrices=False)
    subspace = left[:, :2]
    rotation = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=None)[0]
    poles = np.linalg.eigvals(rotation)
    omega = float(np.mean(np.abs(np.angle(poles)))) / du
    beta = float(np.mean(np.log(np.abs(poles)))) / du
    energy = singular**2
    return omega, beta, float(energy[:2].sum() / energy.sum())


def analyze_record(x: np.ndarray, y: np.ndarray, cfg: Mapping[str, Any]) -> Dict[str, Tuple]:
    """Run every method on one record: ``{method: (omega_hat, beta_hat, statistic)}``."""
    betas, omegas = search_grids(cfg)
    m = cfg["methods"]
    return {
        "drr_mellin_scan": method_drr_scan(x, y, betas, omegas),
        "drr_log_spectrum": method_log_spectrum(x, y, omegas, m["log_spectrum_beta"]),
        "lomb_scargle": method_lomb_scargle(x, y, omegas),
        "nls": method_nls(x, y, omegas, m["nls_beta_bounds"], m["nls_max_nfev"]),
        "esprit": method_esprit(x, y, m["esprit_depth_divisor"]),
    }


# --------------------------------------------------------------------------- conditions


def _rng(cfg: Mapping[str, Any], offset: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), int(offset)]))


def build_conditions(cfg: Mapping[str, Any], split: str) -> Dict[str, List[Dict[str, Any]]]:
    """Signal conditions and null groups of a split, each with its own seed offset."""
    spec = cfg[split]
    base = int(spec["seed_offset"])
    signal_rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for design in spec["designs"]:
        for noise in spec["noises"]:
            for beta in spec["betas"]:
                key = (design["id"], noise, float(beta) if noise == "scale" else 0.0)
                if key not in groups:
                    groups[key] = {
                        "group": f"{key[0]}|{key[1]}|{key[2]:+.1f}",
                        "design": design,
                        "noise": noise,
                        "noise_beta": key[2],
                    }
                for snr in spec["integrated_snr_db"]:
                    signal_rows.append(
                        {
                            "id": f"{design['id']}|{noise}|beta{beta:+.1f}|snr{snr:g}",
                            "design": design,
                            "noise": noise,
                            "beta": float(beta),
                            "integrated_snr_db": float(snr),
                            "group": groups[key]["group"],
                            "seed_offset": base + len(signal_rows),
                        }
                    )
    null_groups = []
    for index, group in enumerate(groups.values()):
        null_groups.append(
            dict(
                group,
                calibration_seed_offset=base + 50000 + index,
                fresh_seed_offset=base + 70000 + index,
            )
        )
    return {"signal": signal_rows, "null": null_groups}


def calibration_threshold(statistics: Sequence[float], alpha: float) -> float:
    """Rank threshold: a fresh exchangeable null exceeds it with probability <= alpha."""
    ordered = np.sort(np.asarray(statistics, dtype=float))
    index = min(ordered.size - 1, int(math.ceil((1.0 - alpha) * (ordered.size + 1))) - 1)
    return float(ordered[index])


def run_null_group(
    group: Mapping[str, Any], cfg: Mapping[str, Any], n_trials: int, seed_offset: int
) -> Dict[str, List[float]]:
    """Statistics of every method on noise-only records of a group."""
    x = design_grid(group["design"])
    rng = _rng(cfg, seed_offset)
    out: Dict[str, List[float]] = {m: [] for m in METHODS}
    for _ in range(n_trials):
        y = simulate_null_trial(x, group["noise"], group["noise_beta"], rng)
        for method, (_, _, stat) in analyze_record(x, y, cfg).items():
            out[method].append(stat)
    return out


def _code(recovered: bool, detected: bool) -> str:
    return str(int(recovered) + 2 * int(detected))


def run_signal_condition(
    spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    n_trials: int,
    thresholds: Mapping[str, float],
    check_fit_log_mode: bool = True,
) -> Dict[str, Any]:
    """Codes per trial: ``0`` missed, ``1`` recovered, ``2`` detected only, ``3`` both."""
    x = design_grid(spec["design"])
    tolerance = recovery_tolerance(x)
    rng = _rng(cfg, spec["seed_offset"])
    codes: Dict[str, List[str]] = {m: [] for m in METHODS}
    omega_err: Dict[str, List[float]] = {m: [] for m in METHODS}
    beta_err: Dict[str, List[float]] = {m: [] for m in METHODS}
    agrees: Optional[bool] = None
    for trial in range(n_trials):
        y, truth = simulate_signal_trial(
            x, spec["noise"], spec["beta"], spec["integrated_snr_db"], cfg["omega_draw"], rng
        )
        results = analyze_record(x, y, cfg)
        if check_fit_log_mode and trial == 0:
            betas, omegas = search_grids(cfg)
            reference = fit_log_mode(x, y, betas, omegas)
            agrees = (
                reference["omega"] == results[PRIMARY][0]
                and reference["beta"] == results[PRIMARY][1]
            )
        for method, (omega_hat, beta_hat, stat) in results.items():
            recovered = bool(abs(omega_hat - truth["omega"]) <= tolerance)
            codes[method].append(_code(recovered, stat > thresholds[method]))
            if recovered:
                omega_err[method].append(omega_hat - truth["omega"])
                if np.isfinite(beta_hat):
                    beta_err[method].append(beta_hat - truth["beta"])

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    return {
        "id": spec["id"],
        "design": spec["design"]["id"],
        "noise": spec["noise"],
        "beta": spec["beta"],
        "integrated_snr_db": spec["integrated_snr_db"],
        "group": spec["group"],
        "seed_offset": spec["seed_offset"],
        "n_trials": n_trials,
        "tolerance": tolerance,
        "codes": {m: "".join(c) for m, c in codes.items()},
        "omega_rmse_recovered": {m: rmse(v) for m, v in omega_err.items()},
        "beta_rmse_recovered": {m: rmse(v) for m, v in beta_err.items() if v},
        "fit_log_mode_agrees": agrees,
    }


def _calibrate(job):
    group, cfg, n = job
    stats_by_method = run_null_group(group, cfg, n, group["calibration_seed_offset"])
    return {m: calibration_threshold(v, cfg["alpha"]) for m, v in stats_by_method.items()}


def _fresh(job):
    group, cfg, n, thresholds = job
    stats_by_method = run_null_group(group, cfg, n, group["fresh_seed_offset"])
    return {
        m: "".join("1" if s > thresholds[m] else "0" for s in v) for m, v in stats_by_method.items()
    }


def _signal(job):
    spec, cfg, n, thresholds = job
    return run_signal_condition(spec, cfg, n, thresholds)


def _map(function, jobs, workers):
    if workers > 1:
        # Spawned workers share no state with the parent process.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(function, jobs))
    return [function(job) for job in jobs]


def run_split(cfg: Mapping[str, Any], split: str, workers: int = 1) -> Dict[str, Any]:
    """Calibrate thresholds on null records, then run signal and fresh-null records."""
    spec = cfg[split]
    built = build_conditions(cfg, split)
    groups = built["null"]
    thresholds = _map(
        _calibrate, [(g, cfg, int(spec["n_null_calibration"])) for g in groups], workers
    )
    by_group = {g["group"]: t for g, t in zip(groups, thresholds)}
    fresh = _map(
        _fresh, [(g, cfg, int(spec["n_null_fresh"]), by_group[g["group"]]) for g in groups], workers
    )
    signal_rows = _map(
        _signal,
        [(s, cfg, int(spec["n_signal"]), by_group[s["group"]]) for s in built["signal"]],
        workers,
    )
    null_rows = [
        {
            "group": g["group"],
            "design": g["design"]["id"],
            "noise": g["noise"],
            "noise_beta": g["noise_beta"],
            "calibration_seed_offset": g["calibration_seed_offset"],
            "fresh_seed_offset": g["fresh_seed_offset"],
            "n_calibration": int(spec["n_null_calibration"]),
            "thresholds": by_group[g["group"]],
            "fresh_codes": f,
        }
        for g, f in zip(groups, fresh)
    ]
    return {"signal": signal_rows, "null": null_rows}


# --------------------------------------------------------------------------- statistics


def decode(codes: str) -> Tuple[np.ndarray, np.ndarray]:
    """Recovered and detected indicators of a code string."""
    values = np.frombuffer(codes.encode("ascii"), dtype=np.uint8) - ord("0")
    return (values & 1).astype(bool), (values & 2).astype(bool)


def endpoint(codes: str, name: str) -> np.ndarray:
    recovered, detected = decode(codes)
    if name == "recovered":
        return recovered
    if name == "detected":
        return detected
    if name == "detected_and_recovered":
        return recovered & detected
    if name == "detected_not_recovered":
        return detected & ~recovered
    raise ValueError(f"unknown endpoint {name!r}")


def select(rows: Iterable[Mapping[str, Any]], subset: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Rows matching a subset filter on ``noise``, ``design``, ``abs_beta``, ``beta`` or SNR."""
    chosen = []
    for row in rows:
        if "noise" in subset and row["noise"] not in subset["noise"]:
            continue
        if "design" in subset and row["design"] not in subset["design"]:
            continue
        if "abs_beta" in subset and not any(
            abs(abs(row["beta"]) - b) < 1e-9 for b in subset["abs_beta"]
        ):
            continue
        if "beta" in subset and not any(abs(row["beta"] - b) < 1e-9 for b in subset["beta"]):
            continue
        if (
            "integrated_snr_db" in subset
            and row["integrated_snr_db"] not in subset["integrated_snr_db"]
        ):
            continue
        chosen.append(row)
    return chosen


def pooled(rows: Sequence[Mapping[str, Any]], method: str, name: str) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=bool)
    return np.concatenate([endpoint(r["codes"][method], name) for r in rows])


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> List[float]:
    if trials == 0:
        return [0.0, 1.0]
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(values: np.ndarray) -> Dict[str, Any]:
    count, trials = int(values.sum()), int(values.size)
    return {
        "rate": count / trials if trials else float("nan"),
        "count": count,
        "trials": trials,
        "ci95": wilson_interval(count, trials),
    }


def mcnemar_greater(first: np.ndarray, second: np.ndarray) -> Dict[str, Any]:
    """Exact one-sided McNemar test that ``first`` succeeds more often than ``second``."""
    wins = int(np.sum(first & ~second))
    losses = int(np.sum(~first & second))
    p = float(stats.binom.sf(wins - 1, wins + losses, 0.5)) if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "p_value": p}


def paired_difference(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    """Mean paired difference with its standard error and one-sided 95% lower bound."""
    d = first.astype(float) - second.astype(float)
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan")
    mean = float(d.mean())
    return {"mean": mean, "se": se, "lower95_one_sided": mean - 1.6448536269514722 * se}


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values in the input order."""
    order = np.argsort(p_values, kind="stable")
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def evaluate_hypotheses(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> List[Dict]:
    """Apply each preregistered decision rule to held-out signal rows."""
    prereg = cfg["preregistration"]
    level = float(prereg["familywise_alpha"])
    out = []
    for h in prereg["hypotheses"]:
        subset = select(rows, h["subset"])
        primary = pooled(subset, PRIMARY, h["endpoint"])
        result: Dict[str, Any] = {
            "id": h["id"],
            "statement": h["statement"],
            "trials": int(primary.size),
        }
        if h["test"] == "superiority":
            tests = {
                c: mcnemar_greater(primary, pooled(subset, c, h["endpoint"])) for c in ESTABLISHED
            }
            adjusted = holm([tests[c]["p_value"] for c in ESTABLISHED])
            for c, p in zip(ESTABLISHED, adjusted):
                tests[c]["holm_p_value"] = p
                tests[c]["difference"] = paired_difference(
                    primary, pooled(subset, c, h["endpoint"])
                )["mean"]
            passed = sum(tests[c]["holm_p_value"] < level for c in ESTABLISHED)
            result["comparisons"] = tests
            result["outcome"] = (
                "supported"
                if passed == len(ESTABLISHED)
                else "partially supported" if passed else "not supported"
            )
        elif h["test"] == "difference_in_differences":
            reference = select(rows, h["reference_subset"])
            c = h["comparator"]
            high = paired_difference(primary, pooled(subset, c, h["endpoint"]))
            low = paired_difference(
                pooled(reference, PRIMARY, h["endpoint"]), pooled(reference, c, h["endpoint"])
            )
            z = (high["mean"] - low["mean"]) / math.sqrt(high["se"] ** 2 + low["se"] ** 2)
            p = float(stats.norm.sf(z))
            result.update(
                {"difference_in_subset": high, "difference_in_reference": low, "z": z, "p_value": p}
            )
            result["outcome"] = "supported" if p < level else "not supported"
        elif h["test"] == "noninferiority":
            diff = paired_difference(primary, pooled(subset, h["comparator"], h["endpoint"]))
            result["difference"] = diff
            result["margin"] = -float(prereg["noninferiority_margin"])
            result["outcome"] = (
                "supported" if diff["lower95_one_sided"] > result["margin"] else "not supported"
            )
        elif h["test"] == "comparator_superiority":
            other = pooled(subset, h["comparator"], h["endpoint"])
            test = mcnemar_greater(other, primary)
            result["comparison"] = dict(test, difference=paired_difference(primary, other)["mean"])
            result["outcome"] = "supported" if test["p_value"] < level else "not supported"
        else:
            raise ValueError(f"unknown test {h['test']!r}")
        out.append(result)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pooled rates of every method and endpoint over signal rows."""
    names = ("recovered", "detected", "detected_and_recovered", "detected_not_recovered")
    return {m: {n: rate(pooled(rows, m, n)) for n in names} for m in METHODS}


def false_alarms(null_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fresh-null false-alarm rate of every method at its calibrated threshold."""
    out = {}
    for m in METHODS:
        flags = "".join(r["fresh_codes"][m] for r in null_rows)
        out[m] = rate(np.frombuffer(flags.encode("ascii"), dtype=np.uint8) == ord("1"))
    return out


# --------------------------------------------------------------------------- study


def preregistration_sha256(cfg: Mapping[str, Any]) -> str:
    """Fingerprint of everything fixed before the held-out run."""
    frozen = {
        key: cfg[key]
        for key in (
            "seed",
            "alpha",
            "omega_grid",
            "beta_grid",
            "omega_draw",
            "methods",
            "holdout",
            "preregistration",
        )
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()


#: Fingerprint of ``DEFAULT_CONFIG`` taken when it was frozen, before the held-out
#: conditions were first run.
FROZEN_PREREGISTRATION_SHA256 = "a6fbf4ed6de44c779ef8f14baae5a46ea85c40a3ff52e397a25aad8a8d31361e"


def integrity_checks(artifact: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, bool]:
    betas, omegas = search_grids(cfg)
    checks = {
        "preregistration_matches_frozen_record": artifact["preregistration_sha256"]
        == FROZEN_PREREGISTRATION_SHA256,
        "omega_draw_inside_search_grid": bool(
            omegas[0] < cfg["omega_draw"][0] and cfg["omega_draw"][1] < omegas[-1]
        ),
        "beta_grid_covers_holdout_betas": bool(
            betas[0] <= min(cfg["holdout"]["betas"]) and max(cfg["holdout"]["betas"]) <= betas[-1]
        ),
        "scan_within_fit_log_mode_limit": bool(betas.size * omegas.size <= 10_000),
    }
    seeds: Dict[str, set] = {"development": set(), "holdout": set()}
    for split in ("development", "holdout"):
        part = artifact.get(split) or {}
        for row in part.get("signal", []):
            seeds[split].add(row["seed_offset"])
        for row in part.get("null", []):
            seeds[split] |= {row["calibration_seed_offset"], row["fresh_seed_offset"]}
        signal_rows = part.get("signal", [])
        if signal_rows:
            checks[f"{split}_profile_matches_fit_log_mode"] = all(
                r["fit_log_mode_agrees"] for r in signal_rows
            )
            tolerance_ok = all(r["tolerance"] > cfg["omega_grid"]["step"] for r in signal_rows)
            checks[f"{split}_tolerance_exceeds_grid_step"] = tolerance_ok
    checks["holdout_seeds_disjoint_from_development"] = not (
        seeds["development"] & seeds["holdout"]
    )
    return checks


def run_benchmark(
    config: Optional[Mapping[str, Any]] = None,
    workers: int = 1,
    splits: Sequence[str] = ("development", "holdout"),
) -> Dict[str, Any]:
    """Run the requested splits and return a JSON-ready artifact."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    started = time.perf_counter()
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "preregistration_sha256": preregistration_sha256(cfg),
        "methods": METHODS,
        "references": REFERENCES,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    for split in ("development", "holdout"):
        artifact[split] = run_split(cfg, split, workers) if split in splits else None
    holdout = artifact["holdout"]
    artifact["hypotheses"] = evaluate_hypotheses(holdout["signal"], cfg) if holdout else []
    artifact["matches_frozen_preregistration"] = (
        artifact["preregistration_sha256"] == FROZEN_PREREGISTRATION_SHA256
    )
    artifact["checks"] = integrity_checks(artifact, cfg)
    artifact["runtime"] = {
        "note": "Wall-clock seconds on the generating machine; not part of the evidence.",
        "workers": int(workers),
        "total_seconds": float(time.perf_counter() - started),
        "machine": platform.machine(),
    }
    return artifact


# --------------------------------------------------------------------------- report

LABELS = {
    "drr_mellin_scan": "DRR-Mellin scan",
    "drr_log_spectrum": "DRR log-spectrum",
    "lomb_scargle": "Lomb-Scargle",
    "nls": "NLS",
    "esprit": "ESPRIT",
}


def _pct(entry: Mapping[str, Any]) -> str:
    return f"{entry['rate']:.3f}"


def _pct_ci(entry: Mapping[str, Any]) -> str:
    low, high = entry["ci95"]
    return f"{entry['rate']:.3f} [{low:.3f}, {high:.3f}]"


def pooled_rmse(rows: Sequence[Mapping[str, Any]], method: str, key: str) -> Optional[float]:
    """Root-mean-square error over recovered trials, pooled across rows."""
    total, count = 0.0, 0
    for row in rows:
        value = row[key].get(method)
        if value is None:
            continue
        n = int(endpoint(row["codes"][method], "recovered").sum())
        total += n * value * value
        count += n
    return math.sqrt(total / count) if count else None


def _factor_table(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Tuple[str, Mapping[str, Any]]],
    name: str = "recovered",
) -> List[str]:
    lines = [
        "| Subset | Trials | " + " | ".join(LABELS[m] for m in METHODS) + " |",
        "| --- | ---: |" + " ---: |" * len(METHODS),
    ]
    for label, subset in groups:
        chosen = select(rows, subset)
        cells = []
        rates = {m: rate(pooled(chosen, m, name)) for m in METHODS}
        best = max(r["rate"] for r in rates.values())
        for m in METHODS:
            text = _pct(rates[m])
            cells.append(f"**{text}**" if rates[m]["rate"] == best else text)
        trials = rates[PRIMARY]["trials"]
        lines.append(f"| {label} | {trials} | " + " | ".join(cells) + " |")
    return lines


def _regime_table(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> List[str]:
    """Paired DRR-minus-comparator differences by design, noise model and envelope strength."""
    lines = [
        "| Design | Noise | abs(beta) | DRR recovered | vs Lomb-Scargle | vs NLS "
        "| DRR detected and recovered | vs Lomb-Scargle | vs NLS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    abs_betas = sorted({abs(b) for b in cfg["holdout"]["betas"]})
    for design in cfg["holdout"]["designs"]:
        for noise in cfg["holdout"]["noises"]:
            for b in abs_betas:
                chosen = select(rows, {"design": [design["id"]], "noise": [noise], "abs_beta": [b]})
                cells = []
                for name in ("recovered", "detected_and_recovered"):
                    primary = pooled(chosen, PRIMARY, name)
                    cells.append(f"{primary.mean():.3f}")
                    for c in ("lomb_scargle", "nls"):
                        other = pooled(chosen, c, name)
                        diff = paired_difference(primary, other)
                        p = mcnemar_greater(primary, other)["p_value"]
                        cells.append(
                            f"{diff['mean']:+.3f} ± {1.959963984540054 * diff['se']:.3f} (p = {p:.1g})"
                        )
                lines.append(f"| {design['id']} | {noise} | {b:g} | " + " | ".join(cells) + " |")
    return lines


def _hypothesis_detail(h: Mapping[str, Any]) -> str:
    if "comparisons" in h:
        parts = []
        for c, t in h["comparisons"].items():
            parts.append(
                f"vs {LABELS[c]}: {t['difference']:+.3f} ({t['wins']} wins / {t['losses']} losses, "
                f"Holm p = {t['holm_p_value']:.2g})"
            )
        return "; ".join(parts)
    if "z" in h:
        hi, lo = h["difference_in_subset"], h["difference_in_reference"]
        return (
            f"advantage {hi['mean']:+.3f} at |beta| = 0.6 vs {lo['mean']:+.3f} at beta = 0; "
            f"z = {h['z']:.2f}, p = {h['p_value']:.2g}"
        )
    if "margin" in h:
        d = h["difference"]
        return (
            f"difference {d['mean']:+.3f}, one-sided 95% lower bound "
            f"{d['lower95_one_sided']:+.3f} vs margin {h['margin']:+.3f}"
        )
    c = h["comparison"]
    return (
        f"DRR minus Lomb-Scargle {c['difference']:+.3f} ({c['wins']} Lomb-Scargle wins / "
        f"{c['losses']} DRR wins, p = {c['p_value']:.2g})"
    )


def render_report(artifact: Mapping[str, Any]) -> str:
    """Render the Markdown companion of an artifact; every number is recomputed from codes."""
    cfg = artifact["config"]
    holdout = artifact["holdout"]
    development = artifact["development"]
    lines = [
        "# DRR-Mellin hidden-mode recovery benchmark",
        "",
        "Generated by `scripts/run_mellin_recovery_benchmark.py`; do not edit by hand. "
        f"Schema `{artifact['schema_version']}`, preregistration SHA-256 "
        f"`{artifact['preregistration_sha256'][:16]}…` (matches the frozen record: "
        f"{'yes' if artifact['matches_frozen_preregistration'] else 'no'}). "
        "Every rate is a Monte Carlo estimate on simulated records, recomputed from the "
        "per-trial codes in the JSON artifact.",
        "",
        "A trial is **recovered** when |omega_hat - omega| <= pi / ln(x_max / x_min) (half a "
        "Rayleigh width in log-frequency). **Detected** means the method's statistic exceeds "
        "the threshold calibrated on noise-only records of the same design and noise model "
        f"(nominal false-alarm rate {cfg['alpha']}).",
        "",
    ]
    if holdout:
        signal_rows = holdout["signal"]
        summary = summarize(signal_rows)
        alarms = false_alarms(holdout["null"])
        lines += [
            "## Preregistered hypotheses (held-out conditions only)",
            "",
            "| ID | Outcome | Statement | Evidence |",
            "| --- | --- | --- | --- |",
        ]
        for h in artifact["hypotheses"]:
            lines.append(
                f"| {h['id']} | **{h['outcome']}** | {h['statement']} | {_hypothesis_detail(h)} |"
            )
        lines += [
            "",
            "## Held-out results, pooled",
            "",
            f"{len(signal_rows)} signal conditions, {summary[PRIMARY]['recovered']['trials']} "
            f"signal trials and {alarms[PRIMARY]['trials']} fresh noise-only trials per method. "
            "Brackets are Wilson 95% intervals; paired comparisons are in the hypothesis table.",
            "",
            "| Method | Family | Recovered | Detected and recovered | Detected, wrong omega "
            "| Fresh-null false alarms |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for m in METHODS:
            s = summary[m]
            lines.append(
                f"| {LABELS[m]} | {METHODS[m]['family']} | {_pct_ci(s['recovered'])} | "
                f"{_pct_ci(s['detected_and_recovered'])} | {_pct(s['detected_not_recovered'])} | "
                f"{_pct_ci(alarms[m])} |"
            )
        noises = cfg["holdout"]["noises"]
        abs_betas = sorted({abs(b) for b in cfg["holdout"]["betas"]})
        mechanism = [
            (f"{noise}, abs(beta) = {b:g}", {"noise": [noise], "abs_beta": [b]})
            for noise in noises
            for b in abs_betas
        ]
        lines += [
            "",
            "## Recovery by noise model and envelope strength",
            "",
            "Bold marks the highest rate in a row (ties are all bold).",
            "",
            *_factor_table(signal_rows, mechanism),
            "",
            "## Recovery by sampling design and envelope strength",
            "",
            *_factor_table(
                signal_rows,
                [
                    (f"{d['id']}, abs(beta) = {b:g}", {"design": [d["id"]], "abs_beta": [b]})
                    for d in cfg["holdout"]["designs"]
                    for b in abs_betas
                ],
            ),
            "",
            "## Recovery by integrated signal-to-noise ratio",
            "",
            *_factor_table(
                signal_rows,
                [
                    (f"{snr:g} dB", {"integrated_snr_db": [snr]})
                    for snr in cfg["holdout"]["integrated_snr_db"]
                ],
            ),
            "",
            "## Detected and recovered by noise model and envelope strength",
            "",
            *_factor_table(signal_rows, mechanism, "detected_and_recovered"),
            "",
            "## Exploratory: where the DRR-Mellin scan differs from Lomb-Scargle and NLS",
            "",
            "Not preregistered; added after the held-out run to locate the regime behind H1 "
            "and H3. Entries are mean paired differences (DRR-Mellin scan minus comparator) "
            "with a 95% normal interval and an unadjusted one-sided exact McNemar p-value "
            "for DRR > comparator; 36 comparisons per endpoint, so read small p-values "
            "against that multiplicity.",
            "",
            *_regime_table(signal_rows, cfg),
            "",
            "## Accuracy among recovered trials",
            "",
            "Root-mean-square error over each method's own recovered trials, so sets differ "
            "between methods. Grid methods carry quantization error up to half a grid step "
            f"({cfg['omega_grid']['step'] / 2:g} in omega).",
            "",
            "| Method | omega RMSE | beta RMSE |",
            "| --- | ---: | ---: |",
        ]
        for m in METHODS:
            w = pooled_rmse(signal_rows, m, "omega_rmse_recovered")
            b = pooled_rmse(signal_rows, m, "beta_rmse_recovered")
            lines.append(f"| {LABELS[m]} | {w:.4f} | {'n/a' if b is None else f'{b:.4f}'} |")
        lines += [
            "",
            "## Fresh-null false alarms by noise model",
            "",
            "| Noise model | Trials | " + " | ".join(LABELS[m] for m in METHODS) + " |",
            "| --- | ---: |" + " ---: |" * len(METHODS),
        ]
        for noise in noises:
            chosen = [r for r in holdout["null"] if r["noise"] == noise]
            fa = false_alarms(chosen)
            lines.append(
                f"| {noise} | {fa[PRIMARY]['trials']} | "
                + " | ".join(_pct(fa[m]) for m in METHODS)
                + " |"
            )
    if development:
        dev_rows = development["signal"]
        lines += [
            "",
            "## Development conditions (not used for hypotheses)",
            "",
            *_factor_table(
                dev_rows,
                [("all", {})]
                + [
                    (f"{d['id']}, beta = {b:+g}", {"design": [d["id"]], "beta": [b]})
                    for d in cfg["development"]["designs"]
                    for b in cfg["development"]["betas"]
                ],
            ),
        ]
    lines += ["", "## Integrity checks", ""]
    for name, holds in artifact["checks"].items():
        lines.append(f"- {'PASS' if holds else 'FAIL'}: `{name}`")
    lines += ["", "## Methods", ""]
    for m, spec in METHODS.items():
        lines.append(f"- `{m}` ({spec['family']}): {spec['description']}.")
    lines += ["", "## References", ""] + [f"- {r}" for r in artifact["references"]]
    return "\n".join(lines) + "\n"


SERIES_STYLE = {
    "drr_mellin_scan": ("#257de0", "o"),
    "lomb_scargle": ("#f47230", "s"),
    "nls": ("#00b894", "^"),
    "esprit": ("#f2a600", "D"),
    "drr_log_spectrum": ("#ee7fac", "v"),
}


def render_figure(artifact: Mapping[str, Any], path) -> None:
    """Held-out recovery against SNR: rows are noise models, columns envelope strength."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = artifact["config"]["holdout"]
    rows = artifact["holdout"]["signal"]
    abs_betas = sorted({abs(b) for b in cfg["betas"]})
    snrs = cfg["integrated_snr_db"]
    noises = cfg["noises"]
    names = {
        "white": "white Gaussian",
        "scale": "scale-proportional",
        "student_t3": "Student-t (3)",
    }
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9c9c9c", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(
        len(noises), len(abs_betas), figsize=(9.6, 7.4), sharex=True, sharey=True
    )
    for i, noise in enumerate(noises):
        for j, b in enumerate(abs_betas):
            ax = axes[i][j]
            ax.grid(True, color="#e5e5e5", linewidth=0.6)
            ax.set_axisbelow(True)
            for m in SERIES_STYLE:
                color, marker = SERIES_STYLE[m]
                values = [
                    rate(
                        pooled(
                            select(
                                rows,
                                {"noise": [noise], "abs_beta": [b], "integrated_snr_db": [s]},
                            ),
                            m,
                            "recovered",
                        )
                    )["rate"]
                    for s in snrs
                ]
                ax.plot(
                    snrs,
                    values,
                    color=color,
                    marker=marker,
                    markersize=5,
                    linewidth=2 if m == PRIMARY else 1.4,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=3 if m == PRIMARY else 2,
                    label=LABELS[m],
                )
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(snrs)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(f"|β| = {b:g}", fontsize=10, color="#191919")
            if j == 0:
                ax.set_ylabel(f"{names[noise]}\nrecovery rate", color="#191919")
            if i == len(noises) - 1:
                ax.set_xlabel("integrated SNR (dB)", color="#191919")
            ax.tick_params(colors="#666666")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(
        "Held-out recovery of a hidden log-periodic mode (pooled over designs and sign of β)",
        y=0.995,
        fontsize=10.5,
        color="#191919",
    )
    fig.legend(
        handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
