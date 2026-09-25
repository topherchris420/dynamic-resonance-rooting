"""Monte Carlo calibration of the DRR rooting test and resonance-depth score.

Every number this module reports comes from simulation with a known truth:

* **Size.** Independent AR(1) series have no directed relationships, so any
  significant edge is a family-wise false positive. A valid test keeps that
  rate near the nominal ``alpha`` at every autocorrelation level.
* **Power.** A source drives a target at a known lag and coupling. The study
  records how often the rooting test recovers that exact edge and how often it
  reports any other edge.
* **Depth reference.** Resonance depth is a descriptive score, not a test. The
  study records its distribution under white and red noise and for noisy
  tones, so a reader can see what a given depth value does and does not rule
  out.

The artifact is deterministic for a fixed configuration and seed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .modules import DEPTH_METHOD_VERSION, DepthCalculator, RootingAnalyzer

SCHEMA_VERSION = "drr-calibration-study-v1"
Z_95 = 1.959963984540054

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20260925,
    "alpha": 0.05,
    "n_samples": 512,
    "max_lag": 4,
    "n_surrogates": 99,
    "size": {
        "n_trials": 400,
        "n_variables": 3,
        "ar_coefficients": [0.0, 0.5, 0.9, 0.97],
        "surrogate_methods": ["circular_shift", "permutation"],
    },
    "power": {
        "n_trials": 200,
        "ar_coefficient": 0.5,
        "lag": 2,
        "couplings": [0.0, 0.1, 0.15, 0.2, 0.3, 0.5],
    },
    "depth": {
        "n_trials": 200,
        "window_size": 256,
        "noise_ar_coefficients": [0.0, 0.5, 0.9],
        "tone_cycles_per_sample": 0.0625,
        "tone_amplitudes": [0.25, 0.5, 1.0],
    },
}


def simulate_ar1(
    n_samples: int,
    n_variables: int,
    coefficient: float,
    rng: np.random.Generator,
    burn_in: int = 200,
) -> np.ndarray:
    """Independent stationary AR(1) series with unit innovation variance."""
    if not -1.0 < coefficient < 1.0:
        raise ValueError("coefficient must lie in (-1, 1) for a stationary AR(1)")
    shocks = rng.standard_normal((n_samples + burn_in, n_variables))
    series = np.empty_like(shocks)
    series[0] = shocks[0] / np.sqrt(1.0 - coefficient**2)
    for t in range(1, len(shocks)):
        series[t] = coefficient * series[t - 1] + shocks[t]
    return series[burn_in:]


def simulate_directed_pair(
    n_samples: int,
    coupling: float,
    lag: int,
    ar_coefficient: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Three AR(1) channels where channel 0 drives channel 1 at ``lag``.

    ``coupling`` is the correlation between the target and the lagged source
    (both have unit variance). Channel 2 is an independent distractor.
    """
    if not 0.0 <= coupling < 1.0:
        raise ValueError("coupling must lie in [0, 1)")
    if lag < 1:
        raise ValueError("lag must be at least 1")
    base = simulate_ar1(n_samples + lag, 3, ar_coefficient, rng)
    base /= np.sqrt(1.0 / (1.0 - ar_coefficient**2))
    source = base[:, 0]
    target = coupling * np.roll(source, lag) + np.sqrt(1.0 - coupling**2) * base[:, 1]
    data = np.column_stack((source, target, base[:, 2]))
    return data[lag:]


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1.0 - p) / trials + z**2 / (4 * trials**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _rate(successes: int, trials: int) -> Dict[str, Any]:
    low, high = wilson_interval(successes, trials)
    return {"rate": successes / trials, "count": successes, "trials": trials, "ci95": [low, high]}


def rooting_size(
    *,
    n_trials: int,
    n_samples: int,
    n_variables: int,
    ar_coefficient: float,
    max_lag: int,
    n_surrogates: int,
    alpha: float,
    surrogate_method: str,
    seed: int,
) -> Dict[str, Any]:
    """Family-wise false-positive rate of the rooting test under independence."""
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    false_positive_trials = []
    for trial in range(n_trials):
        data = simulate_ar1(n_samples, n_variables, ar_coefficient, rng)
        result = analyzer.analyze(
            data,
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            random_state=seed + 1 + trial,
            alpha=alpha,
            surrogate_method=surrogate_method,
        )
        if result["significant_edges"]:
            false_positive_trials.append(trial)
    return {
        "ar_coefficient": ar_coefficient,
        "surrogate_method": surrogate_method,
        "family_wise_error": _rate(len(false_positive_trials), n_trials),
        # Trial indices let a reviewer rerun any prefix of the row and compare exactly.
        "false_positive_trials": false_positive_trials,
    }


def rooting_power(
    *,
    n_trials: int,
    n_samples: int,
    coupling: float,
    lag: int,
    ar_coefficient: float,
    max_lag: int,
    n_surrogates: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    """How often the true edge, and any other edge, is reported."""
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    exact, any_lag, other = 0, 0, 0
    for trial in range(n_trials):
        data = simulate_directed_pair(n_samples, coupling, lag, ar_coefficient, rng)
        result = analyzer.analyze(
            data,
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            random_state=seed + 1 + trial,
            alpha=alpha,
        )
        true_edges = [
            edge
            for edge in result["significant_edges"]
            if edge["source"] == "dim_0" and edge["target"] == "dim_1"
        ]
        any_lag += bool(true_edges)
        exact += any(edge["lag"] == lag for edge in true_edges)
        other += len(result["significant_edges"]) > len(true_edges)
    return {
        "coupling": coupling,
        "true_edge_at_true_lag": _rate(exact, n_trials),
        "true_edge_at_any_lag": _rate(any_lag, n_trials),
        "any_other_edge": _rate(other, n_trials),
    }


def depth_reference(
    *,
    n_trials: int,
    n_samples: int,
    window_size: int,
    noise_ar_coefficients: Sequence[float],
    tone_cycles_per_sample: float,
    tone_amplitudes: Sequence[float],
    seed: int,
) -> List[Dict[str, Any]]:
    """Resonance-depth quantiles for noise and for tones in white noise."""
    rng = np.random.default_rng(seed)
    calculator = DepthCalculator()
    t = np.arange(n_samples)
    cases: List[Tuple[str, Dict[str, float], Any]] = []
    for coefficient in noise_ar_coefficients:
        cases.append(
            (
                "ar1_noise",
                {"ar_coefficient": float(coefficient)},
                lambda c=coefficient: simulate_ar1(n_samples, 1, c, rng)[:, 0],
            )
        )
    for amplitude in tone_amplitudes:
        cases.append(
            (
                "tone_in_white_noise",
                {"amplitude": float(amplitude), "snr_db": float(10 * np.log10(amplitude**2 / 2))},
                lambda a=amplitude: a
                * np.sin(2 * np.pi * tone_cycles_per_sample * t + rng.uniform(0, 2 * np.pi))
                + rng.standard_normal(n_samples),
            )
        )

    rows = []
    for kind, parameters, draw in cases:
        depths = np.array(
            [
                calculator.calculate(draw(), window_size, sampling_rate=1.0)["resonance_depth"]
                for _ in range(n_trials)
            ]
        )
        quantiles = np.quantile(depths, [0.05, 0.5, 0.95])
        rows.append(
            {
                "signal": kind,
                **parameters,
                "depth_q05": float(quantiles[0]),
                "depth_median": float(quantiles[1]),
                "depth_q95": float(quantiles[2]),
            }
        )
    return rows


def _checks(size_rows: Sequence[Mapping], power_rows: Sequence[Mapping], alpha: float) -> Dict:
    """Pass/fail statements derived only from the numbers above them."""
    circular = [row for row in size_rows if row["surrogate_method"] == "circular_shift"]
    permutation = [row for row in size_rows if row["surrogate_method"] == "permutation"]
    null_power = [row for row in power_rows if row["coupling"] == 0.0]
    strongest = max(power_rows, key=lambda row: row["coupling"]) if power_rows else None
    return {
        "circular_shift_size_interval_covers_alpha_or_lies_below": all(
            row["family_wise_error"]["ci95"][0] <= alpha for row in circular
        ),
        "permutation_size_exceeds_alpha_under_autocorrelation": any(
            row["family_wise_error"]["ci95"][0] > alpha
            for row in permutation
            if row["ar_coefficient"] > 0
        ),
        "zero_coupling_true_edge_rate_within_alpha": all(
            row["true_edge_at_any_lag"]["ci95"][0] <= alpha for row in null_power
        ),
        "strongest_coupling_detected_in_most_trials": bool(
            strongest and strongest["true_edge_at_true_lag"]["rate"] >= 0.8
        ),
    }


def run_calibration_study(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run the full size, power, and depth study and return a JSON-ready artifact."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    seed = int(cfg["seed"])
    shared: Dict[str, Any] = {
        "n_samples": int(cfg["n_samples"]),
        "max_lag": int(cfg["max_lag"]),
        "n_surrogates": int(cfg["n_surrogates"]),
        "alpha": float(cfg["alpha"]),
    }

    size_rows = []
    size_cfg = cfg["size"]
    for i, coefficient in enumerate(size_cfg["ar_coefficients"]):
        for j, method in enumerate(size_cfg["surrogate_methods"]):
            size_rows.append(
                rooting_size(
                    n_trials=int(size_cfg["n_trials"]),
                    n_variables=int(size_cfg["n_variables"]),
                    ar_coefficient=float(coefficient),
                    surrogate_method=method,
                    seed=seed + 10_000 * (i + 1) + 1_000 * j,
                    **shared,
                )
            )

    power_cfg = cfg["power"]
    power_rows = [
        rooting_power(
            n_trials=int(power_cfg["n_trials"]),
            coupling=float(coupling),
            lag=int(power_cfg["lag"]),
            ar_coefficient=float(power_cfg["ar_coefficient"]),
            seed=seed + 100_000 + 1_000 * k,
            **shared,
        )
        for k, coupling in enumerate(power_cfg["couplings"])
    ]

    depth_cfg = cfg["depth"]
    depth_rows = depth_reference(
        n_trials=int(depth_cfg["n_trials"]),
        n_samples=int(cfg["n_samples"]),
        window_size=int(depth_cfg["window_size"]),
        noise_ar_coefficients=depth_cfg["noise_ar_coefficients"],
        tone_cycles_per_sample=float(depth_cfg["tone_cycles_per_sample"]),
        tone_amplitudes=depth_cfg["tone_amplitudes"],
        seed=seed + 200_000,
    )

    config_json = json.dumps(cfg, sort_keys=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "depth_method": DEPTH_METHOD_VERSION,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "size": size_rows,
        "power": power_rows,
        "depth": depth_rows,
        "checks": _checks(size_rows, power_rows, float(cfg["alpha"])),
    }


def _pct(interval: Sequence[float]) -> str:
    return f"{interval[0]:.3f}–{interval[1]:.3f}"


def render_calibration_report(artifact: Mapping[str, Any]) -> str:
    """Render the Markdown companion of a calibration artifact."""
    cfg = artifact["config"]
    alpha = cfg["alpha"]
    lines = [
        "# DRR calibration study",
        "",
        "Generated by `python scripts/run_calibration_study.py --output-dir results/expected`",
        f"from `{artifact['schema_version']}` (config SHA-256 `{artifact['config_sha256'][:12]}…`).",
        "Every value below is a Monte Carlo estimate with a known truth. The JSON beside",
        "this file is the machine-readable record.",
        "",
        f"Shared settings: {cfg['n_samples']} samples, lags 1–{cfg['max_lag']},"
        f" {cfg['n_surrogates']} surrogates, alpha = {alpha}, max-statistic correction.",
        "",
        "## Size: false positives on independent series",
        "",
        f"{cfg['size']['n_variables']} independent AR(1) channels, {cfg['size']['n_trials']}"
        " trials per row. Any significant edge is a family-wise error. A valid test stays"
        f" near {alpha}.",
        "",
        "| AR(1) coefficient | Surrogate null | Family-wise error | 95% interval |",
        "| ---: | --- | ---: | --- |",
    ]
    for row in artifact["size"]:
        fwe = row["family_wise_error"]
        lines.append(
            f"| {row['ar_coefficient']:.2f} | {row['surrogate_method']} | {fwe['rate']:.3f} |"
            f" {_pct(fwe['ci95'])} |"
        )
    power_cfg = cfg["power"]
    lines += [
        "",
        "## Power: recovering a known directed edge",
        "",
        f"AR({power_cfg['ar_coefficient']}) source drives the target at lag {power_cfg['lag']};"
        f" a third channel is an independent distractor. {power_cfg['n_trials']} trials per"
        " row. Coupling is the correlation between the target and the lagged source.",
        "",
        "| Coupling | True edge at true lag | True edge at any lag | Any other edge |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in artifact["power"]:
        lines.append(
            f"| {row['coupling']:.2f} | {row['true_edge_at_true_lag']['rate']:.3f} |"
            f" {row['true_edge_at_any_lag']['rate']:.3f} | {row['any_other_edge']['rate']:.3f} |"
        )
    depth_cfg = cfg["depth"]
    lines += [
        "",
        "## Resonance depth reference distribution",
        "",
        f"Depth (`{artifact['depth_method']}`) is descriptive, not a test."
        f" {depth_cfg['n_trials']} draws per row, window {depth_cfg['window_size']}."
        f" Tones sit at {depth_cfg['tone_cycles_per_sample']} cycles per sample in unit-variance"
        " white noise.",
        "",
        "| Signal | Parameter | 5% | Median | 95% |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in artifact["depth"]:
        if row["signal"] == "ar1_noise":
            label = f"AR(1) coefficient {row['ar_coefficient']:.2f}"
        else:
            label = f"amplitude {row['amplitude']:.2f} (SNR {row['snr_db']:.1f} dB)"
        lines.append(
            f"| {row['signal']} | {label} | {row['depth_q05']:.3f} |"
            f" {row['depth_median']:.3f} | {row['depth_q95']:.3f} |"
        )
    lines += ["", "## Checks", "", "| Statement | Holds |", "| --- | --- |"]
    for name, holds in sorted(artifact["checks"].items()):
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if holds else 'no'} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIG",
    "SCHEMA_VERSION",
    "depth_reference",
    "render_calibration_report",
    "rooting_power",
    "rooting_size",
    "run_calibration_study",
    "simulate_ar1",
    "simulate_directed_pair",
    "wilson_interval",
]
