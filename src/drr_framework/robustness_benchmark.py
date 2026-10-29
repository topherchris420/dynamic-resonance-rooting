"""Paired rooting comparisons with frozen calibration and known synthetic truth.

The simple baselines threshold the largest absolute lagged correlation across
all directed pairs. One uses raw samples; the other first fits and removes one
AR(1) coefficient per channel. Thresholds are learned on separate independent
white-noise trials and never retuned on the evaluation cases. Their behavior
under long memory or a regime switch is an empirical transfer test.

Scores and surrogate p-values are not probabilities that an edge is true.
Wilson intervals describe Monte Carlo variation across independent trials,
not uncertainty about an individual edge or physical causation.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import binomial_rate
from .dependent_noise import simulate_fractional_gaussian_noise
from .modules import RootingAnalyzer

SCHEMA_VERSION = "drr-robustness-benchmark-v1"
BASELINES = ("raw_correlation", "ar1_residual_correlation")
METHODS = BASELINES + ("drr_circular_shift", "drr_phase_randomization")
METHOD_NOTES = {
    "raw_correlation": "Max-over-lags correlation; white-noise calibrated family maximum.",
    "ar1_residual_correlation": (
        "Per-window AR(1) residual correlation; separately white-noise calibrated family maximum."
    ),
    "drr_circular_shift": (
        "Existing RootingAnalyzer: circular-shift max-statistic p-values and candidate-edge gate."
    ),
    "drr_phase_randomization": (
        "Existing RootingAnalyzer: phase-randomization max-statistic p-values and candidate-edge gate."
    ),
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20261029,
    "alpha": 0.05,
    "n_samples": 512,
    "max_lag": 4,
    "lag": 2,
    "n_surrogates": 99,
    "n_calibration": 399,
    "n_trials": 200,
    "cases": [
        {
            "id": f"fgn_{hurst:.2f}_coupling_{coupling:.2f}",
            "hurst": hurst,
            "regimes": [{"coupling": coupling, "reverse": False}],
        }
        for hurst in (0.5, 0.85, 0.95)
        for coupling in (0.0, 0.1, 0.3, 0.5)
    ]
    + [
        {
            "id": name,
            "hurst": 0.85,
            "regimes": [
                {"coupling": before, "reverse": False},
                {"coupling": after, "reverse": reverse},
            ],
        }
        for name, before, after, reverse in (
            ("coupling_onset", 0.0, 0.3, False),
            ("coupling_offset", 0.3, 0.0, False),
            ("direction_reversal", 0.3, 0.3, True),
        )
    ],
}


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("seed", "n_samples", "max_lag", "lag", "n_surrogates", "n_calibration", "n_trials"):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"{key} must be an integer in its permitted range")
    alpha = config["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if config["n_samples"] < config["max_lag"] + 4 or config["lag"] > config["max_lag"]:
        raise ValueError("n_samples must support the lag search and lag must be <= max_lag")
    if 1.0 / (config["n_surrogates"] + 1) > alpha:
        raise ValueError("n_surrogates cannot resolve alpha")
    if math.ceil((config["n_calibration"] + 1) * (1 - alpha)) > config["n_calibration"]:
        raise ValueError("n_calibration cannot resolve alpha")
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    names = set()
    for case in cases:
        name = case["id"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("case identifiers must be nonempty and unique")
        names.add(name)
        if not 0 < case["hurst"] < 1 or not case["regimes"]:
            raise ValueError("each case requires a Hurst exponent in (0, 1) and regimes")
        for regime in case["regimes"]:
            if not 0 <= regime["coupling"] < 1 or not isinstance(regime["reverse"], bool):
                raise ValueError("regimes require coupling in [0, 1) and a boolean reverse flag")


def _seed(config: Mapping[str, Any], *parts: Any) -> int:
    material = json.dumps([config["seed"], *parts], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _data_digest(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data, dtype="<f8").tobytes()).hexdigest()


def correlation_edges(data: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Conventional Pearson correlations, independently of the DRR implementation."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or len(data) < max_lag + 3 or max_lag < 1:
        raise ValueError("data must support a multivariate lag search")
    if not np.all(np.isfinite(data)) or np.any(np.std(data, axis=0) == 0):
        raise ValueError("data must be finite with nonconstant channels")
    n_variables = data.shape[1]
    scores = np.zeros((n_variables, n_variables))
    lags = np.zeros((n_variables, n_variables), dtype=int)
    for lag in range(1, max_lag + 1):
        correlation = np.abs(np.corrcoef(data[:-lag].T, data[lag:].T)[:n_variables, n_variables:])
        if not np.all(np.isfinite(correlation)):
            raise ValueError("lagged samples must have nonconstant channels")
        np.fill_diagonal(correlation, 0.0)
        improved = correlation > scores
        scores[improved] = correlation[improved]
        lags[improved] = lag
    return scores, lags


def ar1_residuals(data: np.ndarray) -> np.ndarray:
    """Fit each channel separately inside the observation window; discard its first row."""
    centered = np.asarray(data, dtype=float) - np.mean(data, axis=0)
    previous, current = centered[:-1], centered[1:]
    denominator = np.sum(previous**2, axis=0)
    coefficients = np.divide(
        np.sum(previous * current, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return current - previous * coefficients


def calibrate_baselines(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Freeze family-maximum thresholds using only separate white-noise draws.

    Reject strictly above order statistic ceil((m+1)*(1-alpha)). The rank rule
    is conservative for one exchangeable future family maximum, including ties.
    It is not a guarantee after transferring to a different noise distribution.
    """
    validate_config(config)
    maxima: Dict[str, List[float]] = {method: [] for method in BASELINES}
    hashes = []
    for trial in range(config["n_calibration"]):
        rng = np.random.default_rng(_seed(config, "calibration", trial))
        data = rng.standard_normal((config["n_samples"], 3))
        hashes.append(_data_digest(data))
        for method, samples in zip(BASELINES, (data, ar1_residuals(data))):
            scores, _ = correlation_edges(samples, config["max_lag"])
            maxima[method].append(float(scores.max()))
    rank = math.ceil((config["n_calibration"] + 1) * (1 - config["alpha"]))
    return {
        "reference": "independent standard Gaussian channels; separate seeds; no evaluation data",
        "rank": rank,
        "input_sha256": hashes,
        "methods": {
            method: {"threshold": sorted(values)[rank - 1], "family_maxima": values}
            for method, values in maxima.items()
        },
    }


def simulate_regimes(
    config: Mapping[str, Any], case: Mapping[str, Any], trial: int
) -> List[np.ndarray]:
    """Continuous fGn noise with one complete analysis window per declared regime.

    Drop a lag-length buffer at each boundary, so no scored observation obtains
    its source from a differently labeled regime. No analysis window straddles
    the switch. This measures before/after recovery, not change-detection delay.
    """
    n_samples, lag = config["n_samples"], config["lag"]
    length = n_samples + lag
    rng = np.random.default_rng(_seed(config, "evaluation", case["id"], trial))
    base = simulate_fractional_gaussian_noise(length * len(case["regimes"]), 3, case["hurst"], rng)
    windows = []
    for index, regime in enumerate(case["regimes"]):
        start = index * length
        data = base[start + lag : start + length].copy()
        source, target = (1, 0) if regime["reverse"] else (0, 1)
        coupling = regime["coupling"]
        data[:, target] = (
            coupling * base[start : start + n_samples, source]
            + np.sqrt(1 - coupling**2) * data[:, target]
        )
        windows.append(data)
    return windows


def _truth(config: Mapping[str, Any], regime: Mapping[str, Any]) -> List[List[int]]:
    if regime["coupling"] == 0:
        return []
    source, target = (1, 0) if regime["reverse"] else (0, 1)
    return [[source, target, config["lag"]]]


def evaluate_trial(
    config: Mapping[str, Any], case: Mapping[str, Any], trial: int, calibration: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """All methods see the same windows; the seed schedule permits prefix replay."""
    records = []
    for index, data in enumerate(simulate_regimes(config, case, trial)):
        outcomes = {}
        for method, samples in zip(BASELINES, (data, ar1_residuals(data))):
            scores, lags = correlation_edges(samples, config["max_lag"])
            threshold = calibration["methods"][method]["threshold"]
            outcomes[method] = [
                [int(source), int(target), int(lags[source, target])]
                for source, target in zip(*np.where(scores > threshold))
            ]
        for method in METHODS[2:]:
            result = RootingAnalyzer().analyze(
                data,
                max_lag=config["max_lag"],
                n_surrogates=config["n_surrogates"],
                alpha=config["alpha"],
                surrogate_method=method[4:],
                random_state=_seed(config, "surrogates", case["id"], trial, index, method),
            )
            outcomes[method] = [
                [
                    int(edge["source"].split("_")[-1]),
                    int(edge["target"].split("_")[-1]),
                    int(edge["lag"]),
                ]
                for edge in result["significant_edges"]
            ]
        records.append(
            {
                "case": case["id"],
                "trial": trial,
                "window": index,
                "input_sha256": _data_digest(data),
                "truth": _truth(config, case["regimes"][index]),
                "methods": outcomes,
            }
        )
    return records


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Rebuild every reported proportion from retained per-trial decisions."""
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    seen = set()
    cases = {case["id"]: case for case in config["cases"]}
    for record in records:
        case, trial, window = record["case"], record["trial"], record["window"]
        if type(trial) is not int or type(window) is not int:
            raise ValueError("trial and window identities must be integers")
        digest = record["input_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid input SHA-256")
        if case not in cases or not 0 <= trial < config["n_trials"]:
            raise ValueError("unknown case or trial")
        if not 0 <= window < len(cases[case]["regimes"]):
            raise ValueError("unknown window")
        identity = (case, trial, window)
        if identity in seen:
            raise ValueError("duplicate trial/window record")
        seen.add(identity)
        if record["truth"] != _truth(config, cases[case]["regimes"][window]):
            raise ValueError("record truth differs from the protocol")
        if set(record["methods"]) != set(METHODS):
            raise ValueError("missing or unknown method")
        for edges in record["methods"].values():
            if len({tuple(edge) for edge in edges}) != len(edges):
                raise ValueError("duplicate edge")
            for source, target, lag in edges:
                if any(type(value) is not int for value in (source, target, lag)):
                    raise ValueError("edge endpoints and lag must be integers")
                if source not in range(3) or target not in range(3) or source == target:
                    raise ValueError("invalid edge endpoints")
                if lag not in range(1, config["max_lag"] + 1):
                    raise ValueError("invalid edge lag")
        grouped.setdefault((case, window), []).append(record)
    expected = config["n_trials"] * sum(len(case["regimes"]) for case in cases.values())
    if len(seen) != expected:
        raise ValueError("incomplete trial/window records")
    summaries = []
    for case in config["cases"]:
        for window, regime in enumerate(case["regimes"]):
            batch = grouped[(case["id"], window)]
            truth = _truth(config, regime)
            for method in METHODS:
                detections, false_edges, exact, any_lag = 0, 0, 0, 0
                for record in batch:
                    edges = record["methods"][method]
                    detections += bool(edges)
                    false_edges += any(not truth or edge[:2] != truth[0][:2] for edge in edges)
                    exact += bool(truth and truth[0] in edges)
                    any_lag += bool(truth and any(edge[:2] == truth[0][:2] for edge in edges))
                false_rate = binomial_rate(false_edges, len(batch))
                assessment = None
                if not truth:
                    low, high = false_rate["ci95"]
                    assessment = (
                        "inflated"
                        if low > config["alpha"]
                        else (
                            "upper_bound_at_or_below_alpha"
                            if high <= config["alpha"]
                            else "inconclusive"
                        )
                    )
                summaries.append(
                    {
                        "case": case["id"],
                        "window": window,
                        "hurst": case["hurst"],
                        "coupling": regime["coupling"],
                        "truth": truth,
                        "method": method,
                        "any_detection": binomial_rate(detections, len(batch)),
                        "any_non_injected_direction": false_rate,
                        "true_edge_any_lag": binomial_rate(any_lag, len(batch)) if truth else None,
                        "true_edge_exact_lag": binomial_rate(exact, len(batch)) if truth else None,
                        "null_assessment": assessment,
                    }
                )
    return summaries


def run_robustness_benchmark(
    config: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config, allow_nan=False))
    validate_config(cfg)
    calibration = calibrate_baselines(cfg)
    records = []
    for case in cfg["cases"]:
        for trial in range(cfg["n_trials"]):
            records.extend(evaluate_trial(cfg, case, trial, calibration))
        if progress is not None:
            progress(case["id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "config_sha256": config_digest(cfg),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "method_notes": METHOD_NOTES,
        "calibration": calibration,
        "records": records,
        "summary": summarize_records(records, cfg),
    }


def render_robustness_report(artifact: Mapping[str, Any]) -> str:
    def rate(value: Optional[Mapping[str, Any]]) -> str:
        if value is None:
            return "n/a (no injected edge)"
        low, high = value["ci95"]
        return f"{value['rate']:.3f} [{low:.3f}, {high:.3f}]"

    config = artifact["config"]
    lines = [
        "# DRR robustness benchmark",
        "",
        f"Schema: `{artifact['schema_version']}`. Config SHA-256: `{artifact['config_sha256']}`.",
        f"{config['n_trials']} independent trials per case; {config['n_samples']} samples per window;",
        f"lags 1-{config['max_lag']}; {config['n_surrogates']} surrogates; alpha {config['alpha']}.",
        "",
        "![Null false alarms and injected-lag recovery across noise-memory strengths](comparison.png)",
        "",
        "## Frozen calibration",
        "",
        f"Baselines use {config['n_calibration']} separate white-noise trials. Reject strictly above",
        f"order statistic {artifact['calibration']['rank']} of the family maxima. No evaluation data",
        "selects a threshold. Both baselines search the same directed pairs and lags.",
        "",
    ]
    for method in BASELINES:
        threshold = artifact["calibration"]["methods"][method]["threshold"]
        lines.append(f"- `{method}`: threshold {threshold:.6f}.")
    lines += ["", "## Methods", ""]
    lines += [f"- `{method}`: {artifact['method_notes'][method]}" for method in METHODS]
    lines += [
        "",
        "## Trial-level results",
        "",
        "Rates have pointwise Wilson 95% intervals across trials. Windows within a trial are paired;",
        "they are never pooled as independent replicates. Non-injected directions include the reverse",
        "and distractor pairs. Under coupling, autocorrelation can induce additional associations;",
        "these are off-target relative to the generator, not necessarily false statistical relationships.",
        "On uncoupled windows every reported direction is a family-wise false alarm.",
        "",
        "| Case | Window | Method | Injected direction + lag | Any non-injected direction | Null assessment |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in artifact["summary"]:
        lines.append(
            f"| {row['case']} | {row['window']} | {row['method']} |"
            f" {rate(row['true_edge_exact_lag'])} | {rate(row['any_non_injected_direction'])} |"
            f" {row['null_assessment'] or 'n/a (coupled)'} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- All data are synthetic. No physical or domain-transfer validity is established.",
        "- White-noise calibration does not establish error control under long memory or switches.",
        "- A null row is inflated when its interval lower bound exceeds alpha. An interval crossing",
        "  alpha is inconclusive; failure to detect inflation does not establish control.",
        "- Intervals are pointwise, not simultaneous. No global winner or superiority test is declared.",
        "- Rate intervals condition on the retained baseline thresholds; they do not include",
        "  variation from drawing a new calibration set.",
        "- Regime windows do not straddle switches; a lag-length buffer is discarded at each boundary.",
        "  These are before/after comparisons, not measurements of online change-detection delay.",
        "- Correlation scores and p-values are not calibrated probabilities that an edge is true.",
        "- This is a fixed exploratory benchmark, not an independently registered confirmatory study.",
        "- JSON retains calibration maxima, data hashes and all trial decisions for replay and audit.",
        "",
    ]
    return "\n".join(lines)
