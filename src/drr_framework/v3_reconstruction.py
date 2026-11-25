"""Independent reconstruction of the v3 studies in the September 2026 DRR paper.

The September 2026 paper (SSRN 5285993) reported three operators and their
calibration studies, but its study scripts and JSON outputs were never
committed. This module rebuilds the operators and reruns the studies from the
protocol in ``experiments/v3_reconstruction/protocol.json``, which was
registered before any reconstruction run. Every implementation choice the
manuscript leaves open is fixed there as an assumption (A1 to A14).

The results produced here are reconstruction outputs. They are not recovered
original outputs, and exact numerical agreement with the paper is not expected
because the original random-number streams cannot be recovered.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .calibration import binomial_rate, simulate_ar1, simulate_directed_pair
from .modules import DepthCalculator, RootingAnalyzer

SCHEMA_VERSION = "drr-v3-reconstruction-1"
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Simulation (assumptions A1 to A4)
# ---------------------------------------------------------------------------


def ar2_coefficients(radius: float, frequency: float) -> Tuple[float, float]:
    """AR(2) coefficients of a stochastic oscillator with pole radius ``radius``."""
    return 2.0 * radius * math.cos(2.0 * math.pi * frequency), -(radius**2)


def ar2_variance(a1: float, a2: float) -> float:
    """Stationary variance of x_t = a1 x_{t-1} + a2 x_{t-2} + e_t with unit-variance e."""
    return (1.0 - a2) / ((1.0 + a2) * ((1.0 - a2) ** 2 - a1**2))


def simulate_ar2(
    n_samples: int,
    n_variables: int,
    radius: float,
    rng: np.random.Generator,
    frequency: float = 0.0625,
    burn_in: int = 200,
) -> np.ndarray:
    """Independent AR(2) oscillators scaled to unit theoretical variance (A2)."""
    a1, a2 = ar2_coefficients(radius, frequency)
    shocks = rng.standard_normal((n_samples + burn_in, n_variables))
    series = np.zeros_like(shocks)
    series[0] = shocks[0]
    series[1] = a1 * series[0] + shocks[1]
    for t in range(2, len(shocks)):
        series[t] = a1 * series[t - 1] + a2 * series[t - 2] + shocks[t]
    return series[burn_in:] / math.sqrt(ar2_variance(a1, a2))


def simulate_directed_ar2(
    n_samples: int, coupling: float, lag: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """The coupled design of ``simulate_directed_pair`` with AR(2) channels (A2)."""
    base = simulate_ar2(n_samples + lag, 3, radius, rng)
    source = base[:, 0]
    target = coupling * np.roll(source, lag) + math.sqrt(1.0 - coupling**2) * base[:, 1]
    return np.column_stack((source, target, base[:, 2]))[lag:]


def simulate_process(
    process: Mapping[str, Any], n_samples: int, n_variables: int, rng: np.random.Generator
) -> np.ndarray:
    """Independent channels from a protocol process specification."""
    if process["kind"] == "ar1":
        return simulate_ar1(n_samples, n_variables, float(process["phi"]), rng)
    if process["kind"] == "ar2_oscillator":
        return simulate_ar2(n_samples, n_variables, float(process["r"]), rng)
    raise ValueError(f"unknown process kind {process['kind']!r}")


def simulate_coupled(
    process: Mapping[str, Any],
    n_samples: int,
    coupling: float,
    lag: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Three channels, channel 0 driving channel 1 at ``lag``."""
    if process["kind"] == "ar1":
        return simulate_directed_pair(n_samples, coupling, lag, float(process["phi"]), rng)
    if process["kind"] == "ar2_oscillator":
        return simulate_directed_ar2(n_samples, coupling, lag, float(process["r"]), rng)
    raise ValueError(f"unknown process kind {process['kind']!r}")


def unit_ar1(n_samples: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """One AR(1) series scaled to unit theoretical variance."""
    return simulate_ar1(n_samples, 1, phi, rng)[:, 0] * math.sqrt(1.0 - phi**2)


def simulate_resonance_series(
    family: Mapping[str, Any], n_samples: int, line_frequency: float, rng: np.random.Generator
) -> np.ndarray:
    """One series from a resonance-study family (A3, A4)."""
    kind = family["kind"]
    if kind == "ar1":
        return unit_ar1(n_samples, float(family["phi"]), rng)
    if kind == "ar1_plus_white":
        return unit_ar1(n_samples, float(family["phi"]), rng) + rng.standard_normal(n_samples)
    if kind == "ar2_oscillator":
        return simulate_ar2(n_samples, 1, float(family["r"]), rng, frequency=line_frequency)[:, 0]
    if kind in ("line_in_white", "line_in_ar1"):
        if kind == "line_in_white":
            background = rng.standard_normal(n_samples)
        else:
            background = unit_ar1(n_samples, float(family["phi"]), rng)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        t = np.arange(n_samples)
        line = float(family["amplitude"]) * np.sin(2.0 * math.pi * line_frequency * t + phase)
        return background + line
    raise ValueError(f"unknown family kind {kind!r}")


# ---------------------------------------------------------------------------
# Prewhitening and Granger (A5, A6)
# ---------------------------------------------------------------------------


def _lag_design(series: np.ndarray, order: int, start: int) -> np.ndarray:
    """Intercept and lags 1..order of ``series`` for targets at indices start..T-1."""
    columns = [np.ones(len(series) - start)]
    for lag in range(1, order + 1):
        columns.append(series[start - lag : len(series) - lag])
    return np.column_stack(columns)


def _ols_rss(design: np.ndarray, response: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coef
    return float(residual @ residual), coef, residual


def aic_order(series: np.ndarray, max_order: int, min_order: int = 0) -> int:
    """AR order chosen by AIC = n log(RSS/n) + 2(p+1) on the common sample (A5)."""
    response = series[max_order:]
    n = len(response)
    best_order, best_aic = min_order, math.inf
    for order in range(min_order, max_order + 1):
        rss, _, _ = _ols_rss(_lag_design(series, order, max_order), response)
        aic = n * math.log(max(rss, 1e-300) / n) + 2.0 * (order + 1)
        if aic < best_aic:
            best_order, best_aic = order, aic
    return best_order


def prewhiten(data: np.ndarray, max_order: int = 10) -> Tuple[np.ndarray, List[int]]:
    """Per-channel AR(p) innovations on the common sample t = max_order..T-1 (A5)."""
    data = np.asarray(data, dtype=float)
    innovations = np.empty((data.shape[0] - max_order, data.shape[1]))
    orders = []
    for j in range(data.shape[1]):
        series = data[:, j]
        order = aic_order(series, max_order)
        _, _, residual = _ols_rss(_lag_design(series, order, max_order), series[max_order:])
        innovations[:, j] = residual
        orders.append(order)
    return innovations, orders


def granger_pvalue(
    source: np.ndarray,
    target: np.ndarray,
    max_own_order: int = 10,
    source_lags: int = 4,
) -> Tuple[float, int]:
    """F-test p-value that source lags 1..``source_lags`` help predict ``target`` (A6).

    Returns the p-value and the source lag with the largest absolute coefficient.
    """
    start = max(max_own_order, source_lags)
    own = aic_order(target, max_own_order, min_order=1)
    response = target[start:]
    restricted = _lag_design(target, own, start)
    source_columns = np.column_stack(
        [source[start - lag : len(source) - lag] for lag in range(1, source_lags + 1)]
    )
    unrestricted = np.column_stack((restricted, source_columns))
    rss_r, _, _ = _ols_rss(restricted, response)
    rss_u, coef, _ = _ols_rss(unrestricted, response)
    df_den = len(response) - unrestricted.shape[1]
    f_stat = ((rss_r - rss_u) / source_lags) / (rss_u / df_den)
    p_value = float(stats.f.sf(f_stat, source_lags, df_den))
    lag = int(np.argmax(np.abs(coef[-source_lags:]))) + 1
    return p_value, lag


def granger_edges(
    data: np.ndarray, alpha: float = 0.05, max_own_order: int = 10, source_lags: int = 4
) -> List[Dict[str, Any]]:
    """Pairwise Granger edges with Bonferroni over ordered pairs (A6)."""
    n_variables = data.shape[1]
    n_pairs = n_variables * (n_variables - 1)
    edges = []
    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:
                continue
            p_value, lag = granger_pvalue(data[:, i], data[:, j], max_own_order, source_lags)
            if p_value <= alpha / n_pairs:
                edges.append(
                    {"source": f"dim_{i}", "target": f"dim_{j}", "lag": lag, "p_value": p_value}
                )
    return edges


def v3_rooting(
    data: np.ndarray,
    *,
    max_lag: int,
    n_surrogates: int,
    random_state: int,
    alpha: float,
    max_ar_order: int = 10,
    surrogate_method: str = "circular_shift",
) -> Dict[str, Any]:
    """Prewhitened rooting: the unchanged v2 test on per-channel AR(p) innovations."""
    innovations, orders = prewhiten(data, max_ar_order)
    result = RootingAnalyzer().analyze(
        innovations,
        max_lag=max_lag,
        n_surrogates=n_surrogates,
        random_state=random_state,
        alpha=alpha,
        surrogate_method=surrogate_method,
    )
    result["ar_orders"] = orders
    return result


# ---------------------------------------------------------------------------
# Spectral operators (A7 to A11)
# ---------------------------------------------------------------------------


def periodogram(x: np.ndarray) -> np.ndarray:
    """I_k = |FFT(x - mean)_k|^2 / T at interior bins k = 1..floor(T/2)-1 (A7)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2 / n
    return spectrum[1 : n // 2]


def _pava_nonincreasing(values: np.ndarray) -> np.ndarray:
    """L2 pool-adjacent-violators fit constrained to be non-increasing."""
    means: List[float] = []
    weights: List[int] = []
    for value in values:
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] < means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            means[-2:] = [m]
            weights[-2:] = [w]
    return np.repeat(means, weights)


def monotone_fit(values: np.ndarray) -> np.ndarray:
    """Better of the non-increasing and non-decreasing L2 isotonic fits (A8)."""
    values = np.asarray(values, dtype=float)
    down = _pava_nonincreasing(values)
    up = _pava_nonincreasing(values[::-1])[::-1]
    rss_down = float(np.sum((values - down) ** 2))
    rss_up = float(np.sum((values - up) ** 2))
    return up if rss_up < rss_down else down


def lambda_statistic(x: np.ndarray) -> Tuple[float, int, np.ndarray, np.ndarray]:
    """Spectral excess Lambda = max_k I_k / S-hat_k (paper Eq. 4).

    Returns Lambda, its 1-based Fourier bin, the periodogram and S-hat.
    """
    ordinates = periodogram(x)
    log_fit = monotone_fit(np.log(np.maximum(ordinates, 1e-300)))
    s_hat = np.exp(log_fit + EULER_GAMMA)
    ratio = ordinates / s_hat
    k = int(np.argmax(ratio))
    return float(ratio[k]), k + 1, ordinates, s_hat


def gaussian_surrogate(
    spectrum: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian series whose interior periodogram has mean ``spectrum`` (A9)."""
    coefficients = np.zeros(n_samples // 2 + 1, dtype=complex)
    scale = np.sqrt(n_samples * spectrum / 2.0)
    noise = rng.standard_normal((2, len(spectrum)))
    coefficients[1 : n_samples // 2] = scale * (noise[0] + 1j * noise[1])
    return np.fft.irfft(coefficients, n=n_samples)


def monte_carlo_p(observed: float, null: Sequence[float]) -> float:
    """p = (1 + #{null >= observed}) / (B + 1) (A11)."""
    null_arr = np.asarray(null, dtype=float)
    return float((1 + np.sum(null_arr >= observed)) / (len(null_arr) + 1))


def lambda_test(x: np.ndarray, n_surrogates: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Lambda against refitted Gaussian surrogates with spectrum S-hat."""
    observed, k, ordinates, s_hat = lambda_statistic(x)
    null = [
        lambda_statistic(gaussian_surrogate(s_hat, len(x), rng))[0] for _ in range(n_surrogates)
    ]
    return {
        "lambda": observed,
        "argmax_bin": k,
        "p_value": monte_carlo_p(observed, null),
        "periodogram": ordinates,
        "s_hat": s_hat,
    }


def depth_value(x: np.ndarray, window: int) -> float:
    """drr_composite_v2 resonance depth of the last ``window`` samples."""
    return float(DepthCalculator().calculate(x, window, sampling_rate=1.0)["resonance_depth"])


def lag1_autocorrelation(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.dot(z[:-1], z[1:]) / np.dot(z, z))


def dstar_ar1(
    x: np.ndarray, window: int, n_draws: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Depth p-value against AR(1) nulls matched to the lag-1 autocorrelation (A10)."""
    observed = depth_value(x, window)
    phi = float(np.clip(lag1_autocorrelation(x), -0.99, 0.99))
    null = [depth_value(simulate_ar1(len(x), 1, phi, rng)[:, 0], window) for _ in range(n_draws)]
    return {"depth": observed, "phi_hat": phi, "p_value": monte_carlo_p(observed, null)}


def dstar_isotonic(
    x: np.ndarray, window: int, n_draws: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Depth p-value against Gaussian series with the linear isotonic periodogram fit."""
    observed = depth_value(x, window)
    spectrum = np.maximum(monotone_fit(periodogram(x)), 1e-300)
    null = [depth_value(gaussian_surrogate(spectrum, len(x), rng), window) for _ in range(n_draws)]
    return {"depth": observed, "p_value": monte_carlo_p(observed, null)}


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------


def _edge(edges: Sequence[Mapping[str, Any]], source: str, target: str) -> List[Mapping]:
    return [e for e in edges if e["source"] == source and e["target"] == target]


def rooting_size_row(
    process: Mapping[str, Any], seed: int, n_trials: int, shared: Mapping[str, Any]
) -> Dict[str, Any]:
    """Family-wise error of v2, v3 and Granger on independent channels."""
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    hits: Dict[str, List[int]] = {"v2": [], "v3": [], "granger": []}
    for trial in range(n_trials):
        data = simulate_process(process, shared["n_samples"], shared["n_channels"], rng)
        state = seed + 1 + trial
        common: Dict[str, Any] = dict(
            max_lag=shared["max_lag"],
            n_surrogates=shared["n_surrogates"],
            random_state=state,
            alpha=shared["alpha"],
        )
        v2 = analyzer.analyze(data, surrogate_method="circular_shift", **common)
        v3 = v3_rooting(data, **common)
        granger = granger_edges(data, alpha=shared["alpha"])
        for name, edges in (
            ("v2", v2["significant_edges"]),
            ("v3", v3["significant_edges"]),
            ("granger", granger),
        ):
            if edges:
                hits[name].append(trial)
    return {
        "process": dict(process),
        "seed": seed,
        **{name: binomial_rate(len(trials), n_trials) for name, trials in hits.items()},
        "false_positive_trials": hits,
    }


def rooting_power_row(
    process: Mapping[str, Any],
    seed: int,
    n_trials: int,
    coupling: float,
    lag: int,
    shared: Mapping[str, Any],
) -> Dict[str, Any]:
    """Exact-lag, any-lag and reverse-edge rates of v2, v3 and Granger."""
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    counts = {
        f"{name}_{metric}": 0
        for name in ("v2", "v3", "granger")
        for metric in ("exact", "any", "reverse")
    }
    v2_prefix: Dict[str, int] = {"exact": 0, "any": 0, "reverse": 0, "other": 0}
    for trial in range(n_trials):
        data = simulate_coupled(process, shared["n_samples"], coupling, lag, rng)
        common: Dict[str, Any] = dict(
            max_lag=shared["max_lag"],
            n_surrogates=shared["n_surrogates"],
            random_state=seed + 1 + trial,
            alpha=shared["alpha"],
        )
        v2 = analyzer.analyze(data, **common)
        v3 = v3_rooting(data, **common)
        granger = granger_edges(data, alpha=shared["alpha"])
        for name, edges in (
            ("v2", v2["significant_edges"]),
            ("v3", v3["significant_edges"]),
            ("granger", granger),
        ):
            forward = _edge(edges, "dim_0", "dim_1")
            counts[f"{name}_any"] += bool(forward)
            counts[f"{name}_exact"] += any(e["lag"] == lag for e in forward)
            counts[f"{name}_reverse"] += bool(_edge(edges, "dim_1", "dim_0"))
        if trial < 200:
            forward = _edge(v2["significant_edges"], "dim_0", "dim_1")
            v2_prefix["any"] += bool(forward)
            v2_prefix["exact"] += any(e["lag"] == lag for e in forward)
            v2_prefix["reverse"] += bool(_edge(v2["significant_edges"], "dim_1", "dim_0"))
            v2_prefix["other"] += len(v2["significant_edges"]) > len(forward)
    return {
        "process": dict(process),
        "coupling": coupling,
        "lag": lag,
        "seed": seed,
        **{key: binomial_rate(value, n_trials) for key, value in counts.items()},
        "v2_first_200_counts": v2_prefix,
    }


def long_run_row(phi: float, seed: int, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Raw and prewhitened reverse correlation corr(y_{t-1}, x_t) on a long sample."""
    rng = np.random.default_rng(seed)
    data = simulate_directed_pair(
        int(cfg["n_samples"]), float(cfg["coupling"]), int(cfg["lag"]), phi, rng
    )
    raw = float(np.corrcoef(data[:-1, 1], data[1:, 0])[0, 1])
    innovations, orders = prewhiten(data[:, :2], 10)
    white = float(np.corrcoef(innovations[:-1, 1], innovations[1:, 0])[0, 1])
    return {
        "phi": phi,
        "seed": seed,
        "reverse_corr_raw": raw,
        "reverse_corr_prewhitened": white,
        "ar_orders": orders,
    }


RESONANCE_PROCEDURES = ("depth_fixed_cut", "dstar_ar1", "dstar_isotonic", "lambda")


def resonance_row(
    family: Mapping[str, Any], seed: int, n_trials: int, cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """Rejection rates of the four resonance procedures on one family.

    One generator per family is consumed in a fixed order within each trial:
    the series, then the AR(1) null draws, the isotonic null draws, and the
    Lambda surrogates.
    """
    rng = np.random.default_rng(seed)
    n, window, draws = int(cfg["n_samples"]), int(cfg["depth_window"]), int(cfg["n_null_draws"])
    alpha = float(cfg["alpha"])
    line_bin = int(round(float(cfg["line_frequency"]) * n))
    rejections = {name: 0 for name in RESONANCE_PROCEDURES}
    lambda_bins: List[int] = []
    for _ in range(n_trials):
        x = simulate_resonance_series(family, n, float(cfg["line_frequency"]), rng)
        ar1 = dstar_ar1(x, window, draws, rng)
        iso = dstar_isotonic(x, window, draws, rng)
        lam = lambda_test(x, draws, rng)
        rejections["depth_fixed_cut"] += ar1["depth"] > 0.227
        rejections["dstar_ar1"] += ar1["p_value"] <= alpha
        rejections["dstar_isotonic"] += iso["p_value"] <= alpha
        if lam["p_value"] <= alpha:
            rejections["lambda"] += 1
            lambda_bins.append(lam["argmax_bin"])
    near = sum(abs(k - line_bin) <= 1 for k in lambda_bins)
    return {
        "family": dict(family),
        "seed": seed,
        **{name: binomial_rate(count, n_trials) for name, count in rejections.items()},
        "lambda_rejection_bins": lambda_bins,
        "lambda_within_one_bin_of_line": (near / len(lambda_bins)) if lambda_bins else None,
    }


def qbo_checks(cfg: Mapping[str, Any], depth_window: int = 256) -> Dict[str, Any]:
    """Exploratory known-answer checks on the vendored NOAA QBO indices."""
    from .external_benchmark.qbo import load_qbo_series

    series = load_qbo_series()
    values = np.asarray(series["values"], dtype=float)
    levels = list(series["levels"])
    rooting_cfg, spectral_cfg = cfg["rooting"], cfg["spectral"]
    common: Dict[str, Any] = dict(
        max_lag=int(rooting_cfg["max_lag"]),
        n_surrogates=int(rooting_cfg["n_surrogates"]),
        random_state=int(rooting_cfg["seed"]),
        alpha=0.05,
    )
    v2 = RootingAnalyzer().analyze(values, **common)
    v3 = v3_rooting(values, max_ar_order=int(rooting_cfg["max_ar_order"]), **common)

    def edge_summary(result: Mapping[str, Any], i: int, j: int) -> Dict[str, Any]:
        return {
            "lag": int(result["effective_lag"][i, j]),
            "score": float(result["score_matrix"][i, j]),
            "p_raw": float(result["p_values"][i, j]),
            "p_adj": float(result["adjusted_p_values"][i, j]),
            "passes_candidate_gate": bool(result["score_matrix"][i, j] > result["edge_threshold"]),
            "significant": bool(_edge(result["significant_edges"], f"dim_{i}", f"dim_{j}")),
        }

    rooting = {
        "v2_30_to_50": edge_summary(v2, 0, 1),
        "v2_50_to_30": edge_summary(v2, 1, 0),
        "v3_30_to_50": edge_summary(v3, 0, 1),
        "v3_50_to_30": edge_summary(v3, 1, 0),
        "v3_ar_orders": dict(zip(levels, v3["ar_orders"])),
    }

    rng = np.random.default_rng(int(spectral_cfg["seed"]))
    draws = int(spectral_cfg["n_surrogates"])
    spectral: Dict[str, Any] = {}
    n = values.shape[0]
    for j, level in enumerate(levels):
        x = values[:, j]
        ordinates = periodogram(x)
        band = slice(17, 24)  # 1-based bins 18..24
        ar1 = dstar_ar1(x, depth_window, draws, rng)
        iso = dstar_isotonic(x, depth_window, draws, rng)
        lam = lambda_test(x, draws, rng)
        ratio = lam["periodogram"] / lam["s_hat"]
        band_k = int(np.argmax(ratio[band])) + 18
        spectral[level] = {
            "lag1_autocorrelation": lag1_autocorrelation(x),
            "band_18_24_variance_share": float(ordinates[band].sum() / ordinates.sum()),
            "depth": ar1["depth"],
            "dstar_ar1_p": ar1["p_value"],
            "dstar_isotonic_p": iso["p_value"],
            "lambda": lam["lambda"],
            "lambda_p": lam["p_value"],
            "lambda_argmax_bin": lam["argmax_bin"],
            "lambda_argmax_period_months": n / lam["argmax_bin"],
            "qbo_band_max_ratio": float(ratio[band_k - 1]),
            "qbo_band_max_bin": band_k,
            "qbo_band_max_period_months": n / band_k,
        }
    return {
        "n_months": n,
        "first_month": series["months"][0],
        "last_month": series["months"][-1],
        "rooting": rooting,
        "spectral": spectral,
    }


# ---------------------------------------------------------------------------
# Task list, comparison and findings
# ---------------------------------------------------------------------------


def _key(spec: Mapping[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "ar1":
        return f"ar1_{spec['phi']}"
    if kind == "ar2_oscillator":
        return f"ar2_{spec['r']}"
    if kind == "ar1_plus_white":
        return f"ar1_{spec['phi']}_plus_white"
    if kind == "line_in_white":
        return f"line_white_a{spec['amplitude']}"
    if kind == "line_in_ar1":
        return f"line_ar1_{spec['phi']}_a{spec['amplitude']}"
    raise ValueError(kind)


def build_tasks(protocol: Mapping[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Every independent unit of work as (section, key, kwargs)."""
    shared = dict(protocol["shared"])
    rs = protocol["rooting_study"]
    seed = int(rs["seed"])
    shared["n_channels"] = int(rs["size"]["n_channels"])
    tasks: List[Tuple[str, str, Dict[str, Any]]] = []
    ar1_index = 0
    for process in rs["size"]["processes"]:
        if process["kind"] == "ar1":
            ar1_index += 1
            row_seed = seed + 10_000 * ar1_index
        else:
            row_seed = seed + 50_000
        tasks.append(
            (
                "size",
                _key(process),
                dict(
                    process=process,
                    seed=row_seed,
                    n_trials=int(rs["size"]["n_trials"]),
                    shared=shared,
                ),
            )
        )
    mem = rs["memory"]
    for k, process in enumerate(mem["processes"]):
        tasks.append(
            (
                "memory",
                _key(process),
                dict(
                    process=process,
                    seed=seed + 300_000 + 1_000 * k,
                    n_trials=int(mem["n_trials"]),
                    coupling=float(mem["coupling"]),
                    lag=int(mem["lag"]),
                    shared=shared,
                ),
            )
        )
    pc = rs["power_curve"]
    for k, coupling in enumerate(pc["couplings"]):
        tasks.append(
            (
                "power_curve",
                f"c{coupling}",
                dict(
                    process={"kind": "ar1", "phi": float(pc["phi"])},
                    seed=seed + 400_000 + 1_000 * k,
                    n_trials=int(pc["n_trials"]),
                    coupling=float(coupling),
                    lag=int(pc["lag"]),
                    shared=shared,
                ),
            )
        )
    lr = rs["long_run_check"]
    for k, phi in enumerate(lr["phis"]):
        tasks.append(
            ("long_run", str(phi), dict(phi=float(phi), seed=seed + 600_000 + 1_000 * k, cfg=lr))
        )
    res = protocol["resonance_study"]
    families = [(f, int(res["nulls"]["n_trials"])) for f in res["nulls"]["families"]]
    families += [(f, int(res["alternatives"]["n_trials"])) for f in res["alternatives"]["families"]]
    for j, (family, n_trials) in enumerate(families):
        tasks.append(
            (
                "resonance",
                _key(family),
                dict(family=family, seed=int(res["seed"]) + 1_000 * j, n_trials=n_trials, cfg=res),
            )
        )
    tasks.append(
        ("qbo", "qbo", dict(cfg=protocol["qbo_checks"], depth_window=int(res["depth_window"])))
    )
    return tasks


def run_task(task: Tuple[str, str, Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    section, key, kwargs = task
    if section == "size":
        return section, key, rooting_size_row(**kwargs)
    if section in ("memory", "power_curve"):
        return section, key, rooting_power_row(**kwargs)
    if section == "long_run":
        return section, key, long_run_row(**kwargs)
    if section == "resonance":
        return section, key, resonance_row(**kwargs)
    if section == "qbo":
        return section, key, qbo_checks(**kwargs)
    raise ValueError(section)


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return [float(v) for v in adjusted]


def _cell(label: str, paper_rate: float, n_paper: int, rate: Mapping[str, Any]) -> Dict[str, Any]:
    paper_count = int(round(paper_rate * n_paper))
    table = [[paper_count, n_paper - paper_count], [rate["count"], rate["trials"] - rate["count"]]]
    p = float(stats.fisher_exact(table)[1])
    return {
        "cell": label,
        "paper_rate": paper_rate,
        "paper_count": paper_count,
        "paper_trials": n_paper,
        "reconstructed_rate": rate["rate"],
        "reconstructed_count": rate["count"],
        "reconstructed_trials": rate["trials"],
        "difference": rate["rate"] - paper_rate,
        "fisher_p": p,
    }


def compare(results: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the registered comparison rules."""
    pv = protocol["paper_values"]
    cells: List[Dict[str, Any]] = []
    t2 = pv["table2_size"]
    for row, values in t2["rows"].items():
        for proc, value in values.items():
            cells.append(_cell(f"table2/{row}/{proc}", value, t2["n"], results["size"][row][proc]))
    t3 = pv["table3_memory"]
    names = {
        "v2_exact": "v2_exact",
        "v2_rev": "v2_reverse",
        "v3_exact": "v3_exact",
        "v3_rev": "v3_reverse",
        "granger_any": "granger_any",
        "granger_rev": "granger_reverse",
    }
    for row, values in t3["rows"].items():
        for col, value in values.items():
            cells.append(
                _cell(f"table3/{row}/{col}", value, t3["n"], results["memory"][row][names[col]])
            )
    cells.append(
        _cell(
            "table3_text/ar2_0.95/v2_any",
            t3["text"]["ar2_0.95_v2_any_lag"],
            t3["n"],
            results["memory"]["ar2_0.95"]["v2_any"],
        )
    )
    pc = pv["power_curve_text"]
    pc_names = {
        "v3_any": "v3_any",
        "granger_any": "granger_any",
        "v2_any": "v2_any",
        "v2_rev": "v2_reverse",
        "v3_rev": "v3_reverse",
    }
    for c, values in pc["values"].items():
        for col, value in values.items():
            cells.append(
                _cell(f"power/{c}/{col}", value, pc["n"], results["power_curve"][c][pc_names[col]])
            )
    t4 = pv["table4_resonance"]
    null_rows = {"ar1_0.0", "ar1_0.5", "ar1_0.9", "ar1_0.97", "ar1_0.9_plus_white"}
    for row, values in t4["rows"].items():
        n_paper = t4["n_null"] if row in null_rows else t4["n_alt"]
        for col, value in zip(t4["columns"], values):
            cells.append(
                _cell(f"table4/{row}/{col}", value, n_paper, results["resonance"][row][col])
            )
    adjusted = holm([c["fisher_p"] for c in cells])
    for c, p_adj in zip(cells, adjusted):
        c["holm_p"] = p_adj
        c["verdict"] = "consistent" if p_adj >= 0.05 else "differs"

    long_run = []
    for phi, paper in pv["table1"]["reverse_corr_simulated"].items():
        got = results["long_run"][phi]["reverse_corr_raw"]
        long_run.append(
            {
                "quantity": f"raw reverse corr phi={phi}",
                "paper": paper,
                "reconstructed": got,
                "verdict": "consistent" if abs(got - paper) <= 0.01 else "differs",
            }
        )
    for phi, paper in pv["table1"]["prewhitened_reverse_corr_simulated"].items():
        got = results["long_run"][phi]["reverse_corr_prewhitened"]
        long_run.append(
            {
                "quantity": f"prewhitened reverse corr phi={phi}",
                "paper": paper,
                "reconstructed": got,
                "verdict": "consistent" if abs(got - paper) <= 0.01 else "differs",
            }
        )

    q, qp = results["qbo"], pv["qbo"]
    qbo = []

    def add(quantity: str, paper: Any, got: Any, ok: bool) -> None:
        qbo.append(
            {
                "quantity": quantity,
                "paper": paper,
                "reconstructed": got,
                "verdict": "consistent" if ok else "differs",
            }
        )

    for level in ("u30", "u50"):
        s = q["spectral"][level]
        add(
            f"{level} lag-1 autocorrelation",
            qp["lag1_autocorrelation"][level],
            s["lag1_autocorrelation"],
            round(s["lag1_autocorrelation"], 2) == qp["lag1_autocorrelation"][level],
        )
        add(
            f"{level} share of power in bins 18-24",
            qp["bins_18_24_variance_share"],
            s["band_18_24_variance_share"],
            0.4 <= s["band_18_24_variance_share"] <= 0.6,
        )
        for null in ("ar1", "isotonic"):
            paper_p = qp["dstar_p"][null][level]
            got = s[f"dstar_{null}_p"]
            add(f"{level} D* p ({null})", paper_p, got, (got <= 0.05) == (paper_p <= 0.05))
        add(
            f"{level} Lambda p",
            qp["lambda_p"][level],
            s["lambda_p"],
            (s["lambda_p"] <= 0.05) == (qp["lambda_p"][level] <= 0.05),
        )
        add(
            f"{level} Lambda argmax period (months)",
            qp["lambda_argmax_period_months"][level],
            s["lambda_argmax_period_months"],
            abs(s["lambda_argmax_period_months"] - qp["lambda_argmax_period_months"][level]) <= 0.5,
        )
    for edge, paper in qp["rooting"].items():
        got = q["rooting"][edge]
        add(f"{edge} lag", paper["lag"], got["lag"], got["lag"] == paper["lag"])
        add(
            f"{edge} adjusted p",
            paper["p_adj"],
            got["p_adj"],
            (got["p_adj"] <= 0.05) == (paper["p_adj"] <= 0.05),
        )

    return {
        "monte_carlo_cells": cells,
        "n_cells": len(cells),
        "n_consistent": sum(c["verdict"] == "consistent" for c in cells),
        "long_run": long_run,
        "qbo": qbo,
    }


def exact_reproductions(results: Mapping[str, Any], committed: Mapping[str, Any]) -> List[Dict]:
    """Check the v2 columns against the committed calibration artifact."""
    checks = []
    size_rows = [r for r in committed["size"] if r["surrogate_method"] == "circular_shift"]
    for row in size_rows:
        key = f"ar1_{row['ar_coefficient']}"
        got = results["size"][key]["false_positive_trials"]["v2"]
        checks.append(
            {
                "row": f"size/{key}",
                "committed": row["false_positive_trials"],
                "reconstructed": got,
                "match": got == row["false_positive_trials"],
            }
        )
    for row in committed["memory"]:
        key = f"ar1_{row['ar_coefficient']}"
        got = results["memory"][key]["v2_first_200_counts"]
        want = {
            "exact": row["true_edge_at_true_lag"]["count"],
            "any": row["true_edge_at_any_lag"]["count"],
            "reverse": row["reverse_edge"]["count"],
            "other": row["any_other_edge"]["count"],
        }
        checks.append(
            {"row": f"memory/{key}", "committed": want, "reconstructed": got, "match": got == want}
        )
    return checks


def _lower(rate: Mapping[str, Any]) -> float:
    return float(rate["ci95"][0])


def findings(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Apply the registered decision rules F1 to F12."""
    mem, size, res = results["memory"], results["size"], results["resonance"]
    pcurve, q = results["power_curve"], results["qbo"]
    memory_ar1 = [k for k in mem if k.startswith("ar1_")]
    out = []

    def add(fid: str, ok: bool, evidence: Dict[str, Any]) -> None:
        out.append({"id": fid, "holds": bool(ok), "evidence": evidence})

    add(
        "F1",
        mem["ar1_0.97"]["v2_exact"]["rate"] < mem["ar1_0.5"]["v2_exact"]["rate"]
        and mem["ar1_0.9"]["v2_exact"]["rate"] < 0.6
        and all(mem[k]["v3_exact"]["rate"] >= 0.95 for k in mem),
        {k: (mem[k]["v2_exact"]["rate"], mem[k]["v3_exact"]["rate"]) for k in mem},
    )
    add(
        "F2",
        _lower(mem["ar1_0.9"]["v2_reverse"]) > 0.05
        and all(mem[k]["v3_reverse"]["rate"] <= 0.02 for k in memory_ar1),
        {k: (mem[k]["v2_reverse"]["rate"], mem[k]["v3_reverse"]["rate"]) for k in mem},
    )
    add(
        "F3",
        all(_lower(size[k][p]) <= 0.05 for k in size for p in ("v2", "v3", "granger")),
        {k: {p: size[k][p]["rate"] for p in ("v2", "v3", "granger")} for k in size},
    )
    add(
        "F4",
        all(
            abs(pcurve[c]["v3_any"]["rate"] - pcurve[c]["granger_any"]["rate"]) <= 0.05
            for c in ("c0.1", "c0.2")
        ),
        {c: (pcurve[c]["v3_any"]["rate"], pcurve[c]["granger_any"]["rate"]) for c in pcurve},
    )
    add(
        "F5",
        mem["ar2_0.95"]["v2_exact"]["rate"] < 0.5 and mem["ar2_0.95"]["v3_exact"]["rate"] >= 0.95,
        {
            "v2_exact": mem["ar2_0.95"]["v2_exact"]["rate"],
            "v3_exact": mem["ar2_0.95"]["v3_exact"]["rate"],
        },
    )
    add(
        "F6",
        res["ar1_0.9"]["depth_fixed_cut"]["rate"] >= 0.9
        and res["ar1_0.97"]["depth_fixed_cut"]["rate"] >= 0.9,
        {k: res[k]["depth_fixed_cut"]["rate"] for k in ("ar1_0.9", "ar1_0.97")},
    )
    nulls = ["ar1_0.0", "ar1_0.5", "ar1_0.9", "ar1_0.97", "ar1_0.9_plus_white"]
    add(
        "F7",
        _lower(res["ar1_0.9_plus_white"]["dstar_ar1"]) > 0.05
        and all(_lower(res[k]["dstar_isotonic"]) <= 0.05 for k in nulls)
        and all(_lower(res[k]["lambda"]) <= 0.05 for k in nulls),
        {
            k: {p: res[k][p]["rate"] for p in ("dstar_ar1", "dstar_isotonic", "lambda")}
            for k in nulls
        },
    )
    line = res["line_ar1_0.9_a0.5"]
    add(
        "F8",
        line["lambda"]["rate"] >= 0.5
        and line["dstar_ar1"]["rate"] <= 0.1
        and line["dstar_isotonic"]["rate"] <= 0.1,
        {p: line[p]["rate"] for p in RESONANCE_PROCEDURES},
    )
    loc = line["lambda_within_one_bin_of_line"]
    add("F9", loc is not None and loc >= 0.9, {"within_one_bin": loc})
    add(
        "F10",
        res["ar2_0.8"]["lambda"]["rate"] < 0.15 and res["ar2_0.9"]["lambda"]["rate"] < 0.15,
        {k: res[k]["lambda"]["rate"] for k in ("ar2_0.8", "ar2_0.9")},
    )
    sp = q["spectral"]
    add(
        "F11",
        all(sp[lv]["lambda_p"] <= 0.05 for lv in sp)
        and all(sp[lv][f"dstar_{n}_p"] > 0.05 for lv in sp for n in ("ar1", "isotonic")),
        {lv: {k: sp[lv][k] for k in ("lambda_p", "dstar_ar1_p", "dstar_isotonic_p")} for lv in sp},
    )
    add(
        "F12",
        q["rooting"]["v2_30_to_50"]["p_adj"] <= 0.05
        and q["rooting"]["v3_30_to_50"]["p_adj"] <= 0.05,
        {k: q["rooting"][k]["p_adj"] for k in ("v2_30_to_50", "v3_30_to_50")},
    )
    return out


def assemble(
    protocol: Mapping[str, Any],
    outputs: Sequence[Tuple[str, str, Dict[str, Any]]],
    committed_calibration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect task outputs into one artifact with comparisons and findings."""
    results: Dict[str, Any] = {}
    for section, key, value in outputs:
        if section == "qbo":
            results["qbo"] = value
        else:
            results.setdefault(section, {})[key] = value
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode("utf-8")
    import scipy

    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": protocol["provenance"],
        "protocol_id": protocol["id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "results": results,
        "comparison": compare(results, protocol),
        "findings": findings(results),
    }
    if committed_calibration is not None:
        artifact["exact_reproductions"] = exact_reproductions(results, committed_calibration)
    return artifact


# ---------------------------------------------------------------------------
# Replay check

_REPLAY_IGNORED = ("environment", "runtime_seconds")


def replay_differences(
    replayed: Any, expected: Any, rel_tol: float = 1e-6, path: str = ""
) -> List[str]:
    """Differences between a replayed artifact and the committed one.

    Every count, flag, label and trial index must match exactly. Floats must
    agree to ``rel_tol`` so that last-digit differences between numerical
    libraries do not fail the check while any change of result does. The
    environment and runtime fields are ignored.
    """
    problems: List[str] = []
    where = path or "artifact"
    if isinstance(expected, dict):
        if not isinstance(replayed, dict):
            return [f"{where}: expected an object"]
        keys = set(expected) | set(replayed)
        for key in sorted(keys):
            if not path and key in _REPLAY_IGNORED:
                continue
            child = f"{path}/{key}" if path else str(key)
            if key not in replayed or key not in expected:
                problems.append(f"{child}: present in only one artifact")
                continue
            problems.extend(replay_differences(replayed[key], expected[key], rel_tol, child))
        return problems
    if isinstance(expected, list):
        if not isinstance(replayed, list) or len(replayed) != len(expected):
            return [f"{where}: lists differ in length"]
        for i, (a, b) in enumerate(zip(replayed, expected)):
            problems.extend(replay_differences(a, b, rel_tol, f"{where}[{i}]"))
        return problems
    if isinstance(expected, float) or isinstance(replayed, float):
        if isinstance(expected, bool) or isinstance(replayed, bool):
            return [] if replayed == expected else [f"{where}: {replayed!r} != {expected!r}"]
        if not isinstance(replayed, (int, float)) or not isinstance(expected, (int, float)):
            return [f"{where}: {replayed!r} != {expected!r}"]
        if math.isnan(expected) and math.isnan(replayed):
            return []
        if math.isclose(replayed, expected, rel_tol=rel_tol, abs_tol=1e-12):
            return []
        return [f"{where}: {replayed!r} != {expected!r}"]
    return [] if replayed == expected else [f"{where}: {replayed!r} != {expected!r}"]
