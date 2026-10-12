"""Which null-model assumptions keep lag-search false alarms controlled?

The rooting test reports a directed edge when the largest lagged correlation
between two series exceeds what a surrogate null produces. Each surrogate
method rests on an assumption about the series it is applied to:

* ``permutation`` assumes the samples are exchangeable, which is serial
  independence.
* ``circular_shift`` assumes stationarity. It rolls each series by its own
  random offset, so every series keeps its joint distribution except at one
  seam while the alignment between series is destroyed.
* ``phase_randomization`` assumes a stationary linear Gaussian process. It
  keeps each series' periodogram exactly, hence its sample autocorrelation at
  every lag, and draws new phases.

This study measures the family-wise false-alarm rate of each null on
independent channels whose dependence violates those assumptions by degrees:
fractional Gaussian noise with a Hurst exponent up to 0.95 (long memory, whose
autocorrelation is not summable), GARCH(1,1) noise (uncorrelated but dependent
and heavy-tailed), and a random walk (nonstationary). It then measures the
power to recover a known lag-2 edge when every channel carries long memory.
The uncoupled rows are the negative control for the coupled rows at the same
Hurst exponent: same autocorrelation, no relationship.

The hypotheses and their decision rules are fixed in ``DEFAULT_CONFIG`` before
the run. The artifact records each hypothesis next to its outcome, and
``tests/test_dependent_noise_study.py`` recomputes the outcomes from the rows,
so an outcome cannot be edited without the rows disagreeing.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .calibration import binomial_rate
from .modules import SURROGATE_METHODS, RootingAnalyzer

SCHEMA_VERSION = "drr-dependent-noise-null-study-v1"

Draw = Callable[[np.random.Generator], np.ndarray]

#: What each surrogate null assumes about a series, and what it keeps and destroys.
NULL_ASSUMPTIONS: Dict[str, Dict[str, str]] = {
    "circular_shift": {
        "assumes": "stationarity",
        "keeps": "each series' joint distribution, except at one seam",
        "destroys": "the alignment between series",
    },
    "phase_randomization": {
        "assumes": "a stationary linear Gaussian process",
        "keeps": "each series' periodogram, hence its sample autocorrelation at every lag",
        "destroys": "the phases, so the alignment between series and any non-Gaussian structure",
    },
    "permutation": {
        "assumes": "serial independence",
        "keeps": "each series' marginal distribution",
        "destroys": "each series' memory and the alignment between series",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20261012,
    "alpha": 0.05,
    "n_samples": 512,
    "max_lag": 4,
    "n_surrogates": 99,
    "n_variables": 3,
    "surrogate_methods": ["circular_shift", "phase_randomization", "permutation"],
    "size": {
        "n_trials": 400,
        "processes": [
            {
                "id": "fgn_0.50",
                "family": "fractional_gaussian_noise",
                "hurst": 0.5,
                "stationary": True,
                "gaussian": True,
            },
            {
                "id": "fgn_0.70",
                "family": "fractional_gaussian_noise",
                "hurst": 0.7,
                "stationary": True,
                "gaussian": True,
            },
            {
                "id": "fgn_0.85",
                "family": "fractional_gaussian_noise",
                "hurst": 0.85,
                "stationary": True,
                "gaussian": True,
            },
            {
                "id": "fgn_0.95",
                "family": "fractional_gaussian_noise",
                "hurst": 0.95,
                "stationary": True,
                "gaussian": True,
            },
            {
                "id": "garch",
                "family": "garch",
                "arch": 0.1,
                "garch": 0.85,
                "stationary": True,
                "gaussian": False,
            },
            {"id": "random_walk", "family": "random_walk", "stationary": False, "gaussian": True},
        ],
    },
    "coupled": {
        "n_trials": 200,
        "coupling": 0.3,
        "lag": 2,
        "hurst": [0.5, 0.7, 0.85, 0.95],
        "surrogate_methods": ["circular_shift", "phase_randomization"],
    },
    "preregistration": {
        "decision_rule": (
            "A null is controlled on a process when the Wilson 95% lower bound of its "
            "family-wise error is at most alpha. It is inflated when that lower bound "
            "exceeds alpha. The rule, the processes, the seeds, and the trial counts are "
            "fixed before the run and are not changed after the rows are seen."
        ),
        "hypotheses": [
            {
                "id": "H1",
                "statement": (
                    "The circular-shift null is controlled on every stationary process, "
                    "including fractional Gaussian noise at H = 0.95 and GARCH(1,1) noise."
                ),
            },
            {
                "id": "H2",
                "statement": (
                    "The phase-randomization null is controlled on every stationary Gaussian "
                    "process, which is fractional Gaussian noise at every H."
                ),
            },
            {
                "id": "H3",
                "statement": (
                    "The permutation null is inflated on every long-memory process, which is "
                    "fractional Gaussian noise with H > 0.5."
                ),
            },
            {
                "id": "H4",
                "statement": (
                    "The circular-shift null is inflated on the random walk, because its "
                    "stationarity assumption fails there."
                ),
            },
            {
                "id": "H5",
                "statement": (
                    "The phase-randomization null is inflated on the random walk, because its "
                    "stationarity assumption fails there."
                ),
            },
            {
                "id": "H6",
                "statement": (
                    "Long memory costs power, not size: with the circular-shift null, the lag-2 "
                    "edge at coupling 0.3 is recovered less often at the largest H than at "
                    "H = 0.5."
                ),
            },
        ],
    },
}


# --------------------------------------------------------------------------- simulators


def fgn_autocovariance(lags: Union[Sequence[int], np.ndarray], hurst: float) -> np.ndarray:
    """Autocovariance of unit-variance fractional Gaussian noise at integer lags.

    ``gamma(k) = (|k + 1|^{2H} - 2|k|^{2H} + |k - 1|^{2H}) / 2``. For ``H > 0.5``
    it decays like ``k^{2H - 2}`` and its sum diverges, which is long memory.
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must lie in (0, 1)")
    k = np.abs(np.asarray(lags, dtype=float))
    h2 = 2.0 * hurst
    return 0.5 * ((k + 1.0) ** h2 - 2.0 * k**h2 + np.abs(k - 1.0) ** h2)


def davies_harte_eigenvalues(n_samples: int, hurst: float) -> np.ndarray:
    """Eigenvalues of the circulant embedding used to simulate fractional Gaussian noise.

    The ``2 n`` by ``2 n`` circulant matrix whose first row is the autocovariance
    out to lag ``n`` and back has the fGn covariance as its leading ``n`` by ``n``
    block. Its eigenvalues are the FFT of that row. They must be nonnegative for
    the embedding to be a valid covariance; the run records the minimum.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    gamma = fgn_autocovariance(np.arange(n_samples + 1), hurst)
    row = np.concatenate((gamma, gamma[-2:0:-1]))
    return np.fft.fft(row).real


def simulate_fractional_gaussian_noise(
    n_samples: int, n_variables: int, hurst: float, rng: np.random.Generator
) -> np.ndarray:
    """Exact, independent, unit-variance fractional Gaussian noise series.

    Davies–Harte circulant embedding. Each complex Gaussian draw yields two
    independent real paths, so ``ceil(n_variables / 2)`` draws are made and the
    first ``n_variables`` paths returned. ``hurst = 0.5`` is white noise.
    """
    if n_variables < 1:
        raise ValueError("n_variables must be at least 1")
    eigenvalues = davies_harte_eigenvalues(n_samples, hurst)
    if eigenvalues.min() < -1e-9 * eigenvalues.max():
        raise ValueError(f"circulant embedding is not positive for H = {hurst}")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    m = eigenvalues.shape[0]
    n_draws = (n_variables + 1) // 2
    w = rng.standard_normal((m, n_draws)) + 1j * rng.standard_normal((m, n_draws))
    paths = np.fft.fft(np.sqrt(eigenvalues)[:, None] * w, axis=0) / np.sqrt(m)
    series = np.concatenate((paths.real[:n_samples], paths.imag[:n_samples]), axis=1)
    return np.ascontiguousarray(series[:, :n_variables])


def simulate_garch(
    n_samples: int,
    n_variables: int,
    arch: float,
    garch: float,
    rng: np.random.Generator,
    burn_in: int = 500,
) -> np.ndarray:
    """Independent GARCH(1,1) series with unit unconditional variance.

    ``sigma_t^2 = omega + arch * e_{t-1}^2 + garch * sigma_{t-1}^2`` with Gaussian
    innovations and ``omega = 1 - arch - garch``. The series are uncorrelated at
    every lag, so their periodogram is flat on average, but their squares are
    autocorrelated and their tails are heavier than Gaussian.
    """
    if arch < 0.0 or garch < 0.0 or arch + garch >= 1.0:
        raise ValueError("arch and garch must be nonnegative with arch + garch < 1")
    omega = 1.0 - arch - garch
    shocks = rng.standard_normal((n_samples + burn_in, n_variables))
    series = np.empty_like(shocks)
    variance = np.ones(n_variables)
    for t in range(len(shocks)):
        if t:
            variance = omega + arch * series[t - 1] ** 2 + garch * variance
        series[t] = np.sqrt(variance) * shocks[t]
    return series[burn_in:]


def simulate_random_walk(n_samples: int, n_variables: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Gaussian random walks: the nonstationary boundary case."""
    return np.cumsum(rng.standard_normal((n_samples, n_variables)), axis=0)


def simulate_long_memory_pair(
    n_samples: int, coupling: float, lag: int, hurst: float, rng: np.random.Generator
) -> np.ndarray:
    """Three fGn channels where channel 0 drives channel 1 at ``lag``.

    ``coupling`` is the correlation between the target and the lagged source;
    all channels have unit variance and the same Hurst exponent. Channel 2 is
    an independent distractor. This is the long-memory counterpart of
    :func:`drr_framework.calibration.simulate_directed_pair`.
    """
    if not 0.0 <= coupling < 1.0:
        raise ValueError("coupling must lie in [0, 1)")
    if lag < 1:
        raise ValueError("lag must be at least 1")
    base = simulate_fractional_gaussian_noise(n_samples + lag, 3, hurst, rng)
    source = base[:, 0]
    target = coupling * np.roll(source, lag) + np.sqrt(1.0 - coupling**2) * base[:, 1]
    return np.column_stack((source, target, base[:, 2]))[lag:]


def describe_process(spec: Mapping[str, Any]) -> str:
    """Short human label for a process specification."""
    family = spec["family"]
    if family == "fractional_gaussian_noise":
        label = f"fGn H = {float(spec['hurst']):.2f}"
        return label + " (white noise)" if float(spec["hurst"]) == 0.5 else label
    if family == "garch":
        return f"GARCH(1,1) α = {float(spec['arch']):.2f}, β = {float(spec['garch']):.2f}"
    if family == "random_walk":
        return "Random walk"
    raise ValueError(f"unknown process family {family!r}")


def make_null_process(spec: Mapping[str, Any], n_samples: int, n_variables: int) -> Draw:
    """A draw function for independent channels of the specified process."""
    family = spec["family"]
    if family == "fractional_gaussian_noise":
        hurst = float(spec["hurst"])
        return lambda rng: simulate_fractional_gaussian_noise(n_samples, n_variables, hurst, rng)
    if family == "garch":
        arch, garch = float(spec["arch"]), float(spec["garch"])
        return lambda rng: simulate_garch(n_samples, n_variables, arch, garch, rng)
    if family == "random_walk":
        return lambda rng: simulate_random_walk(n_samples, n_variables, rng)
    raise ValueError(f"unknown process family {family!r}")


# --------------------------------------------------------------------------- experiments


def null_false_alarms(
    draw: Draw,
    *,
    n_trials: int,
    max_lag: int,
    n_surrogates: int,
    alpha: float,
    surrogate_method: str,
    seed: int,
) -> Dict[str, Any]:
    """Family-wise false-alarm rate of one null on independent channels from ``draw``."""
    if surrogate_method not in SURROGATE_METHODS:
        raise ValueError(f"unknown surrogate_method {surrogate_method!r}")
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    false_positive_trials: List[int] = []
    for trial in range(n_trials):
        result = analyzer.analyze(
            draw(rng),
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            random_state=seed + 1 + trial,
            alpha=alpha,
            surrogate_method=surrogate_method,
        )
        if result["significant_edges"]:
            false_positive_trials.append(trial)
    return {
        "surrogate_method": surrogate_method,
        "family_wise_error": binomial_rate(len(false_positive_trials), n_trials),
        # Trial indices let a reviewer rerun any prefix of the row and compare exactly.
        "false_positive_trials": false_positive_trials,
    }


def coupled_edge_recovery(
    draw: Draw,
    *,
    lag: int,
    n_trials: int,
    max_lag: int,
    n_surrogates: int,
    alpha: float,
    surrogate_method: str,
    seed: int,
) -> Dict[str, Any]:
    """How often the edge ``dim_0 -> dim_1`` at ``lag``, and any other edge, is reported."""
    rng = np.random.default_rng(seed)
    analyzer = RootingAnalyzer()
    exact, any_lag, other, reverse = 0, 0, 0, 0
    for trial in range(n_trials):
        result = analyzer.analyze(
            draw(rng),
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            random_state=seed + 1 + trial,
            alpha=alpha,
            surrogate_method=surrogate_method,
        )
        edges = result["significant_edges"]
        true_edges = [e for e in edges if e["source"] == "dim_0" and e["target"] == "dim_1"]
        any_lag += bool(true_edges)
        exact += any(e["lag"] == lag for e in true_edges)
        other += len(edges) > len(true_edges)
        reverse += any(e["source"] == "dim_1" and e["target"] == "dim_0" for e in edges)
    return {
        "surrogate_method": surrogate_method,
        "true_edge_at_true_lag": binomial_rate(exact, n_trials),
        "true_edge_at_any_lag": binomial_rate(any_lag, n_trials),
        "any_other_edge": binomial_rate(other, n_trials),
        "reverse_edge": binomial_rate(reverse, n_trials),
    }


# --------------------------------------------------------------------------- decisions


def is_controlled(row: Mapping[str, Any], alpha: float) -> bool:
    """The preregistered rule: the Wilson 95% lower bound is at most ``alpha``."""
    return bool(row["family_wise_error"]["ci95"][0] <= alpha)


def _rows(size_rows: Sequence[Mapping], method: str, **where: Any) -> List[Mapping]:
    out = []
    for row in size_rows:
        if row["surrogate_method"] != method:
            continue
        if all(row.get(key) == value for key, value in where.items()):
            out.append(row)
    return out


def _long_memory(size_rows: Sequence[Mapping], method: str) -> List[Mapping]:
    return [
        row
        for row in _rows(size_rows, method, family="fractional_gaussian_noise")
        if row["hurst"] > 0.5
    ]


def evaluate_hypotheses(
    size_rows: Sequence[Mapping],
    coupled_rows: Sequence[Mapping],
    cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply each preregistered rule to the rows and record the outcome beside it."""
    alpha = float(cfg["alpha"])
    circ_stationary = _rows(size_rows, "circular_shift", stationary=True)
    phase_gaussian = _rows(size_rows, "phase_randomization", stationary=True, gaussian=True)
    perm_memory = _long_memory(size_rows, "permutation")
    circ_walk = _rows(size_rows, "circular_shift", family="random_walk")
    phase_walk = _rows(size_rows, "phase_randomization", family="random_walk")
    circ_power = [r for r in coupled_rows if r["surrogate_method"] == "circular_shift"]
    hursts = [r["hurst"] for r in circ_power]

    def power_at(h: float) -> Optional[float]:
        for row in circ_power:
            if row["hurst"] == h:
                return float(row["true_edge_at_true_lag"]["rate"])
        return None

    rules = {
        "H1": (
            "every circular_shift row on a stationary process is controlled",
            bool(circ_stationary) and all(is_controlled(r, alpha) for r in circ_stationary),
            [r["process"] for r in circ_stationary],
        ),
        "H2": (
            "every phase_randomization row on a stationary Gaussian process is controlled",
            bool(phase_gaussian) and all(is_controlled(r, alpha) for r in phase_gaussian),
            [r["process"] for r in phase_gaussian],
        ),
        "H3": (
            "every permutation row on fractional Gaussian noise with H > 0.5 is inflated",
            bool(perm_memory) and all(not is_controlled(r, alpha) for r in perm_memory),
            [r["process"] for r in perm_memory],
        ),
        "H4": (
            "the circular_shift row on the random walk is inflated",
            bool(circ_walk) and all(not is_controlled(r, alpha) for r in circ_walk),
            [r["process"] for r in circ_walk],
        ),
        "H5": (
            "the phase_randomization row on the random walk is inflated",
            bool(phase_walk) and all(not is_controlled(r, alpha) for r in phase_walk),
            [r["process"] for r in phase_walk],
        ),
        "H6": (
            "circular_shift true-edge rate at the largest H is below the rate at H = 0.5",
            bool(hursts)
            and power_at(0.5) is not None
            and power_at(max(hursts)) is not None
            and power_at(max(hursts)) < power_at(0.5),  # type: ignore[operator]
            [f"coupled H = {h:.2f}" for h in sorted(hursts)],
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
                "evidence": evidence,
                "outcome": "supported" if holds else "not_supported",
            }
        )
    return outcomes


def integrity_checks(
    size_rows: Sequence[Mapping],
    coupled_rows: Sequence[Mapping],
    embedding: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Dict[str, bool]:
    """Harness checks that must hold for the artifact to be trusted at all.

    These are not hypotheses. They confirm that the white-noise anchor behaves
    like the calibration study, that a real edge is still found when memory is
    absent, that every row ran to completion, and that every fGn embedding was
    a valid covariance.
    """
    alpha = float(cfg["alpha"])
    anchor = [r for r in size_rows if r["process"] == "fgn_0.50"]
    white_power = [
        r for r in coupled_rows if r["surrogate_method"] == "circular_shift" and r["hurst"] == 0.5
    ]
    return {
        "white_noise_anchor_not_inflated_for_any_null": bool(anchor)
        and all(is_controlled(r, alpha) for r in anchor),
        "coupled_edge_recovered_on_white_noise": bool(white_power)
        and all(r["true_edge_at_true_lag"]["rate"] >= 0.8 for r in white_power),
        "every_size_row_complete": all(
            r["family_wise_error"]["trials"] == int(cfg["size"]["n_trials"]) for r in size_rows
        ),
        "every_coupled_row_complete": all(
            r["true_edge_at_true_lag"]["trials"] == int(cfg["coupled"]["n_trials"])
            for r in coupled_rows
        ),
        "circulant_embedding_valid_for_every_hurst": all(
            value["min_eigenvalue"] >= -1e-9 * value["max_eigenvalue"]
            for value in embedding.values()
        ),
    }


# --------------------------------------------------------------------------- the study


def run_dependent_noise_study(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run the size and power rows and return a JSON-ready artifact."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    seed = int(cfg["seed"])
    n_samples = int(cfg["n_samples"])
    n_variables = int(cfg["n_variables"])
    shared: Dict[str, Any] = {
        "max_lag": int(cfg["max_lag"]),
        "n_surrogates": int(cfg["n_surrogates"]),
        "alpha": float(cfg["alpha"]),
    }

    embedding: Dict[str, Dict[str, float]] = {}
    hursts = sorted(
        {float(p["hurst"]) for p in cfg["size"]["processes"] if "hurst" in p}
        | {float(h) for h in cfg["coupled"]["hurst"]}
    )
    for hurst in hursts:
        for length in (n_samples, n_samples + int(cfg["coupled"]["lag"])):
            eig = davies_harte_eigenvalues(length, hurst)
            key = f"H={hurst:.2f},n={length}"
            embedding[key] = {
                "hurst": hurst,
                "n_samples": float(length),
                "min_eigenvalue": float(eig.min()),
                "max_eigenvalue": float(eig.max()),
            }

    size_rows: List[Dict[str, Any]] = []
    size_cfg = cfg["size"]
    for i, spec in enumerate(size_cfg["processes"]):
        draw = make_null_process(spec, n_samples, n_variables)
        for j, method in enumerate(cfg["surrogate_methods"]):
            row = null_false_alarms(
                draw,
                n_trials=int(size_cfg["n_trials"]),
                surrogate_method=method,
                seed=seed + 10_000 * (i + 1) + 1_000 * j,
                **shared,
            )
            size_rows.append(
                {"process": spec["id"], "label": describe_process(spec), **spec, **row}
            )

    coupled_rows: List[Dict[str, Any]] = []
    coupled_cfg = cfg["coupled"]
    coupling, lag = float(coupled_cfg["coupling"]), int(coupled_cfg["lag"])
    for k, hurst in enumerate(coupled_cfg["hurst"]):
        hurst = float(hurst)

        def draw_pair(rng: np.random.Generator, h: float = hurst) -> np.ndarray:
            return simulate_long_memory_pair(n_samples, coupling, lag, h, rng)

        for j, method in enumerate(coupled_cfg["surrogate_methods"]):
            row = coupled_edge_recovery(
                draw_pair,
                lag=lag,
                n_trials=int(coupled_cfg["n_trials"]),
                surrogate_method=method,
                seed=seed + 300_000 + 10_000 * (k + 1) + 1_000 * j,
                **shared,
            )
            coupled_rows.append({"hurst": hurst, "label": f"fGn H = {hurst:.2f}", **row})

    config_json = json.dumps(cfg, sort_keys=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "null_assumptions": NULL_ASSUMPTIONS,
        "embedding": embedding,
        "size": size_rows,
        "coupled": coupled_rows,
        "hypotheses": evaluate_hypotheses(size_rows, coupled_rows, cfg),
        "checks": integrity_checks(size_rows, coupled_rows, embedding, cfg),
    }


# --------------------------------------------------------------------------- report


def _pct(interval: Sequence[float]) -> str:
    return f"{interval[0]:.3f}–{interval[1]:.3f}"


def render_dependent_noise_report(artifact: Mapping[str, Any]) -> str:
    """Render the Markdown companion of a dependent-noise artifact."""
    cfg = artifact["config"]
    alpha = float(cfg["alpha"])
    coupled_cfg = cfg["coupled"]
    lines = [
        "# Null-model assumptions under long memory and dependent noise",
        "",
        "Generated by `python scripts/run_dependent_noise_study.py --output-dir results/expected`",
        f"from `{artifact['schema_version']}` (config SHA-256 `{artifact['config_sha256'][:12]}…`).",
        "Every value below is a Monte Carlo estimate on simulated channels with a known",
        "truth. The JSON beside this file is the machine-readable record, and the",
        "hypotheses and decision rule were fixed before the run.",
        "",
        f"Shared settings: {cfg['n_samples']} samples, {cfg['n_variables']} channels,"
        f" lags 1–{cfg['max_lag']}, {cfg['n_surrogates']} surrogates, alpha = {alpha},"
        " max-statistic correction.",
        "",
        "## The nulls and what they assume",
        "",
        "| Null | Assumes | Keeps | Destroys |",
        "| --- | --- | --- | --- |",
    ]
    for method in cfg["surrogate_methods"]:
        a = artifact["null_assumptions"][method]
        lines.append(f"| `{method}` | {a['assumes']} | {a['keeps']} | {a['destroys']} |")
    lines += [
        "",
        "## Size: false alarms on independent channels",
        "",
        f"{cfg['n_variables']} independent channels, {cfg['size']['n_trials']} trials per row."
        " Any significant edge is a family-wise error. *Controlled* means the Wilson 95%"
        f" lower bound is at most {alpha}; *inflated* means it is above.",
        "",
        "| Process | Null | Family-wise error | 95% interval | Verdict |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in artifact["size"]:
        fwe = row["family_wise_error"]
        verdict = "controlled" if is_controlled(row, alpha) else "inflated"
        lines.append(
            f"| {row['label']} | {row['surrogate_method']} | {fwe['rate']:.3f} |"
            f" {_pct(fwe['ci95'])} | {verdict} |"
        )
    lines += [
        "",
        "## Power: a known edge when every channel has long memory",
        "",
        f"Channel 0 drives channel 1 at lag {coupled_cfg['lag']} with coupling"
        f" {coupled_cfg['coupling']}; a third channel is an independent distractor. All three"
        f" are fractional Gaussian noise with the Hurst exponent shown. {coupled_cfg['n_trials']}"
        " trials per row. The size rows above at the same H are the negative control.",
        "",
        "| Process | Null | True edge at true lag | Any other edge | Reverse edge |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in artifact["coupled"]:
        lines.append(
            f"| {row['label']} | {row['surrogate_method']} |"
            f" {row['true_edge_at_true_lag']['rate']:.3f} | {row['any_other_edge']['rate']:.3f} |"
            f" {row['reverse_edge']['rate']:.3f} |"
        )
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
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIG",
    "NULL_ASSUMPTIONS",
    "SCHEMA_VERSION",
    "coupled_edge_recovery",
    "davies_harte_eigenvalues",
    "describe_process",
    "evaluate_hypotheses",
    "fgn_autocovariance",
    "integrity_checks",
    "is_controlled",
    "make_null_process",
    "null_false_alarms",
    "render_dependent_noise_report",
    "run_dependent_noise_study",
    "simulate_fractional_gaussian_noise",
    "simulate_garch",
    "simulate_long_memory_pair",
    "simulate_random_walk",
]
