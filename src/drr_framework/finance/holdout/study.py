"""Selection, sealed evaluation, evidence export and numerical replay verification."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_features, digest, load_bars
from .models import configurations, walk_forward
from .statistics import (
    block_inference,
    break_even_cost,
    performance,
    portfolio,
    prediction_metrics,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_hashes(root: Path) -> dict:
    paths = sorted((root / "src/drr_framework/finance/holdout").glob("*.py"))
    paths += [root / "src/drr_framework/modules.py", root / "src/drr_framework/_spectral.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def model_names(protocol: dict) -> list:
    models = protocol["models"]
    return models["baselines"] + models["candidates"] + models["ablations"]


def new_output(directory: Path) -> None:
    if directory.exists():
        raise ValueError("Refusing to overwrite an experiment run; use a new output directory")
    directory.mkdir(parents=True)


def select(root: Path, study: Path, output: Path) -> dict:
    new_output(output)
    protocol = json.loads((study / "protocol.json").read_text())
    bars = load_bars(study / "data", protocol)
    timing = protocol["timing"]
    cutoff = pd.Timestamp(timing["test_start"]) - pd.Timedelta(
        days=timing["entry_lag_days"] + timing["label_gap_days"]
    )
    # Test prices never enter feature building, fitting, calibration or selection.
    bars = bars.loc[bars.date < cutoff].copy()
    features, targets = build_features(bars, protocol)
    trials, chosen, mse = [], {}, {}
    for model in model_names(protocol):
        best = None
        for number, params in enumerate(configurations(model, protocol)):
            predictions = walk_forward(
                features,
                targets,
                protocol,
                model,
                params,
                timing["validation_start"],
                timing["validation_end"],
            )
            metrics = prediction_metrics(predictions)
            trial = {"model": model, "trial": number, "params": params, **metrics}
            trials.append(trial)
            write_json(output / "trials.json", trials)
            if best is None or metrics["mse"] < best[0]:
                best = (metrics["mse"], params)
        if best is None:
            raise ValueError(f"No completed validation trials for {model}")
        mse[model], chosen[model] = best
    primary = min(protocol["models"]["candidates"], key=lambda m: mse[m])
    comparator = min(protocol["models"]["baselines"], key=lambda m: mse[m])
    development_cutoff = pd.Timestamp(timing["validation_start"]) - pd.Timedelta(
        days=timing["entry_lag_days"] + timing["label_gap_days"]
    )
    vol = features[[f"{s}_vol20" for s in protocol["data"]["symbols"]]].mean(axis=1)
    lock = {
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": digest(study / "protocol.json"),
        "data_sha256": digest(study / "data/bars.csv"),
        "source_manifest_sha256": digest(study / "data/manifest.json"),
        "source_sha256": source_hashes(root),
        "trials_sha256": digest(output / "trials.json"),
        "selection_label_cutoff_exclusive": str(cutoff.date()),
        "last_validation_entry": str(
            (targets.index[-1] + pd.Timedelta(days=timing["entry_lag_days"])).date()
        ),
        "selected_parameters": chosen,
        "validation_mse": mse,
        "primary_candidate": primary,
        "economic_comparator": comparator,
        "high_volatility_threshold": float(vol.loc[vol.index < development_cutoff].quantile(0.75)),
        "regime_threshold_feature_cutoff_exclusive": str(development_cutoff.date()),
        "expected_test_days": len(pd.date_range(timing["test_start"], timing["test_end"])),
    }
    write_json(output / "selection.json", lock)
    (output / "selection.sha256").write_text(digest(output / "selection.json") + "\n")
    return lock


def verify_lock(root, study, selection):
    if digest(selection / "selection.json") != (selection / "selection.sha256").read_text().strip():
        raise ValueError("Selection checksum mismatch")
    lock = json.loads((selection / "selection.json").read_text())
    expected = {
        "protocol_sha256": digest(study / "protocol.json"),
        "data_sha256": digest(study / "data/bars.csv"),
        "source_manifest_sha256": digest(study / "data/manifest.json"),
        "source_sha256": source_hashes(root),
        "trials_sha256": digest(selection / "trials.json"),
    }
    for key, value in expected.items():
        if lock[key] != value:
            raise ValueError(
                f"Selection lock mismatch: {key}; changes require a disclosed new study"
            )
    return lock


def evaluate(root: Path, study: Path, selection: Path, output: Path) -> dict:
    lock = verify_lock(root, study, selection)
    new_output(output)
    protocol = json.loads((study / "protocol.json").read_text())
    features, targets = build_features(load_bars(study / "data", protocol), protocol)
    timing = protocol["timing"]
    frames = []
    for model in model_names(protocol):
        frames.append(
            walk_forward(
                features,
                targets,
                protocol,
                model,
                lock["selected_parameters"][model],
                timing["test_start"],
                timing["test_end"],
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    symbols = protocol["data"]["symbols"]
    vol = features[[f"{s}_vol20" for s in symbols]].mean(axis=1)
    trend = features[[f"{s}_mom60" for s in symbols]].mean(axis=1)
    regimes = pd.Series(
        np.char.add(
            np.where(vol > lock["high_volatility_threshold"], "high_vol", "low_vol"),
            np.where(trend >= 0, "_up", "_down"),
        ),
        index=features.index,
    )
    predictions["regime"] = predictions.feature_date.map(regimes)
    predictions["year"] = predictions.entry_date.dt.year
    predictions = predictions.sort_values(["model", "entry_date", "symbol"]).reset_index(drop=True)
    expected_rows = lock["expected_test_days"] * len(symbols) * len(model_names(protocol))
    if len(predictions) != expected_rows or predictions.isna().any().any():
        raise ValueError("Incomplete evaluation panel")
    summary, paths = summarize(predictions, protocol, lock)
    write_json(output / "summary.json", summary)
    for name, frame in [("predictions", predictions), ("daily_portfolios", paths)]:
        payload = frame.to_csv(index=False, float_format="%.17g", date_format="%Y-%m-%d").encode()
        (output / f"{name}.csv.gz").write_bytes(gzip.compress(payload, mtime=0))
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "joblib",
                "threadpoolctl",
                "networkx",
            ]
        },
    }
    write_json(output / "environment.json", environment)
    (output / "report.md").write_text(render_report(summary, lock))
    provenance = {
        "protocol_sha256": lock["protocol_sha256"],
        "selection_sha256": digest(selection / "selection.json"),
        "data_sha256": lock["data_sha256"],
        "source_sha256": lock["source_sha256"],
        "artifacts": {
            path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write_json(output / "provenance.json", provenance)
    return summary


def summarize(predictions, protocol, lock):
    symbols = protocol["data"]["symbols"]
    names = model_names(protocol)
    losses, net, weights, metrics, paths = {}, {}, {}, {}, []
    dates = None
    returns = None
    by_stratum = []
    for model in names:
        rows = predictions.loc[predictions.model == model]
        metrics[model] = prediction_metrics(rows)
        actual = rows.pivot(index="entry_date", columns="symbol", values="target")[symbols]
        forecast = rows.pivot(index="entry_date", columns="symbol", values="prediction")[symbols]
        if dates is not None and not actual.index.equals(dates):
            raise ValueError("Models evaluated on different dates")
        dates, returns = actual.index, actual.to_numpy()
        losses[model] = np.mean((forecast.to_numpy() - returns) ** 2, axis=1)
        weights[model] = (forecast.to_numpy() > protocol["execution"]["threshold"]).astype(
            float
        ) / len(symbols)
        for grouping in ["symbol", "year", "regime"]:
            keys = [grouping] if grouping == "symbol" else [grouping, "symbol"]
            for key, subset in rows.groupby(keys):
                key_parts = key if isinstance(key, tuple) else (key,)
                by_stratum.append(
                    {
                        "model": model,
                        "grouping": grouping,
                        "stratum": "/".join(map(str, key_parts)),
                        **prediction_metrics(subset),
                    }
                )
    weights["cash"] = np.zeros_like(returns)
    weights["equal_weight_long"] = np.full_like(returns, 1 / len(symbols))
    economic = []
    economic_strata = []
    day_regimes = (
        predictions.loc[predictions.model == names[0]]
        .groupby("entry_date")
        .regime.first()
        .loc[dates]
    )
    for model, exposure in weights.items():
        for cost in protocol["execution"]["cost_sensitivity_bps"]:
            path = portfolio(returns, exposure, cost)
            record = {"model": model, "one_way_bps": cost, **performance(path, exposure)}
            if cost == protocol["execution"]["base_one_way_bps"]:
                net[model] = path["net"]
                record["break_even_one_way_bps"] = break_even_cost(returns, exposure)
                for group, series in [
                    ("year", pd.Series(dates.year, index=dates)),
                    ("regime", day_regimes),
                ]:
                    for value in sorted(series.unique()):
                        mask = (series == value).to_numpy()
                        economic_strata.append(
                            {
                                "model": model,
                                "grouping": group,
                                "stratum": str(value),
                                **performance(
                                    {k: v[mask] for k, v in path.items()}, exposure[mask]
                                ),
                            }
                        )
            economic.append(record)
            paths.append(
                pd.DataFrame(
                    {
                        "entry_date": dates,
                        "model": model,
                        "one_way_bps": cost,
                        "net_return": path["net"],
                        "gross_return": path["gross"],
                        "turnover": path["turnover"],
                        "cost": path["cost"],
                        "exposure": exposure.sum(axis=1),
                    }
                )
            )
    test_names, differences = [], []
    for candidate in protocol["models"]["candidates"]:
        for baseline in protocol["models"]["baselines"]:
            test_names.append(f"mse:{candidate}>{baseline}")
            differences.append(losses[baseline] - losses[candidate])
        for comparator in ["cash", lock["economic_comparator"]]:
            test_names.append(f"net:{candidate}>{comparator}")
            differences.append(net[candidate] - net[comparator])
    if len(test_names) != 14 or len(set(test_names)) != 14:
        raise ValueError("The registered 14-test family is incomplete")
    inference = []
    for block in protocol["inference"]["block_days"]:
        inference.extend(
            block_inference(
                np.column_stack(differences),
                test_names,
                block,
                protocol["inference"]["bootstrap_replicates"],
                protocol["inference"]["seed"],
            )
        )
    primary = lock["primary_candidate"]
    relative = {
        baseline: 1 - metrics[primary]["mse"] / metrics[baseline]["mse"]
        for baseline in protocol["models"]["baselines"]
    }
    year_wins = sum(
        losses[primary][dates.year == year].mean()
        < losses[lock["economic_comparator"]][dates.year == year].mean()
        for year in sorted(set(dates.year))
    )
    gates = {
        "at_least_one_percent_mse_improvement_against_every_baseline": all(
            value >= protocol["inference"]["minimum_relative_mse_improvement"]
            for value in relative.values()
        ),
        "all_primary_tests_holm_significant_at_every_block_length": all(
            test["p_holm"] <= protocol["inference"]["alpha"]
            for test in inference
            if f":{primary}>" in test["test"]
        ),
        "positive_cumulative_return_at_stress_cost": next(
            item["total_return"] > 0
            for item in economic
            if item["model"] == primary and item["one_way_bps"] == 30
        ),
        "beats_selected_baseline_in_at_least_two_test_years": bool(year_wins >= 2),
        "at_least_85_percent_coverage_on_each_asset": all(
            item["coverage"] >= 0.85
            for item in by_stratum
            if item["model"] == primary and item["grouping"] == "symbol"
        ),
    }
    return {
        "experiment_id": protocol["experiment_id"],
        "verdict": "SUPPORTED" if all(gates.values()) else "NOT SUPPORTED",
        "primary_candidate": primary,
        "economic_comparator": lock["economic_comparator"],
        "test_first_entry": str(dates[0].date()),
        "test_last_entry": str(dates[-1].date()),
        "test_days": len(dates),
        "gates": gates,
        "relative_mse_improvements": relative,
        "year_wins": int(year_wins),
        "forecast_metrics": metrics,
        "forecast_strata": by_stratum,
        "economic_metrics": economic,
        "economic_strata": economic_strata,
        "tests": inference,
        "limitations": [
            protocol["publication"]["prospective_limit"],
            protocol["data"]["vintage_limit"],
            protocol["execution"]["limits"],
            protocol["inference"]["assumptions"],
            protocol["uncertainty"]["limitations"],
            "Two correlated crypto assets on one venue do not establish generalization to equities, futures, other venues or future regimes.",
            "A negative study does not prove no predictive information exists; its power and target are limited.",
            "The DRR lag-correlation and depth scores are descriptive features, not inferred causal edges or evidence of a new financial mechanism.",
        ],
    }, pd.concat(paths, ignore_index=True)


def render_report(summary, lock):
    s = summary
    lines = [
        "# Financial holdout: delayed daily return prediction",
        "",
        f"**Registered conclusion: {s['verdict']}.**",
        "",
        f"Validation selected **{s['primary_candidate']}** as the primary candidate and "
        f"**{s['economic_comparator']}** as the economic comparator. "
        f"The test covers {s['test_days']} entry dates, {s['test_first_entry']} through {s['test_last_entry']}, "
        "on BTCUSDT and ETHUSDT. Positive returns alone are not evidence of incremental forecast value.",
        "",
        "## Decision gates",
        "",
        "| Fixed criterion | Passed |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key.replace('_', ' ')} | {value} |" for key, value in s["gates"].items())
    lines += [
        "",
        "## Every forecasting result",
        "",
        "| Model | MSE | MAE | 90% interval coverage | Mean width | Interval score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, item in s["forecast_metrics"].items():
        lines.append(
            f"| {model} | {item['mse']:.8f} | {item['mae']:.6f} | {item['coverage']:.2%} | {item['mean_interval_width']:.4f} | {item['mean_interval_score']:.4f} |"
        )
    lines += [
        "",
        "## All cost scenarios",
        "",
        "Long/cash positions, fixed 30 bps forecast threshold, full entry/exit costs and drift-aware rebalancing. "
        "Costs are hypothetical one-way scenarios; 15 bps consists of 10 bps fees plus 5 bps slippage. "
        "Cash earns zero in the quote currency. Sharpe uses 365 days and is descriptive.",
        "",
        "| Model | Cost bps | Total net return | CAGR | Sharpe | Max drawdown | Exposure | Turnover |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in s["economic_metrics"]:
        sharpe = "n/a" if item["sharpe"] is None else f"{item['sharpe']:.3f}"
        lines.append(
            f"| {item['model']} | {item['one_way_bps']} | {item['total_return']:.2%} | {item['cagr']:.2%} | {sharpe} | {item['max_drawdown']:.2%} | {item['mean_exposure']:.2%} | {item['total_turnover']:.2f} |"
        )
    lines += [
        "",
        "## Complete multiplicity family",
        "",
        "Positive differences favor the candidate. Loss differences are in squared-return units; "
        "net differences are daily portfolio returns. Every row is retained. The intervals are "
        "pointwise 95% bootstrap intervals; only the Holm-adjusted p-values control the registered "
        "14-test family (approximately, subject to bootstrap assumptions). "
        "All three block lengths must pass; none is selected after seeing the results.",
        "",
        "| Comparison | Block days | Mean difference | CI low | CI high | Raw p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in s["tests"]:
        lines.append(
            f"| {item['test']} | {item['block_days']} | {item['mean_improvement']:.8f} | {item['ci95_low']:.8f} | {item['ci95_high']:.8f} | {item['p_one_sided']:.4f} | {item['p_holm']:.4f} |"
        )
    lines += [
        "",
        "## Regimes and predictive uncertainty",
        "",
        f"The high-volatility cutoff is {lock['high_volatility_threshold']:.6f} daily return standard deviation, "
        "estimated only on development data. Trends use trailing 60-day momentum. "
        "No regime is defined from future test outcomes. The following complete per-asset table "
        "reports every model, calendar year and volatility/trend stratum. These are descriptive comparisons.",
        "",
        "| Model | Group | Stratum | N | MSE | Coverage | Width | Interval score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in s["forecast_strata"]:
        lines.append(
            f"| {item['model']} | {item['grouping']} | {item['stratum']} | {item['rows']} | {item['mse']:.8f} | {item['coverage']:.2%} | {item['mean_interval_width']:.4f} | {item['mean_interval_score']:.4f} |"
        )
    lines += [
        "",
        "## Economic regime breakdown",
        "",
        "Base-cost returns are partitioned from the continuous portfolio path, retaining actual "
        "boundary turnover. Noncontiguous regime growth is a conditional descriptive calculation, "
        "not a separately tradable strategy. Detailed costs, annualized statistics and break-even "
        "costs are in summary.json.",
        "",
        "| Model | Group | Stratum | Days | Mean daily net | Conditional growth | Exposure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in s["economic_strata"]:
        lines.append(
            f"| {item['model']} | {item['grouping']} | {item['stratum']} | {item['days']} | {item['mean_daily_net']:.6f} | {item['total_return']:.2%} | {item['mean_exposure']:.2%} |"
        )
    lines += ["", "## Limits and reproduction", ""]
    lines.extend("- " + item for item in s["limitations"])
    lines += [
        "",
        "Every validation trial is in `experiments/financial_holdout/selection/trials.json`. "
        "The separate selection lock pins the protocol, source files and data. "
        "`predictions.csv.gz` contains targets, forecasts, bounds and fit/calibration timestamps; "
        "`daily_portfolios.csv.gz` contains every daily cost scenario. "
        "`provenance.json` pins the published artifacts and `environment.json` records dependencies.",
        "",
        "Data attribution: Binance Vision, normalized from public daily spot klines. "
        "Data and derived results: CC BY-NC-SA 4.0 and Binance Vision Dataset Terms. "
        "No affiliation or endorsement is implied.",
        "",
    ]
    return "\n".join(lines)


def verify_replay(actual, expected, path="summary"):
    """Tolerance covers platform floating arithmetic, never categorical verdicts."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Replay keys differ: {path}")
        for key in expected:
            verify_replay(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Replay lengths differ: {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            verify_replay(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Replay numeric mismatch: {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Replay mismatch: {path}")
