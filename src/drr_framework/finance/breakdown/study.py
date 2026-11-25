"""Calibration locks, complete simulation ledger and external validation."""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

from ..holdout.data import digest
from ..holdout.statistics import block_inference
from .detectors import first_alarm, score_stream, simulate, threshold
from .market import load_market, market_effects, market_risk, market_scores, warning_thresholds


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, frame):
    data = frame.to_csv(index=False, float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S").encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def new_directory(path):
    if path.exists():
        raise ValueError("Use a new directory; experiment evidence cannot be overwritten")
    path.mkdir(parents=True)


def source_hashes(root):
    files = sorted((root / "src/drr_framework/finance/breakdown").glob("*.py"))
    files += [
        root / name
        for name in [
            "src/drr_framework/modules.py",
            "src/drr_framework/_spectral.py",
            "src/drr_framework/finance/holdout/statistics.py",
            "scripts/run_breakdown_study.py",
            "scripts/fetch_breakdown_data.py",
        ]
    ]
    return {str(path.relative_to(root)): digest(path) for path in files}


def input_hashes(root, study):
    return {
        "protocol": digest(study / "protocol.json"),
        "data": digest(study / "data/bars.csv.gz"),
        "manifest": digest(study / "data/manifest.json"),
        "source": source_hashes(root),
    }


def run_simulations(protocol, calibration=False, cutoffs=None):
    spec = protocol["simulation"]
    scenarios = (
        spec["nulls"] if calibration else spec["nulls"] + spec["changes"] + spec["challenges"]
    )
    n = spec["calibration_replicates"] if calibration else spec["test_replicates"]
    base = spec["calibration_seed"] if calibration else spec["test_seed"]
    episodes, traces = [], []
    for scenario in scenarios:
        for replicate in range(n):
            seed = base + replicate
            times, scores, _, _ = score_stream(simulate(protocol, seed, scenario), protocol)
            traces.append(
                pd.DataFrame(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "seed": seed,
                        "time": times,
                        **scores,
                    }
                )
            )
            for name in protocol["methods"]:
                record = {
                    "scenario": scenario,
                    "replicate": replicate,
                    "seed": seed,
                    "method": name,
                    "max_score": float(max(scores[name])),
                }
                if cutoffs is not None:
                    record.update(
                        first_alarm(
                            times, scores[name], cutoffs[name], spec["change"], spec["length"]
                        )
                    )
                episodes.append(record)
    return pd.DataFrame(episodes), pd.concat(traces, ignore_index=True)


def calibrate(root, study, output):
    new_directory(output)
    protocol = json.loads((study / "protocol.json").read_text())
    episodes, traces = run_simulations(protocol, calibration=True)
    cutoffs = {}
    for name in protocol["methods"]:
        cutoffs[name] = max(
            threshold(
                episodes.loc[
                    (episodes.method == name) & (episodes.scenario == scenario), "max_score"
                ].to_numpy(),
                protocol["simulation"]["false_alarm_budget"],
            )
            for scenario in protocol["simulation"]["nulls"]
        )
    bars = load_market(study / "data", protocol)
    records = market_scores(bars, protocol, protocol["market"]["calibration_months"])
    write_csv(output / "simulation_maxima.csv.gz", episodes)
    write_csv(output / "simulation_scores.csv.gz", traces)
    write_csv(output / "market_scores.csv.gz", records)
    lock = {
        "id": protocol["id"],
        "inputs": input_hashes(root, study),
        "simulation_thresholds": cutoffs,
        "market_thresholds": warning_thresholds(records, protocol),
        "calibration_artifacts": {p.name: digest(p) for p in sorted(output.iterdir())},
    }
    write_json(output / "lock.json", lock)
    (output / "lock.sha256").write_text(digest(output / "lock.json") + "\n")
    return lock


def verify_lock(root, study, calibration):
    if digest(calibration / "lock.json") != (calibration / "lock.sha256").read_text().strip():
        raise ValueError("Calibration lock checksum mismatch")
    lock = json.loads((calibration / "lock.json").read_text())
    if input_hashes(root, study) != lock["inputs"]:
        raise ValueError("Protocol, source or market snapshot changed after calibration")
    for name, expected in lock["calibration_artifacts"].items():
        if digest(calibration / name) != expected:
            raise ValueError("Calibration artifact changed")
    return lock


def wilson_upper(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial count")
    z = 1.6448536269514722
    rate = successes / n
    return float(
        (rate + z * z / (2 * n) + z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)))
        / (1 + z * z / n)
    )


def simulation_summary(episodes, protocol):
    spec = protocol["simulation"]
    outcomes = []
    for scenario in spec["nulls"] + spec["changes"] + spec["challenges"]:
        for name in protocol["methods"]:
            rows = episodes.loc[(episodes.method == name) & (episodes.scenario == scenario)]
            alarms = rows.first_alarm.notna()
            hits = rows.loc[rows.hit, "delay"]
            outcomes.append(
                {
                    "scenario": scenario,
                    "method": name,
                    "n": len(rows),
                    "alarm_fraction": float(alarms.mean()),
                    "alarm_wilson_upper95": wilson_upper(int(alarms.sum()), len(rows)),
                    "pre_change_alarm_fraction": float(rows.pre_change_alarm.mean()),
                    "hit_fraction": float(rows.hit.mean()),
                    "no_alarm_fraction": float((~alarms).mean()),
                    "conditional_mean_delay": float(hits.mean()) if len(hits) else None,
                    "mean_penalized_delay": float(rows.penalized_delay.mean()),
                }
            )
    gains, names = [], []
    for baseline in protocol["baselines"]:
        differences = []
        for scenario in spec["changes"]:
            rows = episodes.loc[episodes.scenario == scenario].pivot(
                index="replicate", columns="method", values="penalized_delay"
            )
            delta = (rows[baseline] - rows.drr).to_numpy(float)
            differences.append(delta)
            gains.append(delta)
            names.append(f"{scenario}:drr>{baseline}")
        gains.append(np.mean(differences, axis=0))
        names.append(f"pooled:drr>{baseline}")
    tests = block_inference(np.column_stack(gains), names, 1, 9999, 9182)
    false_alarm_ok = True
    for scenario in spec["nulls"]:
        stats = {o["method"]: o for o in outcomes if o["scenario"] == scenario}
        false_alarm_ok &= all(
            stats[name]["alarm_wilson_upper95"] <= 0.12 for name in ["drr", *protocol["baselines"]]
        )
        false_alarm_ok &= all(
            stats["drr"]["alarm_fraction"] <= stats[name]["alarm_fraction"] + 0.03
            for name in protocol["baselines"]
        )
    gates = {
        "heldout_false_alarm_budget_comparable": bool(false_alarm_ok),
        "pooled_delay_gain_at_least_8_and_holm_significant_vs_every_baseline": all(
            t["mean_improvement"] >= 8 and t["p_holm"] <= 0.05
            for t in tests
            if t["test"].startswith("pooled:")
        ),
        "no_change_type_more_than_8_observations_worse": all(
            t["mean_improvement"] >= -8 for t in tests if not t["test"].startswith("pooled:")
        ),
    }
    return {
        "verdict": "SUPPORTED IN SIMULATION" if all(gates.values()) else "NOT SUPPORTED",
        "gates": gates,
        "outcomes": outcomes,
        "tests": tests,
    }


def evaluate(root, study, calibration, output):
    lock = verify_lock(root, study, calibration)
    new_directory(output)
    protocol = json.loads((study / "protocol.json").read_text())
    episodes, traces = run_simulations(protocol, cutoffs=lock["simulation_thresholds"])
    simulation = simulation_summary(episodes, protocol)
    bars = load_market(study / "data", protocol)
    records = market_scores(bars, protocol, protocol["market"]["test_months"])
    calibration_records = pd.read_csv(calibration / "market_scores.csv.gz", parse_dates=["time"])
    risk, risk_ledger = market_risk(
        bars,
        pd.concat([calibration_records, records], ignore_index=True),
        lock["market_thresholds"],
        protocol,
    )
    for name in protocol["methods"]:
        records[name + "_warning"] = records[name] > lock["market_thresholds"][name]
    report = {
        "id": protocol["id"],
        "simulation": simulation,
        "market": {
            "status": "DESCRIPTIVE EXTERNAL VALIDATION; TRUE BREAK TIMES UNKNOWN",
            "effects": market_effects(records, lock["market_thresholds"], protocol),
            "scored_timestamps": len(records),
            "excluded_incomplete_forward_horizons": int(records.future_error_ratio.isna().sum()),
            "risk_omitted_terminal_hour_without_next_open": 1,
            "risk": risk,
        },
        "claim_limits": protocol["publication"]["scope"],
    }
    write_json(output / "summary.json", report)
    for name, frame in [
        ("simulation_episodes", episodes),
        ("simulation_scores", traces),
        ("market_scores", records),
        ("risk_ledger", risk_ledger),
    ]:
        write_csv(output / (name + ".csv.gz"), frame)
    environment = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "scikit-learn", "networkx"]
        },
    }
    write_json(output / "environment.json", environment)
    (output / "report.md").write_text(render_report(report, lock))
    write_json(
        output / "provenance.json",
        {
            "inputs": lock["inputs"],
            "lock_sha256": digest(calibration / "lock.json"),
            "artifacts": {path.name: digest(path) for path in sorted(output.iterdir())},
        },
    )
    return report


def render_report(report, lock):
    sim = report["simulation"]
    lines = [
        "# Lead–lag breakdown study",
        "",
        f"**Registered simulation verdict: {sim['verdict']}.**",
        "",
        "Market findings below are descriptive. They cannot establish an earlier true break time, causality or a trading edge.",
        "",
        "## Fixed success gates",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
    ]
    lines += [f"| {key.replace('_', ' ')} | {value} |" for key, value in sim["gates"].items()]
    lines += [
        "",
        "## Every simulation outcome",
        "",
        "First alarm only; early alarms and misses receive the full 256-observation delay penalty. "
        "The common-driver scenario changes observed dependence without changing latent coupling: its alarms are a confounding diagnostic, not proven false alarms. "
        "Delay columns are meaningful for the four true-change scenarios only.",
        "",
        "| Scenario | Method | N | Any alarm | Upper 95% | Early alarm | Hit | No alarm | Penalized delay |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for o in sim["outcomes"]:
        lines.append(
            f"| {o['scenario']} | {o['method']} | {o['n']} | {o['alarm_fraction']:.2%} | {o['alarm_wilson_upper95']:.2%} | {o['pre_change_alarm_fraction']:.2%} | {o['hit_fraction']:.2%} | {o['no_alarm_fraction']:.2%} | {o['mean_penalized_delay']:.2f} |"
        )
    lines += [
        "",
        "## All 15 corrected comparisons",
        "",
        "Positive gains favor DRR. Confidence intervals are pointwise. Holm correction includes all 15 comparisons. "
        "Resampling units are independent simulation replicates, paired across scenarios and methods.",
        "",
        "| Test | Delay gain | CI low | CI high | Raw p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in sim["tests"]:
        lines.append(
            f"| {t['test']} | {t['mean_improvement']:.2f} | {t['ci95_low']:.2f} | {t['ci95_high']:.2f} | {t['p_one_sided']:.4f} | {t['p_holm']:.4f} |"
        )
    lines += [
        "",
        "## Frozen thresholds",
        "",
        "| Method | Simulation path threshold | Market warning threshold |",
        "| --- | ---: | ---: |",
    ]
    for name, cutoff in lock["simulation_thresholds"].items():
        lines.append(f"| {name} | {cutoff:.6f} | {lock['market_thresholds'][name]:.6f} |")
    lines += [
        "",
        "## Held-out 2026 market validation",
        "",
        "BTCUSDT/ETHUSDT hourly bars. Thresholds use February–March; evaluation uses April–September. "
        "Each month refits on only its preceding 256 return observations. Warnings use no future labels. "
        "The next-24-hour target is frozen VAR error relative to zero-return error, in fit-standardized units. "
        "A positive warned-minus-unwarned difference indicates deterioration associated with warnings, not an independently known change point. "
        "Confidence intervals are descriptive circular-block estimates under dependence assumptions; intervals are withheld for sparse warnings.",
        "",
        "| Method | Eligible | Warnings | Warning rate | Error-ratio difference | 95% interval | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for e in report["market"]["effects"]:
        effect = (
            "n/a"
            if e["future_error_ratio_difference"] is None
            else f"{e['future_error_ratio_difference']:.5f}"
        )
        interval = "n/a" if e["ci95"] is None else f"[{e['ci95'][0]:.5f}, {e['ci95'][1]:.5f}]"
        lines.append(
            f"| {e['method']} | {e['eligible_timestamps']} | {e['warnings']} | {e['warning_rate']:.2%} | {effect} | {interval} | {e['inference_status']} |"
        )
    lines += [
        "",
        f"Scored timestamps: {report['market']['scored_timestamps']}; excluded incomplete 24-hour labels: {report['market']['excluded_incomplete_forward_horizons']}. "
        "Warnings without complete labels remain in the risk simulation. The last hourly holding lacks a next open and is explicitly omitted.",
        "",
        "## Risk attribution: all costs and exposure controls",
        "",
        "After warnings, cash begins two hourly opens later and lasts 24 hours. Constant matched-exposure controls use the strategy's realized average exposure: "
        "they are ex-post attribution controls, not deployable strategies. Lower losses alone do not establish useful warning timing. "
        "No market impact, account-specific fees, funding, capacity or stablecoin risk is estimated. Costs are hypothetical. No inferential risk claim is made.",
        "",
        "| Strategy | One-way bps | Net growth | Max drawdown | Worst-5% mean hourly loss | Exposure | Turnover |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["market"]["risk"]:
        lines.append(
            f"| {r['strategy']} | {r['cost_bps']} | {r['total_return']:.2%} | {r['max_drawdown']:.2%} | {r['worst_5pct_mean_hourly_loss']:.4%} | {r['mean_exposure']:.2%} | {r['turnover']:.2f} |"
        )
    lines += [
        "",
        "## Scope and provenance",
        "",
        report["claim_limits"],
        "",
        "All raw score traces, episode outcomes, market scores/labels and hourly risk paths are included as deterministic compressed CSV. "
        "The separate calibration lock pins code, protocol, input data and calibration evidence. "
        "Market data and derived results: Binance Vision, CC BY-NC-SA 4.0 and its dataset terms. Software remains MIT. "
        "Calibration budgets do not guarantee equal realized false-alarm rates under distribution shift. "
        "The simulation family is deliberately limited and public historical data are not independently blinded.",
        "",
    ]
    return "\n".join(lines)
