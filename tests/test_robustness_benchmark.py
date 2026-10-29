"""Known truth, leakage boundaries, honest uncertainty and independent replay."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import runpy

import numpy as np
import pytest

from drr_framework.robustness_benchmark import (
    DEFAULT_CONFIG,
    METHODS,
    ar1_residuals,
    calibrate_baselines,
    config_digest,
    correlation_edges,
    evaluate_trial,
    render_robustness_report,
    run_robustness_benchmark,
    simulate_regimes,
    summarize_records,
    validate_config,
)


def tiny_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(n_samples=128, n_trials=3, n_calibration=39, n_surrogates=19)
    config["cases"] = [config["cases"][0], config["cases"][-1]]
    return config


@pytest.fixture(scope="module")
def artifact():
    return run_robustness_benchmark(tiny_config())


def test_baseline_recovers_known_direction_and_lag_without_drr():
    rng = np.random.default_rng(8)
    source = rng.normal(size=1024)
    target = np.concatenate((rng.normal(size=2), source[:-2]))
    data = np.column_stack((source, target, rng.normal(size=1024)))
    scores, lags = correlation_edges(data, 4)
    assert scores[0, 1] == pytest.approx(1.0)
    assert lags[0, 1] == 2
    assert np.max(scores[1:, :]) < 0.1


def test_prewhitening_removes_ar1_persistence():
    rng = np.random.default_rng(7)
    data = np.zeros((10000, 3))
    innovations = rng.normal(size=data.shape)
    for index in range(1, len(data)):
        data[index] = 0.9 * data[index - 1] + innovations[index]
    residuals = ar1_residuals(data)
    assert np.corrcoef(data[:-1, 0], data[1:, 0])[0, 1] > 0.85
    assert abs(np.corrcoef(residuals[:-1, 0], residuals[1:, 0])[0, 1]) < 0.03


def test_calibration_never_uses_evaluation_cases_or_trial_count():
    first = tiny_config()
    second = copy.deepcopy(first)
    second["cases"][-1]["hurst"] = 0.95
    second["cases"][-1]["regimes"][-1]["coupling"] = 0.8
    second["n_trials"] = 100
    calibration = calibrate_baselines(first)
    assert calibration == calibrate_baselines(second)
    for values in calibration["methods"].values():
        assert values["threshold"] == sorted(values["family_maxima"])[calibration["rank"] - 1]


def test_future_coupling_does_not_change_earlier_window():
    config = tiny_config()
    original = copy.deepcopy(config["cases"][-1])
    altered = copy.deepcopy(original)
    altered["regimes"][-1] = {"coupling": 0.8, "reverse": False}
    before = simulate_regimes(config, original, 0)
    after = simulate_regimes(config, altered, 0)
    np.testing.assert_array_equal(before[0], after[0])
    assert not np.array_equal(before[1], after[1])


def test_regime_generator_recovers_reverse_direction_and_drops_boundary_buffers():
    config = tiny_config()
    config["n_samples"] = 4096
    case = {
        "id": "known",
        "hurst": 0.5,
        "regimes": [{"coupling": 0.9, "reverse": False}, {"coupling": 0.9, "reverse": True}],
    }
    before, after = simulate_regimes(config, case, 0)
    for window, source, target in ((before, 0, 1), (after, 1, 0)):
        assert window.shape == (4096, 3)
        scores, lags = correlation_edges(window, 4)
        assert scores[source, target] == pytest.approx(0.9, abs=0.03)
        assert lags[source, target] == config["lag"]


def test_replay_is_deterministic_and_all_methods_share_retained_inputs(artifact):
    assert artifact == run_robustness_benchmark(tiny_config())
    for case in artifact["config"]["cases"]:
        replay = evaluate_trial(artifact["config"], case, 0, artifact["calibration"])
        retained = [r for r in artifact["records"] if r["case"] == case["id"] and r["trial"] == 0]
        assert replay == retained
    assert all(set(record["methods"]) == set(METHODS) for record in artifact["records"])
    evaluation_hashes = {record["input_sha256"] for record in artifact["records"]}
    assert evaluation_hashes.isdisjoint(artifact["calibration"]["input_sha256"])
    json.dumps(artifact, allow_nan=False)


def test_summary_handles_nulls_wrong_direction_and_wrong_lag_honestly(artifact):
    records = copy.deepcopy(artifact["records"])
    config = artifact["config"]
    for record in records:
        for method in METHODS:
            truth = record["truth"]
            record["methods"][method] = (
                [[0, 1, 1]] if not truth else [[truth[0][0], truth[0][1], 1], [2, 0, 2]]
            )
    rows = summarize_records(records, config)
    for row in rows:
        assert row["any_non_injected_direction"]["rate"] == 1
        if row["truth"]:
            assert row["true_edge_any_lag"]["rate"] == 1
            assert row["true_edge_exact_lag"]["rate"] == 0
        else:
            assert row["true_edge_exact_lag"] is None
            assert row["null_assessment"] == "inflated"
    # Zero false alarms in three trials cannot establish a <= 5% error rate.
    for record in records:
        record["methods"] = {method: [] for method in METHODS}
    for row in summarize_records(records, config):
        if not row["truth"]:
            assert row["null_assessment"] == "inconclusive"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "truth",
        "method",
        "lag",
        "fractional_trial",
        "boolean_window",
        "bad_hash",
        "fractional_endpoint",
    ],
)
def test_summary_rejects_incomplete_or_inconsistent_evidence(artifact, mutation):
    records = copy.deepcopy(artifact["records"])
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "truth":
        records[0]["truth"] = [[0, 1, 2]]
    elif mutation == "method":
        del records[0]["methods"][METHODS[0]]
    elif mutation == "lag":
        records[0]["methods"][METHODS[0]] = [[0, 1, 99]]
    elif mutation == "fractional_trial":
        records[0]["trial"] = 0.5
    elif mutation == "boolean_window":
        records[0]["window"] = False
    elif mutation == "bad_hash":
        records[0]["input_sha256"] = "not-a-hash"
    else:
        records[0]["methods"][METHODS[0]] = [[0.5, 1, 2]]
    with pytest.raises(ValueError):
        summarize_records(records, artifact["config"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_trials", 0),
        ("n_trials", True),
        ("n_surrogates", 3),
        ("n_calibration", 3),
        ("alpha", float("nan")),
        ("lag", 5),
        ("n_samples", 6),
    ],
)
def test_invalid_protocol_is_rejected_before_measurement(key, value):
    config = tiny_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_report_names_uncertainty_limits_and_does_not_claim_calibrated_edge_probabilities(artifact):
    report = render_robustness_report(artifact)
    assert "not calibrated probabilities" in report
    assert "inconclusive" in report
    assert "not simultaneous" in report
    assert "not measurements of online change-detection delay" in report


def test_reference_artifact_summaries_protocol_and_report_agree():
    root = Path(__file__).resolve().parents[1] / "results" / "expected" / "robustness-benchmark"
    protocol_bytes = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    committed = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    assert committed["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert protocol["config"] == committed["config"] == DEFAULT_CONFIG
    assert committed["config_sha256"] == config_digest(DEFAULT_CONFIG)
    assert committed["summary"] == summarize_records(committed["records"], DEFAULT_CONFIG)
    assert (root / "report.md").read_text(encoding="utf-8") == render_robustness_report(committed)


def test_cli_seals_protocol_and_preserves_existing_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_benchmark.py"
    output = tmp_path / "benchmark"
    command = [sys.executable, str(script), "--quick", "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    retained = {path.name: path.read_bytes() for path in output.iterdir()}
    assert retained["comparison.png"].startswith(b"\x89PNG")
    assert b"<svg" in retained["comparison.svg"]
    result = json.loads(retained["benchmark.json"])
    protocol = json.loads(retained["protocol.json"])
    assert protocol["config"] == result["config"]
    assert result["protocol_sha256"] == hashlib.sha256(retained["protocol.json"]).hexdigest()
    assert (
        protocol["source_sha256"]["scripts/run_robustness_benchmark.py"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert "already exists" in second.stderr
    assert retained == {path.name: path.read_bytes() for path in output.iterdir()}


def test_figure_accepts_zero_rate_wilson_roundoff(artifact, tmp_path):
    from drr_framework.calibration import binomial_rate

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_benchmark.py"
    payload = copy.deepcopy(artifact)
    for row in payload["summary"]:
        if not row["truth"]:
            row["any_non_injected_direction"] = binomial_rate(0, 200)
    runpy.run_path(str(script))["write_comparison_figure"](payload, tmp_path)
    assert (tmp_path / "comparison.png").is_file()


@pytest.mark.parametrize("case_index", [0, -1])
def test_reference_trial_decisions_replay(case_index):
    root = Path(__file__).resolve().parents[1] / "results" / "expected" / "robustness-benchmark"
    committed = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    config = committed["config"]
    case = config["cases"][case_index]
    fresh = evaluate_trial(config, case, 0, committed["calibration"])
    retained = [
        record
        for record in committed["records"]
        if record["case"] == case["id"] and record["trial"] == 0
    ]
    # Hashes retain the original numeric bytes. FFT rounding may vary between
    # platforms; cross-environment replay checks the scientific decisions.
    for actual, expected in zip(fresh, retained):
        assert actual["methods"] == expected["methods"]
        assert actual["truth"] == expected["truth"]
