"""Evidence records, the opt-in orchestrator, and its place in the DRR facade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework import DynamicResonanceRooting, generate_coupled_oscillator
from drr_framework.math_evidence import (
    CONCLUSIONS,
    FEATURE_DEFINITIONS,
    PROVEN_STATEMENTS,
    AssumptionCheck,
    MathematicalEvidenceRecord,
    MathematicalStatement,
    array_sha256,
    binomial_upper_tail,
)
from drr_framework.math_features import analyze_mathematical_features
from drr_framework.resonance_math_benchmark import simulate_condition

FS = 100.0


def _oscillators(seed: int, **fields):
    spec = {
        "family": "phase_oscillators",
        "sampling_rate": FS,
        "n_samples": 2048,
        "fx": 10.0,
        "fy": 20.3,
        "coupling": 12.0,
        "ratio": [2, 1],
        "phase_diffusion": 2.0,
        "delay_seconds": 0.0,
        "noise": 0.3,
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
    return simulate_condition(spec, np.random.default_rng(seed))


# ------------------------------------------------------------------ record types


def test_statement_and_assumption_classes_are_enforced():
    MathematicalStatement("x", "proven", "a theorem")
    with pytest.raises(ValueError, match="status"):
        MathematicalStatement("x", "likely", "a hunch")
    with pytest.raises(ValueError, match="non-empty"):
        MathematicalStatement("x", "hypothesis", " ")
    with pytest.raises(ValueError, match="status"):
        AssumptionCheck("stationarity", "probably")


def test_proven_statements_and_definitions_carry_their_support():
    assert PROVEN_STATEMENTS and all(s.status == "proven" for s in PROVEN_STATEMENTS)
    for definition in FEATURE_DEFINITIONS.values():
        assert definition.references and definition.limitations and definition.version
        assert definition.to_dict()["references"] == list(definition.references)


def test_binomial_tail_matches_hand_values():
    assert binomial_upper_tail(0, 19, 0.05) == pytest.approx(1.0)
    assert binomial_upper_tail(1, 19, 0.05) == pytest.approx(1 - 0.95**19)
    assert binomial_upper_tail(20, 19, 0.05) == 0.0


def test_array_hash_depends_on_values_and_shape_only():
    data = np.arange(12.0)
    assert array_sha256(data) == array_sha256(data.astype(np.float32).astype(float))
    assert array_sha256(data) != array_sha256(data.reshape(3, 4))
    assert array_sha256(data.reshape(3, 4)) == array_sha256(np.asfortranarray(data.reshape(3, 4)))


def test_record_identifier_is_a_hash_of_its_content():
    record = MathematicalEvidenceRecord(
        feature_definitions=[FEATURE_DEFINITIONS["phase_locking_value"]],
        data_provenance={"input_sha256": "abc"},
        assumptions=[AssumptionCheck("independence", "unverifiable")],
        configuration={"alpha": 0.05},
        measurements={"plv": float("nan")},
        negative_controls={},
        limitations=["undirected"],
        statements=list(PROVEN_STATEMENTS[:1]),
        conclusion={"status": "inconclusive", "reasons": []},
    )
    payload = record.to_dict()
    assert payload["measurements"]["plv"] is None
    assert json.loads(record.to_json()) == payload
    changed = MathematicalEvidenceRecord(**{**record.__dict__, "configuration": {"alpha": 0.01}})
    assert changed.record_id != record.record_id
    assert record.statements_by_class()["proven"]
    with pytest.raises(ValueError, match="conclusion"):
        MathematicalEvidenceRecord(**{**record.__dict__, "conclusion": {"status": "proved"}})


# ------------------------------------------------------------------ the orchestrator


def test_harmonic_coupling_is_supported_with_a_complete_deterministic_record():
    data, _ = _oscillators(1)
    first = analyze_mathematical_features(data, FS, column_names=("drive", "response"))
    second = analyze_mathematical_features(data, FS, column_names=("drive", "response"))
    record = first["evidence_record"]
    assert record == second["evidence_record"]
    assert record["conclusion"]["status"] == "supported"
    assert "2:1 phase synchronized" in record["conclusion"]["reasons"][0]
    classes = {s["status"] for s in record["statements"]}
    assert classes == {"proven", "empirical", "hypothesis"}
    assert record["data_provenance"]["analysed_sha256"] == array_sha256(data)
    assert {a["status"] for a in record["assumptions"]} <= {
        "checked_passed",
        "checked_failed",
        "unverifiable",
        "not_applicable",
    }
    assert first["negative_controls"]["n_controls"] == 19
    assert not first["negative_controls"]["flagged"]
    json.dumps(first, allow_nan=False)


def test_random_walks_abstain_and_the_record_says_why():
    walks = np.cumsum(np.random.default_rng(2).standard_normal((2048, 2)), axis=0)
    result = analyze_mathematical_features(walks, FS, negative_controls=0)
    assert result["data_quality"]["abstain"]
    record = result["evidence_record"]
    assert record["conclusion"]["status"] == "inconclusive"
    assert any("DFA exponent" in reason for reason in record["conclusion"]["reasons"])
    failed = [a for a in record["assumptions"] if a["status"] == "checked_failed"]
    assert any("random-walk" in a["assumption"] for a in failed)


def test_missing_samples_and_irregular_timestamps_are_regularized_and_recorded():
    data, _ = _oscillators(3)
    holes = data.copy()
    holes[np.random.default_rng(4).random(holes.shape) < 0.05] = np.nan
    result = analyze_mathematical_features(holes, FS, negative_controls=0)
    regularization = result["evidence_record"]["data_provenance"]["regularization"]
    assert regularization["missing_fraction"] == pytest.approx(0.05, abs=0.01)
    jittered, stamps = _oscillators(5, timing_jitter=0.3)
    timed = analyze_mathematical_features(jittered, FS, timestamps=stamps, negative_controls=0)
    assert timed["data_quality"]["regularization"]["interval_cv"] > 0


def test_one_channel_is_characterized_but_no_relationship_is_concluded():
    data, _ = _oscillators(6)
    result = analyze_mathematical_features(data[:, 0], FS)
    assert result["phase_synchrony"] is None and result["trajectory_geometry"] is None
    assert result["evidence_record"]["conclusion"]["status"] == "not_applicable"
    characterization = result["characterization"]["dim_0"]
    assert characterization["second_order_recurrence"]["frequency_hz"] == pytest.approx(
        10.0, abs=1.5
    )
    assert characterization["recurrence"]["analysed_samples"] == 1000


def test_orchestrator_validates_input():
    with pytest.raises(ValueError, match="sampling_rate"):
        analyze_mathematical_features(np.ones((64, 2)), 0.0)
    with pytest.raises(ValueError, match="column_names"):
        analyze_mathematical_features(np.ones((64, 2)), FS, column_names=("a",))
    assert set(CONCLUSIONS) == {"supported", "not_detected", "inconclusive", "not_applicable"}


# ------------------------------------------------------------------ the facade


def _facade(**options):
    _, data = generate_coupled_oscillator(random_state=42)
    return DynamicResonanceRooting(tau=2, sampling_rate=200).analyze_system(
        data,
        multivariate=True,
        window_size=256,
        state_space=False,
        rooting_max_lag=4,
        rooting_n_surrogates=25,
        rooting_random_state=42,
        **options,
    )


def test_features_are_off_by_default_and_leave_every_other_result_unchanged():
    baseline = _facade()
    assert "mathematical_features" not in baseline
    extended = _facade(mathematical_features={"negative_controls": 0})
    assert set(extended) - set(baseline) == {"mathematical_features"}
    assert extended["resonance_depths"] == baseline["resonance_depths"]
    assert extended["rooting_analysis"]["significant_edges"] == (
        baseline["rooting_analysis"]["significant_edges"]
    )
    features = extended["mathematical_features"]
    assert features["experimental"] is True
    assert features["evidence_record"]["conclusion"]["status"] in CONCLUSIONS


def test_the_readme_fixture_is_inconclusive_for_phase_not_negative():
    """Near-periodic channels cannot show phase coupling; the record must not say 'absent'."""
    features = _facade(mathematical_features={"negative_controls": 0})["mathematical_features"]
    pair = features["phase_synchrony"]["pairs"][0]
    assert pair["channels"] == ["dim_0", "dim_1"]
    assert pair["status"] == "inconclusive"
    assert features["evidence_record"]["conclusion"]["status"] != "not_detected"


def test_a_failing_feature_run_is_recorded_not_raised():
    result = _facade(mathematical_features={"phase_options": {"alpha": 2.0}})
    assert "error" in result["mathematical_features"]


# ------------------------------------------------------------------ the CSV workflow


def test_csv_workflow_is_unchanged_by_default_and_reports_features_on_request(tmp_path):
    from drr_framework.csv_analysis import analyze_csv, main

    sample = Path(__file__).resolve().parents[1] / "data/raw/coupled_oscillator_sample.csv"
    options = {
        "columns": ["dim_0", "dim_1", "dim_2"],
        "sampling_rate": 200.0,
        "time_column": "time_seconds",
        "n_surrogates": 25,
    }
    default = analyze_csv(sample, **options)
    assert "mathematical_features" not in default
    assert "mathematical_features" not in default["provenance"]["parameters"]
    extended = analyze_csv(sample, **options, mathematical_features=True)
    assert extended["rooting"] == default["rooting"]
    assert extended["run_id"] != default["run_id"]
    record = extended["mathematical_features"]["evidence_record"]
    assert record["data_provenance"]["source"] == "coupled_oscillator_sample.csv"
    assert record["data_provenance"]["column_names"] == ["dim_0", "dim_1", "dim_2"]
    destination = tmp_path / "bundle"
    main(
        [
            str(sample),
            "--columns",
            "dim_0",
            "dim_1",
            "--sampling-rate",
            "200",
            "--n-surrogates",
            "25",
            "--mathematical-features",
            "--output-dir",
            str(destination),
        ]
    )
    report = (destination / "report.md").read_text(encoding="utf-8")
    assert "## Experimental mathematical features" in report
    assert "dim_0 – dim_1" in report
    json.loads((destination / "analysis.json").read_text(encoding="utf-8"))
