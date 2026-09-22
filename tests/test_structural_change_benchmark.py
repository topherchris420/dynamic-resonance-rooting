"""Preregistered QBO comparison: protocol locks, causality, and the reviewed run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.external_benchmark.protocol import (
    alarm_rate,
    calibrate,
    classify_claim,
    month_span,
)
from drr_framework.external_benchmark.qbo import (
    compute_feature_frame,
    load_preregistration,
    load_qbo_series,
    parse_original_monthly,
    regularized_logistic_scores,
    run_structural_change_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "expected" / "qbo_structural_change_benchmark.json"
GLUED = """
30 mb zonal wind index
                    ORIGINAL        DATA
YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
1979   1.25   2.50-999.90   4.00   5.00   6.00   7.00   8.00   9.00  10.00  11.00  12.00
30 mb zonal wind index
                        ANOMALY
YEAR   JAN   FEB
1979   9.99   8.88
"""


def test_preregistration_locks_the_public_plan():
    spec = load_preregistration()
    analysis = spec["analysis"]
    assert spec["id"] == "qbo-disruption-structural-change-v1"
    assert spec["universal"] is False
    assert spec["claim_class"] == "domain_adapter"
    assert analysis["holdout_start"] == "2011-01"
    assert analysis["trailing_window_months"] == 84
    assert analysis["rooting_max_lag"] == 12
    assert analysis["false_alarm_tolerance"] == 0.05
    assert analysis["var_order"] == 1
    assert analysis["minimum_repeat_count"] == 2
    assert analysis["bootstrap"]["replicates"] == 400
    assert analysis["bootstrap"]["block_length_months"] == 28
    assert analysis["bootstrap"]["seed"] == 20260922
    assert [(item["id"], item["start"], item["end"]) for item in spec["events"]] == [
        ("qbo-2015-2016", "2016-01", "2016-04"),
        ("qbo-2019-2020", "2019-12", "2020-03"),
    ]
    assert [item["id"] for item in spec["hypotheses"]] == ["H1", "H2", "H3"]
    assert spec["dataset"]["first_month"] == "1979-01"
    assert spec["dataset"]["last_month"] == "2026-08"
    assert spec["dataset"]["finite_months"] == 572


def test_original_section_parser_stops_before_the_anomaly_block():
    rows = parse_original_monthly(GLUED)
    assert rows[0] == (1979, 1, 1.25)
    assert rows[1] == (1979, 2, 2.5)
    assert rows[2] == (1979, 3, None)
    assert len(rows) == 12
    assert all(value != 9.99 for _, _, value in rows)


def test_vendored_qbo_record_matches_the_published_first_month():
    series = load_qbo_series()
    assert series["months"][0] == "1979-01"
    assert series["months"][-1] == "2026-08"
    assert len(series["months"]) == 572
    assert series["values"][0, 0] == pytest.approx(1.85)
    assert series["values"][0, 1] == pytest.approx(7.38)


def test_threshold_uses_the_least_strict_admissible_alarm_rate():
    scores = np.arange(100, dtype=float)
    threshold = calibrate(scores, 0.05)
    assert threshold == pytest.approx(95.0)
    assert alarm_rate(scores, threshold, np.ones(100, dtype=bool)) == pytest.approx(0.05)
    assert alarm_rate(scores, 94.0, np.ones(100, dtype=bool)) > 0.05
    tied = np.array([0.0] * 50 + [1.0] * 50)
    assert calibrate(tied, 0.05) is None


def test_claim_rule_does_not_let_the_interval_rescue_a_miss():
    common = dict(
        events_comparable=True,
        drr_calibrated=True,
        drr_holdout_fpr=0.04,
        tolerance=0.05,
        calibrated_challenge_count=3,
        minimum_repeat_count=2,
        ablation_explains_unique_set=False,
        bootstrap_unique_low=1.0,
    )
    assert classify_claim(unique_hit_count=2, **common) == (
        "supported",
        "repeated_unique_detections",
    )
    assert classify_claim(
        unique_hit_count=2,
        bootstrap_unique_low=0.0,
        **{key: value for key, value in common.items() if key != "bootstrap_unique_low"},
    ) == ("inconclusive", "uncertainty_includes_zero")
    assert classify_claim(unique_hit_count=1, **common)[0] == "inconclusive"
    assert classify_claim(unique_hit_count=0, **common) == (
        "not_supported",
        "no_unique_detection",
    )
    assert classify_claim(
        unique_hit_count=2,
        drr_holdout_fpr=0.27,
        **{key: value for key, value in common.items() if key != "drr_holdout_fpr"},
    ) == ("not_supported", "drr_false_alarm_rate")
    assert classify_claim(
        unique_hit_count=2,
        ablation_explains_unique_set=True,
        **{key: value for key, value in common.items() if key != "ablation_explains_unique_set"},
    ) == ("not_supported", "ablation_sufficient")
    assert classify_claim(
        events_comparable=False,
        **{key: value for key, value in common.items() if key != "events_comparable"},
        unique_hit_count=2,
    ) == ("inconclusive", "unscored_event")


def test_logistic_arm_is_withheld_without_estimation_events_and_ignores_holdout_rows():
    generator = np.random.default_rng(0)
    features = generator.normal(size=(40, 3))
    estimation = np.zeros(40, dtype=bool)
    estimation[:30] = True
    withheld, reason = regularized_logistic_scores(features, np.zeros(40), estimation, 1.0)
    assert withheld is None
    assert reason == "no positive labels in the estimation window"
    labels = np.zeros(40)
    labels[:5] = 1.0
    first, fitted_reason = regularized_logistic_scores(features, labels, estimation, 1.0)
    shifted = features.copy()
    shifted[30:] += 10.0
    second, _ = regularized_logistic_scores(shifted, labels, estimation, 1.0)
    assert fitted_reason is None
    assert np.allclose(first[:30], second[:30])
    assert not np.allclose(first[30:], second[30:])


def test_trailing_scores_do_not_read_a_later_month():
    spec = load_preregistration()
    local = json.loads(json.dumps(spec))
    local["analysis"]["trailing_window_months"] = 36
    local["analysis"]["holdout_start"] = "2008-01"
    local["events"] = [
        {"id": "late", "start": "2009-01", "end": "2009-03", "sources": []},
    ]
    months = list(month_span("2000-01", "2009-12"))
    time = np.arange(len(months))
    generator = np.random.default_rng(1)
    values = np.column_stack(
        (
            np.sin(2 * np.pi * time / 28.0),
            np.sin(2 * np.pi * (time - 3) / 28.0),
        )
    )
    values += generator.normal(scale=0.05, size=values.shape)
    mutated = values.copy()
    mutated[-1, 0] += 50.0
    original = compute_feature_frame(values, months, local)
    changed = compute_feature_frame(mutated, months, local)
    assert original["months"][-1] == "2009-12"
    for name in original["scores"]:
        np.testing.assert_allclose(
            original["scores"][name][:-1],
            changed["scores"][name][:-1],
            equal_nan=True,
        )


def test_public_benchmark_obeys_the_holdout_and_the_false_alarm_budget():
    spec = load_preregistration()
    artifact = run_structural_change_benchmark()
    assert artifact["claim"]["universal"] is False
    assert artifact["claim"]["claim_class"] == "domain_adapter"
    assert artifact["preregistration_sha256"]
    scored_months = [row["month"] for row in artifact["series"]]
    assert scored_months[0] == "1986-01"
    assert scored_months[-1] == "2026-08"
    event_months = {
        event["id"]: set(month_span(event["start"], event["end"])) for event in spec["events"]
    }
    for row in artifact["series"]:
        in_event = any(row["month"] in months for months in event_months.values())
        if row["split"] == "estimation":
            assert row["month"] < spec["analysis"]["holdout_start"]
            assert not in_event
        else:
            assert row["month"] >= spec["analysis"]["holdout_start"]
    logistic = next(
        model for model in artifact["models"] if model["name"] == "regularized_logistic"
    )
    assert logistic["status"] == "withheld"
    assert logistic["reason"] == "no positive labels in the estimation window"
    tolerance = spec["analysis"]["false_alarm_tolerance"]
    for model in artifact["models"]:
        if model["status"] != "calibrated":
            continue
        assert model["estimation_false_alarm_rate"] <= tolerance + 1e-12
    repeated = run_structural_change_benchmark()
    assert repeated["claim"] == artifact["claim"]
    assert repeated["models"] == artifact["models"]
    assert repeated["uncertainty"] == artifact["uncertainty"]


def test_reviewed_artifact_matches_a_fresh_run_and_its_own_decision():
    reviewed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fresh = run_structural_change_benchmark()
    assert fresh["claim"]["status"] == reviewed["claim"]["status"] == "not_supported"
    assert fresh["claim"]["reason"] == reviewed["claim"]["reason"] == "drr_false_alarm_rate"
    assert fresh["unique_event_ids"] == reviewed["unique_event_ids"] == []
    for current, stored in zip(fresh["models"], reviewed["models"]):
        assert current["name"] == stored["name"]
        assert current["status"] == stored["status"]
        assert current["events"] == stored["events"]
        if current["holdout_false_alarm_rate"] is None:
            assert stored["holdout_false_alarm_rate"] is None
        else:
            assert current["holdout_false_alarm_rate"] == pytest.approx(
                stored["holdout_false_alarm_rate"]
            )
    drr = next(model for model in reviewed["models"] if model["name"] == "drr_full")
    assert drr["holdout_false_alarm_rate"] > reviewed["analysis"]["false_alarm_tolerance"]
    calls = {item["id"]: item["call"] for item in drr["events"]}
    assert calls == {"qbo-2015-2016": "miss", "qbo-2019-2020": "hit"}
    report = (ARTIFACT.with_suffix(".md")).read_text(encoding="utf-8")
    assert "not_supported" in report
    assert "drr_false_alarm_rate" in report or "false-alarm rate exceeds" in report
