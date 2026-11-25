"""Independent small examples and temporal-integrity checks for Phase 2."""

from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import test_conventional_challenge_models as phase_one

from drr_framework.supervisory import (
    BinaryEventLabel,
    ContingencyDesign,
    ContingencyScore,
    evaluate_contingency,
)
from drr_framework.supervisory.common import canonical_json
from drr_framework.validation_readiness import run_event_backtest
import drr_framework.supervisory.contingency as metrology

var_lab, panel_lab = phase_one.var_lab, phase_one.panel_lab
run_var, run_logit = phase_one.run_var, phase_one.run_logit


# These skips are explicit on the retained Python 3.8-3.11 base-runtime matrix.
# Python 3.12 CI installs the real backend and must run every numerical test.
requires_nist = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="NIST release 0.2.3 requires Python 3.12 syntax"
)
EVENT = "Synthetic independently sampled event in the specified quarter"
TRUTH = (1, 0, 1, 1, 0, 1, 0, 0)
VALUES = (0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0)


def population(
    *,
    values=VALUES,
    truth=TRUTH,
    period="2021-03-31",
    as_of="2020-12-30",
    known="2021-04-01",
    prefix="evaluation",
):
    scores = tuple(
        ContingencyScore(f"SYN_{i}", period, as_of, value, f"{prefix}-{i}", "synthetic-score-v1")
        for i, value in enumerate(values)
    )
    labels = tuple(
        BinaryEventLabel(
            f"SYN_{i}", period, value, known, EVENT, "Seeded synthetic outcome inventory"
        )
        for i, value in enumerate(truth)
    )
    return scores, labels


def design(**changes):
    base = ContingencyDesign(
        "Eight synthetic institutions, one explicitly specified quarter each",
        EVENT,
        "synthetic-score-v1",
        "2019-01-01",
        (0.5,),
        label_semantics="explicit_institution_quarter",
        negative_label_definition="Zero is an independently drawn and recorded non-event, not absence from a list",
    )
    return replace(base, **changes)


def evaluate(scores=None, labels=None, **kwargs):
    default_scores, default_labels = population()
    return evaluate_contingency(
        default_scores if scores is None else scores,
        default_labels if labels is None else labels,
        design=kwargs.pop("design", design()),
        evaluation_as_of=kwargs.pop("evaluation_as_of", "2021-04-30"),
        **kwargs,
    )


@requires_nist
def test_independently_calculated_counts_mcc_f_scores_precision_recall_and_ap():
    result = evaluate()
    assert result.status == "available", result.withholding_reason
    row = result.threshold_results[0]
    assert (row.true_positives, row.false_positives, row.false_negatives, row.true_negatives) == (
        2,
        1,
        2,
        3,
    )
    assert row.mcc == pytest.approx((2 * 3 - 1 * 2) / math.sqrt(3 * 4 * 4 * 5))
    assert row.precision == pytest.approx(2 / 3)
    assert row.recall == pytest.approx(2 / 4)
    assert row.f1 == pytest.approx(4 / 7)
    assert row.f2 == pytest.approx(10 / 19)
    # Positive ranks are 1, 3, 4, 6; ties are handled separately below.
    assert result.average_precision == pytest.approx((1 + 2 / 3 + 3 / 4 + 4 / 6) / 4)
    assert result.positive_label_count == result.negative_label_count == 4
    assert result.evaluated_observation_count == result.observation_count == 8
    assert result.prospective is True
    assert result.pre_specified_threshold == result.selected_threshold == 0.5
    assert result.package_name == "contingency-tools" and result.package_version == "0.2.3"


@requires_nist
def test_threshold_family_equality_determinism_and_retrospective_sensitivity():
    cfg = design(
        thresholds=(0.9, 0.5, 0.3),
        threshold_selection_mode="retrospective_sensitivity",
        registered_at="2021-04-20",
    )
    scores, labels = population()
    first = evaluate(scores, labels, design=cfg)
    second = evaluate(
        tuple(reversed(scores)),
        tuple(reversed(labels)),
        design=replace(cfg, thresholds=(0.3, 0.9, 0.5)),
    )
    assert first == second and first.result_id == second.result_id
    assert first.evaluated_thresholds == (0.3, 0.5, 0.9)
    assert first.threshold_results[-1].true_positives == 1  # >= includes exactly .9
    assert first.selected_threshold is None and first.pre_specified_threshold is None
    assert first.prospective is False
    sensitivity = first.information_set["threshold_sensitivity"]["precision"]
    assert sensitivity == {"minimum": 0.6, "maximum": 1.0, "defined_threshold_count": 3}
    assert first.average_precision == evaluate().average_precision


@requires_nist
def test_unknown_late_and_unscored_observations_are_not_negatives():
    scores, labels = population(values=(None,) + VALUES[1:])
    labels = tuple(
        replace(label, value=None) if i == 1 else label for i, label in enumerate(labels)
    )
    labels = labels[:-1]  # absent last non-event
    labels = tuple(
        replace(label, available_as_of="2022-01-01") if i == 4 else label
        for i, label in enumerate(labels)
    )
    result = evaluate(scores, labels)
    assert result.positive_label_count == 4 and result.negative_label_count == 1
    assert result.unlabeled_observation_count == 3
    assert result.unscored_observation_count == 1
    assert result.evaluated_observation_count == 4
    assert result.evaluated_positive_count == 3 and result.evaluated_negative_count == 1
    row = result.threshold_results[0]
    assert (row.true_positives, row.false_positives, row.false_negatives, row.true_negatives) == (
        1,
        0,
        2,
        1,
    )
    assert result.status == "available_with_exclusions"
    assert len(result.information_set["evaluation_population"]["scores"]) == 8


@pytest.mark.parametrize(
    "changes",
    [
        {"label_semantics": "event_inventory_only"},
        {"label_semantics": "unspecified"},
        {"negative_label_definition": None},
    ],
)
def test_invalid_classification_semantics_withhold_before_loading_backend(changes, monkeypatch):
    def forbidden():
        raise AssertionError("Invalid semantics must not reach numerical evaluation")

    monkeypatch.setattr(metrology, "_load_contingent", forbidden)
    result = evaluate(design=design(**changes))
    assert result.status == "withheld" and "semantics" in result.withholding_reason
    assert result.negative_label_count is None
    assert result.threshold_results == () and result.average_precision is None


def test_no_labeled_or_scored_population_is_explicitly_withheld():
    result = evaluate(labels=())
    assert result.status == "withheld" and result.unlabeled_observation_count == 8
    scores, labels = population(values=(None,) * 8)
    result = evaluate(scores, labels)
    assert result.unscored_observation_count == 8 and result.evaluated_observation_count == 0


@requires_nist
@pytest.mark.parametrize(
    "values,truth,expected",
    [
        ((0.0,) * 8, TRUTH, {"mcc": None, "precision": None, "recall": 0.0, "f1": 0.0}),
        ((1.0,) * 8, TRUTH, {"mcc": None, "precision": 0.5, "recall": 1.0}),
        (VALUES, (0,) * 8, {"mcc": None, "precision": 0.0, "recall": None, "f2": 0.0}),
        (
            (0.0,) * 8,
            (0,) * 8,
            {"mcc": None, "precision": None, "recall": None, "f1": None, "f2": None},
        ),
        (VALUES, (1,) * 8, {"mcc": None, "precision": 1.0, "recall": 3 / 8}),
    ],
)
def test_undefined_metrics_remain_none_instead_of_package_fill_conventions(values, truth, expected):
    result = evaluate(*population(values=values, truth=truth))
    assert result.status == "partially_withheld"
    row = result.threshold_results[0]
    for name, value in expected.items():
        assert (
            getattr(row, name) == pytest.approx(value)
            if value is not None
            else getattr(row, name) is None
        )
    if not any(truth):
        assert result.average_precision is None and result.average_precision_withholding_reason
    elif len(set(values)) == 1:
        assert result.average_precision == pytest.approx(sum(truth) / len(truth))


@requires_nist
def test_raw_extreme_scores_ties_and_average_precision_use_no_minmax_transform(monkeypatch):
    from contingency import Contingent

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw scores must not be min-max rescaled")

    monkeypatch.setattr(Contingent, "from_scalar", forbidden)
    scores, labels = population(
        values=(1e308, 1e308, 1e-300, 0, -1e308, -1e308), truth=(1, 0, 1, 0, 1, 0)
    )
    result = evaluate(scores, labels, design=design(thresholds=(0.0,)))
    row = result.threshold_results[0]
    assert (row.true_positives, row.false_positives, row.false_negatives, row.true_negatives) == (
        2,
        2,
        1,
        1,
    )
    assert result.average_precision == pytest.approx((1 / 2 + 2 / 3 + 3 / 6) / 3)


def tuning_population(**kwargs):
    return population(
        period="2020-09-30", as_of="2020-06-30", known="2020-10-15", prefix="tuning", **kwargs
    )


def tuning_design(**kwargs):
    return design(
        thresholds=(0.1, 0.5, 0.9),
        threshold_selection_mode="tuning_derived",
        tuning_as_of="2020-11-01",
        **kwargs,
    )


@requires_nist
def test_tuning_uses_only_earlier_labels_never_holdout_outcomes():
    tuning_scores, tuning_labels = tuning_population()
    cfg = tuning_design()
    first = evaluate(design=cfg, tuning_scores=tuning_scores, tuning_labels=tuning_labels)
    labels = tuple(replace(label, value=1 - label.value) for label in population()[1])
    second = evaluate(
        labels=labels, design=cfg, tuning_scores=tuning_scores, tuning_labels=tuning_labels
    )
    assert first.selected_threshold == second.selected_threshold == 0.9
    assert first.information_set["tuning"] == second.information_set["tuning"]
    assert first.evaluated_thresholds == (0.9,) and first.prospective
    assert first.threshold_results != second.threshold_results
    unknown_holdout = evaluate(
        labels=(), design=cfg, tuning_scores=tuning_scores, tuning_labels=tuning_labels
    )
    assert unknown_holdout.status == "withheld"
    assert unknown_holdout.selected_threshold == first.selected_threshold
    assert unknown_holdout.information_set["tuning"] == first.information_set["tuning"]
    future = tuple(
        replace(
            label, value=1 - label.value, available_as_of="2020-11-01", source="later correction"
        )
        for label in tuning_labels
    )
    assert first == evaluate(
        design=cfg, tuning_scores=tuning_scores, tuning_labels=tuning_labels + future
    )
    # Even a different future definition cannot invalidate past tuning.
    future = tuple(replace(label, definition="future definition") for label in future)
    assert first == evaluate(
        design=cfg, tuning_scores=tuning_scores, tuning_labels=tuning_labels + future
    )


@requires_nist
def test_tuning_ties_and_objective_are_explicit():
    scores, labels = tuning_population(values=(0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1))
    result = evaluate(
        design=tuning_design(tuning_metric="f2"), tuning_scores=scores, tuning_labels=labels
    )
    assert result.selected_threshold == 0.9
    assert (
        result.information_set["tuning"]["tie_rule"] == "highest threshold among exact metric ties"
    )


@pytest.mark.parametrize(
    "kind", ["future_cutoff", "future_score", "overlapping_period", "reused_source"]
)
def test_invalid_tuning_boundaries_rejected(kind):
    scores, labels = tuning_population()
    cfg = tuning_design()
    if kind == "future_cutoff":
        cfg = replace(cfg, tuning_as_of="2020-12-30")
    elif kind == "future_score":
        scores = (replace(scores[0], as_of=cfg.tuning_as_of),) + scores[1:]
    elif kind == "overlapping_period":
        scores = tuple(replace(row, target_period="2021-03-31") for row in scores)
    else:
        scores = (replace(scores[0], source_id=population()[0][0].source_id),) + scores[1:]
    with pytest.raises(ValueError):
        evaluate(design=cfg, tuning_scores=scores, tuning_labels=labels)


def test_missing_tuning_history_withholds_without_fallback():
    result = evaluate(design=tuning_design())
    assert result.selected_threshold is None and result.status == "withheld"
    assert "No permitted" in result.withholding_reason
    scores, labels = tuning_population()
    late = tuple(replace(label, available_as_of="2020-11-01") for label in labels)
    result = evaluate(design=tuning_design(), tuning_scores=scores, tuning_labels=late)
    assert result.status == "withheld" and result.selected_threshold is None


@requires_nist
def test_no_defined_tuning_objective_withholds():
    scores, labels = tuning_population(values=(0.0,) * 8)
    result = evaluate(design=tuning_design(), tuning_scores=scores, tuning_labels=labels)
    assert result.status == "withheld" and "undefined" in result.withholding_reason
    assert result.selected_threshold is None


@requires_nist
def test_same_quarter_after_event_scores_are_not_described_as_prospective():
    scores, labels = population(as_of="2021-04-02")
    result = evaluate(scores, labels)
    assert result.status == "available" and result.pre_specified_threshold == 0.5
    assert result.prospective is False


@requires_nist
def test_exact_memory_limits_withhold_instead_of_approximating():
    ap_limited = evaluate(design=design(maximum_prediction_cells=40))
    assert ap_limited.threshold_results and ap_limited.average_precision is None
    assert ap_limited.status == "partially_withheld"
    assert "no approximation" in ap_limited.average_precision_withholding_reason
    all_limited = evaluate(design=design(maximum_prediction_cells=7))
    assert all_limited.status == "withheld" and all_limited.threshold_results == ()
    assert "no subsampling" in all_limited.withholding_reason


def test_missing_or_unsupported_dependency_surfaces_without_local_fallback(monkeypatch):
    def missing():
        raise ImportError("Deliberately unavailable NIST backend")

    monkeypatch.setattr(metrology, "_load_contingent", missing)
    result = evaluate()
    assert result.status == "withheld" and "dependency unavailable" in result.withholding_reason
    assert result.threshold_results == () and result.average_precision is None
    assert result.positive_label_count == result.negative_label_count == 4


def test_old_python_is_explicitly_unsupported_by_optional_backend(monkeypatch):
    monkeypatch.setattr(metrology.sys, "version_info", (3, 11))
    result = evaluate()
    assert result.status == "withheld" and "Python 3.12 syntax" in result.withholding_reason


def test_unsupported_package_version_is_surfaced(monkeypatch):
    monkeypatch.setattr(metrology.sys, "version_info", (3, 12))
    monkeypatch.setattr(metrology, "_version", lambda _: "999.0")
    assert "reviewed" in evaluate().withholding_reason


@requires_nist
def test_numerical_warnings_and_programming_errors_are_surfaced(monkeypatch):
    def failure(*args):
        raise FloatingPointError("Deliberate backend failure")

    monkeypatch.setattr(metrology, "_threshold_metrics", failure)
    assert "numerical evaluation withheld" in evaluate().withholding_reason

    def bug(*args):
        raise RuntimeError("Deliberate implementation bug")

    monkeypatch.setattr(metrology, "_threshold_metrics", bug)
    with pytest.raises(RuntimeError, match="implementation bug"):
        evaluate()


@requires_nist
def test_result_immutability_and_json_reconstruction():
    result = evaluate()
    with pytest.raises(FrozenInstanceError):
        result.status = "changed"
    result.information_set["evaluation_population"]["scores"][0]["score"] = 1e9
    assert result == evaluate()
    assert "NaN" not in canonical_json(result) and "Infinity" not in canonical_json(result)
    selected = tuple(
        BinaryEventLabel(**label)
        for label in result.information_set["evaluation_population"]["selected_labels"]
    )
    scores = tuple(
        ContingencyScore(**row) for row in result.information_set["evaluation_population"]["scores"]
    )
    assert result == evaluate(scores, selected)


@requires_nist
def test_binary_metrics_remain_separate_from_lead_window_event_detection(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-02-01", "2021-03-01"],
            "score": [1.0, 0.0, 0.0],
            "event": [False, True, False],
        }
    )
    kwargs = dict(
        date_column="date", score_column="score", event_column="event", threshold=0.5, lead_window=1
    )
    before = run_event_backtest(frame, **kwargs)
    import drr_framework.validation_readiness as readiness

    def forbidden(*args, **kwargs):
        raise AssertionError("Contingency must not call lead-window backtesting")

    monkeypatch.setattr(readiness, "run_event_backtest", forbidden)
    scores, labels = population(values=(1.0, 0.0, 0.0), truth=(0, 1, 0))
    result = evaluate(scores, labels)
    assert before["detected_events"] == 1 and before["mean_lead_time_periods"] == 1
    assert result.threshold_results[0].true_positives == 0
    assert result.evidence_class == "BINARY_CLASSIFICATION_METROLOGY"
    assert run_event_backtest(frame, **kwargs) == before


def test_base_import_does_not_load_optional_contingency():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = f"import sys; sys.path.insert(0, {source!r}); import drr_framework.supervisory; assert 'contingency' not in sys.modules; assert 'beartype' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_phase_one_adapter_does_not_change_econometric_scores(var_lab, panel_lab):
    for results, repeat in (
        (run_var(var_lab), lambda: run_var(var_lab)),
        (run_logit(panel_lab), lambda: run_logit(panel_lab)),
    ):
        original = canonical_json(results)
        rows = tuple(
            ContingencyScore.from_challenge(r, target_period=r.review.reporting_period)
            for r in results
        )
        cfg = design(score_family=rows[0].score_family, registered_at="1999-01-01")
        labels = tuple(
            BinaryEventLabel(
                row.institution_id,
                row.target_period,
                i % 2,
                "2099-01-01",
                EVENT,
                "Synthetic explicit adapter fixture",
            )
            for i, row in enumerate(rows)
        )
        report = evaluate_contingency(rows, labels, design=cfg, evaluation_as_of="2099-01-01")
        assert tuple(row.score for row in rows) == tuple(r.score for r in results)
        assert tuple(row.source_id for row in rows) == tuple(r.result_id for r in results)
        assert not report.prospective
        assert canonical_json(results) == original
        assert results == repeat()
        missing = replace(
            results[0], score=None, status="withheld", withholding_reason="Fixture unavailable"
        )
        adapted = ContingencyScore.from_challenge(
            missing, target_period=missing.review.reporting_period
        )
        assert adapted.score is None and adapted.withholding_reason == missing.withholding_reason


@pytest.mark.parametrize(
    "changes",
    [
        {"thresholds": ()},
        {"thresholds": (0.5, 0.5)},
        {"thresholds": (float("nan"),)},
        {"thresholds": (10**400,)},
        {"thresholds": (True,)},
        {"thresholds": "0.5"},
        {"thresholds": (0.2, 0.8)},
        {"maximum_prediction_cells": True},
        {"threshold_selection_mode": "optimized_holdout"},
        {"threshold_selection_mode": []},
        {"tuning_metric": "arbitrary"},
        {"tuning_as_of": "2020-01-01"},
    ],
)
def test_invalid_configuration_raises(changes):
    with pytest.raises(ValueError):
        design(**changes)


def test_invalid_inputs_and_retrospective_registration_cannot_be_mislabeled():
    scores, labels = population()
    for change in ({"score": float("inf")}, {"score": True}, {"target_period": "2021-03-30"}):
        with pytest.raises(ValueError):
            replace(scores[0], **change)
    with pytest.raises(ValueError, match="registered"):
        evaluate(design=design(registered_at="2021-04-01"))
    with pytest.raises(ValueError, match="one score"):
        evaluate(scores + scores[:1], labels)
    with pytest.raises(ValueError, match="family"):
        evaluate((replace(scores[0], score_family="another-model"),) + scores[1:], labels)
    with pytest.raises(ValueError, match="Conflicting"):
        evaluate(scores, labels + (replace(labels[0], source="conflict"),))
    with pytest.raises(ValueError, match="definition"):
        evaluate(scores, (replace(labels[0], definition="different target"),) + labels[1:])
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(evaluation_as_of="2020-01-01")
