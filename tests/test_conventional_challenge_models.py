"""Synthetic implementation checks, not evidence of supervisory usefulness."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from test_regulatory_foundation import definition, observation
from drr_framework.datasets import RegulatoryAnalysisDataset, SupervisoryPanelDataset
from drr_framework.finance.validation.statistics import benjamini_hochberg_fdr
from drr_framework.supervisory import (
    BinaryEventLabel,
    ChallengeEvaluationDesign,
    ChallengeReview,
    LaggedFeature,
    PanelLogitConfig,
    VARChallengeConfig,
    evaluate_challenge_results,
    walk_forward_panel_logit,
    walk_forward_var,
)
from drr_framework.supervisory.baselines import run_baselines
from drr_framework.supervisory.common import canonical_json
from drr_framework.supervisory.policy_context import PolicyContext, PolicyEvent
from drr_framework.supervisory.semantics import SemanticRegistry
from drr_framework.supervisory.vintage import (
    CalculationLineage,
    ObservationProvenance,
    VintageStore,
)
import drr_framework.supervisory.challenge_models as challenges


VARIABLES = ("SYN_X", "SYN_Y")
EVENT = "Synthetic quarter event; zeros explicitly sampled as non-events"


def calendar(n):
    return tuple(
        p.end_time.date().isoformat() for p in pd.period_range("2001Q1", periods=n, freq="Q")
    )


def record(date, value, metric="SYN_X", institution="A", **kwargs):
    filing = (pd.Timestamp(date) + pd.Timedelta(days=1)).isoformat()
    return observation(
        date,
        value,
        metric=metric,
        institution_id=institution,
        original_filing_date=filing,
        ingestion_date=filing,
        available_as_of=filing,
        perimeter_version="synthetic-consolidated-v1",
        **kwargs,
    )


def review(date):
    return ChallengeReview((pd.Timestamp(date) + pd.Timedelta(days=2)).isoformat(), date)


@pytest.fixture(scope="module")
def var_lab():
    rng = np.random.default_rng(407)
    values = np.zeros((260, 2))
    transition = np.array([[0.5, 0.0], [0.85, 0.2]])
    for t in range(1, len(values)):
        values[t] = transition @ values[t - 1] + rng.normal(0, 0.4, 2)
    dates = calendar(len(values))
    records = tuple(
        record(d, values[t, j], metric)
        for t, d in enumerate(dates)
        for j, metric in enumerate(VARIABLES)
    )
    registry = SemanticRegistry(tuple(definition(v) for v in VARIABLES))
    config = VARChallengeConfig("A", "FR Y-9C", VARIABLES, "2000-01-01", training_periods=210)
    return VintageStore(records), registry, config, dates, values


def run_var(lab, **kwargs):
    store, registry, config, dates, _ = lab
    args = dict(config=config, reviews=[review(dates[-1])], allow_synthetic=True)
    args.update(kwargs)
    return walk_forward_var(store, registry, **args)


def test_var_recovers_known_synthetic_structure_and_reconstructible_forecast(var_lab):
    result = run_var(var_lab)[0]
    assert result.status == "available", result.withholding_reason
    fitted = result.fit
    scale = np.asarray(fitted["normalization_scale"])
    # statsmodels rows are predictors and columns are responses; convert back to raw units.
    beta = np.asarray(fitted["coefficients"])[1:]
    transition = beta.T * scale[:, None] / scale[None, :]
    np.testing.assert_allclose(transition, [[0.5, 0], [0.85, 0.2]], atol=0.15)
    last = np.asarray(fitted["forecast_lag_values"])[-1]
    center = np.asarray(fitted["normalization_mean"])
    forecast = np.r_[1, (last - center) / scale] @ fitted["coefficients"]
    np.testing.assert_allclose(forecast * scale + center, fitted["forecast"])
    expected = np.sqrt(
        np.mean((np.asarray(fitted["forecast_error"]) / fitted["training_residual_sd"]) ** 2)
    )
    assert result.score == pytest.approx(expected)
    assert fitted["residual_df"] == 207
    assert result.information_set["training_end"] < result.review.reporting_period
    assert not (
        set(result.information_set["training_source_ids"])
        & set(result.information_set["scoring_source_ids"])
    )
    assert all(
        o["available_as_of"] < result.review.as_of
        for o in result.information_set["source_observations"]
    )


def test_var_current_does_not_calibrate_fit_or_normalization(var_lab):
    store, registry, config, dates, values = var_lab
    base = run_var(var_lab)[0]
    changed = VintageStore(
        tuple(
            replace(o, value=o.value + 30) if o.reporting_period == dates[-1] else o
            for o in store.observations
        )
    )
    other = run_var((changed, registry, config, dates, values))[0]
    assert base.fit["coefficients"] == other.fit["coefficients"]
    assert base.fit["normalization_mean"] == other.fit["normalization_mean"]
    assert base.fit["forecast"] == other.fit["forecast"]
    assert other.score > base.score + 10


def test_var_walk_forward_ignores_future_filings_revisions_and_peers(var_lab):
    store, registry, config, dates, values = var_lab
    reviews = [review(d) for d in dates[-4:-1]]
    original = run_var(var_lab, reviews=reviews)
    latest = reviews[-1].as_of
    truncated = VintageStore(tuple(o for o in store.observations if o.available_as_of < latest))
    assert original == run_var((truncated, registry, config, dates, values), reviews=reviews)
    old = store.observations[20]
    future_date = (pd.Timestamp(latest) + pd.Timedelta(days=365)).isoformat()
    amended = replace(
        old,
        value=1e8,
        amendment_date=future_date,
        available_as_of=future_date,
        ingestion_date=future_date,
        source_vintage="future",
        supersedes=old.observation_id,
    )
    peer = replace(amended, institution_id="FUTURE_PEER", supersedes=None)
    changed = store.append(amended, peer)
    assert original == run_var((changed, registry, config, dates, values), reviews=reviews)


@pytest.mark.parametrize(
    "kind", ["short", "null", "perimeter", "definition", "unidentified", "condition", "residual"]
)
def test_var_withholds_instead_of_repairing_window_or_variables(var_lab, kind):
    store, registry, config, dates, values = var_lab
    records = list(store.observations)
    if kind == "short":
        records = [o for o in records if o.reporting_period >= dates[-10]]
    elif kind == "condition":
        config = replace(config, max_condition_number=1.0001)
    elif kind == "unidentified":
        lookup = {(o.reporting_period, o.metric): o for o in records}
        records = [replace(o, value=lookup[(o.reporting_period, "SYN_X")].value) for o in records]
    elif kind == "residual":
        lookup = {o.reporting_period: o.value for o in records if o.metric == "SYN_X"}
        # A lagged duplicate gives a deterministic equation, despite full design rank.
        prior = dict(zip(dates[1:], dates[:-1]))
        records = [
            (
                replace(o, value=lookup[prior[o.reporting_period]])
                if o.metric == "SYN_Y" and o.reporting_period in prior
                else o
            )
            for o in records
        ]
    else:
        index = next(i for i, o in enumerate(records) if o.reporting_period == dates[-5])
        change = {
            "null": {"value": None},
            "perimeter": {"perimeter_version": "changed"},
            "definition": {"definition_version": "v2"},
        }[kind]
        records[index] = replace(records[index], **change)
        if kind == "definition":
            registry = SemanticRegistry(
                registry.definitions + (definition(records[index].metric, version="v2"),)
            )
    result = run_var((VintageStore(records), registry, config, dates, values))[0]
    assert result.status == "withheld" and result.score is None and result.flagged is None
    assert result.withholding_reason
    assert result.config.variables == VARIABLES
    assert len(result.information_set["training_periods"]) == config.training_periods


def policy(date, known="2000-01-02"):
    return PolicyContext(
        (
            PolicyEvent(
                "synthetic-break",
                "synthetic",
                "Fixture break",
                "2000-01-01",
                date,
                "fixture",
                (),
                ("FR Y-9C",),
                VARIABLES,
                "reporting",
                "https://example.com/fixture",
                known,
                comparability_break=True,
            ),
        )
    )


def test_policy_break_is_as_of_and_does_not_reset_window(var_lab):
    dates = var_lab[3]
    assert run_var(var_lab, policy_context=policy(dates[-5]))[0].status == "withheld"
    assert run_var(var_lab, policy_context=policy(dates[0]))[0].status == "available"
    future = (pd.Timestamp(review(dates[-1]).as_of) + pd.Timedelta(days=1)).isoformat()
    assert run_var(var_lab) == run_var(var_lab, policy_context=policy(dates[-5], future))


@pytest.fixture(scope="module")
def panel_lab():
    rng = np.random.default_rng(206)
    dates = calendar(72)
    institutions = tuple(f"SYN_FIRM_{i}" for i in range(8))
    records, labels = [], []
    for institution in institutions:
        x = rng.normal(size=(len(dates), 2))
        for t, date in enumerate(dates):
            for j, metric in enumerate(VARIABLES):
                records.append(record(date, x[t, j], metric, institution))
            if t:
                value = int(rng.random() < expit(-0.6 + 1.8 * x[t - 1, 0]))
                labels.append(
                    BinaryEventLabel(
                        institution,
                        date,
                        value,
                        (pd.Timestamp(date) + pd.Timedelta(days=1)).isoformat(),
                        EVENT,
                        "synthetic seeded Bernoulli draw",
                        date if value else None,
                    )
                )
    registry = SemanticRegistry(tuple(definition(v) for v in VARIABLES))
    config = PanelLogitConfig(
        institutions,
        "FR Y-9C",
        tuple(LaggedFeature(v) for v in VARIABLES),
        EVENT,
        "2000-01-01",
        training_periods=55,
    )
    return VintageStore(records), registry, config, dates, tuple(labels)


def run_logit(lab, **kwargs):
    store, registry, config, dates, labels = lab
    args = dict(config=config, reviews=[review(dates[-1])], labels=labels, allow_synthetic=True)
    args.update(kwargs)
    return walk_forward_panel_logit(store, registry, **args)


def test_logit_recovers_precursor_from_raw_panel_and_scores_holdout(panel_lab, monkeypatch):
    def no_retrospective_adapter(*args, **kwargs):
        raise AssertionError("retrospective cleaning is not allowed")

    monkeypatch.setattr(SupervisoryPanelDataset, "from_frame", no_retrospective_adapter)
    results = run_logit(panel_lab)
    assert all(r.status == "available" for r in results), results[0].withholding_reason
    fit = results[0].fit
    raw_slopes = np.asarray(fit["coefficients"])[1:] / fit["normalization_scale"]
    np.testing.assert_allclose(raw_slopes, [1.8, 0], atol=0.4)
    assert fit["converged"] is True and fit["usable_institution_count"] == 8
    assert fit["usable_observation_count"] == 8 * 55
    assert fit["event_count"] + fit["nonevent_count"] == 440
    x = np.asarray(fit["scoring_predictors"])
    design = np.column_stack(
        (np.ones(len(x)), (x - fit["normalization_mean"]) / fit["normalization_scale"])
    )
    np.testing.assert_allclose(expit(design @ fit["coefficients"]), [r.score for r in results])
    assert np.ptp([r.score for r in results]) > 0.5
    for label in results[0].information_set["training_labels"]:
        assert label["reporting_period"] < results[0].review.reporting_period
        assert label["available_as_of"] < results[0].review.as_of


def test_panel_walk_forward_future_normalization_peers_labels_revisions_do_not_leak(panel_lab):
    store, registry, config, dates, labels = panel_lab
    reviews = [review(d) for d in dates[-3:-1]]
    expected = run_logit(panel_lab, reviews=reviews)
    cutoff = reviews[-1].as_of
    trimmed_store = VintageStore(tuple(o for o in store.observations if o.available_as_of < cutoff))
    trimmed_labels = tuple(label for label in labels if label.available_as_of < cutoff)
    assert expected == run_logit(
        (trimmed_store, registry, config, dates, trimmed_labels), reviews=reviews
    )
    old = store.observations[20]
    later = (pd.Timestamp(cutoff) + pd.Timedelta(days=365)).isoformat()
    new_record = replace(
        old,
        value=1e9,
        amendment_date=later,
        ingestion_date=later,
        available_as_of=later,
        source_vintage="revision",
        supersedes=old.observation_id,
    )
    new_peer = replace(new_record, institution_id="OUTSIDE_REGISTERED_SCOPE", supersedes=None)
    revised_label = replace(
        labels[100],
        value=1 - labels[100].value,
        available_as_of=later,
        event_at=None,
        source="later label correction",
    )
    assert expected == run_logit(
        (store.append(new_record, new_peer), registry, config, dates, labels + (revised_label,)),
        reviews=reviews,
    )
    # Even a future label-definition change cannot invalidate an earlier score.
    revised_definition = replace(revised_label, definition="later outcome definition")
    assert expected == run_logit(
        (store, registry, config, dates, labels + (revised_definition,)), reviews=reviews
    )


def test_current_outcomes_and_current_features_do_not_train_logit(panel_lab):
    store, registry, config, dates, labels = panel_lab
    first = run_logit(panel_lab)
    changed_store = VintageStore(
        tuple(
            replace(o, value=o.value * 1e6) if o.reporting_period == dates[-1] else o
            for o in store.observations
        )
    )
    changed_labels = tuple(
        (
            replace(label, value=1 - label.value, event_at=None)
            if label.reporting_period == dates[-1]
            else label
        )
        for label in labels
    )
    second = run_logit((changed_store, registry, config, dates, changed_labels))
    assert [r.score for r in first] == [r.score for r in second]
    assert first[0].fit == second[0].fit


def test_logit_injected_precursor_is_recovered_on_walk_forward_holdouts(panel_lab):
    dates, labels = panel_lab[3:]
    results = run_logit(panel_lab, reviews=[review(date) for date in dates[-6:]])
    truth = {(label.institution_id, label.reporting_period): label.value for label in labels}
    outcomes = np.asarray([truth[(r.institution_id, r.review.reporting_period)] for r in results])
    probabilities = np.asarray([r.score for r in results])
    prevalences = np.asarray([r.fit["event_fraction"] for r in results])
    assert all(r.status == "available" for r in results)
    assert np.mean((probabilities - outcomes) ** 2) < np.mean((prevalences - outcomes) ** 2)


@pytest.mark.parametrize("kind", ["definition", "perimeter", "current_break"])
def test_panel_withholds_incomparable_pooling_and_current_metadata(panel_lab, kind):
    store, registry, config, dates, labels = panel_lab
    institution = config.institutions[-1]
    if kind == "definition":
        registry = SemanticRegistry(
            registry.definitions + tuple(definition(v, version="v2") for v in VARIABLES)
        )
    changed = []
    for fact in store.observations:
        if fact.institution_id == institution:
            if kind == "definition":
                fact = replace(fact, definition_version="v2")
            elif kind == "perimeter" or fact.reporting_period == dates[-1]:
                fact = replace(fact, perimeter_version="different-scope")
        changed.append(fact)
    results = run_logit((VintageStore(changed), registry, config, dates, labels))
    assert all(result.status == "withheld" for result in results)
    assert "break" in results[0].withholding_reason.lower()


def test_panel_separation_and_unexpected_errors_are_surfaced(panel_lab, monkeypatch):
    store, registry, config, dates, labels = panel_lab
    lookup = {
        (fact.institution_id, fact.reporting_period): fact.value
        for fact in store.observations
        if fact.metric == "SYN_X"
    }
    dates_before = dict(zip(dates[1:], dates[:-1]))
    separated = tuple(
        replace(
            label,
            value=int(lookup[(label.institution_id, dates_before[label.reporting_period])] > 0),
            event_at=None,
        )
        for label in labels
    )
    results = run_logit(panel_lab, labels=separated)
    assert all(result.status == "withheld" for result in results)
    assert results[0].fit["fit_attempted"]
    assert results[0].withholding_reason

    def programming_error(*args):
        raise RuntimeError("deliberate implementation error")

    monkeypatch.setattr(challenges, "_fit_logit", programming_error)
    with pytest.raises(RuntimeError, match="implementation error"):
        run_logit(panel_lab)


def test_unknown_or_not_yet_available_labels_are_never_negative(panel_lab):
    labels = panel_lab[-1]
    only_positives = tuple(label for label in labels if label.value == 1)
    result = run_logit(panel_lab, labels=only_positives)[0]
    assert result.status == "withheld"
    assert result.fit["nonevent_count"] == 0 and result.fit["unlabeled_observation_count"] > 0
    assert result.fit["event_count"] < result.fit["configured_training_rows"]
    future = (pd.Timestamp(result.review.as_of) + pd.Timedelta(days=30)).isoformat()
    late_negatives = tuple(
        replace(label, available_as_of=future) for label in labels if not label.value
    )
    assert run_logit(panel_lab, labels=only_positives + late_negatives) == run_logit(
        panel_lab, labels=only_positives
    )
    unknown = tuple(
        replace(label, value=None, event_at=None) for label in labels if not label.value
    )
    assert run_logit(panel_lab, labels=only_positives + unknown)[0].fit["nonevent_count"] == 0


@pytest.mark.parametrize(
    "kind", ["few_events", "nonconvergence", "missing", "perimeter", "collinear", "short"]
)
def test_logit_limitations_are_surfaced(panel_lab, kind):
    store, registry, config, dates, labels = panel_lab
    if kind == "few_events":
        labels = tuple(
            replace(label, value=0, event_at=None) if j > 2 else label
            for j, label in enumerate(labels)
        )
    elif kind == "nonconvergence":
        config = replace(config, max_iterations=1)
    elif kind == "short":
        store = VintageStore(
            tuple(o for o in store.observations if o.reporting_period >= dates[-5])
        )
    elif kind == "collinear":
        lookup = {
            (o.institution_id, o.reporting_period): o.value
            for o in store.observations
            if o.metric == "SYN_X"
        }
        store = VintageStore(
            tuple(
                replace(o, value=lookup[(o.institution_id, o.reporting_period)])
                for o in store.observations
            )
        )
    else:
        chosen = next(o for o in store.observations if o.reporting_period == dates[-2])
        changes = {"value": None} if kind == "missing" else {"perimeter_version": "changed"}
        store = VintageStore(
            tuple(replace(o, **changes) if o == chosen else o for o in store.observations)
        )
    results = run_logit((store, registry, config, dates, labels))
    assert len(results) == len(config.institutions)
    assert all(r.status == "withheld" and r.score is None for r in results)
    assert results[0].withholding_reason
    if kind == "nonconvergence":
        assert results[0].fit["converged"] is False
        assert "converge" in results[0].withholding_reason


def test_dependency_and_solver_failure_are_explicit(var_lab, panel_lab, monkeypatch):
    def missing():
        raise ImportError("deliberately absent statsmodels")

    monkeypatch.setattr(challenges, "_load_statsmodels", missing)
    for result in (run_var(var_lab)[0], run_logit(panel_lab)[0]):
        assert result.status == "withheld" and "dependency unavailable" in result.withholding_reason


def test_determinism_immutability_and_baseline_api_preservation(var_lab, panel_lab):
    store, registry, config, dates, _ = var_lab
    dataset = RegulatoryAnalysisDataset.from_vintage_store(
        store,
        registry,
        institution_id="A",
        form=config.form,
        metrics=config.variables,
        as_of=review(dates[-1]).as_of,
        allow_synthetic=True,
    )
    original = run_baselines(dataset)
    first = run_var(var_lab)[0]
    assert run_var(var_lab)[0] == first
    assert run_logit(panel_lab) == run_logit(panel_lab)
    assert original == run_baselines(dataset)
    assert len(original) == 14 and all(r.training_count == 12 for r in original)
    with pytest.raises(FrozenInstanceError):
        first.status = "changed"
    first.fit["coefficients"][0][0] = 10000
    assert first == run_var(var_lab)[0]
    assert "NaN" not in canonical_json(first) and "Infinity" not in canonical_json(first)


def test_valid_prespecified_test_family_uses_existing_bh_implementation():
    significant, adjusted = benjamini_hochberg_fdr([0.01, 0.04, 0.03, 0.002], alpha=0.025)
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])
    assert significant.tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    "change",
    [
        {"lag_order": 0},
        {"training_periods": True},
        {"max_condition_number": float("inf")},
        {"warning_threshold": float("nan")},
        {"variables": ("SYN_X", "SYN_X")},
    ],
)
def test_invalid_var_configuration_raises(var_lab, change):
    with pytest.raises(ValueError):
        replace(var_lab[2], **change)


def test_registration_and_label_schema_are_checked(var_lab, panel_lab):
    config = replace(var_lab[2], registered_at=review(var_lab[3][-1]).as_of)
    with pytest.raises(ValueError, match="registered"):
        run_var(var_lab, config=config)
    with pytest.raises(ValueError):
        LaggedFeature("SYN_X", 0)
    with pytest.raises(ValueError):
        replace(panel_lab[2], warning_threshold=1.1)
    with pytest.raises(ValueError, match="Conflicting"):
        run_logit(
            panel_lab, labels=panel_lab[-1] + (replace(panel_lab[-1][10], source="conflict"),)
        )
    with pytest.raises(ValueError):
        replace(panel_lab[-1][0], value=float("nan"))


def evaluation_fixture(var_lab, flags):
    base = run_var(var_lab)[0]
    dates = calendar(len(flags))
    rows = tuple(
        (
            replace(
                base,
                review=review(d),
                score=4.0 if flag else 0.0,
                status="available",
                withholding_reason=None,
            )
            if flag is not None
            else replace(
                base,
                review=review(d),
                score=None,
                status="withheld",
                withholding_reason="fixture gap",
            )
        )
        for d, flag in zip(dates, flags)
    )
    labels = tuple(
        BinaryEventLabel(
            "A", d, int(i in (1, 3)), "2010-01-01", EVENT, "fixture", d if i in (1, 3) else None
        )
        for i, d in enumerate(dates)
    )
    design = ChallengeEvaluationDesign(EVENT, "2000-01-01", lead_window=1, minimum_events=2)
    return rows, labels, design


def test_detection_counts_timing_censoring_and_disagreement(var_lab):
    rows, labels, design = evaluation_fixture(var_lab, [True, False, True, False, False, True])
    drr = [
        dict(
            institution_id=r.institution_id,
            as_of=r.review.as_of,
            input_hash=r.information_set["input_hash"],
            drr_alert=False,
        )
        for r in rows
    ]
    report = evaluate_challenge_results(
        rows, labels, design=design, evaluation_as_of="2010-01-01", drr_rows=drr
    )
    metrics = report["by_institution"]["A"]
    assert metrics["status"] == "available"
    assert metrics["event_count"] == metrics["detected_events"] == 2
    assert metrics["missed_events"] == 0 and metrics["lead_time_periods"] == (1, 1)
    assert metrics["right_censored_alerts"] == 1 and metrics["indeterminate_alerts"] == 1
    assert metrics["precision"] is None
    assert metrics["drr_detection"]["missed_events"] == 2
    assert len(metrics["disagreement_reviews"]) == 3
    assert metrics["matched_evaluation"]["baseline_alert_count"] == 3
    late_rows, _, _ = evaluation_fixture(var_lab, [False, True, False, True, False, False])
    late = evaluate_challenge_results(
        late_rows, labels, design=design, evaluation_as_of="2010-01-01"
    )
    assert late["by_institution"]["A"]["detected_events"] == 0


def test_evaluation_unknown_sparse_and_unscored_results_remain_explicit(var_lab):
    rows, labels, design = evaluation_fixture(var_lab, [None, None, False, True, True, False])
    sparse = evaluate_challenge_results(
        rows, labels, design=replace(design, minimum_events=3), evaluation_as_of="2010-01-01"
    )["by_institution"]["A"]
    assert sparse["status"] == "withheld" and sparse["detection_rate"] is None
    assert sparse["event_count"] == 2 and sparse["unscored_event_count"] == 1
    assert sparse["events_without_scored_lead_window"] == 1
    unknown = evaluate_challenge_results(
        rows, [labels[1], labels[3]], design=design, evaluation_as_of="2010-01-01"
    )["by_institution"]["A"]
    assert unknown["unknown_outcome_count"] == 4
    assert unknown["false_positive_alerts"] is None and unknown["indeterminate_alerts"] == 2
    assert unknown["confirmed_false_positive_alerts"] == 0
    untimed = tuple(replace(label, event_at=None) for label in labels)
    report = evaluate_challenge_results(rows, untimed, design=design, evaluation_as_of="2010-01-01")
    assert report["by_institution"]["A"]["detected_events"] is None


def test_evaluation_rejects_mismatched_grids_hashes_and_future_thresholds(var_lab):
    rows, labels, design = evaluation_fixture(var_lab, [True, False, True, False])
    with pytest.raises(ValueError, match="every calendar quarter"):
        evaluate_challenge_results(rows[1::2], labels, design=design, evaluation_as_of="2010-01-01")
    drr = [
        dict(institution_id="A", as_of=r.review.as_of, input_hash="wrong", drr_alert=False)
        for r in rows
    ]
    with pytest.raises(ValueError, match="information sets"):
        evaluate_challenge_results(
            rows, labels, design=design, evaluation_as_of="2010-01-01", drr_rows=drr
        )
    changed = rows[:-1] + (replace(rows[-1], config=replace(rows[-1].config, warning_threshold=2)),)
    with pytest.raises(ValueError, match="fixed"):
        evaluate_challenge_results(changed, labels, design=design, evaluation_as_of="2010-01-01")


@pytest.mark.parametrize(
    "change", [{"definition_version": "unregistered-legacy"}, {"unit": "legacy-unit"}]
)
def test_snapshot_validates_only_the_requested_window(var_lab, panel_lab, change):
    for lab, runner in ((var_lab, run_var), (panel_lab, run_logit)):
        store, registry, config, dates, rest = lab
        expected = runner(lab)
        old = store.observations[0]
        assert old.reporting_period < expected[0].information_set["lag_history_start"]
        changed = VintageStore(
            tuple(replace(o, **change) if o == old else o for o in store.observations)
        )
        assert runner((changed, registry, config, dates, rest)) == expected
        # The same defect on an input inside the window must still withhold.
        current = next(o for o in store.observations if o.reporting_period == dates[-2])
        changed = VintageStore(
            tuple(replace(o, **change) if o == current else o for o in store.observations)
        )
        results = runner((changed, registry, config, dates, rest))
        assert all(r.status == "withheld" and r.score is None for r in results)


def test_windowed_snapshot_preserves_outside_window_lineage(var_lab):
    store, registry, config, dates, values = var_lab
    parent = store.observations[0]
    child = next(o for o in store.observations if o.reporting_period == dates[-5])
    lineage = CalculationLineage(
        "synthetic lineage fixture",
        (parent.observation_id,),
        (parent.metric,),
        (parent.reporting_period,),
        "fixture-v1",
        (),
        child.unit,
        child.available_as_of,
    )
    derived = replace(child, provenance=ObservationProvenance.DERIVED_COMPUTED, lineage=lineage)
    changed = VintageStore(tuple(derived if o == child else o for o in store.observations))
    result = run_var((changed, registry, config, dates, values))[0]
    assert result.status == "available", result.withholding_reason
    assert result.score == run_var(var_lab)[0].score
    assert parent.observation_id not in result.information_set["source_ids"]
    assert derived.observation_id in result.information_set["source_ids"]
    assert any(
        o["lineage"] and parent.observation_id in o["lineage"]["input_ids"]
        for o in result.information_set["source_observations"]
    )


@pytest.mark.parametrize("kind", ["early_x", "late_y", "required_x", "required_y"])
@pytest.mark.parametrize("missing_kind", ["null", "absent"])
def test_mixed_lags_ignore_unused_padding_but_preserve_required_missingness(
    panel_lab, kind, missing_kind
):
    store, registry, config, dates, labels = panel_lab
    config = replace(config, features=(LaggedFeature("SYN_X", 1), LaggedFeature("SYN_Y", 4)))
    lab = (store, registry, config, dates, labels)
    expected = run_logit(lab)
    assert all(r.status == "available" for r in expected)
    target = pd.Period(dates[-1], freq="Q")
    earliest = (target - config.training_periods - 4).end_time.date().isoformat()
    metric, period = {
        "early_x": ("SYN_X", earliest),
        "late_y": ("SYN_Y", dates[-2]),
        "required_x": ("SYN_X", dates[-2]),
        "required_y": ("SYN_Y", earliest),
    }[kind]
    chosen = next(
        o for o in store.observations if o.metric == metric and o.reporting_period == period
    )
    records = tuple(
        replace(o, value=None) if o == chosen else o
        for o in store.observations
        if missing_kind != "absent" or o != chosen
    )
    results = run_logit((VintageStore(records), registry, config, dates, labels))
    fit = results[0].fit
    if kind.startswith("required"):
        assert all(r.status == "withheld" and r.score is None for r in results)
        assert fit["missing_cells"] == fit["missing_cells_by_metric"][metric] == 1
    else:
        assert all(r.status == "available" for r in results)
        assert [r.score for r in results] == [r.score for r in expected]
        assert fit["coefficients"] == expected[0].fit["coefficients"]
        assert fit["missing_cells"] == 0 and fit["unused_missing_cells"] == 1
