"""Typed judgment stays outside DRR mathematics, evidence, and attention ranking."""

from __future__ import annotations

import inspect
import json
import os
import socket
import sqlite3

import pytest

from test_lfbo_validation import history
from test_regulatory_foundation import observation
from drr_framework.analysis import DynamicResonanceRooting
from drr_framework.supervisory import verify_monitoring_snapshot
from drr_framework.supervisory.backtesting import walk_forward_validate
from drr_framework.supervisory.common import canonical, canonical_json, stable_id
from drr_framework.supervisory.demo import CURRENT_REVIEW, PREVIOUS_REVIEW, synthetic_monitoring_lab
from drr_framework.supervisory.evidence_ledger import AnalystReview, EvidenceEntry, EvidenceLedger
from drr_framework.supervisory.judgment import (
    EVIDENCE_PACKET_QUESTIONS,
    POST_HOC_JUDGMENT_EVALUATION,
    QUESTION_SET_VERSION,
    DeterministicMockJudgmentProvider,
    DisabledJudgmentProvider,
    JudgmentResponseError,
    JudgmentResult,
    TypeSafeJudgmentProvider,
    apply_judgment_policy,
    assemble_overlay,
    build_judgment_state,
    describe_post_hoc_judgment,
    make_result,
    parse_system_one_response,
    parse_typed_answer,
    replay_recorded_judgments,
    typesafe_model_risk_extension,
)
from drr_framework.supervisory.monitoring import AttentionBudget
from drr_framework.supervisory.semantics import SemanticRegistry
from drr_framework.supervisory.ui import render_workbench
from drr_framework.supervisory.vintage import VintageStore
from drr_framework.supervisory.workbench import (
    DEFAULT_MODEL_RISK_PROFILE,
    MonitoringWorkbench,
    WorkbenchConfig,
)
from test_regulatory_foundation import definition


SECRET = "unit-test-secret-value"
WHEN = "2026-01-15T00:00:00+00:00"


def packet(**overrides):
    observed = observation()
    body = dict(
        claim="Synthetic assets changed",
        institution="A",
        metric="SYN_ASSETS",
        period=observed.reporting_period,
        source_facts=[dict(observation_id=observed.observation_id, **canonical(observed))],
        calculation={
            "formula": "current-prior",
            "software_version": "test",
            "input_ids": [observed.observation_id],
            "parameters": {"lookback": 12},
        },
        historical_context={"window": "8 quarters"},
        peer_context={"status": "not evaluated"},
        baseline_evidence=[{"method": "robust_z", "flagged": True}],
        drr_evidence={"status": "disabled", "incremental_alert": False},
        robustness={"classification": "mixed", "percentage_surviving": 50},
        contradictory_evidence=["Baselines disagree with one window."],
        policy_context=[],
        data_limitations=["Synthetic demonstration only."],
    )
    body.update(overrides)
    return EvidenceEntry.create(**body)


def choice(question_id, selected, probabilities, confidence):
    from drr_framework.supervisory.judgment import TypedAnswer

    return TypedAnswer(
        question_id,
        "choice",
        choice=selected,
        probabilities=tuple(sorted(probabilities.items())),
        confidence=confidence,
    )


def noul(question_id, probability):
    from drr_framework.supervisory.judgment import TypedAnswer

    return TypedAnswer(question_id, "noul", noul=probability)


def completed_result(
    state, *, adequacy="adequate", complexity="moderate", nouls=None, model_version="mock-v1"
):
    values = {
        "scope_overreach": 0.12,
        "contradiction_material": 0.2,
        "limitations_material": 0.2,
        "additional_review_needed": 0.2,
    }
    values.update(nouls or {})
    answers = (
        choice(
            "evidence_adequacy",
            adequacy,
            {"adequate": 0.7, "limited": 0.2, "insufficient": 0.1},
            0.62,
        ),
        noul("scope_overreach", values["scope_overreach"]),
        noul("contradiction_material", values["contradiction_material"]),
        choice(
            "review_complexity",
            complexity,
            {"routine": 0.2, "moderate": 0.6, "complex": 0.2},
            0.4,
        ),
        noul("limitations_material", values["limitations_material"]),
        noul("additional_review_needed", values["additional_review_needed"]),
    )
    return make_result(
        state,
        provider="mock",
        model="deterministic-mock",
        provider_model_version=model_version,
        status="completed",
        answers=answers,
        requested_at=WHEN,
        completed_at=WHEN,
        latency_ms=0.0,
    )


def sample_state():
    return build_judgment_state(
        packet(), analysis_id="analysis-1", as_of="2025-09-01T00:00:00+00:00"
    )


def deny_network(*_args, **_kwargs):
    raise AssertionError("unexpected network call")


class RecordingClient:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.calls = []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.payload

    def close(self):
        pass


def system_one_payload(**answers):
    body = {
        "evidence_adequacy": {
            "type": "choice",
            "choice": "adequate",
            "probabilities": {"adequate": 0.8, "limited": 0.15, "insufficient": 0.05},
            "confidence": 0.7,
        },
        "scope_overreach": {"type": "noul", "noul": 0.12},
        "contradiction_material": {"type": "noul", "noul": 0.34},
        "review_complexity": {
            "type": "choice",
            "choice": "moderate",
            "probabilities": {"routine": 0.15, "moderate": 0.75, "complex": 0.1},
            "confidence": 0.66,
        },
        "limitations_material": {"type": "noul", "noul": 0.67},
        "additional_review_needed": {"type": "noul", "noul": 0.72},
    }
    body.update(answers)
    return {"model": "jev-1.13.0", "answers": body}


def run_pair(tmp_path, *, enabled, provider="mock", judgment_provider=None, name="run"):
    store, registry, cohort, policy, entities = synthetic_monitoring_lab()
    ledger = EvidenceLedger(tmp_path / f"{name}.sqlite")
    config = WorkbenchConfig(
        allow_synthetic=True,
        enable_drr=False,
        judgment_enabled=enabled,
        judgment_provider=provider,
    )
    workbench = MonitoringWorkbench(
        store,
        registry,
        ledger,
        cohort=cohort,
        policy=policy,
        entities=entities,
        config=config,
        judgment_provider=judgment_provider,
    )
    _, previous, _ = workbench.run(PREVIOUS_REVIEW)
    result, state, passport = workbench.run(CURRENT_REVIEW, previous=previous)
    return result, state, passport, workbench


def test_question_set_is_atomic_and_does_not_judge_the_institution():
    texts = [question.instructions["question"] for question in EVIDENCE_PACKET_QUESTIONS]
    assert texts == [
        "Given only the supplied evidence packet, is there enough evidence to meaningfully evaluate the stated claim?",
        "The stated claim materially goes beyond what the supplied evidence establishes.",
        "The supplied contradictory evidence is materially important to interpreting this claim.",
        "How complex is this evidence packet for a human analyst to review?",
        "The documented limitations materially constrain interpretation of this claim.",
        "The evidence contains enough ambiguity, disagreement, limitation, or uncertainty that additional human scrutiny is warranted.",
    ]
    blob = " ".join(texts).casefold()
    for banned in ("well managed", "poorly managed", "high risk", "supervisory action", "mra"):
        assert banned not in blob
    scope = EVIDENCE_PACKET_QUESTIONS[0].instructions["scope"].casefold()
    assert "do not assess whether an institution" in scope
    assert QUESTION_SET_VERSION == "evidence-packet-v1"
    assert len({question.question_id for question in EVIDENCE_PACKET_QUESTIONS}) == 6


def test_existing_analysis_attention_and_evidence_are_unchanged_when_judgment_is_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    called = DeterministicMockJudgmentProvider()
    disabled, _, _, bench = run_pair(
        tmp_path, enabled=False, judgment_provider=called, name="disabled"
    )
    enabled, _, _, _ = run_pair(tmp_path, enabled=True, provider="mock", name="mock")
    assert called.calls == []
    assert "judgment" not in disabled
    assert disabled["attention"] == enabled["attention"]
    assert disabled["evidence"] == enabled["evidence"]
    assert disabled["state"]["signals"] == enabled["state"]["signals"]
    assert all(signal["review_complexity"] is None for signal in disabled["state"]["signals"])
    assert [item["priority"] for item in disabled["attention"]["review_first"]] == [
        item["priority"] for item in enabled["attention"]["review_first"]
    ]
    assert bench.ledger.reviews() == ()
    verify_monitoring_snapshot(disabled)
    verify_monitoring_snapshot(enabled)


def test_only_review_first_evidence_is_judged_and_no_disposition_is_created(tmp_path):
    provider = DeterministicMockJudgmentProvider()
    result, _, passport, workbench = run_pair(
        tmp_path, enabled=True, judgment_provider=provider, name="selected"
    )
    selected = tuple(item["signal"]["evidence_id"] for item in result["attention"]["review_first"])
    deferred = {item["signal"]["evidence_id"] for item in result["attention"]["deferred"]}
    assert selected
    assert tuple(provider.calls[-len(selected) :]) == selected
    assert result["judgment"]["evidence_ids"] == sorted(selected)
    assert deferred.isdisjoint(result["judgment"]["evidence_ids"])
    assert workbench.ledger.reviews() == ()
    assert result["judgment"]["analysis_id"] == passport.analysis_id
    for evidence_id in selected:
        assert evidence_id in result["evidence"]
    rendered = render_workbench(result)
    assert "Analytical signal" in rendered
    assert "Typed judgment" in rendered
    assert "Analyst disposition" in rendered
    assert "Judgment policy" in rendered
    assert "Model confidence is not statistical confidence" in rendered
    assert "AI confidence" not in rendered
    verify_monitoring_snapshot(result)
    analytical = {
        key: value for key, value in result.items() if key not in {"passport", "judgment"}
    }
    assert stable_id(analytical) == result["passport"]["output_hashes"]["monitoring"]
    assert "TypeSafe" not in json.dumps(result["model_risk_profile"])


def test_local_only_rendering_distinguishes_layers_without_a_blended_score(lab):
    result, _, _ = lab.run(CURRENT_REVIEW)
    rendered = render_workbench(result)
    assert "Analytical signal" in rendered
    assert "Typed judgment" in rendered
    assert "Not requested. Local-only mode" in rendered
    assert "Analyst disposition" in rendered
    assert "AI confidence" not in rendered


def test_canonical_state_is_deterministic_and_excludes_unrelated_material(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    other = packet(claim="Unrelated institution history that must stay local")
    entry = packet(
        claim=f"Synthetic assets changed {SECRET}",
        rationale="analyst rationale must stay out",
        reviewer="Ada Lovelace",
        calculation={
            "formula": f"current-prior {SECRET}",
            "software_version": "test",
            "input_ids": [observation().observation_id],
            "parameters": {"api_key": SECRET, "lookback": 12},
        },
    )
    first = build_judgment_state(entry, analysis_id="analysis-1", as_of="2025-09-01T00:00:00+00:00")
    second = build_judgment_state(
        entry, analysis_id="analysis-1", as_of="2025-09-01T00:00:00+00:00"
    )
    assert first.payload_json == second.payload_json
    assert first.state_id == second.state_id == stable_id(first.payload)
    text = first.payload_json
    assert SECRET not in text
    assert "api_key" not in text
    assert "analyst rationale" not in text
    assert "Ada Lovelace" not in text
    assert other.evidence_id not in text
    assert "Unrelated institution history" not in text
    assert first.payload["evidence_id"] == entry.evidence_id
    assert first.payload["analysis_id"] == "analysis-1"
    assert "decision_boundary" in first.payload
    assert first.truncated is False


def test_truncation_is_explicit_and_oversized_state_is_not_sent():
    huge = packet(drr_evidence={"status": "disabled", "notes": "x" * 5000})
    state = build_judgment_state(
        huge, analysis_id=None, as_of="2025-09-01T00:00:00+00:00", max_characters=800
    )
    assert state.truncated is True
    assert state.payload["truncation"]["truncated"] is True
    assert state.payload["truncation"]["omitted"]
    tiny = build_judgment_state(packet(), analysis_id=None, max_characters=40)
    client = RecordingClient(payload=system_one_payload())
    result = TypeSafeJudgmentProvider(client=client).evaluate(tiny, EVIDENCE_PACKET_QUESTIONS)
    assert result.provider_status == "oversized_state"
    assert client.calls == []
    assert apply_judgment_policy(result)[0] == "JUDGMENT_UNAVAILABLE"


def test_serialization_failure_is_explicit():
    class BadEntry:
        evidence_id = "abc"

        @property
        def payload(self):
            return {"claim": object()}

    state = build_judgment_state(BadEntry(), analysis_id=None)
    assert state.sendable is False
    assert state.failure == "serialization_failure"
    result = DisabledJudgmentProvider().evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    assert result.provider_status == "disabled"


def test_choice_score_and_noul_parsing_keep_primitive_semantics():
    choice_answer = parse_typed_answer(
        "evidence_adequacy",
        {
            "type": "choice",
            "choice": "limited",
            "probabilities": {"limited": 0.5, "adequate": 0.4, "insufficient": 0.1},
            "confidence": 0.25,
        },
    )
    score_answer = parse_typed_answer(
        "severity",
        {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "routine", "1": "moderate", "2": "complex"},
            "probabilities": {"0": 0.1, "1": 0.75, "2": 0.15},
            "confidence": 0.62,
        },
    )
    noul_answer = parse_typed_answer(
        "scope_overreach",
        {"type": "noul", "noul": 0.42, "confidence": 0.99},
    )
    assert choice_answer.choice == "limited"
    assert choice_answer.confidence == 0.25
    assert choice_answer.probabilities[0][0] == "adequate"
    assert score_answer.score == 1.05
    assert score_answer.confidence == 0.62
    assert score_answer.legend[0] == ("0", "routine")
    assert noul_answer.noul == 0.42
    assert noul_answer.confidence is None
    assert "confidence" not in inspect.signature(JudgmentResult).parameters
    model, answers = parse_system_one_response(system_one_payload())
    assert model == "jev-1.13.0"
    assert {answer.question_id for answer in answers} == {
        question.question_id for question in EVIDENCE_PACKET_QUESTIONS
    }
    assert all(answer.confidence is None for answer in answers if answer.primitive == "noul")


def test_malformed_missing_and_unsupported_responses_fail_closed():
    with pytest.raises(JudgmentResponseError) as malformed:
        parse_system_one_response("{")
    assert malformed.value.status == "malformed_response"
    with pytest.raises(JudgmentResponseError) as missing:
        parse_system_one_response({"model": "jev-1.13.0", "answers": {}})
    assert missing.value.status == "missing_fields"
    with pytest.raises(JudgmentResponseError) as unsupported:
        parse_typed_answer("custom", {"type": "rank", "rank": 1})
    assert unsupported.value.status == "unsupported_primitive"
    with pytest.raises(JudgmentResponseError):
        parse_typed_answer(
            "evidence_adequacy",
            {"type": "choice", "choice": "adequate", "probabilities": {"adequate": 1.0}},
        )


def test_provider_failures_are_explicit_and_do_not_become_ready(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    state = sample_state()
    missing = TypeSafeJudgmentProvider().evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    assert missing.provider_status == "missing_credentials"
    assert SECRET not in canonical_json(missing)
    assert apply_judgment_policy(missing)[0] == "JUDGMENT_UNAVAILABLE"

    class TypeSafeAuthenticationError(Exception):
        status = 401

    class TypeSafeAPITimeoutError(TimeoutError):
        pass

    class TypeSafeRateLimitError(Exception):
        status = 429

    class TypeSafeAPIConnectionError(ConnectionError):
        pass

    class TypeSafeInternalServerError(Exception):
        status = 503

    cases = (
        (TypeSafeAuthenticationError(f"bad {SECRET}"), "authentication_error"),
        (TypeSafeAPITimeoutError("timed out"), "timeout"),
        (TypeSafeRateLimitError("slow down"), "rate_limit"),
        (TypeSafeAPIConnectionError("dns"), "network_error"),
        (TypeSafeInternalServerError("down"), "service_failure"),
    )
    for exc, status in cases:
        client = RecordingClient(exc=exc)
        result = TypeSafeJudgmentProvider(client=client).evaluate(state, EVIDENCE_PACKET_QUESTIONS)
        assert result.provider_status == status
        assert apply_judgment_policy(result)[0] == "JUDGMENT_UNAVAILABLE"
        assert SECRET not in canonical_json(result)
        assert SECRET not in " ".join(result.limitations)
    disabled = DisabledJudgmentProvider().evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    assert apply_judgment_policy(disabled)[0] == "JUDGMENT_UNAVAILABLE"


def test_policy_outcomes_follow_the_versioned_rules():
    state = sample_state()
    ready = completed_result(state)
    careful = completed_result(
        state, nouls={"additional_review_needed": 0.72, "limitations_material": 0.67}
    )
    limited = completed_result(state, adequacy="limited")
    insufficient = completed_result(state, adequacy="insufficient", nouls={"scope_overreach": 0.99})
    boundary = completed_result(state, nouls={"scope_overreach": 0.5})
    assert apply_judgment_policy(ready) == ("READY", (), "moderate")
    outcome, warnings, complexity = apply_judgment_policy(careful)
    assert outcome == "REVIEW_CAREFULLY"
    assert complexity == "moderate"
    assert any(item.startswith("additional_review_needed=") for item in warnings)
    assert apply_judgment_policy(limited)[0] == "REVIEW_CAREFULLY"
    assert apply_judgment_policy(insufficient)[0] == "INSUFFICIENT_EVIDENCE"
    assert apply_judgment_policy(boundary)[0] == "READY"


def test_judgments_are_append_only_and_hashes_follow_content_not_timestamps(tmp_path):
    entry = packet()
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(entry)
    state = build_judgment_state(entry, analysis_id="analysis-1", as_of=WHEN)
    first = completed_result(state, model_version="jev-1.13.0")
    later = make_result(
        state,
        provider="mock",
        model="deterministic-mock",
        provider_model_version="jev-1.13.0",
        status="completed",
        answers=first.answers,
        requested_at="2026-02-01T00:00:00+00:00",
        completed_at="2026-02-01T00:00:00+00:00",
        latency_ms=15.0,
    )
    changed = completed_result(state, model_version="jev-9.0.0")
    assert first.judgment_id == later.judgment_id
    assert first.judgment_id != changed.judgment_id
    ledger.append_judgment(assemble_overlay(first))
    ledger.append_judgment(assemble_overlay(later))
    assert len(ledger.judgments()) == 1
    ledger.append_judgment(assemble_overlay(changed))
    assert len(ledger.judgments()) == 2
    assert ledger.reviews() == ()
    with sqlite3.connect(ledger.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE judgments SET payload = '{}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM judgments")
    bogus_id = first.judgment_id
    empty = EvidenceLedger(tmp_path / "empty.sqlite")
    with pytest.raises(KeyError):
        empty.append_judgment(assemble_overlay(first))
    conflict = EvidenceLedger(tmp_path / "conflict.sqlite")
    conflict.append(entry)
    with sqlite3.connect(conflict.path) as db:
        db.execute(
            "INSERT INTO judgments VALUES (?,?,?,?)",
            (bogus_id, entry.evidence_id, WHEN, canonical_json({"evidence_id": "tampered"})),
        )
    with pytest.raises(ValueError, match="integrity"):
        conflict.append_judgment(assemble_overlay(first))


def test_security_records_and_outbound_state_exclude_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    entry = packet(claim=f"Synthetic assets changed {SECRET}")
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(entry)
    state = build_judgment_state(entry, analysis_id="analysis-1", as_of=WHEN)
    overlay = assemble_overlay(completed_result(state))
    ledger.append_judgment(overlay)

    class TypeSafeAuthenticationError(Exception):
        def __str__(self):
            return f"Authorization: Bearer {SECRET}"

    failed = TypeSafeJudgmentProvider(
        client=RecordingClient(exc=TypeSafeAuthenticationError())
    ).evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    ledger.append_judgment(assemble_overlay(failed))
    stored = "\n".join(canonical_json(item) for item in ledger.judgments())
    audit = "\n".join(canonical_json(event) for event in ledger.audit_events())
    assert SECRET not in state.payload_json
    assert SECRET not in stored
    assert SECRET not in audit
    assert "Bearer" not in audit
    client = RecordingClient(payload=system_one_payload())
    TypeSafeJudgmentProvider(client=client).evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    assert SECRET not in json.dumps(client.calls)
    assert set(client.calls[0]["questions"]) == {
        question.question_id for question in EVIDENCE_PACKET_QUESTIONS
    }


def test_replay_and_walk_forward_do_not_call_a_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(
        TypeSafeJudgmentProvider,
        "evaluate",
        lambda self, state, questions: (_ for _ in ()).throw(AssertionError("remote judgment")),
    )
    entry = packet()
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(entry)
    state = build_judgment_state(entry, analysis_id="analysis-1", as_of=WHEN)
    ledger.append_judgment(assemble_overlay(completed_result(state)))
    assert replay_recorded_judgments(ledger, as_of="2026-01-01T00:00:00+00:00") == ()
    found = replay_recorded_judgments(ledger, as_of="2026-01-16T00:00:00+00:00")
    assert len(found) == 1
    assert describe_post_hoc_judgment()["label"] == POST_HOC_JUDGMENT_EVALUATION
    assert describe_post_hoc_judgment()["historical_evidence"] is False
    records = history()
    result = walk_forward_validate(
        VintageStore(records),
        SemanticRegistry((definition(),)),
        institution="A",
        form="FR Y-9C",
        review_dates=[row.available_as_of for row in records[-2:]],
        allow_synthetic=True,
        enable_drr=False,
    )
    assert result["mode"] == "point_in_time_reconstruction"
    assert "judgment" not in result
    assert "typesafe" not in inspect.getsource(walk_forward_validate)
    assert "judgment" not in inspect.getsource(AttentionBudget.select)
    assert "typesafe" not in inspect.getsource(DynamicResonanceRooting.analyze_system)


def test_typesafe_dependency_is_recorded_only_when_that_provider_is_enabled():
    profile = typesafe_model_risk_extension(DEFAULT_MODEL_RISK_PROFILE, model="jev-latest")
    assert profile.risk_tier.value == "unassessed"
    assert profile.validation_status == DEFAULT_MODEL_RISK_PROFILE.validation_status
    assert any("TypeSafe Jev" in item for item in profile.third_party_products)
    assert any("typesafe-sdk" in item for item in profile.dependencies)
    text = " ".join(profile.limitations).casefold()
    assert "does not validate drr" in text


def test_successful_typesafe_parse_does_not_import_requirement_for_offline_tests(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    state = sample_state()
    client = RecordingClient(payload=system_one_payload())
    result = TypeSafeJudgmentProvider(
        model="jev-latest", client=client, clock=lambda: WHEN
    ).evaluate(state, EVIDENCE_PACKET_QUESTIONS)
    assert result.provider_status == "completed"
    assert result.provider_model_version == "jev-1.13.0"
    assert result.model == "jev-latest"
    assert apply_judgment_policy(result)[0] == "REVIEW_CAREFULLY"
    assert client.calls[0]["state"]["evidence_id"] == state.evidence_id
    assert "typesafe_sdk" not in inspect.getsource(build_judgment_state)


def test_invalid_judgment_configuration_fails_fast():
    with pytest.raises(ValueError):
        WorkbenchConfig(judgment_provider="openai")
    with pytest.raises(ValueError):
        WorkbenchConfig(judgment_policy_version="v9")
    with pytest.raises(ValueError):
        WorkbenchConfig(judgment_timeout_seconds=0)


def test_mock_answers_are_a_pure_function_of_the_state():
    state = sample_state()
    provider = DeterministicMockJudgmentProvider()
    assert provider.evaluate(state, EVIDENCE_PACKET_QUESTIONS).judgment_id == (
        provider.evaluate(state, EVIDENCE_PACKET_QUESTIONS).judgment_id
    )


def test_analyst_review_remains_a_human_record(tmp_path):
    entry = packet()
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(entry)
    state = build_judgment_state(entry, analysis_id=None, as_of=WHEN)
    ledger.append_judgment(assemble_overlay(completed_result(state)))
    assert ledger.reviews() == ()
    review = AnalystReview(
        entry.evidence_id, "investigate", "Analyst", "Check the filing", "2025-09-01"
    )
    ledger.review(review)
    assert ledger.latest_reviews()[entry.evidence_id].disposition.value == "investigate"
    assert len(ledger.judgments()) == 1


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
def test_live_typesafe_optional():
    pytest.importorskip("typesafe_sdk")
    state = sample_state()
    result = TypeSafeJudgmentProvider(model="jev-latest", timeout=20).evaluate(
        state, EVIDENCE_PACKET_QUESTIONS
    )
    assert result.provider_status == "completed"
    assert result.provider_model_version
    assert {answer.question_id for answer in result.answers} == {
        question.question_id for question in EVIDENCE_PACKET_QUESTIONS
    }
