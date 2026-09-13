from dataclasses import replace
import sqlite3

import pytest

from test_regulatory_foundation import observation
from drr_framework.supervisory.common import canonical
from drr_framework.supervisory.evidence_ledger import (
    AuditEvent,
    EvidenceEntry,
    EvidenceLedger,
    AnalystReview,
)
from drr_framework.supervisory.feedback import evaluate_signal_usefulness
from drr_framework.supervisory.passport import AnalysisPassport
from drr_framework.supervisory.policy_context import PolicyContext, PolicyEvent
from drr_framework.supervisory.entity_graph import PublicEntity, EntityRelationship, EntityGraph
from drr_framework.supervisory.snc import PublicSNCAggregate, analyze_public_snc
from drr_framework.supervisory.issues import MonitoringIssue, IssueTracker


def entry():
    o = observation()
    return EvidenceEntry.create(
        claim="Synthetic assets changed",
        institution="A",
        metric="SYN_ASSETS",
        period=o.reporting_period,
        source_facts=[dict(observation_id=o.observation_id, **canonical(o))],
        calculation={
            "formula": "current-prior",
            "software_version": "test",
            "input_ids": [o.observation_id],
        },
        historical_context={},
        peer_context={},
        baseline_evidence=[],
        drr_evidence={},
        robustness={},
        contradictory_evidence=[],
        policy_context=[],
        data_limitations=[],
    )


def test_evidence_deep_immutability_roundtrip_and_append_only_reviews(tmp_path):
    e = entry()
    payload = e.payload
    payload["source_facts"][0]["value"] = 999
    assert e.payload["source_facts"][0]["value"] == 100
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(e)
    ledger.append(e)
    assert len(ledger.entries()) == 1
    first = AnalystReview(e.evidence_id, "investigate", "Analyst", "Check filing", "2025-09-01", 3)
    second = AnalystReview(
        e.evidence_id, "explained", "Analyst", "Amendment explains movement", "2025-09-02", 2
    )
    ledger.review(first)
    ledger.review(second)
    assert len(ledger.reviews()) == 2
    assert len(ledger.audit_events()) == 2
    assert {event.outcome for event in ledger.audit_events()} == {"recorded"}
    assert (
        ledger.latest_reviews(as_of="2025-09-01")[e.evidence_id].disposition.value == "investigate"
    )
    assert ledger.get(e.evidence_id) == e
    paths = ledger.export(tmp_path / "exports")
    assert all(p.exists() for p in paths.values())
    assert evaluate_signal_usefulness(ledger).reviewed_signals == 1
    assert evaluate_signal_usefulness(ledger).total_review_minutes == 5
    with pytest.raises(ValueError, match="hash"):
        EvidenceEntry.create(**{k: v for k, v in payload.items() if k != "decision_boundary"})


def test_policy_availability_and_explicit_breakpoints():
    event = PolicyEvent(
        "example",
        "Federal Reserve",
        "Reporting definition update",
        "2025-07-01",
        "2025-06-30",
        "final",
        (),
        ("FR Y-9C",),
        ("SYN_ASSETS",),
        "definition",
        "https://www.federalreserve.gov/supervisionreg.htm",
        "2025-07-02",
        comparability_break=True,
    )
    context = PolicyContext((event,))
    assert context.relevant(as_of="2025-07-01", form="FR Y-9C", metric="SYN_ASSETS") == ()
    assert context.breakpoints(
        as_of="2025-08-01", form="FR Y-9C", metric="SYN_ASSETS", dates=("2025-03-31", "2025-06-30")
    ) == ("2025-06-30",)
    with pytest.raises(ValueError, match="scope"):
        replace(event, applicability="authoritative_applicability")


def test_audit_events_reject_sensitive_details_and_storage_is_append_only(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match="sensitive"):
        ledger.record_audit_event(
            AuditEvent(
                "2025-09-01",
                "review_request",
                "denied",
                "local-client",
                details=(("token", "do-not-store"),),
            )
        )
    event = AuditEvent("2025-09-01", "health_check", "recorded", "local-client")
    ledger.record_audit_event(event)
    with pytest.raises(sqlite3.DatabaseError):
        with ledger._connect() as db:
            db.execute("UPDATE audit_events SET payload='{}' WHERE id=?", (event.event_id,))
    with pytest.raises(sqlite3.DatabaseError):
        with ledger._connect() as db:
            db.execute("DELETE FROM audit_events WHERE id=?", (event.event_id,))


def test_historical_ledger_export_filters_future_reviews(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    evidence = entry()
    ledger.append(evidence)
    ledger.review(
        AnalystReview(
            evidence.evidence_id,
            "investigate",
            "Analyst",
            "Later review",
            "2025-10-01",
        )
    )
    paths = ledger.export(
        tmp_path / "exports",
        as_of="2025-09-01",
        evidence_ids=(evidence.evidence_id,),
    )
    assert paths["reviews"].read_text(encoding="utf-8") == ""


def test_sourced_entity_perimeters_and_future_links():
    entities = tuple(
        PublicEntity(k, k, t, j, "synthetic", "2020-01-01", provenance="synthetic_test")
        for k, t, j in (
            ("parent", "foreign_parent", "GB"),
            ("ihc", "ihc", "US"),
            ("bank", "bank", "US"),
            ("branch", "branch", "US"),
        )
    )
    links = tuple(
        EntityRelationship(
            a, b, "ownership", "2020-01-01", available, "synthetic", provenance="synthetic_test"
        )
        for a, b, available in (
            ("parent", "ihc", "2020-01-01"),
            ("ihc", "bank", "2020-01-01"),
            ("parent", "branch", "2026-01-01"),
        )
    )
    graph = EntityGraph(entities, links)
    assert graph.ihc_perimeter("parent", as_of="2025-01-01") == ("bank", "ihc")
    assert graph.branch_agency_perimeter("parent", as_of="2025-01-01") == ()
    assert graph.ancestors("bank", as_of="2025-01-01") == ("ihc", "parent")
    cycle = EntityRelationship(
        "bank",
        "parent",
        "ownership",
        "2020-01-01",
        "2020-01-01",
        "synthetic",
        provenance="synthetic_test",
    )
    with pytest.raises(ValueError, match="cyclic"):
        EntityGraph(entities, links + (cycle,)).active(as_of="2025-01-01")


def test_public_snc_scope_and_issue_history(tmp_path):
    aggregate = PublicSNCAggregate(
        2025,
        100,
        10,
        "USD billions",
        "https://www.federalreserve.gov/supervisionreg/snc.htm",
        "2026-01-01",
        "2026-01-02",
    )
    assert analyze_public_snc((aggregate,), as_of="2025-12-31")["rows"] == []
    assert analyze_public_snc((aggregate,), as_of="2026-02-01")["rows"][0]["non_pass_share"] == 0.1
    with pytest.raises(ValueError):
        replace(aggregate, non_pass=101)
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    e = entry()
    ledger.append(e)
    tracker = IssueTracker(tmp_path / "issues.sqlite")
    issue = MonitoringIssue(
        "test",
        "A",
        "Check revision",
        "Compare originals",
        "user_entered",
        (e.evidence_id,),
        "normal",
        "open",
        "Analyst",
        "2025-09-01",
        "2025-09-01",
    )
    tracker.append(issue, ledger)
    tracker.append(replace(issue, status="resolved", review_date="2025-09-03"), ledger)
    assert tracker.current(as_of="2025-09-02")[0].status == "open"
    assert tracker.current(as_of="2025-09-04")[0].status == "resolved"
