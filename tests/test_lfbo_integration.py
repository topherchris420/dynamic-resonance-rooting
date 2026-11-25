import copy
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
import http.client
import json
import threading

import pytest

from drr_framework.supervisory.demo import synthetic_monitoring_lab, PREVIOUS_REVIEW, CURRENT_REVIEW
from drr_framework.supervisory.evidence_ledger import EvidenceLedger
from drr_framework.supervisory.workbench import MonitoringWorkbench, WorkbenchConfig
from drr_framework.supervisory.briefs import generate_morning_brief, generate_lfbo_monitoring_brief
from drr_framework.supervisory.cli import make_review_server, export_run
from drr_framework.supervisory.common import canonical_json, stable_id
from drr_framework.supervisory.ui import render_workbench
from drr_framework.supervisory.workbench import review_state_from_dict


def test_complete_workflow_revisions_and_unchanged_morning_brief(lab, tmp_path):
    _, previous, _ = lab.run(PREVIOUS_REVIEW)
    result, state, passport = lab.run(CURRENT_REVIEW, previous=previous)
    assert len(result["analyses"]) == 5
    assert len(result["delta"]["changed_observations"]) == 1
    assert any(
        s["institution"] == "DEMO-B" and s["metric"] == "SYN_CAPITAL"
        for s in result["delta"]["disappeared"]
    )
    assert any(
        i["signal"]["institution"] == "DEMO-A" and i["signal"]["metric"] == "SYN_FUNDING"
        for i in result["attention"]["review_first"]
    )
    same, _, _ = lab.run(CURRENT_REVIEW, previous=state)
    assert not same["attention"]["review_first"]
    assert "Nothing material changed" in generate_morning_brief(same)
    tomorrow, _, _ = lab.run("2026-08-11", previous=state)
    assert not tomorrow["attention"]["review_first"]
    assert "DEMO-A" in generate_lfbo_monitoring_brief(result, "DEMO-A")
    path = export_run(result, state, passport, lab.ledger, tmp_path / "export")
    assert (path / "index.html").is_file()
    assert (path / "morning-brief.md").is_file()
    exported_snapshot = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    assert stable_id({k: v for k, v in exported_snapshot.items() if k != "passport"}) == (
        result["passport"]["output_hashes"]["monitoring"]
    )
    assert export_run(result, state, passport, lab.ledger, tmp_path / "export") == path
    assert all(a["drr"]["status"] == "disabled" for a in result["analyses"])
    assert len(result["passport"]["source_code_sha256"]) == 64
    assert all(len(value) == 64 for value in result["passport"]["source_hashes"])
    payload_without_passport = {k: v for k, v in result.items() if k != "passport"}
    assert stable_id(payload_without_passport) == result["passport"]["output_hashes"]["monitoring"]
    assert result["passport"]["model_risk_profile"]["validation_status"] == ("development_tested")
    assert result["passport"]["previous_state_id"] == previous.state_id
    assert "perspective_configuration" in result["passport"]
    demo_b_brief = generate_lfbo_monitoring_brief(result, "DEMO-B")
    assert "DEMO-A reviewers" not in demo_b_brief


def test_render_escapes_untrusted_claims_and_supports_all_navigation(lab):
    result, _, _ = lab.run(CURRENT_REVIEW)
    oid = next(iter(result["evidence"]))
    result["evidence"][oid]["claim"] = '<script>alert("xss")</script>'
    rendered = render_workbench(result)
    assert '<script>alert("xss")</script>' not in rendered
    assert "&lt;script&gt;" in rendered
    for panel in (
        "today",
        "institutions",
        "peers",
        "evidence",
        "policy",
        "quality",
        "queue",
        "validation",
        "perspectives",
    ):
        assert f'id="{panel}"' in rendered


def test_review_server_blocks_cross_origin_and_records_valid_disposition(lab):
    result, _, _ = lab.run(CURRENT_REVIEW)
    server = make_review_server(lab, result, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=4)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read().decode()
        assert response.status == 200 and "default-src 'none'" in response.getheader(
            "Content-Security-Policy"
        )

        class Tokens(HTMLParser):
            token = None

            def handle_starttag(self, tag, attrs):
                if tag == "form":
                    self.token = dict(attrs).get("data-token", self.token)

        parsed = Tokens()
        parsed.feed(page)
        oid = next(iter(result["evidence"]))
        body = json.dumps(
            dict(
                evidence_id=oid,
                disposition="explained",
                reviewer="Test analyst",
                rationale="Confirmed synthetic amendment",
                review_minutes=2,
            )
        )
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://example.com",
            "X-Review-Token": parsed.token,
        }
        connection.request("POST", "/api/review", body, headers)
        response = connection.getresponse()
        response.read()
        assert response.status == 403 and not lab.ledger.reviews()
        assert lab.ledger.audit_events()[-1].outcome == "denied"
        headers["Origin"] = f"http://127.0.0.1:{server.server_port}"
        connection.request("POST", "/api/review", body, headers)
        response = connection.getresponse()
        response.read()
        assert response.status == 200
        assert lab.ledger.latest_reviews()[oid].disposition.value == "explained"
        assert lab.ledger.audit_events()[-1].action == "analyst_review"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_blocked_data_does_not_produce_a_false_all_clear(lab):
    lab.config = replace(lab.config, allow_synthetic=False)
    result, _, _ = lab.run(CURRENT_REVIEW)
    assert not result["analyses"]
    assert "blockers" in generate_morning_brief(result)
    assert "Nothing material changed" not in generate_morning_brief(result)


def test_supervisory_package_exports_high_level_api():
    import drr_framework
    import drr_framework.supervisory as supervisory

    assert "RegulatoryAnalysisDataset" in drr_framework.__all__
    assert supervisory.MonitoringWorkbench is MonitoringWorkbench
    assert supervisory.FilingContext.__name__ == "FilingContext"
    assert "generate_morning_brief" in supervisory.__all__


def test_incremental_run_reuses_unchanged_institution_inputs(lab):
    _, previous, _ = lab.run(CURRENT_REVIEW)
    initial_cache_entries = len(lab._cache)
    latest = [
        o
        for o in lab.store.as_of(CURRENT_REVIEW)
        if o.institution_id == "DEMO-A" and o.reporting_period == "2026-06-30"
    ]
    new_quarter = tuple(
        replace(
            o,
            reporting_period="2026-09-30",
            value=o.value + 1,
            original_filing_date="2026-11-01",
            ingestion_date="2026-11-02",
            available_as_of="2026-11-02",
            source_vintage="DEMO-A-2026Q3-original",
            amendment_date=None,
            supersedes=None,
        )
        for o in latest
    )
    lab.store = lab.store.append(*new_quarter)
    lab.run("2026-11-02", previous=previous)
    assert len(lab._cache) == initial_cache_entries + 1


def test_review_state_export_envelope_is_verified(lab, tmp_path):
    result, state, passport = lab.run(CURRENT_REVIEW)
    path = export_run(result, state, passport, lab.ledger, tmp_path / "export")
    envelope = json.loads((path / "review-state.json").read_text(encoding="utf-8"))
    assert review_state_from_dict(envelope).state_id == state.state_id
    envelope["state"]["as_of"] = "2026-08-11T00:00:00+00:00"
    with pytest.raises(ValueError, match="integrity"):
        review_state_from_dict(envelope)


def test_export_preserves_ledger_evidence_referenced_only_by_perspectives(lab, tmp_path):
    from test_lfbo_evidence_workflow import entry
    from drr_framework.supervisory.demo import synthetic_perspectives
    from drr_framework.supervisory import PerspectiveInventory, EvidenceEntry

    cited = entry()
    lab.ledger.append(cited)
    unrelated = EvidenceEntry.create(**dict(cited.payload, claim="Unrelated evidence"))
    lab.ledger.append(unrelated)
    rows = synthetic_perspectives(lab.store).snapshot(
        as_of=CURRENT_REVIEW, observations=lab.store.known_records(CURRENT_REVIEW)
    )["perspectives"]
    from drr_framework.supervisory import ScopedPerspective

    values = dict(rows[0])
    values.pop("perspective_id")
    values["evidence_ids"] = (cited.evidence_id,)
    lab.perspectives = PerspectiveInventory((ScopedPerspective(**values),))
    result, state, passport = lab.run(CURRENT_REVIEW)
    assert cited.evidence_id not in result["evidence"]
    path = export_run(result, state, passport, lab.ledger, tmp_path / "export")
    assert (path / "evidence" / f"{cited.evidence_id}.md").is_file()
    assert not (path / "evidence" / f"{unrelated.evidence_id}.md").exists()
