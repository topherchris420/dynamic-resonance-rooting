from contextlib import contextmanager
from dataclasses import replace
from html.parser import HTMLParser
import http.client
import json
import threading

import pytest

from drr_framework.supervisory import verify_monitoring_snapshot
from drr_framework.supervisory.cli import export_run, make_review_server
from drr_framework.supervisory.common import canonical_json
from drr_framework.supervisory.demo import CURRENT_REVIEW, PREVIOUS_REVIEW


@contextmanager
def serve(lab, result, role="analyst"):
    server = make_review_server(lab, result, port=0, role=role)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)

    def get(path):
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 200
        return body

    class Forms(HTMLParser):
        token = ""

        def handle_starttag(self, tag, attrs):
            if tag == "form":
                self.token = dict(attrs).get("data-token", self.token)

    parsed = Forms()
    parsed.feed(get("/"))

    def post(oid, disposition, minutes=2):
        body = json.dumps(
            dict(
                evidence_id=oid,
                disposition=disposition,
                reviewer="Analyst",
                rationale="Reviewed the cited amendment",
                review_minutes=minutes,
            )
        )
        connection.request(
            "POST",
            "/api/review",
            body,
            {
                "Content-Type": "application/json",
                "Origin": f"http://127.0.0.1:{server.server_port}",
                "X-Review-Token": parsed.token,
            },
        )
        response = connection.getresponse()
        response.read()
        return response.status

    try:
        yield get, post
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_live_reviews_preserve_hashes_promote_deferred_and_restore_reopened_items(lab):
    lab.config = replace(lab.config, top_n=1)
    result, _, _ = lab.run(CURRENT_REVIEW)
    original = canonical_json(result)
    first = result["attention"]["review_first"][0]["signal"]["evidence_id"]
    second = result["attention"]["deferred"][0]["signal"]["evidence_id"]
    with serve(lab, result) as (get, post):
        assert post(first, "explained") == 200
        assert get("/api/snapshot") == original == canonical_json(result)
        verify_monitoring_snapshot(json.loads(get("/api/snapshot")))
        activity = json.loads(get("/api/review-activity"))
        assert activity["attention"]["review_first"][0]["signal"]["evidence_id"] == second
        assert (
            next(s for s in activity["signals"] if s["evidence_id"] == first)["disposition"]
            == "explained"
        )
        assert "Latest dispositions" in get("/")
        assert post(first, "unresolved") == 200
        activity = json.loads(get("/api/review-activity"))
        assert activity["attention"]["review_first"][0]["signal"]["evidence_id"] == first
        assert post(first, "useful") == 200
        assert (
            json.loads(get("/api/review-activity"))["attention"]["review_first"][0]["signal"][
                "evidence_id"
            ]
            == second
        )
        assert get("/api/snapshot") == original
    # Restarting the server projects the same persisted analyst activity.
    with serve(lab, result) as (get, _):
        activity = json.loads(get("/api/review-activity"))
        assert activity["attention"]["review_first"][0]["signal"]["evidence_id"] == second
        assert activity["reviews"][0]["disposition"] == "useful"
        assert get("/api/snapshot") == original
    assert not lab.ledger.reviews(as_of=CURRENT_REVIEW)


def test_server_owns_its_snapshot_and_rejects_boolean_review_duration(lab):
    result, _, _ = lab.run(CURRENT_REVIEW)
    original = canonical_json(result)
    oid = next(iter(result["evidence"]))
    with serve(lab, result) as (get, post):
        result["state"]["signals"][0]["claim"] = "external mutation"
        assert get("/api/snapshot") == original
        assert post(oid, "explained", minutes=True) == 400
        assert not lab.ledger.reviews()


def test_viewer_reads_activity_but_cannot_write(lab):
    result, _, _ = lab.run(CURRENT_REVIEW)
    with serve(lab, result, role="viewer") as (get, post):
        assert json.loads(get("/api/review-activity"))["reviews"] == []
        assert post(next(iter(result["evidence"])), "explained") == 403
        assert not lab.ledger.reviews()


def test_investigate_reopens_with_unresolved_priority(lab):
    lab.config = replace(lab.config, top_n=1)
    result, _, _ = lab.run(CURRENT_REVIEW)
    first = result["attention"]["review_first"][0]["signal"]["evidence_id"]
    with serve(lab, result) as (get, post):
        assert post(first, "investigate") == 200
        activity = json.loads(get("/api/review-activity"))
        selected = next(
            item
            for item in activity["attention"]["review_first"]
            if item["signal"]["evidence_id"] == first
        )
        original = next(
            item
            for item in result["attention"]["review_first"]
            if item["signal"]["evidence_id"] == first
        )
        assert selected["priority"] == original["priority"]
        assert (
            next(s for s in activity["signals"] if s["evidence_id"] == first)["disposition"]
            == "investigate"
        )


@pytest.mark.parametrize("field", ["state", "evidence", "passport"])
def test_tampered_snapshots_fail_before_export_creates_files(lab, tmp_path, field):
    result, state, passport = lab.run(CURRENT_REVIEW)
    if field == "state":
        result["state"]["signals"][0]["materiality"] = 0
    elif field == "evidence":
        result["evidence"][next(iter(result["evidence"]))]["claim"] = "altered"
    else:
        result["passport"]["random_seeds"] = [999]
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="integrity"):
        export_run(result, state, passport, lab.ledger, destination)
    assert not destination.exists()


def test_revision_report_explains_removed_alert_and_is_part_of_verified_export(lab, tmp_path):
    _, previous, _ = lab.run(PREVIOUS_REVIEW)
    result, state, passport = lab.run(CURRENT_REVIEW, previous=previous)
    (revision,) = result["observation_revisions"]
    assert revision["current"]["institution_id"] == "DEMO-B"
    assert revision["current"]["metric"] == "SYN_CAPITAL"
    assert revision["raw_change"] == revision["current"]["value"] - revision["previous"]["value"]
    assert any(t["transition"] == "disappeared" for t in revision["concurrent_signal_changes"])
    path = export_run(result, state, passport, lab.ledger, tmp_path / "valid")
    assert (
        json.loads((path / "filing-revisions.json").read_text()) == result["observation_revisions"]
    )
    assert "Filing Revisions" in (path / "morning-brief.md").read_text()
    assert 'id="revisions"' in (path / "index.html").read_text()
    verify_monitoring_snapshot(json.loads((path / "snapshot.json").read_text()))
