"""Content-addressed evidence and append-only analyst dispositions in local SQLite."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .common import (
    canonical,
    canonical_json,
    instant,
    stable_id,
    DECISION_BOUNDARY,
    write_immutable,
)


class AnalystDisposition(str, Enum):
    USEFUL = "useful"
    INVESTIGATE = "investigate"
    EXPLAINED = "explained"
    NOISY = "noisy"
    DISMISSED = "dismissed"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AuditEvent:
    """Minimal structured security/operation event; never stores tokens or rationale."""

    occurred_at: str
    action: str
    outcome: str
    actor: str
    subject_id: str = ""
    details: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "occurred_at", instant(self.occurred_at).isoformat())
        object.__setattr__(self, "details", tuple(tuple(item) for item in self.details))
        if (
            not isinstance(self.action, str)
            or not self.action.strip()
            or len(self.action) > 120
            or not isinstance(self.actor, str)
            or not self.actor.strip()
            or len(self.actor) > 120
            or not isinstance(self.subject_id, str)
            or len(self.subject_id) > 256
        ):
            raise ValueError("Audit action and actor are required")
        if self.outcome not in {"recorded", "denied", "failed"}:
            raise ValueError("Invalid audit outcome")
        if any(len(item) != 2 for item in self.details):
            raise ValueError("Audit details must be key/value pairs")
        sensitive = ("token", "secret", "password", "credential", "rationale")
        for key, value in self.details:
            if (
                not isinstance(key, str)
                or not key.strip()
                or len(key) > 80
                or any(term in key.casefold() for term in sensitive)
                or not isinstance(value, (str, int, float, bool))
                or (isinstance(value, str) and len(value) > 256)
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ValueError("Audit details cannot contain sensitive or oversized values")

    @property
    def event_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class EvidenceEntry:
    """JSON is stored as an immutable string; `payload` returns a fresh copy."""

    payload_json: str

    def __post_init__(self):
        value = json.loads(self.payload_json)
        required = {
            "claim",
            "institution",
            "metric",
            "period",
            "source_facts",
            "calculation",
            "historical_context",
            "peer_context",
            "baseline_evidence",
            "drr_evidence",
            "robustness",
            "contradictory_evidence",
            "policy_context",
            "data_limitations",
            "decision_boundary",
        }
        if required - set(value):
            raise ValueError(f"Incomplete evidence: {sorted(required-set(value))}")
        if (
            not value["source_facts"]
            or not value["claim"]
            or value["decision_boundary"] != DECISION_BOUNDARY
        ):
            raise ValueError(
                "Evidence needs source facts, a claim and the research decision boundary"
            )
        for fact in value["source_facts"]:
            if not all(
                k in fact
                for k in (
                    "observation_id",
                    "metric",
                    "value",
                    "source",
                    "provenance",
                    "source_vintage",
                    "available_as_of",
                )
            ):
                raise ValueError("Incomplete source observation")
            # Digest validates the full observation, not a hand-entered summary.
            from .vintage import RegulatoryObservation

            observation = RegulatoryObservation.from_dict(fact)
            if observation.observation_id != fact["observation_id"]:
                raise ValueError("Source observation hash mismatch")
        source_latest = max(instant(f["available_as_of"]) for f in value["source_facts"])
        evidence_available = value.get("evidence_available_as_of")
        if evidence_available is None:
            # Backward-compatible entries inherit the latest source-fact cutoff.
            # New producers should pass their own point-in-time evidence cutoff.
            evidence_available = source_latest.isoformat()
        else:
            evidence_available = instant(evidence_available).isoformat()
        if instant(evidence_available) < source_latest:
            raise ValueError("Evidence availability cannot precede source facts")
        value["evidence_available_as_of"] = evidence_available
        calculation = value["calculation"]
        if (
            not calculation.get("formula")
            or not calculation.get("software_version")
            or not calculation.get("input_ids")
        ):
            raise ValueError("Calculation formula, software and inputs are required")
        ids = {f["observation_id"] for f in value["source_facts"]}
        if not set(calculation["input_ids"]) <= ids:
            raise ValueError("Calculation references missing source facts")
        object.__setattr__(self, "payload_json", canonical_json(value))

    @property
    def evidence_id(self):
        return stable_id(self.payload)

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @classmethod
    def create(cls, **payload):
        payload["decision_boundary"] = DECISION_BOUNDARY
        return cls(canonical_json(payload))

    def to_markdown(self):
        p = self.payload
        lines = [f"# {p['claim']}", "", f"Evidence: `{self.evidence_id}`", ""]
        for key, value in p.items():
            if key == "claim":
                continue
            lines += [
                "## " + key.replace("_", " ").title(),
                "",
                "```json",
                json.dumps(value, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        return "\n".join(lines)


@dataclass(frozen=True)
class AnalystReview:
    evidence_id: str
    disposition: AnalystDisposition
    reviewer: str
    rationale: str
    reviewed_at: str
    review_minutes: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "disposition", AnalystDisposition(self.disposition))
        object.__setattr__(self, "reviewed_at", instant(self.reviewed_at).isoformat())
        if (
            not isinstance(self.reviewer, str)
            or not self.reviewer.strip()
            or len(self.reviewer) > 120
            or not isinstance(self.rationale, str)
            or not self.rationale.strip()
            or len(self.rationale) > 4000
            or not self.evidence_id
        ):
            raise ValueError("Analyst identity, evidence and rationale are required")
        try:
            minutes = float(self.review_minutes)
        except (TypeError, ValueError):
            raise ValueError("Review minutes must be finite and in [0,1440]") from None
        if (
            isinstance(self.review_minutes, bool)
            or not math.isfinite(minutes)
            or not 0 <= minutes <= 24 * 60
        ):
            raise ValueError("Review minutes must be finite and in [0,1440]")
        object.__setattr__(self, "review_minutes", minutes)

    @property
    def review_id(self):
        return stable_id(self)


class EvidenceLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        if self.path.exists():
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL, payload TEXT NOT NULL, FOREIGN KEY(evidence_id) REFERENCES evidence(id));
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS evidence_no_update
                    BEFORE UPDATE ON evidence
                    BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS evidence_no_delete
                    BEFORE DELETE ON evidence
                    BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS reviews_no_update
                    BEFORE UPDATE ON reviews
                    BEGIN SELECT RAISE(ABORT, 'reviews are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS reviews_no_delete
                    BEFORE DELETE ON reviews
                    BEGIN SELECT RAISE(ABORT, 'reviews are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                    BEFORE UPDATE ON audit_events
                    BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                    BEFORE DELETE ON audit_events
                    BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
            """)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def append(self, entry):
        with self._connect() as db:
            existing = db.execute(
                "SELECT payload FROM evidence WHERE id=?", (entry.evidence_id,)
            ).fetchone()
            if existing and existing[0] != entry.payload_json:
                raise ValueError("Evidence integrity conflict")
            db.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?,?)",
                (entry.evidence_id, entry.payload_json),
            )
        return entry.evidence_id

    def get(self, evidence_id):
        with self._connect() as db:
            row = db.execute("SELECT payload FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        if not row:
            raise KeyError(evidence_id)
        entry = EvidenceEntry(row[0])
        if entry.evidence_id != evidence_id:
            raise ValueError("Evidence integrity check failed")
        return entry

    def entries(self, *, as_of=None, evidence_ids=None):
        allowed = None if evidence_ids is None else set(evidence_ids)
        with self._connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM evidence ORDER BY id")]
        result = []
        cutoff = instant(as_of) if as_of is not None else None
        for oid in ids:
            if allowed is not None and oid not in allowed:
                continue
            entry = self.get(oid)
            if cutoff is not None and instant(entry.payload["evidence_available_as_of"]) > cutoff:
                raise ValueError("Evidence is not available at the requested export cutoff")
            result.append(entry)
        return tuple(result)

    def review(self, review):
        entry = self.get(review.evidence_id)
        latest = max(
            [instant(f["available_as_of"]) for f in entry.payload["source_facts"]]
            + [instant(entry.payload["evidence_available_as_of"])]
        )
        if instant(review.reviewed_at) < latest:
            raise ValueError("Review predates available evidence")
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO reviews VALUES (?,?,?,?)",
                (review.review_id, review.evidence_id, review.reviewed_at, canonical_json(review)),
            )
            event = AuditEvent(
                review.reviewed_at,
                "analyst_review",
                "recorded",
                review.reviewer,
                review.evidence_id,
                (("disposition", review.disposition.value), ("review_id", review.review_id)),
            )
            db.execute(
                "INSERT OR IGNORE INTO audit_events VALUES (?,?,?)",
                (event.event_id, event.occurred_at, canonical_json(event)),
            )
        return review.review_id

    def record_audit_event(self, event):
        """Append an operational event without changing analytical evidence."""
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO audit_events VALUES (?,?,?)",
                (event.event_id, event.occurred_at, canonical_json(event)),
            )
        return event.event_id

    def audit_events(self, *, as_of=None):
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,payload FROM audit_events ORDER BY occurred_at,id"
            ).fetchall()
        result = []
        for oid, payload in rows:
            event = AuditEvent(**json.loads(payload))
            if event.event_id != oid:
                raise ValueError("Audit event integrity check failed")
            if as_of is None or instant(event.occurred_at) <= instant(as_of):
                result.append(event)
        return tuple(result)

    def reviews(self, *, as_of=None):
        with self._connect() as db:
            rows = db.execute("SELECT id,payload FROM reviews ORDER BY reviewed_at,id").fetchall()
        result = []
        for oid, payload in rows:
            review = AnalystReview(**json.loads(payload))
            if review.review_id != oid:
                raise ValueError("Review integrity check failed")
            if as_of is None or instant(review.reviewed_at) <= instant(as_of):
                result.append(review)
        return tuple(result)

    def latest_reviews(self, *, as_of=None):
        return {review.evidence_id: review for review in self.reviews(as_of=as_of)}

    def export(self, directory, *, as_of=None, evidence_ids=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        entries = self.entries(as_of=as_of, evidence_ids=evidence_ids)
        snapshot_id = stable_id([e.evidence_id for e in entries])
        rows = [dict(evidence_id=e.evidence_id, **e.payload) for e in entries]
        json_path = write_immutable(
            directory / f"ledger-{snapshot_id[:16]}.json", canonical_json(rows) + "\n"
        )
        jsonl_path = write_immutable(
            directory / f"ledger-{snapshot_id[:16]}.jsonl",
            "".join(canonical_json(row) + "\n" for row in rows),
        )
        for entry in entries:
            write_immutable(directory / f"{entry.evidence_id}.md", entry.to_markdown())
        allowed = None if evidence_ids is None else set(evidence_ids)
        reviews = tuple(
            review
            for review in self.reviews(as_of=as_of)
            if allowed is None or review.evidence_id in allowed
        )
        review_path = write_immutable(
            directory / f"reviews-{stable_id(reviews)[:16]}.jsonl",
            "".join(canonical_json(r) + "\n" for r in reviews),
        )
        audit = self.audit_events(as_of=as_of)
        audit_path = write_immutable(
            directory / f"audit-{stable_id(audit)[:16]}.jsonl",
            "".join(canonical_json(event) + "\n" for event in audit),
        )
        return {
            "json": json_path,
            "jsonl": jsonl_path,
            "reviews": review_path,
            "audit": audit_path,
        }
