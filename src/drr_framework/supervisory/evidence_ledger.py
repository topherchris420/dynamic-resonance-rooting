"""Content-addressed evidence and append-only analyst dispositions in local SQLite."""

from __future__ import annotations

import json
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
        if not self.reviewer.strip() or not self.rationale.strip() or not self.evidence_id:
            raise ValueError("Analyst identity, evidence and rationale are required")
        if not 0 <= self.review_minutes <= 24 * 60:
            raise ValueError("Review minutes must be finite and in [0,1440]")

    @property
    def review_id(self):
        return stable_id(self)


class EvidenceLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL, payload TEXT NOT NULL, FOREIGN KEY(evidence_id) REFERENCES evidence(id));
            """
            )

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

    def entries(self):
        with self._connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM evidence ORDER BY id")]
        return tuple(self.get(oid) for oid in ids)

    def review(self, review):
        entry = self.get(review.evidence_id)
        latest = max(instant(f["available_as_of"]) for f in entry.payload["source_facts"])
        if instant(review.reviewed_at) < latest:
            raise ValueError("Review predates available evidence")
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO reviews VALUES (?,?,?,?)",
                (review.review_id, review.evidence_id, review.reviewed_at, canonical_json(review)),
            )
        return review.review_id

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

    def export(self, directory):
        directory = Path(directory)
        entries = self.entries()
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
        reviews = self.reviews()
        review_path = write_immutable(
            directory / f"reviews-{stable_id(reviews)[:16]}.jsonl",
            "".join(canonical_json(r) + "\n" for r in reviews),
        )
        return {"json": json_path, "jsonl": jsonl_path, "reviews": review_path}
