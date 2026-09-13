"""Analyst-owned work items with append-only issue history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .common import canonical_json, instant, stable_id


@dataclass(frozen=True)
class MonitoringIssue:
    issue_key: str
    institution: str
    title: str
    description: str
    source: str
    evidence_ids: Tuple[str, ...]
    priority: str
    status: str
    owner: str
    created_date: str
    review_date: str
    due_date: Optional[str] = None
    analyst_notes: Tuple[str, ...] = ()
    unresolved_questions: Tuple[str, ...] = ()
    data_quality_blockers: Tuple[str, ...] = ()
    disposition: str = "unresolved"

    def __post_init__(self):
        for name in (
            "evidence_ids",
            "analyst_notes",
            "unresolved_questions",
            "data_quality_blockers",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all((self.issue_key, self.title, self.owner)) or self.source not in {
            "public_data",
            "analytical_exception",
            "synthetic_test",
            "user_entered",
        }:
            raise ValueError("Issue needs identity, owner and an allowed source")
        if self.priority not in {"low", "normal", "high"} or self.status not in {
            "open",
            "investigating",
            "blocked",
            "resolved",
            "closed",
        }:
            raise ValueError("Invalid issue priority/status")
        if instant(self.review_date) < instant(self.created_date):
            raise ValueError("Issue review predates creation")
        if self.due_date:
            instant(self.due_date)

    @property
    def revision_id(self):
        return stable_id(self)


class IssueTracker:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.path)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS issues (id TEXT PRIMARY KEY, issue_key TEXT, review_date TEXT, payload TEXT)"
            )

    def append(self, issue, ledger):
        for oid in issue.evidence_ids:
            ledger.get(oid)
        with sqlite3.connect(str(self.path)) as db:
            db.execute(
                "INSERT OR IGNORE INTO issues VALUES (?,?,?,?)",
                (
                    issue.revision_id,
                    issue.issue_key,
                    instant(issue.review_date).isoformat(),
                    canonical_json(issue),
                ),
            )
        return issue.revision_id

    def current(self, *, as_of):
        with sqlite3.connect(str(self.path)) as db:
            rows = db.execute(
                "SELECT id,payload FROM issues WHERE review_date<=? ORDER BY review_date,id",
                (instant(as_of).isoformat(),),
            ).fetchall()
        current = {}
        for oid, payload in rows:
            issue = MonitoringIssue(**json.loads(payload))
            if issue.revision_id != oid:
                raise ValueError("Issue history integrity check failed")
            current[issue.issue_key] = issue
        return tuple(current[k] for k in sorted(current))
