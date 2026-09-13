"""Preserve scoped observations and disagreement without ranking ways of knowing."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Optional, Tuple

from .common import canonical, instant, stable_id


class PerspectiveKind(str, Enum):
    INSTITUTIONAL_ASSESSMENT = "institutional_assessment"
    QUANTITATIVE_MODEL = "quantitative_model"
    EXPERT_JUDGMENT = "expert_judgment"
    LOCAL_MEASUREMENT = "local_measurement"
    DISTRIBUTIONAL_INDICATOR = "distributional_indicator"
    LIVED_EXPERIENCE = "lived_experience"
    CONTEXTUAL_EVIDENCE = "contextual_evidence"


@dataclass(frozen=True)
class ScopedPerspective:
    system: str
    kind: PerspectiveKind
    claim: str
    author: str
    population: str
    geography: str
    scale: str
    dimension: str
    period_start: str
    period_end: str
    horizon: str
    available_as_of: str
    evidence_ids: Tuple[str, ...]
    method: str
    limitations: Tuple[str, ...]
    outside_scope: Tuple[str, ...]
    support_assessment: str
    support_assessor: str
    interpretation: str = ""
    confidence: Optional[float] = None
    institution_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "kind", PerspectiveKind(self.kind))
        for name in ("evidence_ids", "limitations", "outside_scope"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "institution_ids", tuple(sorted(set(self.institution_ids))))
        for name in (
            "system",
            "claim",
            "author",
            "population",
            "geography",
            "scale",
            "dimension",
            "horizon",
            "method",
            "support_assessor",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("Perspective requires explicit identity, scope and method")
        for name in ("period_start", "period_end", "available_as_of"):
            object.__setattr__(self, name, instant(getattr(self, name)).isoformat())
        if instant(self.period_start) > instant(self.period_end):
            raise ValueError("Perspective period is reversed")
        if not self.evidence_ids or not self.limitations or not self.outside_scope:
            raise ValueError("Evidence, limitations and representation boundaries are required")
        if self.support_assessment not in {"supported", "mixed", "unsupported", "not_evaluated"}:
            raise ValueError("Invalid attributed support assessment")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite and in [0,1]")

    @property
    def perspective_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class DocumentedDisagreement:
    perspective_ids: Tuple[str, ...]
    interpretation: str
    recorded_by: str
    available_as_of: str

    def __post_init__(self):
        object.__setattr__(self, "perspective_ids", tuple(sorted(set(self.perspective_ids))))
        object.__setattr__(self, "available_as_of", instant(self.available_as_of).isoformat())
        if (
            len(self.perspective_ids) < 2
            or not self.interpretation.strip()
            or not self.recorded_by.strip()
        ):
            raise ValueError(
                "Disagreement requires distinct perspectives and attributed interpretation"
            )

    @property
    def disagreement_id(self):
        return stable_id(self)


class PerspectiveInventory:
    """Immutable records; append returns a new inventory. No consensus score."""

    def __init__(self, perspectives=(), disagreements=()):
        self._perspectives = tuple(sorted(set(perspectives), key=lambda p: p.perspective_id))
        self._disagreements = tuple(sorted(set(disagreements), key=lambda d: d.disagreement_id))
        by_id = {p.perspective_id: p for p in self._perspectives}
        for disagreement in self._disagreements:
            if not set(disagreement.perspective_ids) <= set(by_id):
                raise ValueError("Disagreement references missing perspectives")
            members = [by_id[i] for i in disagreement.perspective_ids]
            if len({p.system for p in members}) != 1:
                raise ValueError("Disagreement must identify perspectives on the same system")
            if any(
                instant(p.available_as_of) > instant(disagreement.available_as_of) for p in members
            ):
                raise ValueError("Disagreement predates available perspectives")

    def append(self, *perspectives, disagreements=()):
        return type(self)(
            self._perspectives + perspectives, self._disagreements + tuple(disagreements)
        )

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = payload if isinstance(payload, dict) else {"perspectives": payload}
        perspectives = []
        for row in payload.get("perspectives", ()):
            values = dict(row)
            values.pop("perspective_id", None)
            perspectives.append(ScopedPerspective(**values))
        disagreements = []
        for row in payload.get("disagreements", ()):
            values = dict(row)
            for key in (
                "disagreement_id",
                "differing_scope_fields",
                "comparison_status",
                "conclusion",
            ):
                values.pop(key, None)
            disagreements.append(DocumentedDisagreement(**values))
        return cls(tuple(perspectives), tuple(disagreements))

    def snapshot(self, ledger=None, *, as_of, observations=()):
        perspectives = [
            p for p in self._perspectives if instant(p.available_as_of) <= instant(as_of)
        ]
        observed_by_id = {o.observation_id: o for o in observations}
        for p in perspectives:
            for oid in p.evidence_ids:
                entry = None
                if ledger is not None:
                    try:
                        entry = ledger.get(oid)
                    except KeyError:
                        entry = None
                if entry is not None:
                    evidence_available = [
                        entry.payload.get("evidence_available_as_of")
                        or max(f["available_as_of"] for f in entry.payload["source_facts"])
                    ]
                elif oid in observed_by_id:
                    evidence_available = [observed_by_id[oid].available_as_of]
                else:
                    raise ValueError("Perspective references missing evidence")
                if any(instant(value) > instant(p.available_as_of) for value in evidence_available):
                    raise ValueError("Perspective leaks future evidence")
        disagreements = []
        by_id = {p.perspective_id: p for p in perspectives}
        for d in self._disagreements:
            if instant(d.available_as_of) > instant(as_of):
                continue
            members = [by_id[i] for i in d.perspective_ids]
            differences = [
                name
                for name in (
                    "population",
                    "geography",
                    "scale",
                    "dimension",
                    "period_start",
                    "period_end",
                    "horizon",
                    "method",
                )
                if len({getattr(p, name) for p in members}) > 1
            ]
            disagreements.append(
                dict(
                    disagreement_id=d.disagreement_id,
                    **canonical(d),
                    differing_scope_fields=differences,
                    comparison_status=(
                        "different_observational_scopes"
                        if differences
                        else "shared_scope_requires_review"
                    ),
                    conclusion="Preserve all referenced perspectives. Scope differences may explain divergence; they do not establish agreement or refute either claim.",
                )
            )
        return dict(
            perspectives=[
                dict(perspective_id=p.perspective_id, **canonical(p)) for p in perspectives
            ],
            disagreements=disagreements,
            principle="A model can be excellent without being exhaustive. Accuracy within a representation does not establish completeness.",
            boundary="Support assessments are attributed judgments, not automatically verified truth. Authority, statistical confidence and lived experience do not determine precedence.",
        )
