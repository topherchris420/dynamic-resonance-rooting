"""Public policy/reporting context with explicit availability and applicability basis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .common import day, instant, source_url, stable_id


class ApplicabilityBasis(str, Enum):
    AUTHORITATIVE_APPLICABILITY = "authoritative_applicability"
    ANALYST_INTERPRETATION = "analyst_interpretation"
    POSSIBLE_RELEVANCE = "possible_relevance"


@dataclass(frozen=True)
class PolicyEvent:
    identifier: str
    authority: str
    title: str
    publication_date: str
    effective_date: Optional[str]
    status: str
    institution_classes: Tuple[str, ...]
    forms: Tuple[str, ...]
    metrics: Tuple[str, ...]
    topic: str
    source: str
    available_as_of: str
    analyst_interpretation: str = ""
    confidence: float = 1.0
    applicability: ApplicabilityBasis = ApplicabilityBasis.POSSIBLE_RELEVANCE
    applicability_evidence: str = ""
    comparability_break: bool = False

    def __post_init__(self):
        object.__setattr__(self, "applicability", ApplicabilityBasis(self.applicability))
        for name in ("institution_classes", "forms", "metrics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        source_url(self.source)
        if instant(self.available_as_of) < instant(self.publication_date):
            raise ValueError("Policy event cannot be available before publication")
        if self.effective_date:
            day(self.effective_date)
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0,1]")
        if (
            self.applicability == ApplicabilityBasis.AUTHORITATIVE_APPLICABILITY
            and not self.applicability_evidence
        ):
            raise ValueError("Authoritative applicability requires a quoted/source-specific scope")
        if self.comparability_break and not self.effective_date:
            raise ValueError("Reporting break needs an effective date")

    @property
    def event_id(self):
        return stable_id(self)


class PolicyContext:
    def __init__(self, events=()):
        self.events = tuple(sorted(events, key=lambda e: (e.publication_date, e.event_id)))

    def relevant(self, *, as_of, form=None, metric=None, institution_class=None):
        return tuple(
            e
            for e in self.events
            if instant(e.available_as_of) <= instant(as_of)
            and (not e.forms or form in e.forms)
            and (not e.metrics or metric in e.metrics)
            and (not e.institution_classes or institution_class in e.institution_classes)
        )

    def breakpoints(self, *, as_of, form, metric, dates, institution_class=None):
        points = []
        for event in self.relevant(
            as_of=as_of, form=form, metric=metric, institution_class=institution_class
        ):
            if event.comparability_break:
                first = next((d for d in dates if day(d) >= day(event.effective_date)), None)
                if first:
                    points.append(first)
        return tuple(sorted(set(points)))
