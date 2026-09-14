"""Review-state deltas and an explicit analyst attention budget."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Optional, Tuple

from .common import canonical, instant, stable_id


@dataclass(frozen=True)
class MonitoringSignal:
    key: str
    institution: str
    metric: str
    period: str
    claim: str
    evidence_id: str
    materiality: float
    confidence: float
    persistence: int = 0
    peer_divergence: float = 0.0
    robustness: Optional[float] = None
    drr_incremental: bool = False
    disposition: str = "unresolved"
    review_complexity: Optional[str] = None

    def __post_init__(self):
        if not self.evidence_id or not self.key:
            raise ValueError("Analyst-facing signals require evidence")
        if (
            not 0 <= self.materiality <= 100
            or not 0 <= self.confidence <= 1
            or not 0 <= self.peer_divergence <= 1
        ):
            raise ValueError("Invalid signal score/confidence")
        if self.robustness is not None and not 0 <= self.robustness <= 1:
            raise ValueError("Robustness must be in [0,1]")


@dataclass(frozen=True)
class ReviewState:
    as_of: str
    observations: Tuple[Tuple[str, str], ...] = ()
    signals: Tuple[MonitoringSignal, ...] = ()
    exceptions: Tuple[str, ...] = ()
    policy_events: Tuple[str, ...] = ()
    entity_relationships: Tuple[str, ...] = ()
    drr_relationships: Tuple[str, ...] = ()
    failed_relationships: Tuple[str, ...] = ()

    def __post_init__(self):
        instant(self.as_of)
        for name in (
            "observations",
            "signals",
            "exceptions",
            "policy_events",
            "entity_relationships",
            "drr_relationships",
            "failed_relationships",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len({s.key for s in self.signals}) != len(self.signals):
            raise ValueError("Duplicate signal key in review state")
        if any(
            not isinstance(row, (tuple, list))
            or len(row) != 2
            or any(not isinstance(value, str) or not value for value in row)
            for row in self.observations
        ):
            raise ValueError("Review observations require nonempty key/ID pairs")
        object.__setattr__(self, "observations", tuple(tuple(row) for row in self.observations))
        if len(dict(self.observations)) != len(self.observations):
            raise ValueError("Duplicate observation key in review state")

    @property
    def state_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class MonitoringDelta:
    previous_as_of: Optional[str]
    current_as_of: str
    new_observations: Tuple[str, ...]
    changed_observations: Tuple[str, ...]
    new_signals: Tuple[MonitoringSignal, ...]
    strengthened: Tuple[MonitoringSignal, ...]
    weakened: Tuple[MonitoringSignal, ...]
    changed_evidence: Tuple[MonitoringSignal, ...]
    disappeared: Tuple[MonitoringSignal, ...]
    new_exceptions: Tuple[str, ...]
    resolved_exceptions: Tuple[str, ...]
    new_policy_events: Tuple[str, ...]
    entity_changes: Tuple[str, ...]
    new_relationships: Tuple[str, ...]
    failed_relationships: Tuple[str, ...]


def compare_review_states(previous_review_state, current_state):
    prev = previous_review_state or ReviewState(current_state.as_of)
    if instant(current_state.as_of) < instant(prev.as_of):
        raise ValueError("Current review precedes previous state")
    old, new = {s.key: s for s in prev.signals}, {s.key: s for s in current_state.signals}
    po, co = dict(prev.observations), dict(current_state.observations)

    def added(name):
        return tuple(sorted(set(getattr(current_state, name)) - set(getattr(prev, name))))

    return MonitoringDelta(
        previous_review_state.as_of if previous_review_state else None,
        current_state.as_of,
        tuple(sorted(k for k in co if k not in po)),
        tuple(sorted(k for k in co if k in po and co[k] != po[k])),
        tuple(new[k] for k in sorted(new.keys() - old.keys())),
        tuple(
            new[k]
            for k in sorted(new.keys() & old.keys())
            if new[k].materiality > old[k].materiality
        ),
        tuple(
            new[k]
            for k in sorted(new.keys() & old.keys())
            if new[k].materiality < old[k].materiality
        ),
        tuple(
            new[k]
            for k in sorted(new.keys() & old.keys())
            if new[k].evidence_id != old[k].evidence_id and new[k].materiality == old[k].materiality
        ),
        tuple(old[k] for k in sorted(old.keys() - new.keys())),
        added("exceptions"),
        tuple(sorted(set(prev.exceptions) - set(current_state.exceptions))),
        added("policy_events"),
        tuple(sorted(set(prev.entity_relationships) ^ set(current_state.entity_relationships))),
        added("drr_relationships"),
        added("failed_relationships"),
    )


@dataclass(frozen=True)
class AttentionItem:
    signal: MonitoringSignal
    priority: float
    components: Tuple[Tuple[str, float], ...]
    novelty: str


@dataclass(frozen=True)
class AttentionSelection:
    review_first: Tuple[AttentionItem, ...]
    deferred: Tuple[AttentionItem, ...]
    deprioritized: Tuple[MonitoringSignal, ...]


@dataclass(frozen=True)
class AttentionBudget:
    top_n: int = 5
    minimum_materiality: float = 20.0
    minimum_confidence: float = 0.5

    def __post_init__(self):
        if (
            isinstance(self.top_n, bool)
            or not isinstance(self.top_n, Integral)
            or self.top_n < 0
            or not 0 <= self.minimum_materiality <= 100
            or not 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("Invalid attention budget")

    def select(self, delta):
        candidates = []
        deprioritized = list(delta.weakened + delta.disappeared)
        for novelty, signals in (
            ("new", delta.new_signals),
            ("strengthened", delta.strengthened),
            ("evidence changed", delta.changed_evidence),
        ):
            for signal in signals:
                if (
                    signal.materiality < self.minimum_materiality
                    or signal.confidence < self.minimum_confidence
                    or signal.disposition in {"useful", "explained", "noisy", "dismissed"}
                ):
                    deprioritized.append(signal)
                    continue
                components = (
                    ("materiality", signal.materiality * 0.45),
                    ("novelty", 15.0 if novelty == "new" else 8.0),
                    ("persistence", min(signal.persistence, 4) * 2.0),
                    ("peer_divergence", signal.peer_divergence * 10),
                    ("robustness", (signal.robustness or 0) * 10),
                    ("data_confidence", signal.confidence * 10),
                    ("unresolved", 2.0 if signal.disposition == "unresolved" else 0),
                    ("incremental_information", 5.0 if signal.drr_incremental else 0),
                )
                candidates.append(
                    AttentionItem(signal, sum(v for _, v in components), components, novelty)
                )
        ranked = tuple(sorted(candidates, key=lambda item: (-item.priority, item.signal.key)))
        return AttentionSelection(ranked[: self.top_n], ranked[self.top_n :], tuple(deprioritized))


def build_review_activity(snapshot, reviews, *, as_of):
    """Project current human activity onto an immutable analytical snapshot.

    Rebuild the attention budget from its original candidates so resolving an item
    promotes the next deferred item and reopening it restores its original rank.
    Reviews of evidence outside this snapshot are deliberately excluded.
    """
    cutoff = instant(as_of)
    latest = {}
    allowed = set(snapshot["evidence"])
    for review in sorted(reviews, key=lambda r: (instant(r.reviewed_at), r.review_id)):
        if review.evidence_id in allowed and instant(review.reviewed_at) <= cutoff:
            latest[review.evidence_id] = review

    def signal(value):
        item = MonitoringSignal(**value)
        review = latest.get(item.evidence_id)
        return replace(item, disposition=review.disposition.value) if review else item

    delta = dict(snapshot["delta"])
    for name in ("new_signals", "strengthened", "weakened", "changed_evidence", "disappeared"):
        delta[name] = tuple(signal(s) for s in delta[name])
    attention = AttentionBudget(top_n=snapshot["config"]["top_n"]).select(MonitoringDelta(**delta))
    return canonical(
        {
            "analysis_id": snapshot["passport"]["analysis_id"],
            "analytical_as_of": snapshot["as_of"],
            "review_as_of": cutoff.isoformat(),
            "reviews": [
                dict(review_id=r.review_id, **canonical(r)) for _, r in sorted(latest.items())
            ],
            "signals": [signal(s) for s in snapshot["state"]["signals"]],
            "attention": attention,
        }
    )
