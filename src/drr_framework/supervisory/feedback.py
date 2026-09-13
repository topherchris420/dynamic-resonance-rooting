"""Measure analyst feedback without fitting models to human dispositions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AnalystFeedbackMetrics:
    reviewed_signals: int
    review_events: int
    usefulness_rate: Optional[float]
    noise_rate: Optional[float]
    dismissal_rate: Optional[float]
    investigation_rate: Optional[float]
    total_review_minutes: float
    metric_usefulness: Tuple[Tuple[str, int, int], ...]
    peer_group_usefulness: Tuple[Tuple[str, int, int], ...]
    drr_incremental_usefulness: Optional[float]
    repeated_noise_patterns: Tuple[Tuple[str, int], ...]
    average_lead_time: Optional[float] = None
    limitation: str = (
        "Descriptive analyst feedback, subject to selection and reviewer bias. Lead time needs independent event dates."
    )


def evaluate_signal_usefulness(ledger, *, as_of=None):
    reviews = ledger.reviews(as_of=as_of)
    latest = ledger.latest_reviews(as_of=as_of)
    n = len(latest)

    def rate(status):
        return sum(r.disposition.value == status for r in latest.values()) / n if n else None

    metrics, peers, noisy = {}, {}, {}
    incremental = []
    for oid, review in latest.items():
        p = ledger.get(oid).payload
        useful = review.disposition.value == "useful"
        for store, key in (
            (metrics, p["metric"]),
            (peers, p["peer_context"].get("definition", {}).get("name", "unspecified")),
        ):
            current = store.setdefault(key, [0, 0])
            current[0] += 1
            current[1] += int(useful)
        if review.disposition.value in {"noisy", "dismissed"}:
            noisy[p["metric"]] = noisy.get(p["metric"], 0) + 1
        if p["drr_evidence"].get("incremental_alert", False):
            incremental.append(useful)
    return AnalystFeedbackMetrics(
        n,
        len(reviews),
        rate("useful"),
        rate("noisy"),
        rate("dismissed"),
        rate("investigate"),
        sum(r.review_minutes for r in reviews),
        tuple((k, *v) for k, v in sorted(metrics.items())),
        tuple((k, *v) for k, v in sorted(peers.items())),
        sum(incremental) / len(incremental) if incremental else None,
        tuple((k, v) for k, v in sorted(noisy.items()) if v > 1),
    )
