"""Outcome analysis kept separate from conventional challenge-model estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import matched_evaluation
from .challenge_models import _integer, _labels_as_of, _quarter, _text
from .common import canonical, instant, stable_id


@dataclass(frozen=True)
class ChallengeEvaluationDesign:
    """Caller-declared design; labels concern explicitly observed institution-quarters.

    The lead window is measured in calendar quarters on a complete review grid.
    Positive labels need ``event_at`` to establish whether a warning preceded an
    event. A small event inventory withholds a performance comparison while
    retaining descriptive counts and alert burden.
    """

    event_definition: str
    registered_at: str
    lead_window: int = 2
    minimum_events: int = 5

    def __post_init__(self):
        _text(self.event_definition, "event_definition")
        object.__setattr__(self, "registered_at", instant(self.registered_at).isoformat())
        _integer(self.lead_window, "lead_window", 0)
        _integer(self.minimum_events, "minimum_events")


def evaluate_challenge_results(results, labels, *, design, evaluation_as_of, drr_rows=None):
    """Detection performance comparison under a pre-registered evaluation design.

    Model fitting is already complete. Later labels enter this evaluation only.
    ``drr_rows``, when supplied, must carry institution_id, as_of, input_hash and
    an explicit boolean-or-None drr_alert. Their input hashes must bind to the
    challenge snapshots. Matched alert burden reuses ``matched_evaluation``;
    disagreement is exposed separately. No outcome absence is made a negative.
    """
    results, labels = tuple(results), tuple(labels)
    if not results:
        raise ValueError("At least one challenge result is required")
    if len({stable_id(r.config) for r in results}) != 1:
        raise ValueError("Evaluation requires one fixed challenge specification")
    if instant(design.registered_at) >= min(instant(r.review.as_of) for r in results):
        raise ValueError("Evaluation design must be registered before the first review")
    if instant(evaluation_as_of) < max(instant(r.review.as_of) for r in results):
        raise ValueError("Evaluation cutoff precedes a score's availability")
    if hasattr(results[0].config, "event_definition"):
        if results[0].config.event_definition != design.event_definition:
            raise ValueError("Evaluation and fitted outcome definitions differ")
    label_map = _labels_as_of(labels, design.event_definition, evaluation_as_of)
    keys = [(r.institution_id, r.review.as_of) for r in results]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate institution/review result")
    comparison = None
    if drr_rows is not None:
        drr_rows = tuple(drr_rows)
        comparison = {(r["institution_id"], instant(r["as_of"]).isoformat()): r for r in drr_rows}
        if len(comparison) != len(drr_rows) or set(comparison) != set(keys):
            raise ValueError("DRR and challenge institution/review grids must match exactly")
        for result in results:
            other = comparison[(result.institution_id, result.review.as_of)]
            if other["input_hash"] != result.information_set["input_hash"]:
                raise ValueError("DRR and challenge source information sets differ")
            if other["drr_alert"] is not None and not isinstance(other["drr_alert"], bool):
                raise ValueError("drr_alert must be boolean or None (unavailable)")
    institutions = {}
    for institution in sorted({r.institution_id for r in results}):
        rows = sorted(
            (r for r in results if r.institution_id == institution),
            key=lambda r: r.review.reporting_period,
        )
        periods = [_quarter(r.review.reporting_period) for r in rows]
        if any(b.ordinal - a.ordinal != 1 for a, b in zip(periods, periods[1:])):
            raise ValueError(
                "Evaluation requires every calendar quarter, including withheld reviews"
            )
        if any(instant(b.review.as_of) <= instant(a.review.as_of) for a, b in zip(rows, rows[1:])):
            raise ValueError("Review times must increase with reporting periods")
        outcomes = [label_map.get((institution, r.review.reporting_period)) for r in rows]
        flags = [r.flagged for r in rows]
        metrics = _event_metrics(rows, flags, outcomes, design)
        metrics["withholding_reasons"] = tuple(
            {"as_of": r.review.as_of, "reason": r.withholding_reason}
            for r in rows
            if r.status == "withheld"
        )
        if comparison is not None:
            drr_flags = [comparison[(institution, r.review.as_of)]["drr_alert"] for r in rows]
            metrics["drr_detection"] = _event_metrics(rows, drr_flags, outcomes, design)
            metrics["disagreement_reviews"] = tuple(
                r.review.as_of
                for r, a, b in zip(rows, flags, drr_flags)
                if a is not None and b is not None and a != b
            )
            if any(f is None for f in flags + drr_flags):
                metrics["matched_evaluation"] = {
                    "status": "withheld",
                    "reason": "At least one matched score is unavailable; no review was dropped",
                }
            else:
                metrics["matched_evaluation"] = matched_evaluation(
                    [
                        dict(as_of=r.review.as_of, baseline_alert=a, drr_alert=b)
                        for r, a, b in zip(rows, flags, drr_flags)
                    ],
                    events=None,
                    lead_window=design.lead_window,
                )
        institutions[institution] = metrics
    return dict(
        evidence_class="EVENT_DETECTION_EVALUATION",
        description="detection performance comparison under a pre-registered evaluation design",
        design=canonical(design),
        evaluation_as_of=instant(evaluation_as_of).isoformat(),
        result_ids=tuple(r.result_id for r in results),
        by_institution=institutions,
        label_ids=tuple(
            sorted(
                label.label_id
                for label in label_map.values()
                if (label.institution_id, label.reporting_period)
                in {(r.institution_id, r.review.reporting_period) for r in results}
            )
        ),
        limitations=(
            "Registration timestamps are caller declarations, not verified external registrations.",
            "Counts concern labeled institution-quarters and may share economic episodes.",
            "Detection is measured against the supplied inventory; unknown outcomes are not non-events.",
            "Source-hash equality is necessary, but cannot attest to the caller's DRR fitting procedure.",
            "Synthetic recovery and numerical stability do not establish real-world usefulness.",
        ),
    )


def _event_metrics(rows, flags, outcomes, design):
    from ..validation_readiness import run_event_backtest

    n, lead = len(rows), design.lead_window
    positives = [i for i, label in enumerate(outcomes) if label is not None and label.value == 1]
    alerts = [i for i, flag in enumerate(flags) if flag is True]
    timing_missing = any(outcomes[i].event_at is None for i in positives)
    base = dict(
        event_count=len(positives),
        labeled_nonevent_count=sum(label is not None and label.value == 0 for label in outcomes),
        unknown_outcome_count=sum(label is None or label.value is None for label in outcomes),
        alert_count=len(alerts),
        alert_burden=len(alerts) / n,
        scored_observation_count=sum(flag is not None for flag in flags),
        unscored_observation_count=sum(flag is None for flag in flags),
        unscored_event_count=sum(flags[i] is None for i in positives),
        right_censored_reviews=min(lead, n),
        right_censored_alerts=sum(i + lead >= n for i in alerts),
        detected_events=None,
        missed_events=None,
        events_without_scored_lead_window=None,
        false_positive_alerts=None,
        confirmed_false_positive_alerts=None,
        indeterminate_alerts=None,
        mean_lead_time_periods=None,
        lead_time_periods=(),
        lead_time_days=(),
        detection_rate=None,
        precision=None,
    )
    if timing_missing:
        return dict(
            base,
            status="withheld",
            reason="Positive outcome timing is unavailable; post-event alerts cannot be credited",
        )
    detected, leads, elapsed = set(), [], []
    false_positives = indeterminate = true_alerts = 0
    for alert in alerts:
        # An event's grid position alone cannot prove temporal precedence. Pass
        # only events not already over at this alert's actual availability time.
        eligible_events = [
            i
            for i in positives
            if instant(outcomes[i].event_at) >= instant(rows[alert].review.as_of)
        ]
        scores = np.full(n, np.nan)
        scores[alert] = 1.0
        frame = pd.DataFrame(
            {
                "date": [r.review.reporting_period for r in rows],
                "score": scores,
                "known_event": [i in eligible_events for i in range(n)],
            }
        )
        raw = run_event_backtest(
            frame,
            date_column="date",
            score_column="score",
            event_column="known_event",
            threshold=0.5,
            lead_window=lead,
        )
        if raw["true_positive_alerts"]:
            event = min(i for i in eligible_events if alert <= i <= alert + lead)
            detected.add(event)
            true_alerts += 1
            leads.append(event - alert)
            elapsed.append(
                (
                    instant(outcomes[event].event_at) - instant(rows[alert].review.as_of)
                ).total_seconds()
                / 86400
            )
        elif alert + lead < n and all(
            label is not None and label.value is not None
            for label in outcomes[alert : alert + lead + 1]
        ):
            false_positives += 1
        else:
            indeterminate += 1
    no_scored_window = sum(
        not any(
            flags[j] is not None and instant(rows[j].review.as_of) <= instant(outcomes[i].event_at)
            for j in range(max(0, i - lead), i + 1)
        )
        for i in positives
    )
    base.update(
        detected_events=len(detected),
        missed_events=len(positives) - len(detected),
        events_without_scored_lead_window=no_scored_window,
        false_positive_alerts=false_positives if not indeterminate else None,
        confirmed_false_positive_alerts=false_positives,
        indeterminate_alerts=indeterminate,
        lead_time_periods=tuple(leads),
        lead_time_days=tuple(elapsed),
        mean_lead_time_periods=float(np.mean(leads)) if leads else None,
    )
    if len(positives) < design.minimum_events:
        return dict(
            base,
            status="withheld",
            reason="Too few labeled events for the configured performance comparison",
        )
    base.update(
        detection_rate=len(detected) / len(positives),
        precision=true_alerts / len(alerts) if alerts and not indeterminate else None,
    )
    return dict(base, status="available", reason=None)
