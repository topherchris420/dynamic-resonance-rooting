"""Deterministic exceptions; questionable observations are never repaired silently."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .common import stable_id


@dataclass(frozen=True)
class DataQualityException:
    severity: str
    institution: str
    metric: str
    period: str
    issue_type: str
    evidence: Tuple[str, ...]
    likely_explanation: Optional[str] = None
    required_analyst_review: bool = True
    source_links: Tuple[str, ...] = ()
    affected_calculations: Tuple[str, ...] = ()

    @property
    def exception_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class ReconciliationRule:
    """Analyst-supplied comparable-metric identity: total=sum(parts) or ratio=num/den."""

    metric: str
    inputs: Tuple[str, ...]
    operation: str = "sum"
    multiplier: float = 1.0
    tolerance: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if self.operation not in {"sum", "ratio"} or not self.inputs:
            raise ValueError("Invalid reconciliation rule")
        if self.operation == "ratio" and len(self.inputs) != 2:
            raise ValueError("Ratio needs exactly two inputs")
        if (
            not np.isfinite(self.multiplier)
            or not np.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError("Invalid reconciliation multiplier/tolerance")


def reconcile_dataset(dataset, *, rules=(), extreme_change=0.5, carry_forward_periods=4):
    if extreme_change <= 0 or carry_forward_periods < 2:
        raise ValueError("Invalid reconciliation thresholds")
    issues = []
    records = {(o.metric, o.reporting_period): o for o in dataset.observations}
    frame = dataset.frame

    def add(kind, metric, period, severity="warning", explanation=None, inputs=()):
        observations = [
            records[(m, p)] for m, p in ((metric, period), *inputs) if (m, p) in records
        ]
        issues.append(
            DataQualityException(
                severity,
                dataset.institution_id,
                metric,
                period,
                kind,
                tuple(o.observation_id for o in observations),
                explanation,
                True,
                tuple(sorted({o.source for o in observations})),
                (f"{metric}:change", f"{metric}:DRR"),
            )
        )

    for metric in dataset.variable_names:
        series = frame[metric]
        run = 1
        for i, (period, value) in enumerate(series.items()):
            if pd.isna(value):
                add(
                    (
                        "missing_quarter"
                        if period not in {o.reporting_period for o in dataset.observations}
                        else "missing_value"
                    ),
                    metric,
                    period,
                    "important",
                )
                run = 1
                continue
            o = records.get((metric, period))
            if o and o.amendment_date:
                add(
                    "amended_filing",
                    metric,
                    period,
                    "info",
                    "Source labels this observation as an amendment",
                )
            if i == 0:
                continue
            prior_period, prior_value = series.index[i - 1], series.iloc[i - 1]
            prior = records.get((metric, prior_period))
            evidence = ((metric, prior_period),)
            if o and prior:
                for field, kind in (
                    ("unit", "unit_change"),
                    ("definition_version", "definition_change"),
                    ("perimeter_version", "perimeter_change"),
                ):
                    if getattr(o, field) != getattr(prior, field):
                        add(
                            kind, metric, period, "important", f"Reported {field} changed", evidence
                        )
            if pd.isna(prior_value):
                continue
            if value == 0 and prior_value != 0:
                add("unexpected_zero", metric, period, inputs=evidence)
            if prior_value != 0 and abs((value - prior_value) / prior_value) >= extreme_change:
                add("extreme_qoq", metric, period, inputs=evidence)
            run = run + 1 if value == prior_value else 1
            if run >= carry_forward_periods:
                add(
                    "suspicious_carry_forward",
                    metric,
                    period,
                    explanation=f"{run} consecutive identical reported values",
                    inputs=evidence,
                )
        if dataset.dates:
            age = (
                pd.Period(dataset.available_as_of[:10], freq="Q").ordinal
                - pd.Period(dataset.dates[-1], freq="Q").ordinal
            )
            if age > 1:
                add(
                    "stale_series",
                    metric,
                    dataset.dates[-1],
                    explanation=f"Latest reporting period is {age} quarters before review",
                )
    for rule in rules:
        if any(m not in frame.columns for m in (rule.metric, *rule.inputs)):
            raise ValueError("Reconciliation rule references an unknown metric")
        for period, row in frame.iterrows():
            inputs = row[list(rule.inputs)].to_numpy(dtype=float)
            if not np.isfinite(inputs).all() or pd.isna(row[rule.metric]):
                continue
            expected = (
                float(inputs.sum())
                if rule.operation == "sum"
                else None if inputs[1] == 0 else float(inputs[0] / inputs[1])
            )
            if expected is None:
                add(
                    "broken_ratio",
                    rule.metric,
                    period,
                    "important",
                    "Denominator is zero",
                    tuple((m, period) for m in rule.inputs),
                )
            elif abs(row[rule.metric] - expected * rule.multiplier) > rule.tolerance:
                add(
                    "inconsistent_total" if rule.operation == "sum" else "broken_ratio",
                    rule.metric,
                    period,
                    "important",
                    f"Expected {expected*rule.multiplier:g} from explicit {rule.operation} rule",
                    tuple((m, period) for m in rule.inputs),
                )
    return tuple(sorted(issues, key=lambda e: (e.period, e.metric, e.issue_type)))


def reconcile_store(store, as_of):
    """Report duplicates/conflicts before attempting a matrix pivot."""
    groups = {}
    for o in store.known_records(as_of):
        groups.setdefault((o.key, o.filing_date), []).append(o)
    result = []
    for (_, _), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        o = rows[0]
        conflict = (
            len({(r.value, r.unit, r.definition_version, r.perimeter_version) for r in rows}) > 1
        )
        result.append(
            DataQualityException(
                "important" if conflict else "warning",
                o.institution_id,
                o.metric,
                o.reporting_period,
                "source_conflict" if conflict else "duplicate_record",
                tuple(r.observation_id for r in rows),
                source_links=tuple(sorted({r.source for r in rows})),
            )
        )
    return tuple(result)
