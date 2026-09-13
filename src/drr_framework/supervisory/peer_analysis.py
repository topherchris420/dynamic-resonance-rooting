"""Explicit, vintage-matched horizontal cohorts; context is never causal attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .common import instant, stable_id
from .change_detection import percentile


class SignalContext(str, Enum):
    FIRM_SPECIFIC = "firm_specific"
    PEER_CLUSTER = "peer_cluster"
    BROAD_COMMON_FACTOR = "broad_common_factor"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class PeerGroupDefinition:
    name: str
    institutions: Tuple[str, ...]
    rationale: str
    as_of: str
    effective_from: str
    exclusions: Tuple[Tuple[str, str], ...] = ()
    filters: Tuple[str, ...] = ()
    minimum_peers: int = 3

    def __post_init__(self):
        object.__setattr__(self, "institutions", tuple(sorted(self.institutions)))
        object.__setattr__(self, "exclusions", tuple(tuple(e) for e in self.exclusions))
        object.__setattr__(self, "filters", tuple(self.filters))
        if (
            not self.name
            or not self.rationale
            or len(self.institutions) != len(set(self.institutions))
        ):
            raise ValueError("Cohort needs a name, rationale and unique members")
        if self.minimum_peers < 3:
            raise ValueError("At least three peers required for context classification")
        instant(self.as_of)
        instant(self.effective_from)

    @property
    def definition_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class PeerAnalysis:
    definition: PeerGroupDefinition
    institution: str
    metric: str
    period: str
    as_of: str
    included: Tuple[str, ...]
    excluded: Tuple[Tuple[str, str], ...]
    source_vintages: Tuple[str, ...]
    metric_coverage: float
    median: Optional[float]
    quartiles: Tuple[Optional[float], Optional[float]]
    percentile: Optional[float]
    mad: Optional[float]
    robust_deviation: Optional[float]
    dispersion: Optional[float]
    change_dispersion: Optional[float]
    moving_together: int
    paired_peer_count: int
    share_moving_together: Optional[float]
    rank_migration: Optional[float]
    persistent_outlier: bool
    cross_sectional_break_candidate: bool
    context: SignalContext
    evidence: Tuple[str, ...]
    limitations: Tuple[str, ...]


def analyze_peers(
    store,
    registry,
    cohort,
    *,
    institution,
    form,
    metric,
    period,
    as_of,
    allow_synthetic=False,
    movement_threshold=5.0,
):
    if instant(cohort.as_of) > instant(as_of) or instant(cohort.effective_from) > instant(period):
        raise ValueError("Peer definition was not available/effective at the analysis point")
    observations = store.as_of(as_of)
    target = next((o for o in observations if o.key == (institution, form, metric, period)), None)
    if target is None:
        raise ValueError("Target observation missing")
    definition = registry.resolve(
        form,
        metric,
        period,
        as_of=as_of,
        version=target.definition_version,
        allow_synthetic=allow_synthetic,
    )
    prior_period = (pd.Period(period, freq="Q") - 1).end_time.date().isoformat()
    lookup = {o.key: o for o in observations}
    excluded = dict(cohort.exclusions)
    included, values, changes, prior_values, evidence, vintages = [], [], [], [], [], set()
    for peer in cohort.institutions:
        if peer == institution:
            excluded[peer] = "Target excluded from its comparison distribution"
            continue
        if peer in excluded:
            continue
        o = lookup.get((peer, form, metric, period))
        if o is None or o.value is None:
            excluded[peer] = "Missing current-period observation"
            continue
        if o.unit != target.unit or o.definition_version != target.definition_version:
            excluded[peer] = "Different units or definition version"
            continue
        if o.perimeter_version != target.perimeter_version:
            excluded[peer] = "Different reporting perimeter version"
            continue
        registry.resolve(
            form,
            metric,
            period,
            as_of=as_of,
            version=o.definition_version,
            allow_synthetic=allow_synthetic,
        )
        if o.provenance.value == "imputed_causal":
            excluded[peer] = "Imputed comparison value"
            continue
        included.append(peer)
        values.append(o.value)
        evidence.append(o.observation_id)
        vintages.add(o.source_vintage)
        old = lookup.get((peer, form, metric, prior_period))
        if (
            old
            and old.value is not None
            and old.unit == o.unit
            and old.definition_version == o.definition_version
            and old.perimeter_version == o.perimeter_version
        ):
            changes.append(
                100 * (o.value - old.value) / abs(old.value) if old.value != 0 else np.nan
            )
            prior_values.append(old.value)
            evidence.append(old.observation_id)
        else:
            changes.append(np.nan)
            prior_values.append(np.nan)
    old = lookup.get((institution, form, metric, prior_period))
    target_change = None
    if (
        target.value is not None
        and old
        and old.value not in (None, 0)
        and old.unit == target.unit
        and old.definition_version == target.definition_version
        and old.perimeter_version == target.perimeter_version
    ):
        target_change = 100 * (target.value - old.value) / abs(old.value)
    arr, delta = np.array(values), np.array(changes)
    paired = np.isfinite(delta)
    moving = (
        int(
            np.sum(
                paired
                & (np.abs(delta) >= movement_threshold)
                & (np.sign(delta) == np.sign(target_change))
            )
        )
        if target_change is not None
        else 0
    )
    share = moving / int(paired.sum()) if paired.any() else None
    pct = percentile(target.value if target.value is not None else np.nan, arr)
    med = float(np.median(arr)) if len(arr) else None
    mad = float(np.median(np.abs(arr - med))) if len(arr) else None
    deviation = (
        (target.value - med) / (1.4826 * mad)
        if target.value is not None and mad is not None and mad > 0
        else None
    )
    previous_pct = (
        percentile(old.value, np.asarray(prior_values)[paired])
        if old and old.value is not None and paired.any()
        else None
    )
    current_paired_pct = (
        percentile(target.value, arr[paired]) if target.value is not None and paired.any() else None
    )
    context = SignalContext.INDETERMINATE
    limitations = []
    if len(arr) < cohort.minimum_peers or int(paired.sum()) < cohort.minimum_peers:
        limitations.append("Insufficient comparable paired peers for signal context")
    elif target_change is not None and abs(target_change) >= movement_threshold:
        context = (
            SignalContext.BROAD_COMMON_FACTOR
            if share >= 0.7
            else SignalContext.PEER_CLUSTER if share >= 0.3 else SignalContext.FIRM_SPECIFIC
        )
    return PeerAnalysis(
        cohort,
        institution,
        metric,
        period,
        instant(as_of).isoformat(),
        tuple(included),
        tuple(sorted(excluded.items())),
        tuple(sorted(vintages)),
        len(arr)
        / max(1, len(set(cohort.institutions) - {institution} - set(dict(cohort.exclusions)))),
        med,
        tuple(float(v) for v in np.quantile(arr, [0.25, 0.75])) if len(arr) else (None, None),
        pct,
        mad,
        float(deviation) if deviation is not None else None,
        float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
        float(np.std(delta[paired], ddof=1)) if paired.sum() > 1 else None,
        moving,
        int(paired.sum()),
        share,
        (
            current_paired_pct - previous_pct
            if current_paired_pct is not None and previous_pct is not None
            else None
        ),
        bool(
            pct is not None
            and previous_pct is not None
            and ((pct >= 95 and previous_pct >= 95) or (pct <= 5 and previous_pct <= 5))
        ),
        bool(share is not None and share >= 0.7 and int(paired.sum()) >= cohort.minimum_peers),
        context,
        tuple([target.observation_id] + evidence),
        tuple(limitations) + ("Shared direction does not identify a causal source.",),
    )
