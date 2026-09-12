"""Transparent quarter-based material changes using only the supplied as-of view."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def percentile(value, history):
    values = np.asarray(history, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(value):
        return None
    return float(100 * (np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def robust_z(value, history):
    values = np.asarray(history, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or not np.isfinite(value):
        return None
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    if scale <= np.finfo(float).eps * max(abs(float(center)), 1.0):
        return 0.0 if value == center else None
    return float((value - center) / scale)


@dataclass(frozen=True)
class MaterialChange:
    metric: str
    label: str
    institution: str
    period: str
    unit: str
    value: Optional[float]
    raw_change: Optional[float]
    normalized_change: Optional[float]
    yoy_change: Optional[float]
    yoy_percent: Optional[float]
    rolling_mean: Optional[float]
    rolling_median: Optional[float]
    rolling_volatility: Optional[float]
    robust_z_score: Optional[float]
    historical_percentile: Optional[float]
    trend_slope: Optional[float]
    acceleration: Optional[float]
    persistence: int
    reversal: bool
    structural_break_candidate: bool
    materiality_score: float
    data_confidence: float
    evidence_references: Tuple[str, ...]
    reason_flagged: Tuple[str, ...]
    peer_percentile: Optional[float] = None
    peer_context: str = "indeterminate"
    comparison_limitations: Tuple[str, ...] = ()

    @property
    def flagged(self):
        return bool(self.reason_flagged)

    @property
    def signal_key(self):
        return f"{self.institution}:{self.metric}:material_change"


def detect_material_changes(
    dataset, *, lookback=12, percent_threshold=5.0, z_threshold=3.0, min_history=4, breakpoints=()
):
    if lookback < 4 or min_history < 4 or percent_threshold <= 0 or z_threshold <= 0:
        raise ValueError("Invalid change-detection configuration")
    results = []
    for metric in dataset.variable_names:
        series = dataset.frame[metric].astype(float)
        records = {o.reporting_period: o for o in dataset.observations if o.metric == metric}
        versions = [
            (
                (records[p].definition_version, records[p].unit, records[p].perimeter_version)
                if p in records
                else None
            )
            for p in series.index
        ]
        start = max(0, len(series) - lookback - 1)
        limitations = []
        for i in range(1, len(series)):
            if (
                versions[i] is not None
                and versions[i - 1] is not None
                and versions[i] != versions[i - 1]
            ) or series.index[i] in breakpoints:
                start = max(start, i)
                limitations.append(f"Comparability breakpoint at {series.index[i]}")
        comparable = series.iloc[start:]
        current = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else None
        history = comparable.iloc[:-1]
        finite = history.dropna().to_numpy()
        prev = comparable.iloc[-2] if len(comparable) >= 2 else np.nan
        raw = None if current is None or pd.isna(prev) else float(current - prev)
        pct = None if raw is None or prev == 0 else float(100 * raw / abs(prev))
        year = comparable.iloc[-5] if len(comparable) >= 5 else np.nan
        yoy = None if current is None or pd.isna(year) else float(current - year)
        yoy_pct = None if yoy is None or year == 0 else float(100 * yoy / abs(year))
        z = robust_z(current if current is not None else np.nan, finite)
        changes = comparable.diff().to_numpy()
        direction = np.sign(raw) if raw is not None else 0
        persistence = 0
        for change in changes[::-1]:
            if not np.isfinite(change) or direction == 0 or np.sign(change) != direction:
                break
            persistence += 1
        prior_change = changes[-2] if len(changes) >= 2 else np.nan
        acceleration = (
            float(raw - prior_change) if raw is not None and np.isfinite(prior_change) else None
        )
        reversal = bool(raw is not None and np.isfinite(prior_change) and raw * prior_change < 0)
        trend = comparable.iloc[-8:]
        mask = trend.notna().to_numpy()
        slope = (
            float(np.polyfit(np.arange(len(trend))[mask], trend.to_numpy()[mask], 1)[0])
            if mask.sum() >= 3
            else None
        )
        reason = []
        enough = len(finite) >= min_history
        if enough and pct is not None and abs(pct) >= percent_threshold:
            reason.append(
                f"Quarter change {pct:+.2f}% exceeds {percent_threshold:g}% review threshold"
            )
        if enough and z is not None and abs(z) >= z_threshold:
            reason.append(f"Robust historical deviation {z:+.2f} exceeds {z_threshold:g}")
        if not enough:
            limitations.append(
                f"Only {len(finite)} comparable prior observations; {min_history} required"
            )
        if raw is not None and prev == 0:
            limitations.append("Prior value is zero; percent change is undefined")
        obs = tuple(records[p] for p in comparable.index if p in records)
        confidence = float(
            sum(o.value is not None and o.provenance.value != "imputed_causal" for o in obs)
            / max(len(comparable), 1)
        )
        components = [
            abs(pct) / percent_threshold if pct is not None else 0,
            abs(z) / z_threshold if z is not None else 0,
        ]
        score = float(min(100, 40 * max(components))) if reason else 0.0
        meta = next(
            (m for k, m in dataset.metric_metadata.items() if k.startswith(metric + ":")), {}
        )
        results.append(
            MaterialChange(
                metric,
                meta.get("label", metric),
                dataset.institution_id,
                series.index[-1],
                meta.get("unit", "unknown"),
                current,
                raw,
                pct,
                yoy,
                yoy_pct,
                float(np.mean(finite)) if len(finite) else None,
                float(np.median(finite)) if len(finite) else None,
                float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                z,
                percentile(current if current is not None else np.nan, finite),
                slope,
                acceleration,
                persistence,
                reversal,
                bool(z is not None and abs(z) >= z_threshold),
                score,
                confidence,
                tuple(o.observation_id for o in obs),
                tuple(reason),
                comparison_limitations=tuple(limitations),
            )
        )
    return tuple(results)


def attach_peer_context(change, peer):
    return replace(change, peer_percentile=peer.percentile, peer_context=peer.context.value)
