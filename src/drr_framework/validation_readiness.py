"""Validation-readiness helpers for supervisory DRR workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

VALIDATION_REFERENCE_BASIS = (
    {
        "label": "SR 11-7 model risk management guidance",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm",
    },
    {
        "label": "Federal Reserve supervision and regulation",
        "url": "https://www.federalreserve.gov/supervisionreg.htm",
    },
    {
        "label": "Updated Statement of Supervisory Operating Principles",
        "url": "https://www.federalreserve.gov/supervisionreg/updated-statement-of-supervisory-operating-principles.htm",
    },
)

SR_11_7_SECTIONS = (
    "intended_use",
    "conceptual_soundness",
    "implementation_verification",
    "outcomes_analysis",
    "ongoing_monitoring",
    "governance_and_change_control",
)

VALIDATION_READINESS_CHECKLIST = {
    "intended_use": "Document approved use, users, data population, and prohibited uses.",
    "conceptual_soundness": "Document theory, assumptions, limitations, and why DRR is appropriate for the monitoring use case.",
    "implementation_verification": "Independently review code, tests, reproducibility, numerical stability, and release controls.",
    "outcomes_analysis": "Backtest alerts against historical events, benchmark methods, and analyst review outcomes.",
    "ongoing_monitoring": "Track drift, false positives, false negatives, lead time, reviewer usefulness, and recurring limitations.",
    "governance_and_change_control": "Record owners, approvals, version changes, validation status, and retirement criteria.",
    "independent_validation": "Required before validated use",
}


def build_model_risk_card(
    *,
    model_name: str,
    version: str,
    intended_use: str,
    prohibited_uses: Sequence[str],
    owners: Optional[Sequence[str]] = None,
    data_lineage: Optional[Mapping[str, Any]] = None,
    assumptions: Optional[Sequence[str]] = None,
    limitations: Optional[Sequence[str]] = None,
    validation_status: str = "candidate; not validated supervisory methodology",
    monitoring_plan: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a model-risk card for a DRR supervisory candidate workflow."""

    return _drop_empty(
        {
            "model_name": model_name,
            "version": version,
            "intended_use": intended_use,
            "prohibited_uses": list(prohibited_uses),
            "owners": list(owners or ()),
            "data_lineage": dict(data_lineage or {}),
            "assumptions": list(assumptions or ()),
            "limitations": list(limitations or ()),
            "validation_status": validation_status,
            "monitoring_plan": list(monitoring_plan or _default_monitoring_plan()),
            "decision_boundary": "review evidence only; not a supervisory conclusion",
        }
    )


def build_validation_readiness_packet(
    *,
    model_card: Mapping[str, Any],
    analysis_results: Mapping[str, Any],
    benchmark_results: Optional[Mapping[str, Any]] = None,
    shadow_mode_summary: Optional[Mapping[str, Any]] = None,
    explanations: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a validation-readiness packet for model-risk review."""

    summary = _analysis_summary(analysis_results)
    readiness = {
        "status": "validation-ready candidate; not validated methodology",
        "model_card": dict(model_card),
        "analysis_summary": summary,
        "sr_11_7_sections": list(SR_11_7_SECTIONS),
        "checklist": dict(VALIDATION_READINESS_CHECKLIST),
        "benchmark_results": dict(benchmark_results or {}),
        "shadow_mode_summary": dict(shadow_mode_summary or {}),
        "explainability_summary": dict(explanations or {}),
        "reference_basis": [dict(reference) for reference in VALIDATION_REFERENCE_BASIS],
        "required_before_validated_use": [
            "independent conceptual-soundness review",
            "implementation verification by a party independent of developers",
            "historical outcomes analysis against approved event labels",
            "ongoing monitoring thresholds and owner signoff",
            "formal governance approval for a precisely scoped intended use",
        ],
    }
    return {"validation_readiness": readiness}


def run_event_backtest(
    frame: pd.DataFrame,
    *,
    date_column: str,
    score_column: str,
    event_column: str,
    threshold: float,
    lead_window: int,
) -> Dict[str, Any]:
    """Backtest whether threshold alerts occur before labeled events."""

    if lead_window < 0:
        raise ValueError("lead_window must be non-negative")
    required = [date_column, score_column, event_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"required backtest columns not found: {missing}")

    prepared = frame.loc[:, required].copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
    prepared[score_column] = pd.to_numeric(prepared[score_column], errors="coerce")
    prepared[event_column] = prepared[event_column].astype(bool)
    prepared = (
        prepared.dropna(subset=[score_column]).sort_values(date_column).reset_index(drop=True)
    )

    alert_indices = [int(index) for index in prepared.index[prepared[score_column] >= threshold]]
    event_indices = [int(index) for index in prepared.index[prepared[event_column]]]
    event_set = set(event_indices)

    true_positive_alerts = []
    false_positive_alerts = []
    detected_events = set()
    lead_times = []

    for alert_index in alert_indices:
        future_events = [
            event_index
            for event_index in event_indices
            if 0 <= event_index - alert_index <= lead_window
        ]
        if future_events:
            event_index = min(future_events, key=lambda item: item - alert_index)
            true_positive_alerts.append(alert_index)
            detected_events.add(event_index)
            lead_times.append(event_index - alert_index)
        else:
            false_positive_alerts.append(alert_index)

    missed_events = event_set - detected_events
    precision = len(true_positive_alerts) / len(alert_indices) if alert_indices else 0.0
    recall = len(detected_events) / len(event_indices) if event_indices else 0.0

    return {
        "threshold": float(threshold),
        "lead_window": int(lead_window),
        "observation_count": int(len(prepared)),
        "alert_count": int(len(alert_indices)),
        "event_count": int(len(event_indices)),
        "true_positive_alerts": int(len(true_positive_alerts)),
        "false_positive_alerts": int(len(false_positive_alerts)),
        "detected_events": int(len(detected_events)),
        "missed_events": int(len(missed_events)),
        "precision": float(precision),
        "recall": float(recall),
        "mean_lead_time_periods": float(np.mean(lead_times)) if lead_times else None,
        "alert_dates": [
            prepared.loc[index, date_column].strftime("%Y-%m-%d") for index in alert_indices
        ],
        "event_dates": [
            prepared.loc[index, date_column].strftime("%Y-%m-%d") for index in event_indices
        ],
    }


def create_shadow_review_record(
    *,
    signal_id: str,
    reviewer: str,
    disposition: str,
    rationale: str,
    analyst_action: str = "none",
    review_time: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an append-only shadow-mode review record."""

    if disposition not in {"useful", "noisy", "missed", "needs_review"}:
        raise ValueError("disposition must be one of: useful, noisy, missed, needs_review")
    return _drop_empty(
        {
            "mode": "shadow",
            "signal_id": signal_id,
            "reviewer": reviewer,
            "disposition": disposition,
            "rationale": rationale,
            "analyst_action": analyst_action,
            "review_time": review_time or datetime.now(timezone.utc).isoformat(),
            "decision_boundary": "review feedback only; no supervisory action",
            "metadata": dict(metadata or {}),
        }
    )


def append_shadow_review_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append a shadow-mode review record to a JSONL audit file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return output_path


def explain_supervisory_signal(
    results: Mapping[str, Any],
    *,
    variable_names: Optional[Sequence[str]] = None,
    data_lineage: Optional[Mapping[str, Any]] = None,
    max_items: int = 5,
) -> Dict[str, Any]:
    """Create a compact explanation of what drove a supervisory DRR signal."""

    variable_names = tuple(variable_names or ())
    depths = results.get("resonance_depths", {}) if isinstance(results, Mapping) else {}
    resonances = results.get("resonances", {}) if isinstance(results, Mapping) else {}
    rooting = results.get("rooting_analysis", {}) if isinstance(results, Mapping) else {}
    state_space = results.get("state_space_analysis", {}) if isinstance(results, Mapping) else {}
    diagnostics = state_space.get("diagnostics", {}) if isinstance(state_space, Mapping) else {}

    dominant_modes = []
    for dimension, depth in sorted(depths.items(), key=lambda item: float(item[1]), reverse=True)[
        :max_items
    ]:
        resonance = resonances.get(dimension, {}) if isinstance(resonances, Mapping) else {}
        dominant_modes.append(
            {
                "dimension": dimension,
                "variable": _resolve_variable_name(dimension, variable_names),
                "resonance_depth": float(depth),
                "dominant_frequency": resonance.get("dominant_freq"),
            }
        )

    top_edges = []
    for edge in (
        rooting.get("significant_edges", [])[:max_items] if isinstance(rooting, Mapping) else []
    ):
        source = edge.get("source")
        target = edge.get("target")
        top_edges.append(
            {
                "source": source,
                "target": target,
                "source_variable": _resolve_variable_name(source, variable_names),
                "target_variable": _resolve_variable_name(target, variable_names),
                "lag": edge.get("lag"),
                "weight": edge.get("weight"),
                "p_value": edge.get("p_value"),
            }
        )

    return {
        "dominant_modes": dominant_modes,
        "top_edges": top_edges,
        "state_space_summary": {
            "is_stable": diagnostics.get("is_stable"),
            "spectral_radius": diagnostics.get("spectral_radius"),
            "mean_log_likelihood": diagnostics.get("mean_log_likelihood"),
        },
        "data_lineage": dict(data_lineage or {}),
        "analyst_questions": [
            "Does the signal survive alternative transformations, windows, and peer groups?",
            "Do source-data owners confirm the metric movement and lineage?",
            "Is the signal material relative to the institution's size, complexity, and risk profile?",
            "Does shadow-mode review show useful lead time without excessive false positives?",
        ],
        "caveats": [
            "This explanation is not a finding, rating, MRA, MRIA, enforcement recommendation, or policy decision.",
            "Directed edges are diagnostic lead-lag signals, not causal proof.",
        ],
    }


def _analysis_summary(results: Mapping[str, Any]) -> Dict[str, Any]:
    depths = results.get("resonance_depths", {}) if isinstance(results, Mapping) else {}
    depth_values = (
        [float(value) for value in depths.values()] if isinstance(depths, Mapping) else []
    )
    rooting = results.get("rooting_analysis", {}) if isinstance(results, Mapping) else {}
    edges = rooting.get("significant_edges", []) if isinstance(rooting, Mapping) else []
    return {
        "mean_resonance_depth": float(np.mean(depth_values)) if depth_values else 0.0,
        "max_resonance_depth": float(np.max(depth_values)) if depth_values else 0.0,
        "significant_relationships": len(edges),
        "is_rooted": (
            bool(results.get("is_rooted", False)) if isinstance(results, Mapping) else False
        ),
    }


def _resolve_variable_name(dimension: Any, variable_names: Sequence[str]) -> str:
    if not variable_names:
        return str(dimension)
    try:
        index = int(str(dimension).split("_")[-1])
    except (TypeError, ValueError):
        return str(dimension)
    if 0 <= index < len(variable_names):
        return str(variable_names[index])
    return str(dimension)


def _default_monitoring_plan() -> Sequence[str]:
    return (
        "Run rolling-window backtests after each data vintage update.",
        "Review shadow-mode usefulness and noise rates with analysts.",
        "Escalate model changes through documented governance before approved use.",
    )


def _drop_empty(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }
