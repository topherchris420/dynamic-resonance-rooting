"""Audience-aware DRR report generation."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import networkx as nx
import numpy as np

from .state_space import KalmanResult, Measurement, StateSpaceSystem, Transition


Audience = Literal["general", "physics", "policy"]


def serialize_analysis_results(value: Any) -> Any:
    """Convert DRR analysis output into JSON-serializable structures."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Transition):
        return {
            "TTT": serialize_analysis_results(value.TTT),
            "RRR": serialize_analysis_results(value.RRR),
            "CCC": serialize_analysis_results(value.CCC),
        }
    if isinstance(value, Measurement):
        return {
            "ZZ": serialize_analysis_results(value.ZZ),
            "DD": serialize_analysis_results(value.DD),
            "QQ": serialize_analysis_results(value.QQ),
            "EE": serialize_analysis_results(value.EE),
        }
    if isinstance(value, StateSpaceSystem):
        return {
            "transition": serialize_analysis_results(value.transition),
            "measurement": serialize_analysis_results(value.measurement),
            "state_names": list(value.state_names),
            "observable_names": list(value.observable_names),
            "shock_names": list(value.shock_names),
            "diagnostics": serialize_analysis_results(value.diagnostics()),
        }
    if isinstance(value, KalmanResult):
        return {
            "log_likelihood": serialize_analysis_results(value.log_likelihood),
            "predicted_state": serialize_analysis_results(value.predicted_state),
            "filtered_state": serialize_analysis_results(value.filtered_state),
            "final_state": serialize_analysis_results(value.final_state),
            "final_covariance": serialize_analysis_results(value.final_covariance),
            "innovations": serialize_analysis_results(value.innovations),
            "summary": serialize_analysis_results(value.summary()),
        }
    if isinstance(value, nx.Graph):
        return {
            "nodes": list(value.nodes(data=True)),
            "edges": [
                {"source": source, "target": target, **attributes}
                for source, target, attributes in value.edges(data=True)
            ],
        }
    if isinstance(value, dict):
        return {str(key): serialize_analysis_results(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_analysis_results(item) for item in value]
    if is_dataclass(value):
        return serialize_analysis_results(value.__dict__)
    return str(value)


def write_analysis_report(
    results: Dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "drr_report",
    audience: Audience = "general",
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write JSON and Markdown reports for a DRR analysis result."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    serializable = serialize_analysis_results(results)
    payload = {
        "audience": audience,
        "title": title or _default_title(audience),
        "metadata": serialize_analysis_results(metadata or {}),
        "results": serializable,
        "summary": summarize_analysis_results(results, audience=audience),
    }

    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(
        render_markdown_report(payload, audience=audience),
        encoding="utf-8",
    )

    return {"json": json_path, "markdown": markdown_path}


def summarize_analysis_results(results: Dict[str, Any], audience: Audience = "general") -> Dict[str, Any]:
    depths = results.get("resonance_depths", {})
    depth_values = [float(value) for value in depths.values()] if depths else []
    rooting = results.get("rooting_analysis", {})
    state_space = results.get("state_space_analysis", {})
    diagnostics = state_space.get("diagnostics", {}) if isinstance(state_space, dict) else {}

    return {
        "headline": _headline(audience, depth_values, diagnostics),
        "mean_resonance_depth": float(np.mean(depth_values)) if depth_values else 0.0,
        "max_resonance_depth": float(np.max(depth_values)) if depth_values else 0.0,
        "is_rooted": bool(results.get("is_rooted", False)),
        "significant_relationships": len(rooting.get("significant_edges", [])),
        "state_space_stable": diagnostics.get("is_stable"),
        "spectral_radius": diagnostics.get("spectral_radius"),
        "recommended_next_step": _recommended_next_step(audience),
        "caveat": _audience_caveat(audience),
    }


def render_markdown_report(payload: Dict[str, Any], audience: Audience = "general") -> str:
    results = payload["results"]
    summary = payload["summary"]
    lines = [
        f"# {payload['title']}",
        "",
        f"Audience: `{audience}`",
        "",
        "## Executive Summary",
        "",
        summary["headline"],
        "",
        _audience_caveat(audience),
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean resonance depth | {_format_float(summary['mean_resonance_depth'])} |",
        f"| Max resonance depth | {_format_float(summary['max_resonance_depth'])} |",
        f"| Rooted flag | {summary['is_rooted']} |",
        f"| Significant directed relationships | {summary['significant_relationships']} |",
        f"| State-space stable | {summary['state_space_stable']} |",
        f"| Spectral radius | {_format_float(summary['spectral_radius'])} |",
        "",
    ]

    lines.extend(_resonance_section(results))
    lines.extend(_rooting_section(results))
    lines.extend(_state_space_section(results))
    lines.extend(
        [
            "## Recommended Next Step",
            "",
            summary["recommended_next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def _resonance_section(results: Dict[str, Any]) -> list[str]:
    resonances = results.get("resonances", {})
    if not resonances:
        return []

    lines = [
        "## Resonance Map",
        "",
        "| Dimension | Dominant Frequency | Detected Frequencies |",
        "| --- | ---: | --- |",
    ]
    for dimension, details in resonances.items():
        frequencies = details.get("frequencies", [])
        dominant = details.get("dominant_freq", 0.0)
        lines.append(
            f"| {dimension} | {_format_float(dominant)} | {_format_sequence(frequencies)} |"
        )
    lines.append("")
    return lines


def _rooting_section(results: Dict[str, Any]) -> list[str]:
    rooting = results.get("rooting_analysis", {})
    edges = rooting.get("significant_edges", []) if isinstance(rooting, dict) else []
    if not edges:
        return [
            "## Directed Rooting",
            "",
            "No statistically significant directed relationships were detected under the configured threshold.",
            "",
        ]

    lines = [
        "## Directed Rooting",
        "",
        "| Source | Target | Weight | Lag | p-value |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for edge in edges:
        lines.append(
            f"| {edge.get('source')} | {edge.get('target')} | "
            f"{_format_float(edge.get('weight'))} | {edge.get('lag')} | "
            f"{_format_float(edge.get('p_value'))} |"
        )
    lines.append("")
    return lines


def _state_space_section(results: Dict[str, Any]) -> list[str]:
    state_space = results.get("state_space_analysis", {})
    diagnostics = state_space.get("diagnostics", {}) if isinstance(state_space, dict) else {}
    if not diagnostics:
        return []

    metrics = [
        "spectral_radius",
        "is_stable",
        "mean_log_likelihood",
        "final_uncertainty_trace",
        "state_count",
        "shock_count",
    ]
    lines = [
        "## State-Space Diagnostics",
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
    ]
    for metric in metrics:
        if metric in diagnostics:
            lines.append(f"| {metric} | {_format_value(diagnostics[metric])} |")
    lines.append("")
    return lines


def _headline(audience: Audience, depth_values: Iterable[float], diagnostics: Dict[str, Any]) -> str:
    values = list(depth_values)
    mean_depth = float(np.mean(values)) if values else 0.0
    stable = diagnostics.get("is_stable")

    if audience == "physics":
        stability = "stable" if stable else "not yet classified as stable"
        return (
            f"The system shows mean resonance depth {_format_float(mean_depth)} and a "
            f"state-space transition that is {stability}. Use the resonance map and impulse "
            "responses to inspect mode coupling and perturbation behavior."
        )
    if audience == "policy":
        stability = "stable" if stable else "not yet classified as stable"
        return (
            f"The observable set has mean resonance depth {_format_float(mean_depth)} and a "
            f"{stability} fitted transition system. Treat this as a diagnostic for lagged "
            "structure, shock propagation, and scenario design, not as a policy recommendation."
        )
    return (
        f"DRR detected mean resonance depth {_format_float(mean_depth)}. Review the directed "
        "rooting and state-space sections for system structure and uncertainty diagnostics."
    )


def _recommended_next_step(audience: Audience) -> str:
    if audience == "physics":
        return "Run the same workflow across controlled perturbation amplitudes and compare impulse-response decay rates."
    if audience == "policy":
        return "Repeat the analysis across vintages or regimes, then compare whether directed lags and shock responses remain stable."
    return "Validate the result on a held-out period or synthetic benchmark with known coupling."


def _audience_caveat(audience: Audience) -> str:
    if audience == "physics":
        return "This report is a computational diagnostic; physical interpretation requires domain calibration and experimental controls."
    if audience == "policy":
        return "This report is a research diagnostic; it is not a forecast, causal policy claim, or recommendation."
    return "This report is a diagnostic summary and should be validated against domain-specific assumptions."


def _default_title(audience: Audience) -> str:
    if audience == "physics":
        return "DRR Physics Lab Report"
    if audience == "policy":
        return "DRR Policy Lab Report"
    return "DRR Analysis Report"


def _format_sequence(values: Any, max_items: int = 5) -> str:
    if values is None:
        return ""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, list):
        values = list(values) if isinstance(values, tuple) else [values]
    shown = ", ".join(_format_float(value) for value in values[:max_items])
    if len(values) > max_items:
        shown += ", ..."
    return shown


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, np.number)):
        return _format_float(value)
    if isinstance(value, list):
        return _format_sequence(value)
    return str(value)
