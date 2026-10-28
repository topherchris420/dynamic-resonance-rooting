"""Validated, reproducible analysis of user-supplied, regularly sampled CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .modules import DepthCalculator, ResonanceDetector, RootingAnalyzer

NULL_ASSUMPTIONS = {
    "circular_shift": "Stationarity; circular shifts preserve within-channel dependence except at the seam.",
    "phase_randomization": "Stationary linear Gaussian processes; each channel's periodogram is preserved.",
    "permutation": "Serial independence; permutation destroys within-channel temporal dependence.",
}


def _json_safe(value: Any) -> Any:
    """Use JSON null for unavailable numerical values (never NaN/Infinity)."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("drr-framework", "numpy", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def analyze_csv(
    path: Path,
    *,
    columns: Sequence[str],
    sampling_rate: float,
    time_column: Optional[str] = None,
    window_size: int = 256,
    max_lag: int = 4,
    n_surrogates: int = 99,
    random_state: int = 42,
    alpha: float = 0.05,
    surrogate_method: str = "circular_shift",
    difference: bool = False,
    mathematical_features: bool = False,
) -> Dict[str, Any]:
    """Analyze explicit signal columns without dropping, imputing, or sorting rows.

    Optional timestamps must be numeric seconds, strictly increasing, and agree
    with the declared sample rate to relative tolerance 1e-4 (absolute 1e-9 s).
    First differencing is opt-in and recorded; it does not prove stationarity.
    ``mathematical_features`` adds the experimental data-quality checks, phase
    synchronization tests, and evidence record of
    :func:`drr_framework.math_features.analyze_mathematical_features`; without
    it the output is unchanged.
    """
    if len(columns) < 2 or len(set(columns)) != len(columns):
        raise ValueError("Select at least two distinct signal columns with --columns.")
    if time_column in columns:
        raise ValueError("The time column cannot also be a signal column.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive.")
    for name, integer_value, minimum in (
        ("window_size", window_size, 8),
        ("max_lag", max_lag, 1),
        ("n_surrogates", n_surrogates, 0),
        ("random_state", random_state, 0),
    ):
        if (
            isinstance(integer_value, bool)
            or not isinstance(integer_value, int)
            or integer_value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if surrogate_method not in NULL_ASSUMPTIONS:
        raise ValueError(f"surrogate_method must be one of {', '.join(NULL_ASSUMPTIONS)}.")

    # Read once: the checksum describes exactly the bytes used in this run.
    raw = Path(path).read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
    header = next(reader, [])
    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
        raise ValueError("CSV must have nonempty, unique column headers.")
    requested = list(columns) + ([time_column] if time_column else [])
    missing = [name for name in requested if name not in header]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}.")
    indices = [header.index(name) for name in requested]
    rows = []
    for row in reader:
        if len(row) != len(header):
            raise ValueError(
                f"CSV row ending at line {reader.line_num} has the wrong number of fields."
            )
        values = []
        for name, index in zip(requested, indices):
            try:
                value = float(row[index])
            except ValueError as exc:
                raise ValueError(
                    f"Column {name!r}, line {reader.line_num}: expected a finite number."
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"Column {name!r}, line {reader.line_num}: non-finite value.")
            values.append(value)
        rows.append(values)
    if len(rows) < 32:
        raise ValueError("At least 32 observations are required.")
    values_array = np.asarray(rows, dtype=float)
    if time_column:
        intervals = np.diff(values_array[:, -1])
        if np.any(intervals <= 0) or not np.allclose(
            intervals, 1.0 / sampling_rate, rtol=1e-4, atol=1e-9
        ):
            raise ValueError(
                "Timestamps must increase uniformly and match --sampling-rate; "
                "resolve gaps, duplicates, or rate mismatches before analysis."
            )
    data = values_array[:, : len(columns)]
    if difference:
        data = np.diff(data, axis=0)
    if window_size > len(data):
        raise ValueError("window_size exceeds the observations available after preprocessing.")
    if max_lag > len(data) - 3:
        raise ValueError("max_lag must leave at least three paired observations.")
    if not np.isfinite(data).all():
        raise ValueError("Preprocessing produced non-finite values; rescale the input.")
    for index, name in enumerate(columns):
        if np.all(data[:, index] == data[0, index]):
            raise ValueError(f"Signal column {name!r} is constant after preprocessing.")

    channels = {}
    for index, name in enumerate(columns):
        series = data[:, index]
        spectrum = ResonanceDetector().detect(series, method="fft", sampling_rate=sampling_rate)
        frequencies = spectrum["dominant_freq"]
        channels[name] = {
            "dominant_frequency_hz": float(frequencies[0]) if len(frequencies) else None,
            "spectrum": spectrum,
            "depth": DepthCalculator().calculate(
                series, window_size, sampling_rate=sampling_rate, resonance_frequencies=frequencies
            ),
        }
    rooting = RootingAnalyzer().analyze(
        data,
        max_lag=max_lag,
        n_surrogates=n_surrogates,
        random_state=random_state,
        alpha=alpha,
        surrogate_method=surrogate_method,
        correction="max_statistic",
    )
    # Preserve matrix order while presenting original names on edges.
    dimension_names = {f"dim_{index}": name for index, name in enumerate(columns)}
    for collection in ("candidate_edges", "significant_edges"):
        for edge in rooting[collection]:
            edge["source"] = dimension_names[edge["source"]]
            edge["target"] = dimension_names[edge["target"]]
            edge["lag_seconds"] = edge["lag"] / sampling_rate
    rooting["matrix_columns"] = list(columns)
    warnings = [
        "Statistical lead-lag relationships do not establish causation.",
        "Resonance depth is descriptive, not a significance test or a calibrated probability.",
        "Null assumptions are not automatically verified; nonstationary data can inflate false alarms.",
    ]
    if not time_column:
        warnings.append("Sampling regularity was not checked: no time column was supplied.")
    if difference:
        warnings.append("Results describe first differences, not original signal levels.")
    if n_surrogates == 0:
        warnings.append(
            "Inference is disabled. Candidate edges are exploratory; no significance claim is available."
        )
    elif 1.0 / (n_surrogates + 1) > alpha:
        warnings.append(
            "Too few surrogates to reach alpha; increase --n-surrogates before interpreting absent edges."
        )
    parameters = {
        "columns": list(columns),
        "sampling_rate": sampling_rate,
        "time_column": time_column,
        "window_size": window_size,
        "max_lag": max_lag,
        "n_surrogates": n_surrogates,
        "random_state": random_state,
        "alpha": alpha,
        "surrogate_method": surrogate_method,
        "difference": difference,
        "spectral_method": "fft",
        "rooting_method": "lagged_correlation",
        "correction": "max_statistic",
    }
    features = None
    if mathematical_features:
        from .math_features import analyze_mathematical_features

        parameters["mathematical_features"] = True
        features = analyze_mathematical_features(
            data,
            sampling_rate,
            column_names=list(columns),
            random_state=random_state,
            data_source=Path(path).name,
        )
        warnings.append(
            "Mathematical features are experimental; their measured behavior on synthetic data "
            "is in results/expected/resonance_math_benchmark.json."
        )
    provenance = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_rows": len(rows),
        "analyzed_rows": len(data),
        "versions": _versions(),
        "parameters": parameters,
    }
    run_id = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "provenance": provenance,
        "null_assumption": NULL_ASSUMPTIONS[surrogate_method],
        "warnings": warnings,
        "channels": channels,
        "rooting": rooting,
    }
    if features is not None:
        payload["mathematical_features"] = features
    return _json_safe(payload)


def _number(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else "n/a"


def _feature_lines(features: Dict[str, Any], cell: Callable[[Any], str]) -> List[str]:
    """Report section for the experimental mathematical features."""
    quality = features["data_quality"]
    record = features["evidence_record"]
    lines = [
        "",
        "## Experimental mathematical features",
        "",
        f"Evidence record `{record['record_id']}`: **{record['conclusion']['status']}**.",
        "",
        *[f"- {cell(reason)}" for reason in record["conclusion"]["reasons"]],
        "",
        "Data-quality gate: "
        + (
            "**abstains**: relationship tests are not interpretable here."
            if quality["abstain"]
            else "passes."
        ),
        "",
        *[f"- {cell(item)}" for item in quality["reasons"] + quality["warnings"]],
    ]
    phase = features.get("phase_synchrony")
    if phase:
        lines += [
            "",
            "| Channel pair | Status | Ratio n:m | PLV | Holm-adjusted p | Effective phase samples |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for pair in phase["pairs"]:
            ratio = pair["selected_ratio"]
            lines.append(
                f"| {cell(pair['channels'][0])} – {cell(pair['channels'][1])} | {pair['status']} |"
                f" {f'{ratio[0]}:{ratio[1]}' if ratio else 'n/a'} | {_number(pair['plv'])} |"
                f" {_number(pair['adjusted_p_value'])} | {_number(pair['effective_samples'])} |"
            )
        lines += [
            "",
            "Phase synchronization is undirected. An inconclusive pair is not evidence of no "
            "relationship.",
        ]
    return lines


def render_report(payload: Dict[str, Any]) -> str:
    """Render a compact review report alongside the complete machine-readable output."""

    def cell(value: Any) -> str:
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")

    provenance = payload["provenance"]
    rooting = payload["rooting"]
    lines = [
        "# DRR analysis report",
        "",
        f"Run: `{payload['run_id']}`",
        "",
        f"Input SHA-256: `{provenance['input_sha256']}`",
        "",
        f"Analyzed **{provenance['analyzed_rows']} observations** across **{len(payload['channels'])} channels**.",
        "",
        "## Interpretation boundaries",
        "",
        *[f"- {warning}" for warning in payload["warnings"]],
        "",
        "## Channel diagnostics",
        "",
        "| Channel | Dominant frequency (Hz) | Resonance depth |",
        "| --- | ---: | ---: |",
    ]
    for name, result in payload["channels"].items():
        frequency = result["dominant_frequency_hz"]
        frequency_label = f"{frequency:.6g}" if frequency is not None else "No detected peak"
        depth = result["depth"]["resonance_depth"]
        lines.append(f"| {cell(name)} | {frequency_label} | {depth:.6g} |")
    lines.extend(
        [
            "",
            "## Lead-lag inference",
            "",
            f"Null assumption: {payload['null_assumption']}",
            "",
            f"Max-statistic correction; alpha = {rooting['alpha']}; surrogates = {rooting['n_surrogates']}.",
            "",
            f"Minimum attainable p-value: {rooting['minimum_attainable_p_value']:.6g}.",
            "",
        ]
    )
    if not rooting["inference_available"]:
        lines.append(
            "**Inference disabled.** Inspect candidate edges in analysis.json for exploration only."
        )
    elif not rooting["significant_edges"]:
        lines.append(
            "No candidate edge passed the selected test. This does not demonstrate absence of a relationship."
        )
    else:
        lines.extend(
            [
                "| Source | Target | Lag (samples) | Lag (seconds) | Adjusted p-value |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for edge in rooting["significant_edges"]:
            lines.append(
                f"| {cell(edge['source'])} | {cell(edge['target'])} | {edge['lag']} | "
                f"{edge['lag_seconds']:.6g} | {edge['adjusted_p_value']:.6g} |"
            )
    if "mathematical_features" in payload:
        lines.extend(_feature_lines(payload["mathematical_features"], cell))
    lines.extend(
        [
            "",
            "## Reproduction settings",
            "",
            "```json",
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "Full spectra, depth components, candidate edges, test matrices, and versions are in analysis.json.",
            "",
            "Before interpreting a result, check the null assumptions, compare an appropriate baseline, "
            "and validate on a separate dataset. These diagnostics do not transfer evidence between domains.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(payload: Dict[str, Any], output_dir: Path) -> None:
    """Publish both files together; refuse existing destinations rather than overwrite evidence."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new directory."
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = render_report(payload)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".drr-", dir=output_dir.parent) as staging:
        staged = Path(staging) / "bundle"
        staged.mkdir()
        (staged / "analysis.json").write_text(encoded, encoding="utf-8")
        (staged / "report.md").write_text(report, encoding="utf-8")
        staged.rename(output_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze regularly sampled CSV signals and export a reproducible DRR report."
    )
    parser.add_argument("input", type=Path, help="UTF-8 CSV with a header row.")
    parser.add_argument(
        "--columns", nargs="+", required=True, help="At least two signal column names."
    )
    parser.add_argument(
        "--sampling-rate", type=float, required=True, help="Samples per second (Hz)."
    )
    parser.add_argument("--time-column", help="Optional numeric time column in seconds.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for JSON and Markdown."
    )
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument(
        "--n-surrogates", type=int, default=99, help="Surrogate draws; 0 disables inference."
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--surrogate-method", choices=list(NULL_ASSUMPTIONS), default="circular_shift"
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help="Analyze first differences; recorded in provenance.",
    )
    parser.add_argument(
        "--mathematical-features",
        action="store_true",
        help="Add experimental data-quality checks, phase-synchronization tests, and an evidence record.",
    )
    args = parser.parse_args(argv)
    options = vars(args).copy()
    path = options.pop("input")
    destination = options.pop("output_dir")
    try:
        if destination.exists():
            raise FileExistsError(
                f"Output directory already exists: {destination}. Choose a new directory."
            )
        payload = analyze_csv(path, **options)
        write_bundle(payload, destination)
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        parser.error(str(exc))
    print(
        f"Analyzed {payload['provenance']['analyzed_rows']} rows; report: {destination / 'report.md'}"
    )
    print(f"Complete results: {destination / 'analysis.json'}")
    for warning in payload["warnings"]:
        print(f"Note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
