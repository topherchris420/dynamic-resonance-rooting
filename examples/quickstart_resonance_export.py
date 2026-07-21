"""Load, analyze, visualize, and export a compact DRR resonance example.

Run from the repository root:

    python examples/quickstart_resonance_export.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drr_framework import DepthCalculator, ResonanceDetector, RootingAnalyzer

DATA_PATH = ROOT / "data" / "raw" / "coupled_oscillator_sample.csv"
OUTPUT_DIR = ROOT / "results" / "quickstart"


def load_sample(path: Path = DATA_PATH) -> tuple[float, np.ndarray]:
    """Load the checked-in coupled-oscillator sample dataset."""

    frame = pd.read_csv(path)
    sampling_rate = float(frame.attrs.get("sampling_rate_hz", 200.0))
    data = frame[["dim_0", "dim_1", "dim_2"]].to_numpy(dtype=float)
    return sampling_rate, data


def analyze_sample(data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
    """Run resonance, depth, and rooting analysis on the sample dataset."""

    detector_result = ResonanceDetector().detect(
        data[:, 0],
        method="welch",
        sampling_rate=sampling_rate,
        peak_height_ratio=0.2,
    )
    detected_frequency = float(detector_result["dominant_freq"][0])
    depth_result = DepthCalculator().calculate(
        data[:, 0],
        window_size=256,
        sampling_rate=sampling_rate,
        resonance_frequencies=np.array([detected_frequency]),
    )
    rooting_result = RootingAnalyzer().analyze(
        data,
        max_lag=4,
        n_surrogates=25,
        random_state=42,
        alpha=0.05,
    )

    return {
        "dataset": DATA_PATH.relative_to(ROOT).as_posix(),
        "sampling_rate_hz": sampling_rate,
        "detected_frequency_hz": detected_frequency,
        "resonance_depth": float(depth_result["resonance_depth"]),
        "resonance_depth_components": {
            key: float(value) for key, value in depth_result["components"].items()
        },
        "rooting_method": rooting_result["method"],
        "significant_edges": rooting_result["significant_edges"],
    }


def save_trace_plot(data: np.ndarray, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save a quick visual trace for the first two oscillator dimensions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "coupled_oscillator_trace.png"

    plt.figure(figsize=(10, 4))
    plt.plot(data[:, 0], label="dim_0 source", linewidth=1.0)
    plt.plot(data[:, 1], label="dim_1 lagged response", linewidth=1.0, alpha=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("DRR Quickstart Coupled Oscillator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def export_summary(summary: Dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    """Export analysis summary as JSON and flat CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quickstart_summary.json"
    csv_path = output_dir / "quickstart_metrics.csv"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})

    return {"json": json_path, "csv": csv_path}


def main() -> Dict[str, Path]:
    """Run the quickstart workflow and return written artifact paths."""

    sampling_rate, data = load_sample()
    summary = analyze_sample(data, sampling_rate)
    figure_path = save_trace_plot(data)
    paths = export_summary(summary)
    paths["figure"] = figure_path

    print(f"Wrote quickstart JSON: {paths['json']}")
    print(f"Wrote quickstart CSV: {paths['csv']}")
    print(f"Wrote quickstart figure: {paths['figure']}")
    return paths


if __name__ == "__main__":
    main()
