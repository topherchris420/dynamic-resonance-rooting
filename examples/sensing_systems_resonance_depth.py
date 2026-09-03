"""Sensing Systems DRR Benchmark Example.

Inspired by signal-processing architectures in US Patent 8,169,362 B2
("Mobile Sense Through The Wall Radar System", Cook et al., Raytheon, issued 2012).

This script uses radar DSP pipeline operations as a cross-domain stress test
for DRR's four diagnostic categories:
1. Resonance detection (range/Doppler spectral analysis)
2. Rooting analysis (inter-channel/range-bin lead-lag correlation)
3. Resonance depth (stability of low-frequency target mode against clutter)
4. State-space diagnostics (Kalman filter tracking and change detection)

Run from the repository root:

    python examples/sensing_systems_resonance_depth.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drr_framework import DynamicResonanceRooting, generate_micro_doppler_analog

OUTPUT_DIR = ROOT / "results" / "sensing_systems"


def generate_sample(
    duration: float = 30.0,
    sampling_rate: float = 100.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate multi-channel micro-Doppler radar DSP analog time series."""
    return generate_micro_doppler_analog(
        duration=duration,
        sampling_rate=sampling_rate,
        n_channels=3,
        target_frequency_hz=0.3,
        clutter_frequency_hz=0.02,
        lag=2,
        regime_change_time=15.0,
        noise_scale=0.05,
        random_state=random_state,
    )


def analyze_sample(t: np.ndarray, data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Run DRR resonance, depth, rooting, and state-space analysis on sensing data."""
    sampling_rate = float(metadata["sampling_rate_hz"])
    drr = DynamicResonanceRooting(embedding_dim=3, tau=1, sampling_rate=sampling_rate)

    analysis_results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=200,
        state_space=True,
        state_space_horizon=10,
        method="welch",
    )

    resonance_depths = analysis_results.get("resonance_depths", {})
    rooting_results = analysis_results.get("rooting_analysis", {})
    significant_edges = (
        rooting_results.get("significant_edges", []) if isinstance(rooting_results, dict) else []
    )
    state_space_res = analysis_results.get("state_space_analysis", {})

    # Extract state-space innovations / log likelihood if available
    state_space_summary = {}
    if isinstance(state_space_res, dict) and "log_likelihood" in state_space_res:
        state_space_summary = {
            "log_likelihood": float(state_space_res["log_likelihood"]),
            "n_states": int(state_space_res.get("n_states", data.shape[1])),
            "spectral_radius": float(state_space_res.get("spectral_radius", 0.0)),
        }

    return {
        "benchmark_system": metadata["benchmark_system"],
        "inspiration_source": "US Patent 8,169,362 B2 (Raytheon, 2012)",
        "sampling_rate_hz": sampling_rate,
        "duration_seconds": float(metadata["duration_seconds"]),
        "n_channels": int(metadata["n_channels"]),
        "ground_truth_target_freq_hz": float(metadata["target_frequency_hz"]),
        "ground_truth_clutter_freq_hz": float(metadata["clutter_frequency_hz"]),
        "ground_truth_inter_channel_lag_samples": int(metadata["inter_channel_lag_samples"]),
        "ground_truth_regime_change_time_s": float(metadata["regime_change_time_seconds"]),
        "resonance_depths": {k: float(v) for k, v in resonance_depths.items()},
        "average_resonance_depth": (
            float(np.mean(list(resonance_depths.values()))) if resonance_depths else 0.0
        ),
        "significant_edges": significant_edges,
        "significant_edges_count": len(significant_edges),
        "state_space_diagnostics": state_space_summary,
        "is_rooted": bool(analysis_results.get("is_rooted", False)),
        "caveat": "Cross-domain DSP software engineering benchmark analogy only. Not an operational radar system.",
    }


def save_trace_plot(
    t: np.ndarray, data: np.ndarray, metadata: Dict[str, Any], output_dir: Path = OUTPUT_DIR
) -> Path:
    """Save visualization plot of multi-channel radar returns with regime transition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "micro_doppler_trace.png"

    plt.figure(figsize=(11, 6))

    plt.subplot(2, 1, 1)
    for c in range(data.shape[1]):
        plt.plot(t, data[:, c], label=f"Channel {c} (Range bin {c})", alpha=0.8, linewidth=0.8)
    plt.axvline(
        x=metadata["regime_change_time_seconds"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Regime Shift (Standing -> Walking)",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")
    plt.title("Sensing System Analog: Multi-Channel Micro-Doppler Time Series")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    # Highlight high-frequency bandpass/diff to visualize micro-Doppler walking onset
    diff_signal = np.diff(data[:, 0], prepend=data[0, 0])
    plt.plot(
        t,
        diff_signal,
        color="purple",
        alpha=0.7,
        linewidth=0.7,
        label="Channel 0 Onset / Innovation Proxy",
    )
    plt.axvline(
        x=metadata["regime_change_time_seconds"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Regime Shift Onset",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered Derivative")
    plt.title("Coherent Change Detection Proxy (High-Pass / State Innovation)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def export_summary(summary: Dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    """Export summary metrics as JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sensing_systems_summary.json"
    csv_path = output_dir / "sensing_systems_metrics.csv"

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
    """Run the sensing system DRR analysis workflow and save results."""
    t, data, metadata = generate_sample()
    summary = analyze_sample(t, data, metadata)
    figure_path = save_trace_plot(t, data, metadata)
    paths = export_summary(summary)
    paths["figure"] = figure_path

    print(f"Wrote sensing systems JSON: {paths['json']}")
    print(f"Wrote sensing systems CSV: {paths['csv']}")
    print(f"Wrote sensing systems figure: {paths['figure']}")
    return paths


if __name__ == "__main__":
    main()
