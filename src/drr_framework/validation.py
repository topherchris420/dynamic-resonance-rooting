"""Synthetic validation routines for Dynamic Resonance Rooting.

The functions in this module generate deterministic benchmark artifacts that can
be regenerated from a clean checkout. They are intentionally lightweight: no
external datasets, no notebooks, and no plotting required for CI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .modules import DepthCalculator, ResonanceDetector, RootingAnalyzer


def generate_coupled_oscillator(
    sampling_rate: float = 200.0,
    duration: float = 4.0,
    target_frequency_hz: float = 12.5,
    lag: int = 2,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a system with known resonance and directed coupling.

    Returns:
        t: Time axis in seconds.
        data: Three-column system where dim_0 drives dim_1 at ``lag`` samples
            and dim_2 is an independent distractor.
    """
    rng = np.random.default_rng(random_state)
    n = int(sampling_rate * duration)
    t = np.arange(n) / sampling_rate

    source = (
        np.sin(2 * np.pi * target_frequency_hz * t)
        + 0.2 * np.sin(2 * np.pi * (target_frequency_hz * 2.0) * t)
        + noise_scale * rng.normal(size=n)
    )
    target = np.roll(source, lag) + noise_scale * rng.normal(size=n)
    target[:lag] = noise_scale * rng.normal(size=lag)
    distractor = rng.normal(scale=0.4, size=n)

    return t, np.column_stack([source, target, distractor])


def run_reproduction_experiment(
    output_dir: str | Path = "results/reproduction",
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, object]:
    """Run the deterministic DRR validation experiment used by the README.

    The benchmark encodes two known truths: dim_0 has a dominant 12.5 Hz
    resonance, and dim_0 drives dim_1 with a two-sample lag. The returned
    summary is JSON-serializable and suitable for paper supplement artifacts.
    """
    sampling_rate = 200.0
    target_frequency_hz = 12.5
    lag = 2
    _, data = generate_coupled_oscillator(
        sampling_rate=sampling_rate,
        target_frequency_hz=target_frequency_hz,
        lag=lag,
        random_state=random_state,
    )

    detector_result = ResonanceDetector().detect(
        data[:, 0],
        method="welch",
        sampling_rate=sampling_rate,
        peak_height_ratio=0.2,
    )
    detected_frequency_hz = float(detector_result["dominant_freq"][0])

    depth_result = DepthCalculator().calculate(
        data[:, 0],
        window_size=256,
        sampling_rate=sampling_rate,
        resonance_frequencies=np.array([detected_frequency_hz]),
    )
    rooting_result = RootingAnalyzer().analyze(
        data,
        max_lag=4,
        n_surrogates=25,
        random_state=random_state,
        alpha=0.05,
    )

    expected_edge_detected = any(
        edge["source"] == "dim_0" and edge["target"] == "dim_1"
        for edge in rooting_result["significant_edges"]
    )

    summary: Dict[str, object] = {
        "benchmark": "coupled_oscillator_known_frequency_and_direction",
        "sampling_rate_hz": sampling_rate,
        "target_frequency_hz": target_frequency_hz,
        "detected_frequency_hz": detected_frequency_hz,
        "frequency_error_hz": abs(detected_frequency_hz - target_frequency_hz),
        "resonance_depth": float(depth_result["resonance_depth"]),
        "resonance_depth_components": {
            key: float(value) for key, value in depth_result["components"].items()
        },
        "resonance_depth_confidence_interval": [
            float(depth_result["confidence_interval"][0]),
            float(depth_result["confidence_interval"][1]),
        ],
        "rooting_method": rooting_result["method"],
        "expected_edge": "dim_0 -> dim_1",
        "expected_lag_samples": lag,
        "expected_edge_detected": bool(expected_edge_detected),
        "significant_edges_count": len(rooting_result["significant_edges"]),
    }

    if save_artifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "drr_reproduction_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with (output_path / "drr_reproduction_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            for key, value in summary.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                writer.writerow({"metric": key, "value": value})

    return summary
