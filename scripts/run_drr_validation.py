#!/usr/bin/env python3
"""
Automated Reproducibility Validation Runner for Dynamic Resonance Rooting (DRR).

This script executes the complete end-to-end evaluation suite:
1. Deterministic reproduction benchmark (from validation.py)
2. Temporal out-of-sample benchmark (from benchmarks_suite.py)
3. Sensitivity sweeps and placebo/null tests (from sensitivity_tests.py)
4. Evidence card generation (from evidence_card.py)

Outputs JSON and CSV artifacts with cryptographic SHA-256 hashes for reproducibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from drr_framework.benchmarks import BenchmarkSystems
from drr_framework.benchmarks_suite import run_temporal_out_of_sample_benchmark
from drr_framework.evidence_card import create_drr_evidence_card
from drr_framework.sensitivity_tests import (
    run_parameter_sensitivity_experiment,
    run_placebo_and_null_tests,
)
from drr_framework.validation import run_reproduction_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drr_validation_runner")


def run_full_validation_suite(
    output_dir: str = "results/validation", random_state: int = 42
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("1. Running Deterministic Reproduction Experiment...")
    reproduction_res = run_reproduction_experiment(
        output_dir=output_path, random_state=random_state
    )

    logger.info("2. Running Temporal Out-of-Sample Benchmark Suite...")
    rng = np.random.default_rng(random_state)
    t, synth_data = BenchmarkSystems.generate_lorenz_data(duration=20, dt=0.01)
    events = (np.abs(synth_data[:, 0]) > 15.0).astype(int)
    oos_res = run_temporal_out_of_sample_benchmark(synth_data, events, window_size=100)

    logger.info("3. Running Automated Parameter Sensitivity Sweeps...")
    sensitivity_res = run_parameter_sensitivity_experiment(
        synth_data[:300],
        sampling_rate=100.0,
        window_sizes=[32, 64],
        methods=["welch"],
        tau_values=[1, 2],
    )

    logger.info("4. Running Placebo and Null Experiments...")
    null_res = run_placebo_and_null_tests(
        n_samples=300, sampling_rate=100.0, random_state=random_state
    )

    logger.info("5. Generating Supervisory Evidence Card...")
    # The card describes the coupled-oscillator reproduction benchmark: every
    # statistic on it is measured by step 1, and the Lorenz sweeps are cited
    # only as the robustness and baseline context they are.
    edge_p = float(reproduction_res["expected_edge_adjusted_p_value"])
    card = create_drr_evidence_card(
        signal_id=f"coupled_oscillator_{random_state}",
        variables=["dim_0", "dim_1", "dim_2"],
        methodology="welch_depth_lagged_correlation_circular_shift",
        parameter_configuration={
            "window_size": 256,
            "max_lag": 4,
            "n_surrogates": 25,
            "correction": "max_statistic",
        },
        p_value=edge_p,
        effect_size_dict={"resonance_depth": reproduction_res["resonance_depth"]},
        robustness_score=1.0 - sensitivity_res["fragility_score"],
        benchmark_comparison={
            "lorenz_auroc_vs_volatility": oos_res["DRR_Resonance_Depth"]["auroc"]
            - oos_res["Rolling_Volatility"]["auroc"]
        },
        confidence_interval=(
            reproduction_res["resonance_depth_confidence_interval"][0],
            reproduction_res["resonance_depth_confidence_interval"][1],
        ),
        data_provenance={"dataset": "synthetic coupled oscillator (known 12.5 Hz, lag 2)"},
        detection_statement=(
            f"Peak at {reproduction_res['detected_frequency_hz']:.2f} Hz; "
            f"dim_0 -> dim_1 adjusted p = {edge_p:.3f} "
            f"(minimum attainable {reproduction_res['minimum_attainable_p_value']:.3f})."
        ),
        interpretation_statement=(
            "Specification check: the injected frequency and lead-lag edge are recovered. "
            "Synthetic ground truth only; no claim about real systems."
        ),
    )

    full_payload = {
        "random_state": random_state,
        "reproduction_summary": reproduction_res,
        "out_of_sample_benchmark": oos_res,
        "parameter_sensitivity": sensitivity_res,
        "placebo_and_null_tests": null_res,
        "evidence_card": card.to_dict(),
    }

    payload_json = json.dumps(full_payload, indent=2, sort_keys=True)
    reproducibility_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    full_payload["suite_reproducibility_hash"] = reproducibility_hash

    (output_path / "drr_full_validation_summary.json").write_text(
        json.dumps(full_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.info("Validation suite complete. SHA-256: %s", reproducibility_hash)
    return full_payload


if __name__ == "__main__":
    run_full_validation_suite()
