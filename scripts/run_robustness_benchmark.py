#!/usr/bin/env python3
"""Compare rooting methods with separately calibrated simple baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drr_framework.robustness_benchmark import (
    DEFAULT_CONFIG,
    METHODS,
    SCHEMA_VERSION,
    config_digest,
    render_robustness_report,
    run_robustness_benchmark,
    validate_config,
)


def write_comparison_figure(artifact, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    colors = ("#6b7280", "#b45309", "#0369a1", "#047857")
    labels = (
        "Raw correlation",
        "AR(1) residual correlation",
        "DRR circular shift",
        "DRR phase randomization",
    )
    stationary = {case["id"] for case in artifact["config"]["cases"] if len(case["regimes"]) == 1}
    for axis, coupling, metric, title in zip(
        axes,
        (0.0, 0.1, 0.3),
        ("any_non_injected_direction", "true_edge_exact_lag", "true_edge_exact_lag"),
        (
            "Uncoupled: family-wise false alarms",
            "Weak coupling 0.10: injected-lag recovery",
            "Coupling 0.30: injected-lag recovery",
        ),
    ):
        for method, color, label in zip(METHODS, colors, labels):
            rows = sorted(
                [
                    row
                    for row in artifact["summary"]
                    if row["case"] in stationary
                    and row["method"] == method
                    and row["coupling"] == coupling
                ],
                key=lambda row: row["hurst"],
            )
            rates = [row[metric]["rate"] for row in rows]
            # The existing Wilson helper may round its zero-event lower bound
            # to a tiny positive epsilon. Matplotlib requires nonnegative spans.
            lower = [max(0.0, row[metric]["rate"] - row[metric]["ci95"][0]) for row in rows]
            upper = [max(0.0, row[metric]["ci95"][1] - row[metric]["rate"]) for row in rows]
            axis.errorbar(
                [row["hurst"] for row in rows],
                rates,
                yerr=[lower, upper],
                color=color,
                label=label,
                marker="o",
                capsize=3,
                linewidth=1.5,
            )
        if coupling == 0:
            axis.axhline(artifact["config"]["alpha"], color="#111827", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Hurst exponent", ylim=(-0.03, 1.03))
        axis.set_xticks((0.5, 0.85, 0.95))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Proportion of independent trials")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.text(
        0.5,
        0.015,
        "Synthetic data; pointwise Wilson 95% intervals conditional on frozen thresholds. Read recovery alongside null error.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.88))
    figure.savefig(output_dir / "comparison.png", dpi=180)
    figure.savefig(output_dir / "comparison.svg")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "robustness-benchmark"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Small smoke run, not the reference study."
    )
    args = parser.parse_args(argv)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.quick:
        config.update(n_samples=128, n_trials=8, n_calibration=39, n_surrogates=19)
    validate_config(config)
    if args.output_dir.exists():
        parser.error(
            "output directory already exists; select a fresh directory to preserve earlier runs"
        )

    sources = [
        "src/drr_framework/robustness_benchmark.py",
        "src/drr_framework/calibration.py",
        "src/drr_framework/dependent_noise.py",
        "src/drr_framework/modules.py",
        "scripts/run_robustness_benchmark.py",
    ]
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "study_status": "fixed exploratory comparison; not external preregistration",
        "config": config,
        "config_sha256": config_digest(config),
        "source_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources
        },
    }
    # Seal the protocol before producing any evaluation measurements. Interrupted
    # runs retain it; every subsequent attempt must use a different directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_bytes = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (args.output_dir / "protocol.json").write_bytes(protocol_bytes)
    print(f"Protocol retained in {args.output_dir}", flush=True)
    artifact = run_robustness_benchmark(
        config, progress=lambda name: print(f"Completed {name}", flush=True)
    )
    artifact["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_robustness_report(artifact), encoding="utf-8")
    write_comparison_figure(artifact, args.output_dir)
    print(
        f"Retained {len(artifact['records'])} trial/window records and {len(artifact['summary'])} comparisons"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
