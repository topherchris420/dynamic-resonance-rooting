#!/usr/bin/env python3
"""Run the phase and harmonic mathematics benchmark and write its artifacts.

    python scripts/run_resonance_math_benchmark.py --output-dir results/expected --workers 4

The benchmark simulates channel pairs with a known relationship, applies
existing DRR rooting, conventional baselines, the proposed phase-synchrony
test, its ablations, an oriented-area test, and the gated combinations, and
writes ``resonance_math_benchmark.json`` with a rendered ``.md`` companion.
Rows are identical for any number of workers. The exit code reports the
integrity checks only; the preregistered hypotheses are recorded with their
outcomes, whatever they are.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drr_framework.resonance_math_benchmark import (  # noqa: E402
    DEFAULT_CONFIG,
    render_benchmark_report,
    run_benchmark,
)

ARTIFACT_STEM = "resonance_math_benchmark"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "expected")
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel processes (rows unchanged)."
    )
    parser.add_argument(
        "--trials-scale",
        type=float,
        default=1.0,
        help="Multiply every trial count (use < 1 for a quick look; the committed run uses 1).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("development", "holdout"),
        default=["development", "holdout"],
    )
    args = parser.parse_args(argv)

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.trials_scale != 1.0:
        for split in ("development", "holdout"):
            for kind, count in config[split]["n_trials"].items():
                config[split]["n_trials"][kind] = max(10, int(round(count * args.trials_scale)))

    started = time.perf_counter()
    artifact = run_benchmark(config, workers=args.workers, splits=args.splits)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{ARTIFACT_STEM}.json"
    md_path = args.output_dir / f"{ARTIFACT_STEM}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_benchmark_report(artifact), encoding="utf-8")

    print(f"Wrote {json_path} and {md_path} in {elapsed:.0f}s")
    print(
        f"  preregistration matches the frozen record: {artifact['matches_frozen_preregistration']}"
    )
    for hypothesis in artifact["hypotheses"]:
        print(f"  {hypothesis['outcome']:>13}  {hypothesis['id']}: {hypothesis['statement']}")
    for name, holds in artifact["checks"].items():
        print(f"  {'PASS' if holds else 'FAIL'}  {name}")
    return 0 if all(artifact["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
