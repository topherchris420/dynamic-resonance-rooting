#!/usr/bin/env python3
"""Run the dependent-noise null study and write its artifacts.

    python scripts/run_dependent_noise_study.py --output-dir results/expected

The study measures the family-wise false-alarm rate of each surrogate null
(circular shift, phase randomization, permutation) on independent channels of
fractional Gaussian noise, GARCH(1,1) noise, and random walks, then the power
to recover a known edge when every channel has long memory. It writes
``dependent_noise_null_study.json`` and a rendered ``.md`` companion. The exit
code reports the integrity checks only; the preregistered hypotheses are
recorded with their outcomes, whatever they are.
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

from drr_framework.dependent_noise import (  # noqa: E402
    DEFAULT_CONFIG,
    render_dependent_noise_report,
    run_dependent_noise_study,
)

ARTIFACT_STEM = "dependent_noise_null_study"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "expected")
    parser.add_argument(
        "--trials-scale",
        type=float,
        default=1.0,
        help="Multiply every trial count (use < 1 for a quick look; the committed run uses 1).",
    )
    args = parser.parse_args(argv)

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.trials_scale != 1.0:
        for section in ("size", "coupled"):
            scaled = int(round(config[section]["n_trials"] * args.trials_scale))
            config[section]["n_trials"] = max(10, scaled)

    started = time.perf_counter()
    artifact = run_dependent_noise_study(config)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{ARTIFACT_STEM}.json"
    md_path = args.output_dir / f"{ARTIFACT_STEM}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_dependent_noise_report(artifact), encoding="utf-8")

    print(f"Wrote {json_path} and {md_path} in {elapsed:.0f}s")
    for hypothesis in artifact["hypotheses"]:
        print(f"  {hypothesis['outcome']:>13}  {hypothesis['id']}: {hypothesis['statement']}")
    for name, holds in artifact["checks"].items():
        print(f"  {'PASS' if holds else 'FAIL'}  {name}")
    return 0 if all(artifact["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
