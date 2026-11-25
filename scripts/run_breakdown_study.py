#!/usr/bin/env python3
"""Calibrate or replay the registered lead-lag breakdown study offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drr_framework.finance.breakdown.study import calibrate, evaluate
from drr_framework.finance.holdout.study import verify_replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "evaluate"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir", type=Path, default=ROOT / "experiments/lead_lag_breakdown/calibration"
    )
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    study = ROOT / "experiments/lead_lag_breakdown"
    if args.stage == "calibrate":
        result = calibrate(ROOT, study, args.output_dir)
        filename = "lock.json"
        print("Calibrated and locked; no held-out scenarios or market outcomes scored.")
    else:
        result = evaluate(ROOT, study, args.calibration_dir, args.output_dir)
        filename = "summary.json"
        print(
            json.dumps(
                {"verdict": result["simulation"]["verdict"], "gates": result["simulation"]["gates"]}
            )
        )
    if args.verify_against:
        verify_replay(result, json.loads((args.verify_against / filename).read_text()))
        print("Numerical replay matches published evidence.")


if __name__ == "__main__":
    main()
