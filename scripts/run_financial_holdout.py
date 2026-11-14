#!/usr/bin/env python3
"""Select without test data, then evaluate a sealed financial forecasting study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drr_framework.finance.holdout.study import evaluate, select, verify_lock, verify_replay


ROOT = Path(__file__).resolve().parents[1]


def verify_selection_replay(root: Path, study: Path, actual: Path, expected: Path) -> None:
    actual_lock = verify_lock(root, study, actual)
    expected_lock = verify_lock(root, study, expected)
    # Each digest authenticates its own trial file. Across platforms, compare
    # the trial values with the same numerical tolerance as the other outputs.
    del actual_lock["trials_sha256"]
    del expected_lock["trials_sha256"]
    verify_replay(actual_lock, expected_lock, "selection")
    verify_replay(
        json.loads((actual / "trials.json").read_text()),
        json.loads((expected / "trials.json").read_text()),
        "trials",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "evaluate"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-dir", type=Path, default=ROOT / "experiments/financial_holdout/selection"
    )
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    study = ROOT / "experiments/financial_holdout"
    if args.stage == "select":
        result = select(ROOT, study, args.output_dir)
        print(
            json.dumps(
                {
                    "primary_candidate": result["primary_candidate"],
                    "comparator": result["economic_comparator"],
                }
            )
        )
    else:
        result = evaluate(ROOT, study, args.selection_dir, args.output_dir)
        print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}))
    if args.verify_against:
        if args.stage == "select":
            verify_selection_replay(ROOT, study, args.output_dir, args.verify_against)
        else:
            verify_replay(result, json.loads((args.verify_against / "summary.json").read_text()))
        print("Numerical replay matches the published artifact.")


if __name__ == "__main__":
    main()
