#!/usr/bin/env python3
"""Select without test data, then evaluate a sealed financial forecasting study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drr_framework.finance.holdout.study import evaluate, select, verify_replay


ROOT = Path(__file__).resolve().parents[1]


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
        filename = "selection.json" if args.stage == "select" else "summary.json"
        verify_replay(result, json.loads((args.verify_against / filename).read_text()))
        print("Numerical replay matches the published artifact.")


if __name__ == "__main__":
    main()
