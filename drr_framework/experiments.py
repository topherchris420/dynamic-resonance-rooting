"""Command-line entry points for DRR research experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .validation import run_reproduction_experiment


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible DRR research experiments.")
    parser.add_argument(
        "--output-dir",
        default="results/reproduction",
        help="Directory for generated JSON and CSV artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Deterministic random seed for synthetic validation data.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Run the experiment and print JSON without writing files.",
    )
    args = parser.parse_args(argv)

    summary = run_reproduction_experiment(
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
        save_artifacts=not args.no_artifacts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
