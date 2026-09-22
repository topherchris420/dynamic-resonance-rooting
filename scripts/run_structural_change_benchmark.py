#!/usr/bin/env python3
"""Run the preregistered QBO structural-change benchmark.

From the repository root::

    python scripts/run_structural_change_benchmark.py --output-dir results/expected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drr_framework.external_benchmark.qbo import (
    run_structural_change_benchmark,
    write_benchmark_artifact,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/expected",
        help="Directory for the JSON artifact and the Markdown report.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Print the claim and skip writing files.",
    )
    args = parser.parse_args(argv)
    artifact = run_structural_change_benchmark()
    claim = artifact["claim"]
    print(json.dumps({"status": claim["status"], "reason": claim["reason"], "text": claim["text"]}))
    if not args.no_artifacts:
        json_path, markdown_path = write_benchmark_artifact(artifact, Path(args.output_dir))
        print(f"wrote {json_path}")
        print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
