#!/usr/bin/env python3
"""Copy the landing page's evidence numbers out of the committed artifacts.

    python scripts/sync_site_evidence.py          # rewrite index.html in place
    python scripts/sync_site_evidence.py --check  # exit 1 if the page is stale

``index.html`` carries a ``<script type="application/json" id="evidence-data">``
block. Every number the page shows about calibration or external evidence is
read from that block, and that block is generated here from
``results/expected``. ``tests/test_site_evidence.py`` fails when they drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "index.html"
CALIBRATION = ROOT / "results" / "expected" / "rooting_calibration_study.json"
QBO = ROOT / "results" / "expected" / "qbo_structural_change_benchmark.json"
BLOCK = re.compile(
    r'(<script type="application/json" id="evidence-data">)(.*?)(</script>)', re.DOTALL
)


def _round(value: float) -> float:
    return round(float(value), 4)


def build_evidence() -> Dict[str, Any]:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    qbo = json.loads(QBO.read_text(encoding="utf-8"))
    config = calibration["config"]

    size: Dict[float, Dict[str, Any]] = {}
    for row in calibration["size"]:
        entry = size.setdefault(row["ar_coefficient"], {"ar": row["ar_coefficient"]})
        entry[row["surrogate_method"]] = _round(row["family_wise_error"]["rate"])
        entry[row["surrogate_method"] + "_ci"] = [
            _round(bound) for bound in row["family_wise_error"]["ci95"]
        ]

    depth = {}
    for row in calibration["depth"]:
        if row["signal"] == "ar1_noise":
            key = f"ar1_{row['ar_coefficient']:.2f}"
        else:
            key = f"tone_amp_{row['amplitude']:.2f}"
        depth[key] = {
            "median": _round(row["depth_median"]),
            "q05": _round(row["depth_q05"]),
            "q95": _round(row["depth_q95"]),
        }

    drr = next(model for model in qbo["models"] if model["role"] == "drr")
    return {
        "calibration": {
            "source": "results/expected/rooting_calibration_study.json",
            "alpha": config["alpha"],
            "n_samples": config["n_samples"],
            "n_surrogates": config["n_surrogates"],
            "size_trials": config["size"]["n_trials"],
            "power_trials": config["power"]["n_trials"],
            "power_lag": config["power"]["lag"],
            "size": [size[key] for key in sorted(size)],
            "power": [
                {
                    "coupling": row["coupling"],
                    "rate": _round(row["true_edge_at_true_lag"]["rate"]),
                    "ci": [_round(bound) for bound in row["true_edge_at_true_lag"]["ci95"]],
                }
                for row in calibration["power"]
            ],
            "depth": depth,
        },
        "qbo": {
            "source": "results/expected/qbo_structural_change_benchmark.json",
            "status": qbo["claim"]["status"],
            "reason": qbo["claim"]["reason"],
            "domain": qbo["claim"]["domain"],
            "tolerance": qbo["analysis"]["false_alarm_tolerance"],
            "drr_holdout_false_alarm_rate": _round(drr["holdout_false_alarm_rate"]),
            "unique_event_ids": qbo["unique_event_ids"],
        },
    }


def render_block(evidence: Dict[str, Any]) -> str:
    return "\n" + json.dumps(evidence, indent=2, sort_keys=True) + "\n    "


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting")
    args = parser.parse_args(argv)

    page = PAGE.read_text(encoding="utf-8")
    match = BLOCK.search(page)
    if match is None:
        print("index.html has no evidence block", file=sys.stderr)
        return 1
    fresh = render_block(build_evidence())
    if match.group(2) == fresh:
        print("index.html evidence is current")
        return 0
    if args.check:
        print("index.html evidence is stale; run scripts/sync_site_evidence.py", file=sys.stderr)
        return 1
    PAGE.write_text(page[: match.start(2)] + fresh + page[match.end(2) :], encoding="utf-8")
    print("index.html evidence updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
