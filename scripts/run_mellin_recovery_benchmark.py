#!/usr/bin/env python3
"""Run the DRR-Mellin hidden-mode recovery benchmark and write its artifacts.

    python scripts/run_mellin_recovery_benchmark.py --workers 4

writes ``results/expected/mellin_recovery/`` with ``mellin_recovery_benchmark.json``
(per-trial codes), ``report.md`` (every table recomputed from the codes) and
``recovery.png``. ``--verify-against DIR`` replays the run and fails unless
every per-trial code, calibrated threshold and accuracy summary matches the
artifact in ``DIR``. One numerical thread per process is enforced so that
parallel runs neither oversubscribe the CPU nor change the rows; rows are
identical for any number of workers. The exit code reports integrity and
replay checks only. Preregistered hypotheses are printed with their outcomes,
whatever they are.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drr_framework.mellin_recovery_benchmark import (  # noqa: E402
    render_figure,
    render_report,
    run_benchmark,
)

ARTIFACT = "mellin_recovery_benchmark.json"
DEFAULT_OUTPUT = ROOT / "results" / "expected" / "mellin_recovery"


def _close(a, b, rel=1e-9):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


def compare(replayed, expected):
    """Differences between the evidence of two artifacts (runtime and environment ignored)."""
    problems = []
    for key in ("schema_version", "config", "preregistration_sha256"):
        if replayed[key] != expected[key]:
            problems.append(f"{key} differs")
    for split in ("development", "holdout"):
        mine, theirs = replayed.get(split), expected.get(split)
        if mine is None or theirs is None:
            if mine is not theirs:
                problems.append(f"{split}: present in only one artifact")
            continue
        for a, b in zip(mine["null"], theirs["null"]):
            if a["fresh_codes"] != b["fresh_codes"]:
                problems.append(f"{split} null {a['group']}: fresh codes differ")
            for m, t in a["thresholds"].items():
                if not _close(t, b["thresholds"][m]):
                    problems.append(f"{split} null {a['group']}: {m} threshold differs")
        for a, b in zip(mine["signal"], theirs["signal"]):
            for m, codes in a["codes"].items():
                if codes != b["codes"][m]:
                    diff = sum(x != y for x, y in zip(codes, b["codes"][m]))
                    problems.append(f"{split} {a['id']} {m}: {diff} trial codes differ")
            for key in ("omega_rmse_recovered", "beta_rmse_recovered"):
                for m, v in a[key].items():
                    if not _close(v, b[key].get(m)):
                        problems.append(f"{split} {a['id']} {m}: {key} differs")
        if len(mine["signal"]) != len(theirs["signal"]):
            problems.append(f"{split}: condition counts differ")
    if [h["outcome"] for h in replayed["hypotheses"]] != [
        h["outcome"] for h in expected["hypotheses"]
    ]:
        problems.append("hypothesis outcomes differ")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("development", "holdout"),
        default=["development", "holdout"],
    )
    parser.add_argument("--verify-against", type=Path, default=None)
    parser.add_argument("--no-figure", action="store_true", help="Skip the PNG figure.")
    args = parser.parse_args(argv)

    artifact = run_benchmark(workers=args.workers, splits=args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=1, sort_keys=True) + "\n"
    # Render from the stored form so the report is reproducible from the JSON alone.
    artifact = json.loads(payload)
    (args.output_dir / ARTIFACT).write_text(payload, encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_report(artifact), encoding="utf-8")
    if artifact["holdout"] and not args.no_figure:
        render_figure(artifact, args.output_dir / "recovery.png")
    print(f"Wrote {args.output_dir} in {artifact['runtime']['total_seconds']:.0f}s")
    rows_digest = hashlib.sha256(
        json.dumps({s: artifact[s] for s in ("development", "holdout")}, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    print(f"  evidence rows SHA-256: {rows_digest}")
    print(
        f"  preregistration matches the frozen record: {artifact['matches_frozen_preregistration']}"
    )
    for h in artifact["hypotheses"]:
        print(f"  {h['outcome']:>19}  {h['id']}: {h['statement']}")
    ok = True
    for name, holds in artifact["checks"].items():
        print(f"  {'PASS' if holds else 'FAIL'}  {name}")
        ok &= bool(holds)
    if args.verify_against is not None:
        expected = json.loads((args.verify_against / ARTIFACT).read_text(encoding="utf-8"))
        problems = compare(artifact, expected)
        for problem in problems[:50]:
            print(f"  MISMATCH  {problem}")
        print(f"  {'PASS' if not problems else 'FAIL'}  replay matches {args.verify_against}")
        ok &= not problems
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
