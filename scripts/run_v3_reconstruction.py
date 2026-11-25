"""Run the registered v3 reconstruction and write its artifact.

Usage::

    python scripts/run_v3_reconstruction.py --workers 4 \
        --output results/expected/v3_reconstruction/v3_reconstruction.json

The protocol in ``experiments/v3_reconstruction/protocol.json`` fixes every
design choice. ``--smoke`` shrinks trial counts for a quick wiring check; its
output is not a study result. ``--verify-against FILE`` exits non-zero unless
the new artifact matches FILE in every count, label and trial index, with
floats equal to a relative tolerance of 1e-6 (environment and runtime ignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from drr_framework.v3_reconstruction import (  # noqa: E402
    assemble,
    build_tasks,
    replay_differences,
    run_task,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "v3_reconstruction" / "protocol.json"
CALIBRATION = ROOT / "results" / "expected" / "rooting_calibration_study.json"


def _shrink(protocol: dict) -> dict:
    p = copy.deepcopy(protocol)
    rs = p["rooting_study"]
    rs["size"]["n_trials"] = 4
    rs["memory"]["n_trials"] = 4
    rs["power_curve"]["n_trials"] = 4
    rs["long_run_check"]["n_samples"] = 4000
    p["resonance_study"]["nulls"]["n_trials"] = 2
    p["resonance_study"]["alternatives"]["n_trials"] = 2
    p["resonance_study"]["n_null_draws"] = 9
    p["qbo_checks"]["rooting"]["n_surrogates"] = 19
    p["qbo_checks"]["spectral"]["n_surrogates"] = 19
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-against", type=Path, default=None)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_protocol = _shrink(protocol) if args.smoke else protocol
    tasks = build_tasks(run_protocol)
    # Longest tasks first so the pool stays busy.
    order = {"resonance": 0, "memory": 1, "power_curve": 2, "size": 3, "qbo": 4, "long_run": 5}
    tasks.sort(key=lambda t: order[t[0]])
    start = time.time()
    outputs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for section, key, value in pool.map(run_task, tasks):
            outputs.append((section, key, value))
            print(f"[{time.time() - start:7.1f}s] {section}/{key}", flush=True)
    committed = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    artifact = assemble(protocol, outputs, None if args.smoke else committed)
    artifact["smoke"] = bool(args.smoke)
    artifact["runtime_seconds"] = round(time.time() - start, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    held = sum(bool(f["holds"]) for f in artifact["findings"])
    print(f"registered findings that hold: {held} of {len(artifact['findings'])}")
    if args.verify_against is None:
        return 0
    expected = json.loads(args.verify_against.read_text(encoding="utf-8"))
    stored = json.loads(args.output.read_text(encoding="utf-8"))
    problems = replay_differences(stored, expected)
    for problem in problems[:50]:
        print(f"  MISMATCH  {problem}")
    if len(problems) > 50:
        print(f"  ... and {len(problems) - 50} more")
    print(f"{'PASS' if not problems else 'FAIL'}  replay matches {args.verify_against}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
