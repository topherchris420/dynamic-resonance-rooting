#!/usr/bin/env python3
"""Verify independent calibration replays against the sealed breakdown study."""

import argparse
from pathlib import Path

import pandas as pd

from drr_framework.finance.breakdown.study import verify_lock
from drr_framework.finance.holdout.study import verify_replay

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/lead_lag_breakdown"


def verify_calibration_replay(actual: Path, expected: Path) -> None:
    actual_lock = verify_lock(ROOT, STUDY, actual)
    expected_lock = verify_lock(ROOT, STUDY, expected)
    actual_artifacts = actual_lock.pop("calibration_artifacts")
    expected_artifacts = expected_lock.pop("calibration_artifacts")
    verify_replay(sorted(actual_artifacts), sorted(expected_artifacts), "calibration.artifacts")
    verify_replay(actual_lock, expected_lock, "calibration")
    for name in sorted(expected_artifacts):
        replay = pd.read_csv(actual / name, keep_default_na=False)
        published = pd.read_csv(expected / name, keep_default_na=False)
        pd.testing.assert_frame_equal(
            replay, published, check_exact=False, rtol=1e-7, atol=1e-10, obj=name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("published", type=Path)
    args = parser.parse_args()
    verify_calibration_replay(args.replay, args.published)
    print("Numerical calibration replay matches the sealed evidence.")
