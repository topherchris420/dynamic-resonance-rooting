"""Policy-facing DRR workflow.

Run from the repository root:

    python examples/policy_lab.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drr_framework import DynamicResonanceRooting, PolicyResonanceDataset, write_analysis_report


def synthetic_policy_frame(periods: int = 120, random_state: int = 2026) -> pd.DataFrame:
    """Create a small macro-style dataset for demonstration.

    The series are synthetic and intentionally modest: they show how to use DRR
    with policy observables without implying a real policy forecast.
    """

    rng = np.random.default_rng(random_state)
    dates = pd.date_range("1996-01-01", periods=periods, freq="QS")
    cycle = np.sin(np.linspace(0, 8 * np.pi, periods))
    demand_gap = 0.65 * cycle + rng.normal(scale=0.15, size=periods)
    inflation = 2.0 + 0.35 * np.roll(demand_gap, 2) + rng.normal(scale=0.08, size=periods)
    policy_rate = 2.5 + 0.55 * np.roll(inflation - 2.0, 1) + 0.25 * demand_gap
    policy_rate += rng.normal(scale=0.06, size=periods)

    return pd.DataFrame(
        {
            "date": dates,
            "inflation": inflation,
            "demand_gap": demand_gap,
            "policy_rate": policy_rate,
        }
    )


def main() -> None:
    dataset = PolicyResonanceDataset.from_frame(
        synthetic_policy_frame(),
        date_column="date",
        value_columns=["inflation", "demand_gap", "policy_rate"],
        sampling_rate=4.0,
        transform="diff",
        interpolate=True,
        standardize=True,
        source="synthetic quarterly policy observables",
    )

    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=dataset.sampling_rate)
    results = drr.analyze_system(
        dataset.to_drr_input(),
        multivariate=True,
        window_size=64,
        state_space=True,
        state_space_horizon=12,
    )

    paths = write_analysis_report(
        results,
        Path("results") / "policy_lab",
        stem="synthetic_policy_observables",
        audience="policy",
        title="DRR Policy Lab: Synthetic Observables",
        metadata={
            **dataset.metadata,
            "variables": dataset.variable_names,
            "note": "Synthetic example for method demonstration only.",
        },
    )
    print(f"Wrote policy report: {paths['markdown']}")
    print(f"Wrote policy data: {paths['json']}")


if __name__ == "__main__":
    main()
