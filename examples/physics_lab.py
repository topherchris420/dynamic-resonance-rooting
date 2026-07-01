"""Physics-facing DRR workflow.

Run from the repository root:

    python examples/physics_lab.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drr_framework import (
    DynamicResonanceRooting,
    generate_coupled_oscillator,
    write_analysis_report,
)


def main() -> None:
    sampling_rate = 200.0
    _, data = generate_coupled_oscillator(
        sampling_rate=sampling_rate,
        duration=5.0,
        target_frequency_hz=12.5,
        lag=2,
        noise_scale=0.04,
        random_state=2026,
    )

    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)
    results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=256,
        state_space=True,
        state_space_horizon=16,
    )

    paths = write_analysis_report(
        results,
        Path("results") / "physics_lab",
        stem="coupled_oscillator",
        audience="physics",
        title="DRR Physics Lab: Coupled Oscillator",
        metadata={
            "system": "synthetic coupled oscillator",
            "sampling_rate_hz": sampling_rate,
            "known_driver": "dim_0",
            "known_response": "dim_1",
        },
    )
    print(f"Wrote physics report: {paths['markdown']}")
    print(f"Wrote physics data: {paths['json']}")


if __name__ == "__main__":
    main()
