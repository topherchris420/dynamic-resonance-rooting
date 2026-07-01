# User Guide

## Installation

```bash
git clone https://github.com/topherchris420/dynamic-resonance-rooting.git
cd dynamic-resonance-rooting
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Quick Start

```python
from drr_framework import DynamicResonanceRooting, generate_coupled_oscillator

sampling_rate = 200.0
_, data = generate_coupled_oscillator(
    sampling_rate=sampling_rate,
    target_frequency_hz=12.5,
    random_state=42,
)

drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)
results = drr.analyze_system(data, multivariate=True, window_size=256)

print(results["resonance_depths"])
print(results["rooting_analysis"]["significant_edges"])
```

## Choosing A Workflow

- Use `examples/physics_lab.py` for coupled oscillator and physical-system
  diagnostics.
- Use `examples/policy_lab.py` for tabular policy observables.
- Use `examples/supervisory_policy_lab.py` for institution panels, supervisory
  caveats, validation-readiness packets, and Tableau exports.
- Use `examples/quickstart_resonance_export.py` for a compact load-detect-export
  demonstration.

## Interpreting Outputs

- `resonances`: dominant frequencies and spectral evidence.
- `resonance_depths`: scalar persistence/stability scores by dimension.
- `resonance_depth_details`: component-level scores and confidence intervals.
- `rooting_analysis`: directed lagged relationships and edge metadata.
- `state_space_analysis`: transition, measurement, likelihood, stability, and
  impulse-response diagnostics.

DRR outputs are research diagnostics. Domain conclusions require separate
validation, calibration, and review.
