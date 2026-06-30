# Dynamic Resonance Rooting (DRR) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml/badge.svg)](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml)

Reference implementation for the research paper **"Dynamic Resonance Rooting: A Computational Framework for Complex Adaptive Systems"** by Christopher Woodyard.

DRR analyzes complex adaptive systems by detecting dominant oscillatory modes, estimating directed rooting structure, and computing a normalized Resonance Depth score for stability and persistence.

## What Is Implemented

- **Resonance detection** with FFT and Welch power spectral density methods.
- **Composite Resonance Depth** with component-level output and confidence interval.
- **Rooting analysis** with lag-aware directed scoring, surrogate p-values, and significant edge metadata.
- **Synthetic validation harness** that regenerates benchmark JSON/CSV artifacts from a clean checkout.
- **Benchmark systems** for Lorenz, Rossler, Heston, and FitzHugh-Nagumo style workflows.
- **Real-time sliding-window analyzer** for streaming-style experiments.
- **DSGE-inspired state-space diagnostics** with transition/measurement matrices, Kalman filtering, likelihood summaries, stability checks, and impulse responses.
- **Audience-specific workflow adapters** for physics labs, policy analysis, supervisory panels, SQL ingestion, caveated reports, and Tableau-ready CSV exports.
- **Fed supervisory alignment pack** with institution presets, risk-domain tags, source-data lineage metadata, report checklists, and official-reference links.

Wavelet detection and the acoustic metamaterial generative design suite are currently experimental/planned surfaces. The generative design suite documents architecture and configuration, but it is not yet a full audio-to-mesh fabrication pipeline.

## Install

```bash
git clone https://github.com/topherchris420/dynamic-resonance-rooting.git
cd dynamic-resonance-rooting
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

## Reproduce The Research Smoke Benchmark

Run the deterministic validation harness:

```bash
python -m drr_framework.experiments --output-dir results/reproduction
```

This writes:

- `results/reproduction/drr_reproduction_summary.json`
- `results/reproduction/drr_reproduction_metrics.csv`

Current benchmark truth:

- Known dominant frequency: `12.5 Hz`
- Known directed edge: `dim_0 -> dim_1`
- Expected lag: `2` samples

Verified local output after this upgrade:

```text
detected_frequency_hz: 12.5
frequency_error_hz: 0.0
expected_edge_detected: true
resonance_depth: 0.9042969836929234
rooting_method: lagged_correlation
```

## Run The Example Config

The checked-in `config.yml` is self-contained and generates Lorenz data:

```bash
python scripts/run_analysis.py --config config.yml
```

## Resonance Depth Metric

The current implementation computes a normalized composite score:

```text
RD = 0.35 * spectral_concentration
   + 0.25 * temporal_persistence
   + 0.25 * phase_coherence
   + 0.15 * amplitude_stability
```

The calculator returns both the scalar score and the component details:

```python
from drr_framework.modules import DepthCalculator

result = DepthCalculator().calculate(
    data,
    window_size=256,
    sampling_rate=200.0,
    resonance_frequencies=[12.5],
)

print(result["resonance_depth"])
print(result["components"])
print(result["confidence_interval"])
```

## Quick Start

```python
from drr_framework import DynamicResonanceRooting, BenchmarkSystems

# Generate benchmark data
t, data = BenchmarkSystems.generate_lorenz_data(duration=5, dt=0.01)

# Analyze the multivariate system
drr = DynamicResonanceRooting(embedding_dim=3, tau=1, sampling_rate=100.0)
results = drr.analyze_system(data, multivariate=True, window_size=256)

print(results["resonance_depths"])
print(results["resonance_depth_details"])
print(results.get("rooting_analysis", {}))
```

## Core APIs

### ResonanceDetector

```python
from drr_framework.modules import ResonanceDetector

result = ResonanceDetector().detect(
    signal,
    method="welch",
    sampling_rate=200.0,
    peak_height_ratio=0.2,
)
```

Returns dominant frequencies, peak magnitudes, confidence estimates, noise floor, and the spectrum used by the detector.

### RootingAnalyzer

```python
from drr_framework.modules import RootingAnalyzer

result = RootingAnalyzer().analyze(
    multivariate_data,
    max_lag=4,
    n_surrogates=25,
    random_state=42,
)
```

Returns a directed score matrix, effective lag matrix, p-values, edge threshold, and `significant_edges` records. Transfer entropy can be requested with `method="transfer_entropy"` when `pyinform` is installed; otherwise the analyzer uses the deterministic lagged-correlation fallback and reports that method explicitly.

### Reproduction Harness

```python
from drr_framework import run_reproduction_experiment

summary = run_reproduction_experiment(output_dir="results/reproduction")
print(summary)
```

### DSGE-Inspired State-Space Diagnostics

DRR now includes a Python-native state-space layer inspired by the modeling
discipline used in FRBNY's `DSGE.jl`: explicit transition equations,
measurement equations, Kalman filtering, and impulse-response diagnostics. It
does not require Julia and does not copy DSGE.jl internals.

```python
from drr_framework import analyze_resonance_state_space

analysis = analyze_resonance_state_space(data, impulse_horizon=12)

print(analysis["system"]["TTT"])          # transition matrix
print(analysis["diagnostics"])            # stability, likelihood, innovations
print(analysis["impulse_responses"][0])   # shock response path
```

`DynamicResonanceRooting.analyze_system(...)` attaches this bundle under
`results["state_space_analysis"]` by default when enough samples are available.

## Audience Workflows

DRR now has three explicit front doors:

- **Physics Lab**: mode detection, coupling diagnostics, state-space stability,
  and perturbation response. Run `python examples/physics_lab.py`.
- **Policy Lab**: tabular observable loading, DSGE-style diagnostics, lagged
  influence review, and caveated research reports. Run `python examples/policy_lab.py`.
- **Supervisory Analytics Lab**: institution panel loading, SQL-friendly adapters,
  peer-group diagnostics, supervisory caveats, and Tableau-ready CSV exports. Run
  `python examples/supervisory_policy_lab.py`.

Use `build_supervisory_alignment_metadata(...)` to attach institution segment, risk domain,
MDRM/FFIEC/NIC lineage, review-owner, and diagnostic-boundary context to supervision reports.

See `docs/audience-guide.md` and `docs/method-crosswalk.md` for vocabulary,
review expectations, and domain-specific caveats.

## Testing

```bash
pytest
```

Current local verification after this upgrade:

```text
36 passed
```

## Citation

See `CITATION.cff`. If you use this code in research, cite both the associated DRR paper and this software artifact.

```text
Woodyard, C. (2026). Dynamic Resonance Rooting Framework. MIT licensed software.
https://github.com/topherchris420/dynamic-resonance-rooting
```

## Notes For Reviewers

This repository is a research implementation. It is designed for reproducible computational experiments and methodological review. Domain-specific clinical, financial, defense, or infrastructure use requires independent validation, calibration, and risk review.
