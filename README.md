# Dynamic Resonance Rooting (DRR) Framework

[![PyPI Version](https://img.shields.io/pypi/v/drr-framework?style=flat)](https://pypi.org/project/drr-framework/)
[![Python Versions](https://img.shields.io/pypi/pyversions/drr-framework?style=flat)](https://pypi.org/project/drr-framework/)
[![License](https://img.shields.io/pypi/l/drr-framework?style=flat)](https://pypi.org/project/drr-framework/)
[![CI](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml/badge.svg)](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml)

A research framework for studying complex adaptive systems through resonance detection, causal rooting analysis, and stability diagnostics.

**Version:** 4.2.0  
**License:** MIT  
**Author:** Christopher Woodyard

---

## What is DRR?

The **Dynamic Resonance Rooting (DRR) Framework** is a computational pipeline that analyzes time-series and panel data to answer three core questions about complex adaptive systems:

1. **Resonance Detection** — What oscillatory modes exist in the data?
2. **Rooting Analysis** — What are the directional lead-lag relationships between variables?
3. **Stability Assessment** — How stable are the resonance structures over time?

DRR combines spectral analysis (FFT, Welch PSD), causal rooting (transfer entropy, lagged correlation), and state-space diagnostics to provide evidence for hypothesis generation about system behavior.

---

## Key Features

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Resonance Detection** | Identifies oscillatory patterns via FFT and Welch power spectral density |
| **Causal Rooting** | Maps directional lead-lag relationships using transfer entropy or lagged correlation |
| **Resonance Depth** | Composite scoring combining spectral concentration, temporal persistence, phase coherence, and amplitude stability |
| **State-Space Diagnostics** | Transition, measurement, and stability analysis |
| **Phase-Transition Detection** | Evidence for drift or abrupt changes in system behavior |

### Supported Data Sources

| Source | Description |
|--------|-------------|
| **Physics Systems** | Coupled oscillators, Lorenz, Rössler, Heston, FitzHugh-Nagumo benchmarks |
| **Policy Data** | Tabular time-series from FRED, policy observables |
| **Supervisory Panels** | Banking data (FFIEC 002, FR Y-9C), institutional metrics |
| **Custom Time-Series** | Any multivariate numerical array |

### Export Formats

- **JSON** — Structured analysis results
- **CSV** — Tableau-ready artifacts
- **Markdown** — Human-readable reports
- **Plots** — Visualization of resonance, rooting graphs, state-space

---

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

---

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

---

## Use Cases

### Physics Lab
**File:** `examples/physics_lab.py`

Analyze coupled oscillator systems and benchmark generators:
- Lorenz attractor
- Rössler system
- Heston volatility model
- FitzHugh-Nagumo neurons

### Policy Lab
**File:** `examples/policy_lab.py`

Analyze tabular policy observables:
- Time-series from FRED
- Economic indicators
- Policy shock response

### Supervisory Policy Lab
**File:** `examples/supervisory_policy_lab.py`

Federal Reserve banking supervision:
- FFIEC 002 / FR Y-9C data integration
- Institutional risk metrics
- Validation-readiness packets
- **Tableau exports for executive dashboards**

### Quick Start
**File:** `examples/quickstart_resonance_export.py`

Compact load-detect-export demonstration

---

## Macro Stability & Banking Skin Cockpit

The framework includes a specialized supervisory application for Federal Reserve oversight:

### Purpose
Real-time early-warning system for detecting non-linear liquidity panics ("Dash for Cash" loop) across Large Foreign Banking Organizations (LFBOs).

### Key Capabilities
1. **Resonance Detection** — Identifies hidden cyclical funding stress via Welch/FFT
2. **Rooting Analysis** — Maps lead-lag structures between Treasury shocks and bank liquidity drains
3. **Composite Scoring** — Resonance Depth combining spectral concentration, temporal persistence, phase coherence, amplitude stability
4. **Stress Simulation** — 200 bps parallel rate shift (Full AOCI vs. Opt-Out)

### Files
- `layer1_regulatory_backend.py` — Python/SQL regulatory backend
- `layer2_tableau_blueprint.twb` — Tableau dashboard blueprint
- `tableau_calculated_fields.py` — Tableau calculated fields
- `tableau_output/` — Generated CSV exports

### Compliance
Implements **SR 11-7 Model Risk Management** guidelines from the Federal Reserve.

---

## Project Structure

```
dynamic-resonance-rooting/
├── src/drr_framework/          # Core package
│   ├── analysis.py             # DynamicResonanceRooting class
│   ├── _spectral.py            # Spectral analysis
│   ├── benchmarks.py           # Benchmark generators
│   ├── datasets.py              # Data adapters
│   ├── reporting.py            # Export utilities
│   ├── state_space.py          # State-space diagnostics
│   ├── supervision.py           # Supervisory components
│   └── validation_readiness.py # Validation packets
├── examples/                    # Usage examples
│   ├── physics_lab.py
│   ├── policy_lab.py
│   ├── supervisory_policy_lab.py
│   └── quickstart_resonance_export.py
├── tests/                       # Test suite
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── user-guide.md
│   ├── developer-guide.md
│   ├── api.md
│   └── faq.md
├── data/                        # Datasets
├── results/                     # Output artifacts
└── pyproject.toml              # Package config
```

---

## Output Interpretation

| Key | Description |
|-----|-------------|
| `resonances` | Dominant frequencies and spectral evidence |
| `resonance_depths` | Scalar persistence/stability scores by dimension |
| `resonance_depth_details` | Component-level scores and confidence intervals |
| `rooting_analysis` | Directed lagged relationships and edge metadata |
| `state_space_analysis` | Transition, measurement, likelihood, stability, impulse-response |

**Important:** DRR outputs are research diagnostics. Domain conclusions require separate validation, calibration, and review.

---

## Documentation

- [User Guide](docs/user-guide.md) — Getting started
- [Architecture](docs/architecture.md) — System design
- [API Reference](docs/api.md) — Function documentation
- [Developer Guide](docs/developer-guide.md) — Contributing
- [FAQ](docs/faq.md) — Common questions
- [Reproducibility](docs/reproducibility.md) — Ensuring reproducible results
- [Validation Readiness](docs/validation-readiness-guide.md) — Model validation

---

## Citation

If you use DRR in your research, please cite:

```bibtex
@software{drr-framework,
  author = {Christopher Woodyard},
  title = {Dynamic Resonance Rooting (DRR) Framework},
  url = {https://github.com/topherchris420/dynamic-resonance-rooting},
  version = {4.2.0},
  year = {2026}
}
```

---

## License

MIT License — see LICENSE file for details.

---

## References

- [FFIEC 002 Reports](https://www.ffiec.gov/NPW)
- [Federal Reserve Supervision](https://www.federalreserve.gov/supervisionreg.htm)
- [SR 11-7: Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
