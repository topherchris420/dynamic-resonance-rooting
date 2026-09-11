# Dynamic Resonance Rooting (DRR) Framework

[![CI](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml/badge.svg)](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research framework for studying complex adaptive systems through resonance detection, causal rooting analysis, and stability diagnostics.

**Version:** 4.3.0  
**License:** MIT  
**Author:** Christopher Woodyard

---

## 📄 Flier

A one-page overview of DRR — what it detects, who it's for, and how to get started.

<p align="center">
  <a href="docs/assets/drr-flier.png">
    <img src="docs/assets/drr-flier.png" alt="Dynamic Resonance Rooting flier — resonance detection, causal rooting, stability diagnostics" width="600">
  </a>
</p>

Print-ready (US Letter). Learn more at [Vers3Dynamics.com](https://vers3dynamics.com).

---

## What is DRR?

The **Dynamic Resonance Rooting (DRR) Framework** is a computational pipeline that analyzes time-series and panel data to answer three core questions about complex adaptive systems:

1. **Resonance Detection** — What oscillatory modes exist in the data?
2. **Rooting Analysis** — What are the directional lead-lag relationships between variables?
3. **Stability Assessment** — How stable are the resonance structures over time?

DRR combines spectral analysis (FFT, Welch PSD, Morlet wavelet scalograms), causal rooting (transfer entropy, lagged correlation), and state-space diagnostics to provide evidence for hypothesis generation about system behavior.

---

## Key Features

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Resonance Detection** | Identifies oscillatory patterns via FFT, Welch power spectral density, or Morlet wavelet scalograms (time-localized, for nonstationary signals) |
| **Causal Rooting** | Maps directional lead-lag relationships using transfer entropy or lagged correlation, with candidate edges, significant edges, and surrogate-tested p-values |
| **Resonance Depth** | Composite scoring combining spectral concentration, temporal persistence, phase coherence, and amplitude stability |
| **State-Space Diagnostics** | Transition, measurement, and stability analysis with a Kalman filter |
| **State-Space Smoothing** | Retrospective states and structural shocks via Hamilton (RTS) and Koopman disturbance smoothers, plus Durbin–Koopman and Carter–Kohn posterior draws |
| **Nonlinear Filtering** | Tempered particle filter (Herbst–Schorfheide) for likelihood evaluation of nonlinear resonance systems, plus fast Chandrasekhar recursions for linear ones |
| **Closed-Loop Control** | State observation, sensitivity estimation, and bounded intervention targeting via `ResonanceNavigationEngine` |
| **Real-Time Streaming** | Pre-allocated circular buffer O(1) rolling DRR updates via `RealTimeDRR` for low-latency streaming data |
| **Validation Readiness** | Model risk cards, shadow review records, supervisory profiles, and event backtesting via `validation_readiness` |
| **Phase-Transition Detection** | Evidence for drift or abrupt changes in system behavior |

### Supported Data Sources

| Source | Description |
|--------|-------------|
| **Physics Systems** | Coupled oscillators, Lorenz, Rössler, Heston, FitzHugh-Nagumo benchmarks |
| **Sonoluminescence** | Multimodal acoustic driver, cavitation bubble dynamics, optical flash emission, and electrical transduction |
| **Sensing Systems** | Multi-channel radar micro-Doppler DSP analog time series |
| **Policy Data** | Tabular time-series from FRED, policy observables |
| **Supervisory Panels** | Banking data (FFIEC 002, FR Y-9C), institutional metrics |
| **Financial Markets** | Multi-asset macro panels (SPY, TLT, GLD, HYG, VIXY) via OpenBB / Qlib / DataFrame providers |
| **Custom Time-Series** | Any multivariate numerical array |

### Export Formats

- **JSON** — Structured analysis results
- **CSV** — Tableau-ready artifacts & walk-forward results
- **Markdown** — Human-readable reports
- **Plots** — Visualization of resonance, rooting graphs, state-space, strategy drawdowns

---

## Installation

### From PyPI

```bash
pip install drr-framework
```

### Optional Extras

```bash
# Data Infrastructure (OpenBB)
pip install "drr-framework[quant-data]"

# Portfolio & Risk Engine (Riskfolio-Lib)
pip install "drr-framework[quant-risk]"

# Machine Learning Research (Microsoft Qlib)
pip install "drr-framework[quant-ml]"

# Robustness & Validation Engine (VectorBT)
pip install "drr-framework[quant-validation]"

# Full Quantitative Research Suite
pip install "drr-framework[quant]"
```

### From Source

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
results = drr.analyze_system(
    data,
    multivariate=True,
    window_size=256,
    rooting_method="lagged_correlation",
    rooting_n_surrogates=25,
    rooting_random_state=42,
)

print(results["resonance_depths"])
print(results["rooting_analysis"]["candidate_edges"])
print(results["rooting_analysis"]["significant_edges"])
```

The rooting graph only includes significant edges. Use
`results["rooting_analysis"]["candidate_edges"]` when you want the exploratory
effect-size list without the surrogate test.

---

## DRR Quant Research Lab

DRR can be evaluated as a structural market-state representation using established open-source quantitative research infrastructure:

```text
Market Data (OpenBB / Qlib / DataFrame)
        ↓
   DRR Framework
        ↓
Structural Market State (MarketResonanceState)
        ↓
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Qlib  Riskfolio      VectorBT
ML    Portfolio      Validation
```

Its purpose is to test whether Dynamic Resonance Rooting produces useful, reproducible structural representations of financial markets and whether those representations contain incremental information beyond conventional financial features out of sample.

### Key Capabilities

1. **Microsoft Qlib ML Integration** (`drr_framework.finance.qlib`):
   - Combines conventional market factors with DRR state features.
   - Evaluates Matched Experiments: Control (Qlib model + conventional features) vs. Experiment (Qlib model + conventional + DRR features).
   - Computes Information Coefficient (IC), Rank IC, ICIR, Rank ICIR, and conducts feature ablation studies.

2. **Riskfolio-Lib Integration** (`drr_framework.finance.portfolio`):
   - DRR-conditioned dynamic portfolio regime switching (Mean-Variance vs. CVaR 95% tail risk).
   - SciPy SLSQP optimization fallback.
   - Strictly causal no-lookahead expanding/rolling percentile regime policies.

3. **VectorBT Validation Adapter** (`drr_framework.finance.validation`):
   - Independent backtest reconstruction and multi-parameter robustness sweeps (lookbacks, quantiles, horizons, time-delays).

4. **Strict Anti-Leakage & Statistical Rigor**:
   - Automated lookahead invariant verification (`assert_no_lookahead_leakage`).
   - Newey-West HAC standard errors and Benjamini-Hochberg FDR corrections.
   - Date-shuffled negative controls.

> [!NOTE]
> **Research & Scientific Disclaimer**: DRR structural resonance is an empirical complex-systems state metric, not a guaranteed profitable trading strategy or direct alpha signal. High structural resonance measures strong persistent structured dynamics across assets, which must be evaluated empirically. A null result is scientifically acceptable.

---

## Use Cases

### Sonoluminescence Lab
**File:** `examples/sonoluminescence_lab.py`

Multimodal coupled resonant system benchmark spanning acoustic, cavitation, optical, and electrical domains:
- Ultrasonic acoustic driver ($f_a \approx 25\text{ kHz}$)
- Acoustic waveguide horn resonator & impedance concentration
- Rayleigh-Plesset nonlinear bubble oscillator & Blake cavitation threshold
- Ultrafast UV-blue sonoluminescence flash emission ($\lambda_{\text{EM}} \approx 350\text{ nm}$)
- Downstream optical/electrical transduction & energy bookkeeping
- Resonant Transduction Efficiency Index (RTEI) metric

### Sensing Systems Lab
**File:** `examples/sensing_systems_resonance_depth.py`

Cross-domain micro-Doppler radar DSP analog time-series benchmark evaluating:
- Multi-channel radar returns with regime transition (standing to walking)
- Coherent change detection proxy and high-pass state innovation
- Resonance depth stability of low-frequency target mode against clutter
- State-space Kalman filter tracking and change detection

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

### State-Space Lab
**File:** `examples/state_space_lab.py`

End-to-end state-space workflow — fit, Kalman filter, Koopman smooth (with
structural shocks), Durbin–Koopman posterior bands, Chandrasekhar likelihood,
and a tempered particle filter on a nonlinear observation.

### Quant Macro Lab
**File:** `examples/quant_macro_lab.py`

Run full DRR Quant Research Lab walk-forward portfolio evaluation:

```bash
python examples/quant_macro_lab.py --offline --rebalance monthly
```

Runs matched strategy evaluations (Equal Weight, Mean-Variance, Static CVaR, DRR Conditioned, SPY Buy-Hold) and exports structured research reports and summary plots.

### Quickstart Export Example
**File:** `examples/quickstart_resonance_export.py`

Compact workflow demonstrating sample dataset loading, resonance detection, depth calculation, rooting analysis, and export of JSON, CSV, and trace figures.

### Primary Qlib Matched Experiment Example
**File:** `examples/qlib_drr_experiment.py`

```bash
python examples/qlib_drr_experiment.py --offline --model linear
```

### DRR Riskfolio Portfolio Example
**File:** `examples/drr_riskfolio_portfolio.py`

```bash
python examples/drr_riskfolio_portfolio.py --offline
```

### VectorBT Robustness Sweeps Example
**File:** `examples/drr_vectorbt_robustness.py`

```bash
python examples/drr_vectorbt_robustness.py --offline
```

---

## Project Structure

```
dynamic-resonance-rooting/
├── src/drr_framework/          # Core package
│   ├── analysis.py             # DynamicResonanceRooting orchestration class
│   ├── modules.py              # ResonanceDetector, RootingAnalyzer, DepthCalculator
│   ├── _spectral.py            # Spectral analysis utilities
│   ├── benchmarks.py           # Physics & radar benchmark generators
│   ├── sonoluminescence.py     # Sonoluminescence multimodal benchmark
│   ├── control_engine.py       # Closed-loop ResonanceNavigationEngine
│   ├── realtime.py             # Circular buffer RealTimeDRR streaming
│   ├── state_space.py          # State-space modeling & Kalman filter
│   ├── smoothers.py            # Hamilton, Koopman & simulation smoothers
│   ├── particle_filter.py      # Tempered particle filter
│   ├── resonance_geometry.py   # Differential geometry & manifold dynamics
│   ├── cross_resonance.py      # Cross-resonance tensor & collective modes
│   ├── topology_dynamics.py    # Rooting topology drift & migration
│   ├── structural_surprise.py  # Information-theoretic structural surprise
│   ├── datasets.py              # Data adapters for policy & supervisory data
│   ├── supervision.py          # Regulatory profiles & supervisory risk domains
│   ├── validation_readiness.py # Model risk cards & validation packets
│   ├── finance/                # DRR Quant Research Lab
│   │   ├── config.py           # Experiment configuration
│   │   ├── types.py            # MarketResonanceState & result containers
│   │   ├── data/               # Data adapters (OpenBB, Qlib, DataFrame)
│   │   ├── features/           # DRR & conventional feature engineering
│   │   ├── qlib/               # Qlib dataset & matched experiment engine
│   │   ├── portfolio/          # Riskfolio adapter & regime policies
│   │   ├── validation/         # VectorBT adapter, walk-forward & anti-leakage
│   │   └── reporting/          # Metrics, plots & Markdown research reports
│   └── visualizations.py       # Plotting utilities
├── examples/                    # Usage examples & lab scripts
├── tests/                       # Unit and integration test suite
├── docs/                        # Framework documentation
│   ├── quant-research.md       # Quant Research Lab architecture guide
│   ├── validation-readiness-guide.md # Model risk & supervisory guide
│   ├── resonance-geometry.md   # Differential geometry guide
│   └── user-guide.md            # Getting started user guide
└── pyproject.toml              # Package configuration
```

---

## Output Interpretation

| Key | Description |
|-----|-------------|
| `resonances` | Dominant frequencies and spectral evidence |
| `resonance_depths` | Scalar persistence/stability scores by dimension |
| `resonance_depth_details` | Component-level scores and confidence intervals |
| `rooting_analysis` | Directed lagged relationships, `score_matrix`/`transfer_entropy`, raw/adjusted p-values, candidate/significant edges |
| `state_space_analysis` | Transition, measurement, likelihood, stability, impulse-response |

**Important:** DRR outputs are research diagnostics. Domain conclusions require separate validation, calibration, and review.

---

## Documentation

- [Quant Research Lab Architecture](docs/quant-research.md) — Quantitative finance integration guide
- [Validation Readiness Guide](docs/validation-readiness-guide.md) — Model risk management and supervisory compliance
- [Resonance Geometry](docs/resonance-geometry.md) — Differential geometry and state space manifold dynamics
- [Method Crosswalk](docs/method-crosswalk.md) — Cross-walk comparison with traditional time-series methods
- [Audience Guide](docs/audience-guide.md) — Guidance for researchers, quants, and risk managers
- [User Guide](docs/user-guide.md) — Getting started
- [Architecture](docs/architecture.md) — System design
- [API Reference](docs/api.md) — Function documentation
- [Developer Guide](docs/developer-guide.md) — Contributing
- [FAQ](docs/faq.md) — Common questions
- [Reproducibility](docs/reproducibility.md) — Ensuring reproducible results

---

## Citation

If you use DRR in your research, please cite:

```bibtex
@software{drr-framework,
  author = {Christopher Woodyard},
  title = {Dynamic Resonance Rooting (DRR) Framework},
  url = {https://github.com/topherchris420/dynamic-resonance-rooting},
  version = {4.3.0},
  year = {2026}
}
```

---

## License

MIT License — see LICENSE file for details.
