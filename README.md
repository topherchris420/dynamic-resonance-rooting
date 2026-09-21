# Dynamic Resonance Rooting (DRR) Framework

[![CI](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml/badge.svg)](https://github.com/topherchris420/dynamic-resonance-rooting/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A modular, high-performance research framework and computational suite for complex adaptive systems — combining spectral resonance detection, directional transfer entropy rooting, state-space filtering, differential geometry, closed-loop control, and quantitative research adapters.

**Version:** 4.3.0  
**License:** MIT  
**Author:** Christopher Woodyard

---

## 📄 Framework Overview Flier

A one-page executive summary of the DRR framework — what it detects, who it's for, and how to get started.

<p align="center">
  <a href="docs/assets/drr-flier.png">
    <img src="docs/assets/drr-flier.png" alt="Dynamic Resonance Rooting flier — resonance detection, directional rooting, stability diagnostics" width="600">
  </a>
</p>

Print-ready (US Letter). Learn more at [Vers3Dynamics.com](https://vers3dynamics.com).

---

## 📌 What is Dynamic Resonance Rooting?

The **Dynamic Resonance Rooting (DRR) Framework** provides a unified computational pipeline to quantify, model, and navigate complex adaptive systems across physics, sensing, policy, banking supervision, and financial markets. It addresses three foundational questions:

1. **Resonance Detection** — What oscillatory modes and phase-coherent structures exist in nonstationary multivariate data?
2. **Directional Rooting** — What are the directed lead-lag relationships and information flows governing system dynamics?
3. **Stability & Navigation** — How stable are these resonance structures over time, and how can closed-loop control steer system states?

DRR combines multi-resolution spectral decomposition (FFT, Welch PSD, Morlet wavelets), non-parametric directional rooting (transfer entropy, surrogate null-hypothesis testing), state-space estimation (Kalman filtering, Hamilton/Koopman smoothers, tempered particle filters), and information-theoretic diagnostics to yield reproducible empirical evidence.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT DATA SOURCES                                    │
│    Physical Systems  •  Sensing DSP  •  Policy Series  •  Financial Markets  •  Panels   │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               DRR CORE ANALYTICS ENGINE                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │   Resonance Detector    │  │    Rooting Analyzer     │  │    Depth Calculator     │  │
│  │ FFT • Welch • Wavelets  │  │ Transfer Entropy • Corr │  │ Coherence • Persistence │  │
│  └────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘  │
└───────────────┼────────────────────────────┼────────────────────────────┼───────────────┘
                │                            │                            │
                ▼                            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                             ADVANCED DYNAMICAL ENGINE                                   │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  State-Space & Smooth   │  │   Resonance Geometry    │  │ Closed-Loop Control     │  │
│  │ Kalman • Hamilton • DK  │  │ Manifolds • Curvature   │  │ Navigation • Guidance   │  │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘  │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               APPLICATION ADAPTER LAYERS                                │
│  ┌─────────────────────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │        Quant Research Lab               │  │  LFBO Supervisory Monitoring         │  │
│  │ Qlib ML • Riskfolio • VectorBT Backtest │  │ As-Of Store • Disagreement • Ledger  │  │
│  └─────────────────────────────────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Foundations

| Concept | Formulation | Description |
|---------|-------------|-------------|
| **Resonance Depth** | $D_r = S_{\text{power}} \cdot T_{\text{persist}} \cdot C_{\text{phase}} \cdot A_{\text{stability}}$ | Composite score measuring spectral energy concentration, temporal persistence, channel phase coherence, and amplitude stability |
| **Transfer Entropy** | $TE_{X \to Y} = \sum P(y_{t+1}, y_t, x_t) \log \frac{P(y_{t+1} \mid y_t, x_t)}{P(y_{t+1} \mid y_t)}$ | Directional information flow from variable $X$ to $Y$, filtered via surrogate null-distribution p-values |
| **State-Space Filtering** | $x_t = A x_{t-1} + w_t, \quad y_t = C x_t + v_t$ | Gaussian state transitions and measurement equations evaluated via Kalman filter & RTS/Koopman disturbance smoothers |
| **Structural Surprise** | $S = D_{\text{KL}}(P_{t} \parallel P_{t-1})$ | Kullback-Leibler divergence measuring unexpected shifts in system topology and parameter dynamics |

---

## 💡 Key Capabilities Matrix

### 1. Spectral & Oscillation Analysis
- **Multimodal Wavelets**: Time-localized Morlet scalograms for nonstationary time series.
- **Welch & FFT Power Spectra**: High-resolution spectral density estimation with real-valued FFT optimizations.
- **Cross-Resonance Tensors**: Inter-channel phase coupling and collective oscillatory modes.

### 2. Directional Rooting & Topology Dynamics
- **Transfer Entropy**: Non-parametric directed information flow with PyInform acceleration and fallback estimators.
- **Surrogate Testing**: Shuffle-based statistical significance testing for candidate directed graph edges.
- **Topology Migration**: Rooting graph drift tracking, centrality shifts, and edge creation/destruction metrics.

### 3. State-Space, Smoothers & Control
- **Kalman Filtering**: Low-latency linear state estimation, innovation sequence diagnostics, and likelihood evaluation.
- **Koopman & Hamilton Smoothers**: Disturbance smoothers extracting structural shocks and retrospective state estimates.
- **Tempered Particle Filtering**: Herbst–Schorfheide particle filters for highly nonlinear resonance manifolds.
- **Closed-Loop Navigation Engine**: `ResonanceNavigationEngine` for state estimation, sensitivity calculation, and bounded control interventions.

### 4. Quant Research Lab
- **Microsoft Qlib Integration**: Matched experiment engine evaluating feature incremental value (IC, Rank IC, ICIR) with ablation controls.
- **Riskfolio-Lib Portfolio Engine**: DRR-conditioned regime switching (Mean-Variance vs. CVaR 95% tail risk minimization).
- **VectorBT Robustness Engine**: Multi-parameter backtesting sweeps with Newey-West HAC standard errors and Benjamini-Hochberg FDR adjustments.
- **Strict Anti-Leakage Invariants**: Automated point-in-time checks ensuring zero lookahead leakage.

### 5. LFBO Supervisory Research Workbench
- **Continuous Monitoring**: Public-data supervisory workflow answering *what changed, what deserves attention, and why?*
- **Disagreement Principle & Scope Resolver**: `DRR_ScopeResolver` retains divergent observations across scales (`MACRO`, `MESO`, `MICRO`, `LOCAL`) without forced statistical erasure.
- **Immutable Evidence Ledger**: Point-in-time reconstruction, content-addressed SHA-256 source hashes, and append-only audit passports.
- **Optional typed judgment**: Disabled by default. When enabled, a provider-neutral overlay asks bounded questions about evidence the attention budget already selected. It does not change DRR mathematics, statistical tests, evidence hashes, or attention rank.

---

## ⚡ Quick Start

### 1. Basic Resonance & Directional Rooting Analysis

```python
from drr_framework import DynamicResonanceRooting, generate_coupled_oscillator

# Generate multi-channel synthetic benchmark data
sampling_rate = 200.0
_, data = generate_coupled_oscillator(
    sampling_rate=sampling_rate,
    target_frequency_hz=12.5,
    random_state=42,
)

# Initialize DRR pipeline
drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)

# Run full multivariate system analysis
results = drr.analyze_system(
    data,
    multivariate=True,
    window_size=256,
    rooting_method="lagged_correlation",
    rooting_n_surrogates=25,
    rooting_random_state=42,
)

print("Resonance Depths:", results["resonance_depths"])
print("Significant Edges:", results["rooting_analysis"]["significant_edges"])
```

### 2. Real-Time Circular Buffer Streaming

```python
import numpy as np
from drr_framework.realtime import RealTimeDRR

# Initialize real-time streaming engine for 4 input channels
rt_drr = RealTimeDRR(n_variables=4, window_size=128, sampling_rate=100.0)

# Simulate streaming data processing
for t in range(500):
    point = np.random.randn(4)
    result = rt_drr.process_data_point(point)
    if result["status"] == "ready":
        print(f"Step {t}: Dominant Frequencies = {result['dominant_frequencies']}")
```

### 3. Closed-Loop Resonance Control

```python
from drr_framework.control_engine import ResonanceNavigationEngine

# Initialize closed-loop navigation engine
engine = ResonanceNavigationEngine(
    state_dim=2,
    control_dim=1,
    target_depth=0.85,
    max_intervention=0.5,
)

# Observe state, estimate sensitivity, and compute intervention
current_state = [0.42, 0.18]
control_output = engine.step(current_state=current_state)
print("Recommended Control Action:", control_output["intervention"])
```

### 4. LFBO Workbench & Non-Erasure Scope Resolver

```python
from drr_framework.disagreement import DRR_ScopeResolver, ObservationalScale, ObservationalPerspective

resolver = DRR_ScopeResolver()

# Register macro quantitative perspective
resolver.add_perspective(ObservationalPerspective(
    name="Macro Quant Model",
    scale=ObservationalScale.MACRO,
    indicator="Capital Ratio",
    value=0.125,
    method="Call Report Ingestion",
))

# Register local operational perspective
resolver.add_perspective(ObservationalPerspective(
    name="Local Examiner Assessment",
    scale=ObservationalScale.LOCAL,
    indicator="Capital Ratio",
    value=0.108,
    method="On-site Examination",
))

# Resolve scopes while preserving disagreement
resolution = resolver.resolve()
print("Scope Disagreements Retained:", len(resolution["disagreements"]))
```

---

## 🛠️ Installation

### From PyPI

```bash
pip install drr-framework
```

### Optional Extras

```bash
# Data Adapters (OpenBB)
pip install "drr-framework[quant-data]"

# Risk Engine (Riskfolio-Lib)
pip install "drr-framework[quant-risk]"

# Machine Learning Engine (Microsoft Qlib)
pip install "drr-framework[quant-ml]"

# Robustness & Backtesting (VectorBT)
pip install "drr-framework[quant-validation]"

# Full Quantitative Research Suite
pip install "drr-framework[quant]"
```

### From Source

```bash
git clone https://github.com/topherchris420/dynamic-resonance-rooting.git
cd dynamic-resonance-rooting
python -m pip install -e ".[dev]"
pre-commit install
```

---

## 💻 Command Line Interface (CLI)

The framework installs two CLI executables:

```bash
# Run end-to-end deterministic reproduction & validation suite
drr-reproduce --all

# Launch LFBO Supervisory Research Workbench
drr-monitor --demo --serve
```

---

## 🔬 Lab Benchmarks & Applications Catalog

| Lab Script | Domain | Description | Run Command |
|------------|--------|-------------|-------------|
| **`sonoluminescence_lab.py`** | Multimodal Physics | Multimodal acoustic driver, cavitation dynamics, optical flash emission, and electrical transduction | `python examples/sonoluminescence_lab.py` |
| **`sensing_systems_resonance_depth.py`** | Sensing & Radar DSP | Micro-Doppler radar DSP analog time series, clutter rejection, and target mode tracking | `python examples/sensing_systems_resonance_depth.py` |
| **`physics_lab.py`** | Dynamical Systems | Coupled oscillators, Lorenz, Rössler, Heston, and FitzHugh-Nagumo benchmark models | `python examples/physics_lab.py` |
| **`policy_lab.py`** | Macro Policy | Macroeconomic indicator time series (FRED) and policy shock response analysis | `python examples/policy_lab.py` |
| **`supervisory_policy_lab.py`** | Banking Supervision | FR Y-9C / FFIEC 002 data integration, risk domain scoring, and Tableau export generation | `python examples/supervisory_policy_lab.py` |
| **`state_space_lab.py`** | State-Space Modeling | Kalman filtering, Koopman disturbance smoothers, and tempered particle filtering | `python examples/state_space_lab.py` |
| **`quant_macro_lab.py`** | Quantitative Macro | Full multi-asset walk-forward portfolio allocation and research report generation | `python examples/quant_macro_lab.py --offline` |
| **`qlib_drr_experiment.py`** | Machine Learning | Qlib matched experiments evaluating IC/Rank IC lift of DRR features | `python examples/qlib_drr_experiment.py --offline` |
| **`drr_riskfolio_portfolio.py`** | Portfolio Risk | Riskfolio-Lib mean-variance and CVaR 95% regime switching optimization | `python examples/drr_riskfolio_portfolio.py --offline` |
| **`drr_vectorbt_robustness.py`** | Validation & Backtest | VectorBT parameter sensitivity sweeps and anti-leakage verification | `python examples/drr_vectorbt_robustness.py --offline` |
| **`lfbo_monitoring_workbench.py`** | Supervisory Workbench | Interactive supervisory workbench with point-in-time filing revisions and dispositions | `python examples/lfbo_monitoring_workbench.py` |
| **`quickstart_resonance_export.py`** | Quickstart / Export | End-to-end dataset generation, analysis, and JSON/CSV/figure export | `python examples/quickstart_resonance_export.py` |

---

## 📂 Project Structure

```text
dynamic-resonance-rooting/
├── src/drr_framework/          # Core DRR Framework package
│   ├── analysis.py             # DynamicResonanceRooting main orchestration class
│   ├── modules.py              # ResonanceDetector, RootingAnalyzer, DepthCalculator
│   ├── _spectral.py            # Optimized FFT & Morlet wavelet algorithms
│   ├── benchmarks.py           # Physics, radar, & oscillator generators
│   ├── sonoluminescence.py     # Multimodal sonoluminescence coupled system
│   ├── control_engine.py       # Closed-loop ResonanceNavigationEngine
│   ├── realtime.py             # Circular buffer O(1) RealTimeDRR streaming
│   ├── state_space.py          # Linear state-space modeling & Kalman filter
│   ├── smoothers.py            # Hamilton, Koopman, & Durbin-Koopman smoothers
│   ├── particle_filter.py      # Tempered particle filter (Herbst–Schorfheide)
│   ├── resonance_geometry.py   # Differential geometry & manifold curvature
│   ├── cross_resonance.py      # Cross-resonance tensor & collective modes
│   ├── topology_dynamics.py    # Rooting topology drift & migration metrics
│   ├── structural_surprise.py  # Information-theoretic structural surprise
│   ├── disagreement.py         # Non-erasure scope resolution & perspectives
│   ├── supervision.py          # Regulatory profiles & supervisory risk domains
│   ├── validation_readiness.py # Model risk cards & validation packets
│   ├── finance/                # DRR Quant Research Lab
│   │   ├── data/               # Adapters (OpenBB, Qlib, DataFrame)
│   │   ├── features/           # DRR state & factor feature engineering
│   │   ├── qlib/               # Qlib dataset & matched experiment engine
│   │   ├── portfolio/          # Riskfolio adapter & regime allocation
│   │   ├── validation/         # VectorBT adapter, walk-forward & HAC stats
│   │   └── reporting/          # Research report generators & plotting
│   └── supervisory/            # LFBO Supervisory Workbench backend & CLI
├── examples/                    # Runnable lab scripts and demonstrations
├── tests/                       # Unit and integration test suite
├── docs/                        # Framework documentation and guides
└── pyproject.toml              # Package configuration and dependencies
```

---

## 📚 Documentation Catalog

Detailed technical documentation is available in the [`docs/`](docs/) directory:

- [**User Guide**](docs/user-guide.md) — Getting started with DRR.
- [**Architecture**](docs/architecture.md) — High-level design and module interaction.
- [**API Reference**](docs/api.md) — Function and class signatures.
- [**Quant Research Lab Guide**](docs/quant-research.md) — Qlib, Riskfolio, and VectorBT integration.
- [**LFBO Workbench Guide**](docs/lfbo-workbench.md) — Supervisory workbench, data contract, and evidence model.
- [**Scope Resolution Guide**](docs/scope-resolution.md) — Non-erasure scope resolution and observational perspectives.
- [**Validation Readiness Guide**](docs/validation-readiness-guide.md) — Model risk cards and SR 26-2 compliance profiles.
- [**Resonance Geometry Guide**](docs/resonance-geometry.md) — Differential geometry, manifold dynamics, and curvature.
- [**Method Crosswalk**](docs/method-crosswalk.md) — Comparison against traditional time-series methods.
- [**Audience Guide**](docs/audience-guide.md) — Domain-specific entry points for researchers, quants, and risk managers.
- [**Security Posture**](docs/security-posture.md) — Trust boundaries, local binding, and audit logging.
- [**Developer Guide**](docs/developer-guide.md) — Guidelines for contributing and extending DRR.
- [**Reproducibility Guide**](docs/reproducibility.md) — Ensuring deterministic reproduction across platforms.
- [**FAQ**](docs/faq.md) — Frequently asked questions.

---

## 📖 Citation

If you use the DRR framework in your academic research or quantitative work, please cite:

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

## ⚖️ License

Distributed under the **MIT License**. See `LICENSE` for details.
