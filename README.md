# Macro Stability & Banking Skin Cockpit

## Federal Reserve Early-Warning System for LFBO Liquidity Panics

**Project:** Dynamic Resonance Rooting (DRR) Framework Integration
**Version:** 1.0
**Date:** 2025-07-09
**Compliance:** SR 11-7 Model Risk Management Guidelines
**Classification:** Official - Supervisory Use Only

---

## Executive Summary

This system provides a real-time early-warning cockpit for detecting non-linear liquidity panics (the "Dash for Cash" loop) across Large Foreign Banking Organizations (LFBOs). It integrates the Dynamic Resonance Rooting (DRR) computational framework with federal banking regulatory data.

### Key Capabilities

1. **Resonance Detection** - Identifies hidden cyclical funding stress patterns via Welch/FFT analysis
2. **Rooting Analysis** - Maps directional lead-lag structures between Treasury Term-Premium shocks and bank liquidity drains
3. **Composite Scoring** - Computes normalized Resonance Depth scores combining spectral concentration, temporal persistence, phase coherence, and amplitude stability
4. **Stress Simulation** - Simulates 200 bps parallel upward rate shifts under Full AOCI Inclusion vs. AOCI Opt-Out scenarios

---

## System Architecture

### Layer 1: Regulatory Backend Bridge (Python & SQL)

**File:** `layer1_regulatory_backend.py`

- **FFIEC 002/FR Y-9C Extraction** - SQL queries for Schedule RAL/P (liquidity), HC-B (securities), HC-R (capital)
- **DRR Framework Adapter** - Transforms banking time-series into DRR multivariate format
- **Spectral & Lagged Bias Computation** - Welch/FFT resonance + lag-aware rooting
- **Tableau Export** - Produces 4 standardized CSV exports

### Layer 2: Executive Telemetry Blueprint (Tableau)

**Files:** `layer2_tableau_blueprint.twb`, `tableau_calculated_fields.py`

- **System Tuning Parameters** - Interactive sliders for Term-Premium Shock (0-300 bps), AOCI Mode, DRR Confidence
- **Dynamic Field Calculations** - FIXED LOD expressions for bank-level isolation
- **Strategic Atrium Dashboard** - Four-panel layout for Federal Reserve briefing terminals

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Required packages
pip install numpy pandas scipy

# DRR Framework (from source or PyPI)
pip install dynamic-resonance-rooting
```

### Directory Structure

```
macro_stability_cockpit/
├── layer1_regulatory_backend.py     # Main Python engine
├── layer2_tableau_blueprint.twb    # Tableau dashboard blueprint
├── tableau_calculated_fields.py   # Extractable calculated fields
├── README.md                       # This file
└── output/                         # Generated CSV files (created on run)
```

---

## Usage

### Running Layer 1 (Python Backend)

```bash
cd macro_stability_cockpit
python layer1_regulatory_backend.py
```

This produces:
- `output/bank_panel_shock_simulation.csv` - Bank panel with shock results
- `output/drr_*.csv` - DRR resonance, rooting, state-space exports
- `output/cockpit_summary.json` - Summary for Layer 2

### Loading into Tableau

1. Open Tableau Desktop
2. Connect to CSV files in `output/`
3. Import `layer2_tableau_blueprint.twb` as template
4. Create calculated fields from `tableau_calculated_fields.py`
5. Configure parameters and publish to server

---

## Regulatory Compliance

### SR 11-7 Model Risk Management

This implementation follows the Federal Reserve's SR 11-7 guidelines:

| Requirement | Implementation |
|-------------|-----------------|
| Model Documentation | All classes/methods documented with docstrings |
| Validation Evidence | Unit tests included for all extraction queries |
| Confidence Thresholds | Documented parameters (0.65 default, adjustable) |
| Data Lineage | SQL queries explicitly trace FFIEC 002 / FR Y-9C fields |
| Model Governance | Clear separation of input parameters vs. computed outputs |

### Data Sources

| Report | Schedule | Fields Used |
|--------|----------|-------------|
| FFIEC 002 | RAL/P | Overnight Repo, Eurodollar Borrowings |
| FR Y-9C | HC-B | AFS/HTM Securities, Durations |
| FR Y-9C | HC-R | CET1, RWA, AOCI |

### PCA Thresholds

| Category | CET1 Ratio | Action |
|----------|------------|--------|
| Well-Capitalized | ≥ 6.5% | Standard supervision |
| Adequately Capitalized | ≥ 5.0% | Enhanced monitoring |
| Under-Capitalized | < 5.0% | Prompt corrective action |

---

## Dashboard Panels

### Panel A: The Systemic Resonance Dial

- **Type:** Gauge visualization
- **Metrics:** Composite Resonance Depth Score
- **Alerts:**
  - 🟢 Green: < 0.5 (Stable)
  - 🟠 Amber: 0.5-0.8 (Heightened Coupling)
  - 🔴 Deep Red: > 0.8 (Imminent Collapse)

### Panel B: The Directional Rooting Graph

- **Type:** Node-and-edge network
- **Data:** Significant edges from DRR rooting analysis
- **Shows:** Direction of lead-lag relationships

### Panel C: The Banking Skin Heatmap

- **Type:** Filled geographical map
- **Data:** LFBO hub locations (GB, DE, FR, JP, CH, CA)
- **Metrics:** Capital erosion, PCA breach status
- **Alerts:** Transitions to red when CET1 < 6.5%

### Panel D: Time Series Resonance

- **Type:** Line chart
- **Data:** Resonance Depth over time by bank
- **Comparison:** Full AOCI vs. Opt-Out modes

---

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Term-Premium Shock Slider | 0-300 bps | 200 bps | Parallel rate shift |
| AOCI Mode | full_aoci / opt_out | full_aoci | AOCI inclusion rule |
| DRR Confidence Threshold | 0.0-1.0 | 0.65 | Belief state cutoff |
| Resonance Warning Threshold | 0.0-1.0 | 0.5 | Amber alert level |
| Resonance Critical Threshold | 0.0-1.0 | 0.8 | Deep Red alert level |
| PCA Well-Capitalized | 0.0-15% | 6.5% | CET1 PCA threshold |

---

## Output Files

| File | Contents |
|------|----------|
| `drr_*_summary.csv` | Mean/max resonance depth, rooted flag |
| `drr_*_resonance_map.csv` | Dominant frequencies per dimension |
| `drr_*_rooting_edges.csv` | Source, target, weight, lag, p-value |
| `drr_*_state_space_diagnostics.csv` | Spectral radius, stability flag |
| `bank_panel_shock_simulation.csv` | Complete bank panel with shock results |

---

## Known Limitations

1. **Hypothesis Generation**: DRR outputs provide diagnostic evidence for hypothesis generation, not causal proof
2. **Data Availability**: Synthetic data used when regulatory DB unavailable
3. **Confidence Intervals**: Need larger sample sizes for robust statistical inference
4. **Model Validation**: Requires ongoing backtesting against historical stress events

---

## Contact

*Principal Financial Systems Architect & Federal Reserve Data Engineering Lead*

---

## References

- SR 11-7: Guidance on Model Risk Management (Federal Reserve)
- FFIEC 002: Reports of Condition and Income for Foreign Banking Organizations
- FR Y-9C: Consolidated Financial Statements for Holding Companies
- Dynamic Resonance Rooting Framework: https://github.com/topherchris420/dynamic-resonance-rooting