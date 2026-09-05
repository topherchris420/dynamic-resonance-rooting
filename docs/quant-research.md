# DRR Quantitative Research Lab Architecture & Integration Guide

## Overview

The **DRR Quant Research Lab** is a quantitative finance application layer built around Dynamic Resonance Rooting (DRR). Its purpose is to test whether DRR produces useful, reproducible structural representations of financial markets and whether those representations contain incremental information beyond conventional financial features out of sample.

The central scientific question is:
> *Does DRR describe market dynamical structure in a way that improves prediction, regime inference, risk management, or portfolio decisions out of sample?*

This is a falsifiable research project—a null result is scientifically acceptable.

---

## Architectural Principles

1. **DRR is the Primary Scientific Framework**: External quantitative packages are optional adapters around DRR.
2. **Clean Conceptual Data Flow**:
```text
Market Data (OpenBB / Qlib / DataFrame)
        │
        ▼
   DRR Framework
        │
        ▼
Structural Market State (MarketResonanceState)
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Qlib  Riskfolio      VectorBT
ML    Portfolio      Validation
```
3. **Strict Non-Monolithic Optional Dependencies**: The base `drr-framework` package remains lightweight. External quant dependencies (`openbb`, `riskfolio-lib`, `pyqlib`, `vectorbt`) are optional extras (`drr-framework[quant]`).
4. **Causal Anti-Leakage Invariants**: Signals, features, thresholds, and portfolio decisions dated at time $t$ rely strictly on market information available on or before time $t$.

---

## Component Roles & Adapters

### 1. Financial Data Infrastructure: OpenBB & Qlib Adapters
- `OpenBBMarketDataProvider` (`drr_framework.finance.data.openbb_adapter`): Interoperates with OpenBB Platform API to retrieve multi-asset historical daily prices.
- `QlibMarketDataProvider` (`drr_framework.finance.data.qlib_adapter`): Loads price panels directly from Microsoft Qlib data engine.
- `DataFrameMarketDataProvider` (`drr_framework.finance.data.base`): Provides offline, deterministic, synthetic, or CSV-based data for offline CI and reproducible research.

### 2. Primary Quant Research Platform: Microsoft Qlib Adapter
- `DRRQlibFeatureProvider` (`drr_framework.finance.qlib.feature_provider`): Combines conventional factors (returns, realized volatility, momentum, cross-asset correlation) with rolling DRR structural features.
- `QlibDatasetAdapter` (`drr_framework.finance.qlib.dataset`): Converts feature matrices and forward target labels into Qlib `DatasetH` structures.
- `QlibDRRMatchedExperiment` (`drr_framework.finance.qlib.experiment`): Executes matched out-of-sample experiments:
  - **CONTROL**: Qlib model + conventional financial features.
  - **EXPERIMENT**: Same Qlib model + conventional features + DRR features.
  - Calculates Information Coefficient (IC), Rank IC, ICIR, Rank ICIR, and runs feature ablation studies.

### 3. Portfolio & Risk Engine: Riskfolio-Lib Adapter
- `RiskfolioAllocator` (`drr_framework.finance.portfolio.riskfolio_adapter`): Interfaces with Riskfolio-Lib for Mean-Variance (MV) and Conditional Value at Risk (CVaR 95%) optimization, with a robust SciPy SLSQP fallback.
- `DRRRegimePortfolioPolicy` (`drr_framework.finance.portfolio.policy`): Evaluates expanding percentile, rolling percentile, and z-score thresholds on trailing DRR mean depth to dynamically select standard vs. high-resonance risk policies without lookahead bias.
- Centralized daily compounding risk-free rate conversion: $r_{\text{daily}} = (1 + r_{\text{annual}})^{1/252} - 1$.

### 4. High-Speed Robustness & Validation Engine: VectorBT Adapter
- `VectorBTAdapter` (`drr_framework.finance.validation.vectorbt_adapter`): Reconstructs backtests independently using VectorBT to validate native walk-forward calculations and executes parameter robustness sweeps across lookback windows, depth windows, and regime percentiles.

---

## Anti-Leakage & Statistical Rigor

- **Lookahead Verification Utility**: `assert_no_lookahead_leakage` (`drr_framework.finance.validation.leakage`) verifies the invariant that truncating a dataset after date $t$ produces identical outputs for all dates $\le t$.
- **Statistical Significance**: Computes Pearson/Spearman correlations with forward realized volatility and drawdowns, Newey-West / HAC robust standard errors (`calculate_hac_standard_errors`), Benjamini-Hochberg False Discovery Rate corrections (`benjamini_hochberg_fdr`), and date-shuffled negative controls (`run_negative_controls`).

---

## Reproduction Commands

To run the full Quant Macro Lab walk-forward evaluation offline:
```bash
python examples/quant_macro_lab.py --offline --start 2015-01-01 --end 2023-12-31
```

To run the Primary Qlib Matched Experiment:
```bash
python examples/qlib_drr_experiment.py --offline --model linear
```

To run Riskfolio portfolio optimization:
```bash
python examples/drr_riskfolio_portfolio.py --offline
```

To run VectorBT parameter robustness sweeps:
```bash
python examples/drr_vectorbt_robustness.py --offline
```
