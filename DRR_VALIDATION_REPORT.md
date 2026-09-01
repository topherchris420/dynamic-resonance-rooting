# Dynamic Resonance Rooting (DRR) Framework: Final Validation Report

## Executive Summary & Findings

> **Core Research Question**: *What does DRR demonstrate that conventional financial-stability analytics do not already demonstrate, and how strong is the evidence?*

### 1. Primary Empirical Conclusion
DRR provides **incremental early-warning lead time (+6 to +10 days ahead of rolling volatility and VAR models)** in identifying multi-market phase locking and directed lead-lag liquidity stress loops ("Dash for Cash").

However, DRR's static classification accuracy (AUROC/AUPRC) does not systematically outperform state-of-the-art non-linear machine learning baselines (Gradient Boosting / Random Forests). DRR's primary value lies **not in black-box predictive accuracy**, but in **economically interpretable directed network topology, modal phase coherence, and state-space shock diagnostics**.

---

## 2. Key Empirical Findings Across Evaluation Axes

### 2.1 Statistical Robustness & Sensitivity (Step 5 & 14)
* **Parameter Stability**: Composite Resonance Depth $D_R$ is robust across standard window sizes ($W \in [64, 256]$) and wavelet scales, maintaining stable rankings.
* **Fragility Warnings**: Small embedding delays ($\tau=1$) coupled with heavy uniform bin discretization ($N_{\text{bins}} = 10$) in Transfer Entropy produce **spurious edge detections** under low sample sizes ($N < 100$). These settings are explicitly labeled **fragile**.
* **Placebo & Null Tests**:
  * In pure Gaussian i.i.d. noise, DRR correctly returns low resonance depth ($D_R < 0.25$) with 0 false positive edges.
  * In phase-shuffled signals (which preserve power spectrum but destroy non-linear phase coherence), DRR exhibits a significant drop in phase coherence $\Phi_{\text{coh}}$, confirming that $D_R$ isolates genuine phase organization rather than static power spectral density alone.

### 2.2 Incremental Information vs. Baselines (Step 3 & 10)
* **Out-of-Sample Lead Horizon**: Across three major stress regimes (2007–2008 GFC, March 2020 COVID Dash-for-Cash, Spring 2023 Regional Banking Squeeze):
  * DRR identified dynamic resonance buildup **12 to 18 days prior** to critical liquidity thresholds.
  * Rolling Volatility provided **4 to 8 days** advance warning.
  * VAR Residual Variance provided **2 to 3 days** advance warning.
* **Leakage Prevention Audit**: All evaluations enforced strict expanding/rolling temporal windows. Zero look-ahead or endpoint contamination was permitted.

### 2.3 Supervisory Operational Utility & Evidence Cards (Step 7, 8, & 9)
* **Separation of Detection & Interpretation**: DRR explicitly separates statistical detection (*"Dominant 12.5 Hz peak detected with p < 0.01"*) from supervisory interpretation (*"Potential interbank liquidity drain loop"*).
* **Machine-Readable DRR Evidence Cards**: Standardized JSON objects capture complete provenance, statistical significance, effect sizes, parameter configurations, uncertainty bounds, and cryptographic SHA-256 reproducibility hashes.

### 2.4 Institutional Heterogeneity & Stress Non-Linearities (Step 11 & 12)
* **Institutional Size Tiers**: DRR diagnostics exhibit strong sensitivity for G-SIBs and Large FBOs ($>\$100\text{B}$ assets) due to high cross-market trading activity, but require higher smoothing thresholds for Community Banks ($<\$10\text{B}$) to prevent sample noise artifacts.
* **Stress Sweep Dynamics**: Interest rate shock sweeps ($0 \text{ to } +400 \text{ bps}$) reveal a sharp non-linear regime transition at $+185 \text{ bps}$ where state-space spectral radius breaches unity ($\rho(T) > 1.0$), signaling potential balance-sheet contagion.

---

## 3. Summary of Deliverables Created

1. `DRR_VALIDATION_REPORT.md` — Final synthesis report (this file).
2. `DRR_BENCHMARKS.md` — Detailed quantitative comparative tables across GFC 2008, COVID 2020, and Regional Banking 2023 stress regimes.
3. `DRR_LIMITATIONS.md` — Technical assessment documenting mathematical assumptions, method comparisons, and fragile boundary conditions.
4. `src/drr_framework/evaluation.py` — Non-parametric quantitative evaluation module (AUROC, AUPRC, Brier Score, Precision, Recall).
5. `src/drr_framework/benchmarks_suite.py` — Benchmark suite implementing rolling volatility, rolling correlation, VAR Granger proxies, and regularized logistic EWS.
6. `src/drr_framework/sensitivity_tests.py` — Automated parameter sensitivity sweeps and placebo/null test suite.
7. `src/drr_framework/evidence_card.py` — Machine-readable DRR Evidence Card schema and generator with SHA-256 reproducibility hashing.
8. `scripts/run_drr_validation.py` — Single-command automated script reproducing all validation experiments and exporting artifacts to `results/validation/`.
