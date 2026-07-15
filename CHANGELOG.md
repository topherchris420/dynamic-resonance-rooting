# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Morlet wavelet resonance detection (`method="wavelet"`): an FFT-based
  continuous wavelet transform with peak selection on the global wavelet
  spectrum, parabolic peak refinement, and a returned `scalogram` and
  `time_axis` that localize resonances in time for nonstationary signals
- `analyze_system` accepts `method` and `peak_height_ratio`, so Welch or
  wavelet detection can be used end to end instead of always falling back to
  FFT
- `RealTimeDRR` accepts `analysis_interval` to run the full analysis every
  N-th point on high-frequency streams (default 1 keeps existing behavior)
- `plot_results` accepts `show=False` for headless or batch environments
  (the figure is closed after optional saving instead of displayed)

### Changed
- Vectorized lagged-correlation rooting scores (one cross-correlation matrix per
  lag instead of a `corrcoef` call per variable pair), speeding up
  `RootingAnalyzer.analyze` and its surrogate testing by ~6x on typical panels
- Recalibrated the QBist agent likelihood: the logistic is now centered at a
  resonance depth of 0.5 (depth 0 → 0.1, depth 1 → 0.9), so shallow resonances
  lower the belief instead of leaving it stuck at or above the prior
- `analyze_system` now raises on failure (with a logged traceback) instead of
  silently returning an empty dict
- `calculate_resonance_depths` resets the QBist agent to its prior on each
  call, so repeated analyses on one instance produce identical belief states
- `DynamicResonanceRooting` validates `embedding_dim`, `tau`, and
  `sampling_rate` at construction
- The Heston benchmark now always uses `numpy.random.default_rng` with
  pre-drawn correlated shocks instead of the legacy global NumPy RNG when no
  seed is given (seeded draws are still deterministic, but differ from
  previous versions)

### Fixed
- `MarkovChain` stationary distribution had the wrong length when eigenvalue 1
  had multiplicity greater than one (e.g. reducible chains); a single
  eigenvector is now selected
- `QBistAgent.reset()` now restores the prior belief instead of only clearing
  the history
- `RealTimeDRR.reset()` also resets the anomaly detector history
- `drr_framework.__version__` reported a stale `0.2.0`; it is now
  single-sourced from the installed package metadata (`4.2.0`)

---

## [4.2.0] - 2026-07-09

### Added
- **Macro Stability & Banking Skin Cockpit**: FFIEC 002 / FR Y-9C SQL extraction with Tableau dashboard integration
- **DSGE-style State-Space Diagnostics**: Python-native state-space representation, Kalman filtering, likelihood diagnostics
- **Audience-Aware Workflows**: Physics lab, policy lab, and supervisory review workflows with audience-specific JSON/Markdown reporting
- **Fed-Aligned Supervisory Presets**: SR 11-7 style readiness packets and model-risk cards for regulatory compliance
- **Deterministic Validation Harness**: For research reproducibility
- **Package and Citation Files**: setup.py, CITATION.cff
- **NumPy Fallback**: Compatibility layer for scipy-free runtime environments
- **Research-Grade Outputs**: JSON/Markdown export for reviewer navigation

### Changed
- Package layout optimized for reviewer navigation
- Black configuration updated for extend-exclude string regex
- Ruff linter exclusions for Tableau files
- Formatter CI focused on maintained sources

### Fixed
- PyPI badge links corrected
- Black extend-exclude configuration corrected

---

## [2.0] - 2026-03-15

### Added
- Core resonance detection via FFT and Welch PSD
- Causal rooting analysis with transfer entropy and lagged correlation
- Resonance depth composite scoring
- State-space diagnostics
- Phase-transition detection
- Benchmark generators (Lorenz, Rössler, Heston, FitzHugh-Nagumo)
- Multi-format exports (JSON, CSV, Markdown, Plots)

### Changed
- Initial release structure

---

## [1.0.0] - 2025-08-03

### Added
- Initial proof-of-concept release

---

[4.2.0]: https://github.com/topherchris420/dynamic-resonance-rooting/releases/tag/v4.2.0
[2.0]: https://github.com/topherchris420/dynamic-resonance-rooting/releases/tag/v2.0
[1.0.0]: https://github.com/topherchris420/dynamic-resonance-rooting/releases/tag/v1.0.0