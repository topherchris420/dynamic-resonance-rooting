# Uncertainty quantification review: rooting robustness

**System:** DRR rooting analysis. **Date:** 2026-10-29.
**Scope:** repository evidence and a synthetic comparison harness, with no
changes to runtime inference or operational decisions.

## Executive summary

**Overall assessment: partial, with substantial simulation evidence.** DRR already
uses surrogate inference, maximum-statistic correction and Monte Carlo rate
intervals. The existing dependent-noise study examines long memory, volatility
clustering and nonstationarity. The new benchmark makes baseline transfer,
weak coupling and regime comparisons inspectable through shared trial records.

The principal interpretation risks are treating p-values as edge probabilities,
equating failure to detect error inflation with established control, and treating
reverse associations induced by memory as necessarily false statistical findings.
The new report explicitly separates these quantities and preserves inconclusive
results. No independently validated decision confidence is established.

## System context

- Task: retrospective lead–lag detection in multivariate time series.
- Output: candidate/significant edges, lag scores and surrogate p-values.
- Decision consequence in this change: research diagnostics and offline comparison.
- Evidence: `src/drr_framework/modules.py`, `calibration.py`, `dependent_noise.py`,
  `robustness_benchmark.py`, their tests, and retained simulation artifacts.

## Current UQ posture

**Implemented:** distinct surrogate assumptions, family-maximum adjustment,
known-generator size/power studies, and Wilson intervals for trial-level rates.
See `RootingAnalyzer.analyze`, `calibration.binomial_rate` and
`dependent_noise.run_dependent_noise_study`.

**Calibration evidence:** error-rate studies for specified synthetic nulls.
The new `calibrate_baselines` freezes simple-baseline thresholds using independent
white-noise draws. This is not calibrated edge-truth confidence or a guarantee
after distribution shift.

**Shift/corruption evidence:** the existing study includes random walks and
dependent noise. The new `simulate_regimes` adds coupling onset, offset and
direction reversal at known boundaries. No adversarially optimized perturbation
or unknown-boundary online delay study was performed.

**Operational policy:** this addition produces offline artifacts. It introduces
no confidence-based automation or change to existing significance decisions.

## Evidence-based findings

1. **Implemented — baseline comparison with isolated calibration.**
   `calibrate_baselines`, `evaluate_trial` and the calibration-isolation tests
   separate calibration data from evaluation and share windows across methods.
   This enables comparisons of declared procedures without retuning on test cases.
2. **Implemented — auditable uncertainty and negative results.**
   `summarize_records` reconstructs rates and pointwise Wilson intervals from
   trial decisions, rejects incomplete records, and labels uncertain null results
   inconclusive. The CLI seals protocol/source hashes before execution.
3. **Partial — validation across dependence and shift.**
   The generator grid covers selected long-memory strengths and labeled regime
   changes. It does not cover every process, adaptive attack, physical system,
   or unknown-boundary deployment. These limits remain visible in the report.
4. **Unknown — independently validated operational confidence.**
   No physical/domain-transfer validation or independent replication was produced
   by this change. Existing domain-specific artifacts do not transfer their
   evidence claims automatically to these synthetic comparisons.

## Recommended approaches

1. Retain paired trial comparisons with frozen thresholds; track null family-wise
   false alarms, injected direction/lag recovery and non-injected directions.
2. Repeat independent calibration sets in a future study; distinguish conditional
   test-set rate variation from variation introduced by recalibration.
3. Freeze external datasets and comparator settings before evaluation; preserve
   failed comparisons and distinguish independent replication from replay.

## Tooling and implementation notes

The implementation uses existing NumPy and DRR simulator/inference code. No new
runtime dependency or probabilistic model is introduced. Two simple baselines
precede more complex alternatives. The rank threshold's exchangeability boundary
and the complete protocol are documented in `docs/ROBUSTNESS_BENCHMARK.md`.

## Operational guidance

Use these results to identify assumption failures and select further studies.
Keep inconclusive intervals visible; do not promote them to a validated operating
threshold. Scores and p-values remain research outputs. Any later deployment
policy needs its own intended use, independent validation data and monitoring
criteria. Invalid/incomplete benchmark evidence raises an error rather than
silently contributing a favorable rate.

## Unknowns and assumptions

Trials are independent conditional on retained calibration; windows inside a
trial are dependent. Numeric byte hashes may vary across FFT/platform versions.
Rate intervals are pointwise and exclude calibration-set uncertainty. Boundaries
are known and buffered. A generator's injected direction is distinct from its
complete association graph. Replaying synthetic results does not establish
physical validity, independent replication, or calibrated edge probabilities.
