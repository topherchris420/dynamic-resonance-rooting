# Rooting robustness comparison

This fixed exploratory benchmark compares the existing circular-shift and
phase-randomization rooting tests with two simple baselines. It extends the
dependent-noise study with a coupling-strength grid, shared trial inputs,
separate baseline calibration, and before/after regime comparisons.

## Run and inspect

```bash
python scripts/run_robustness_benchmark.py --quick --output-dir results/robustness-smoke
python scripts/run_robustness_benchmark.py --output-dir results/robustness-full
```

Use a fresh directory for every run. The command refuses existing directories,
including incomplete attempts. It writes `protocol.json` before computing the
study, then `benchmark.json`, `report.md` and PNG/SVG comparison figures. The protocol retains the complete
configuration, configuration digest, source commit, and hashes of the actual
analysis sources. Source hashes cover local changes even before they are committed.
The result retains the protocol's digest, Python/NumPy versions, calibration
maxima, per-window input hashes and every method's reported edges.

The [retained reference run](../results/expected/robustness-benchmark/report.md)
uses the full defaults. A quick run uses fewer trials, samples and surrogates;
its wide intervals and coarse p-value resolution are intended for smoke testing.
An interrupted attempt retains its protocol but is not a completed result.

## Fixed design

- Three independent latent fractional Gaussian noise channels, 512 scored samples
  per window, lags 1–4, injected lag 2, 200 independent trials per case.
- Hurst exponents 0.50, 0.85 and 0.95 crossed with coupling 0, 0.10, 0.30 and 0.50.
  Coupling is the target's population correlation with its lagged source;
  innovation mixing preserves unit marginal variance.
- At H = 0.85, three additional paired cases turn coupling on, turn it off,
  or reverse its direction between windows. Continuous latent noise is shared
  across both windows; a lag-length buffer is discarded at each regime boundary.
  No scored window straddles a switch. This does not estimate online detection delay.
- Every method receives the same underlying trial/window. Case names and trial
  indices determine evaluation seeds independently of calibration seeds. Changing
  a later regime's coupling cannot change the earlier window.
- The existing DRR methods retain their candidate-edge gate, maximum-statistic
  correction, alpha 0.05, and 99 surrogates. Their implementation is unchanged.

The protocol is retained before execution, but this is not an independent
preregistration or a confirmatory test of a claim selected before development.

## Simple baselines and calibration

`raw_correlation` computes absolute Pearson correlation at every allowed lag
and directed pair. `ar1_residual_correlation` first estimates one AR(1)
coefficient per channel inside the current observation window, then runs the
same search on its residuals. Removing one AR coefficient does not remove all
long memory. These are retrospective window analyses, not forecasting models.

For each baseline separately, 399 independent white-noise calibration windows
produce a distribution of the maximum score across every pair and lag. Freeze
the order statistic at rank `ceil((m + 1) * (1 - alpha))`; report an edge only
when its score strictly exceeds that threshold. No evaluation window selects
the threshold. This finite-sample rank correction follows the exchangeable-score
argument described by [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

The rank argument applies to one exchangeable future family maximum, marginal
over calibration and evaluation draws. It does not guarantee error control for
a particular frozen threshold or under transfer to long memory or switches.
Those transfer cases are precisely what the benchmark measures. DRR and the
baselines also differ in the existing DRR candidate-edge gate, so the comparison
is between the complete declared procedures, not an isolated null-model ablation.

## Outcomes and uncertainty

The JSON `truth` field identifies the generator's injected directions and lags.
It is not a complete description of the correlation graph: long-memory coupling
can induce other real lead–lag associations, including a reverse direction.

Each case/window/method reports:

- Recovery of the injected direction at any lag and at the injected lag.
  These are unavailable on null windows, rather than reported as zero power.
- Any reported direction and any non-injected direction. On independent null
  windows, any report is a family-wise false alarm. Under coupling, a non-injected
  direction is off-target relative to the generator and need not be a false
  statistical relationship or evidence of a causal error.
- Counts, denominators and pointwise Wilson 95% intervals across independent
  trials, using the existing `calibration.binomial_rate` implementation. Paired
  windows are never pooled as independent observations. Rate intervals condition
  on the frozen calibration thresholds and do not include recalibration variation.
- Null assessment: **inflated** if the interval's lower bound exceeds alpha;
  **upper bound at or below alpha** if its upper bound is at most alpha;
  otherwise **inconclusive**. Failure to detect inflation does not establish control.

Correlation magnitudes and surrogate p-values are not probabilities that an edge
is true. The study therefore reports no ECE or Brier score for those quantities.
Intervals are pointwise across this grid, not simultaneous. No global winner,
superiority claim, causal conclusion, or physical validation is declared.

## Verification and next steps

`tests/test_robustness_benchmark.py` checks known direction/lag recovery,
prewhitening, calibration isolation, future-regime isolation, deterministic
replay, incomplete-evidence rejection, uncertainty interpretation, CLI preservation,
and reconstruction of the reference summaries from retained trial decisions.

The next research steps are repeated calibration sets, additional independent
generators, and frozen external-data comparisons. Independent replication and
domain-specific validation remain separate work.
