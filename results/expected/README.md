# Expected Outputs

This directory contains compact expected outputs for examples and smoke-test
workflows. They are intentionally small so reviewers can compare a fresh run
against committed reference artifacts without downloading external data.

`qbo_structural_change_benchmark.json` is the preregistered external comparison
on the vendored NOAA CPC QBO snapshots. The Markdown file beside it is rendered
from that JSON. Regenerate both with
`python scripts/run_structural_change_benchmark.py --output-dir results/expected`.

`rooting_calibration_study.json` is the Monte Carlo calibration of the rooting
test and the resonance-depth score on simulated systems with a known truth.
Regenerate it and its Markdown companion with
`python scripts/run_calibration_study.py --output-dir results/expected`.

`dependent_noise_null_study.json` is the preregistered null-model study: the
family-wise error of the circular-shift, phase-randomization, and permutation
nulls on fractional Gaussian noise, GARCH(1,1) noise, and random walks, with
the power to recover a known edge under long memory and the outcome of each
hypothesis fixed before the run. Regenerate it and its Markdown companion with
`python scripts/run_dependent_noise_study.py --output-dir results/expected`.

`resonance_math_benchmark.json` is the preregistered phase and harmonic
benchmark: false-relationship and detection rates of DRR rooting, conventional
baselines, the experimental phase-synchrony test and its ablations on held-out
simulated channel pairs, one character per trial, with nine hypotheses fixed
before the held-out run. Regenerate it and its Markdown companion with
`python scripts/run_resonance_math_benchmark.py --output-dir results/expected --workers 4`.

`robustness-benchmark/` retains the fixed exploratory comparison of DRR rooting
with raw and AR(1)-residual correlation baselines. It contains the pre-execution
protocol, independent calibration maxima, all trial/window decisions, a rendered
report, and PNG/SVG figures. It covers a memory/coupling grid and paired coupling
onset, offset and reversal cases. Reproduce into a **fresh** directory with
`python scripts/run_robustness_benchmark.py --output-dir results/robustness-full`.
See [the design and interpretation limits](../../docs/ROBUSTNESS_BENCHMARK.md).
