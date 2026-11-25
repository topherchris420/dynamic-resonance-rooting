# Reproducibility Guide

DRR includes deterministic examples and smoke benchmarks so reviewers can
compare a clean checkout against expected artifacts.

## Reproduction Harness

```bash
python -m drr_framework.experiments --output-dir results/reproduction
```

For a smoke run that prints the summary without writing artifacts, add
`--no-artifacts`. When you do want files, point `--output-dir` at a temporary
directory so the run stays isolated from the working tree.

Expected files:

- `results/reproduction/drr_reproduction_summary.json`
- `results/reproduction/drr_reproduction_metrics.csv`

The harness uses a synthetic coupled oscillator with:

- Known dominant frequency: `12.5 Hz`
- Known directed edge: `dim_0 -> dim_1`
- Expected lag: `2` samples

## Rooting Configuration

The public rooting path is deterministic when you keep the default settings:

- `rooting_method="lagged_correlation"`
- `rooting_n_surrogates=25`
- `rooting_random_state=0`
- `rooting_surrogate_method="circular_shift"`
- `rooting_correction="max_statistic"`

That configuration gives a minimum attainable p-value of `1 / 26`
(`~0.0385`). The returned `candidate_edges` list is exploratory, while
`significant_edges` and the influence graph only include edges that survive the
selected p-value test.

Circular-shift surrogates preserve each series' marginal distribution and
cyclic ordering, and they approximately preserve within-series dependence for
reasonably stationary series. They are a good default for the current fixture,
but they are not a good null model for strongly nonstationary series. If your
data drifts hard over time, use a different null model.

## Random Seeds

Examples that generate stochastic data use explicit `random_state` values.
When adding new examples, expose a seed parameter or document why the workflow is
deterministic without one.

## Example Data

Small example datasets live under `data/raw/`. Processed reference artifacts
live under `data/processed/` or `results/expected/`.

Large external datasets should be prepared by scripts and documented with:

- Source URL
- License or terms
- Retrieval date
- Preprocessing steps
- Expected file names and checksums

## External evidence benchmark

The preregistered QBO comparison is separate from the synthetic reproduction
harness. From the repository root:

```bash
python scripts/run_structural_change_benchmark.py --output-dir results/expected
```

The command reads the vendored NOAA CPC snapshots and
`src/drr_framework/external_benchmark/data/preregistration.json`. It writes
`qbo_structural_change_benchmark.json` and a Markdown report that states only
the claim contained in that JSON. The protocol is described in
[External evidence](external-evidence.md).

## Calibration study

```bash
python scripts/run_calibration_study.py --output-dir results/expected
```

The study simulates systems with a known truth and writes
`rooting_calibration_study.json` with a rendered Markdown companion: the rooting
test's family-wise error on independent AR(1) series, its power to recover a
known edge, the effect of memory in every channel, and resonance-depth reference
distributions. It takes about four minutes. `--trials-scale 0.1` gives a quick
look. `tests/test_calibration_study.py` reruns a prefix of the committed rows and
requires an exact match.

## Dependent-noise null study

```bash
python scripts/run_dependent_noise_study.py --output-dir results/expected
```

The study answers the question in [MATH_RESEARCH.md](MATH_RESEARCH.md): which
surrogate assumptions keep lag-search false alarms controlled under long memory
or dependent noise. It simulates independent channels of fractional Gaussian
noise (exact Davies–Harte embedding, Hurst exponent 0.5 to 0.95), GARCH(1,1)
noise, and random walks, measures the family-wise error of the circular-shift,
phase-randomization, and permutation nulls on each, and measures power to
recover a lag-2 edge when every channel has long memory. The six hypotheses
and the decision rule are fixed in `drr_framework.dependent_noise.DEFAULT_CONFIG`
before the run, and the artifact `dependent_noise_null_study.json` records each
outcome beside its rule. It takes about ten minutes. `--trials-scale 0.1` gives
a quick look. The exit code reports the integrity checks only; a hypothesis
that is not supported is recorded, not hidden. `tests/test_dependent_noise_study.py`
reruns a prefix of committed rows exactly and recomputes every outcome from the
rows.

## Phase and harmonic benchmark

```bash
python scripts/run_resonance_math_benchmark.py --output-dir results/expected --workers 4
```

The study asks whether the experimental `n:m` phase-synchrony test adds value to
DRR's relationship tests. It simulates coupled and deliberately misleading
channel pairs, applies DRR rooting, conventional baselines, the phase test, its
ablations, an oriented-area test, and gated combinations, and writes
`resonance_math_benchmark.json` with a rendered Markdown companion. Development
conditions set the methods; held-out conditions, methods, thresholds and nine
hypotheses were committed with a SHA-256 fingerprint before the held-out run,
and the artifact records whether its configuration still matches. Rows are
identical for any number of workers; each trial has its own seed stream, so any
trial reruns alone. It takes about fourteen minutes on four workers.
`--trials-scale 0.1` gives a quick look. `tests/test_resonance_math_benchmark.py`
recomputes every rate and outcome from the per-trial codes and reruns a prefix of
committed rows exactly.

## Landing page evidence

`index.html` embeds the calibration, null-study, and QBO numbers it charts.
After regenerating any artifact, run:

```bash
python scripts/sync_site_evidence.py
```

`tests/test_site_evidence.py` fails when the page and the artifacts disagree.

## Reviewer Checklist

- Can the example run from a clean checkout?
- Are random seeds fixed?
- Are expected outputs committed or generated by a documented command?
- Are caveats visible in reports?
- Are any domain claims supported by reproducible artifacts?
