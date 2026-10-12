# Which null assumptions keep lag-search false alarms controlled?

**Status:** preregistered study, executed. The record is
[`results/expected/dependent_noise_null_study.json`](../results/expected/dependent_noise_null_study.json)
with a [rendered companion](../results/expected/dependent_noise_null_study.md).
Reproduce it with
`python scripts/run_dependent_noise_study.py --output-dir results/expected`
(about ten minutes). `tests/test_dependent_noise_study.py` reruns a prefix of
committed rows exactly and recomputes every hypothesis outcome from the rows,
so the outcomes below cannot be edited without the rows disagreeing.

## The question

DRR reports a directed edge when the largest lagged correlation between two
series exceeds what a surrogate null produces. A surrogate null is a
statistical assumption about the series, made concrete as a way of generating
look-alike data with no cross-series relationship. If the assumption is wrong,
the null is too narrow and the test raises false alarms.

**Which of those assumptions keep lag-search false alarms controlled under
long-memory or dependent noise?** Long memory is the hard case: the
autocorrelation of fractional Gaussian noise with Hurst exponent H > 0.5 decays
like k^(2H−2) and its sum diverges, so a series never "forgets", and standard
effective-sample-size arguments fail.

## Assumptions mapped before importing a mathematical idea

These three were written before any row was run and are unchanged.

1. **A surrogate distribution respects the tested process' serial structure.**
   Made testable: each null is named for the structure it keeps, and the
   processes below violate those assumptions by degrees.
2. **A discovered directed edge is an estimate of lead–lag, not a causal arrow.**
   The power rows report reverse edges (the target appearing to lead its own
   source) as a cost, not as a finding.
3. **All thresholds, lag windows and multiple-test corrections are fixed before
   the holdout.** The configuration, seeds, trial counts, hypotheses and the
   decision rule live in `drr_framework.dependent_noise.DEFAULT_CONFIG`, and
   the artifact records the configuration's SHA-256.

## Protocol

**Nulls.** Each is a statement about a series.

| Null | Assumes | Keeps | Destroys |
| --- | --- | --- | --- |
| `circular_shift` (default) | stationarity | each series' joint distribution, except at one seam | the alignment between series |
| `phase_randomization` (new) | a stationary linear Gaussian process | each series' periodogram, hence its sample autocorrelation at every lag | the phases, so the alignment between series and any non-Gaussian structure |
| `permutation` (legacy) | serial independence | each series' marginal distribution | each series' memory and the alignment between series |

**Processes.** Independent channels, so every reported edge is a false alarm.

| Process | Breaks | Generator |
| --- | --- | --- |
| Fractional Gaussian noise, H = 0.50 | nothing (white noise anchor) | exact Davies–Harte circulant embedding |
| Fractional Gaussian noise, H = 0.70, 0.85, 0.95 | serial independence; memory is not summable | exact Davies–Harte circulant embedding |
| GARCH(1,1), α = 0.10, β = 0.85 | Gaussianity and independence while staying uncorrelated | recursion with Gaussian innovations, unit unconditional variance |
| Random walk | stationarity | cumulative sum of Gaussian increments |

**Settings.** 512 samples, 3 channels, lags 1–4, 99 surrogates, α = 0.05,
max-statistic correction, 400 systems per size row, 200 per power row.

**Negative control.** The coupled rows drive channel 0 → channel 1 at lag 2 with
coupling 0.3 while every channel is fractional Gaussian noise at the same H.
The uncoupled size rows at that H are their negative control: same memory, no
relationship.

**Decision rule, fixed before the run.** A null is controlled on a process when the Wilson 95% lower bound of its family-wise error is at most alpha. It is inflated when that lower bound exceeds alpha. The rule, the processes, the seeds, and the trial counts are fixed before the run and are not changed after the rows are seen.

## Results

### Size: false alarms on independent channels

| Process | Null | Family-wise error | 95% interval | Verdict |
| --- | --- | ---: | --- | --- |
| fGn H = 0.50 (white noise) | circular_shift | 0.055 | 0.037–0.082 | controlled |
| fGn H = 0.50 (white noise) | phase_randomization | 0.052 | 0.035–0.079 | controlled |
| fGn H = 0.50 (white noise) | permutation | 0.037 | 0.023–0.061 | controlled |
| fGn H = 0.70 | circular_shift | 0.048 | 0.031–0.073 | controlled |
| fGn H = 0.70 | phase_randomization | 0.043 | 0.027–0.067 | controlled |
| fGn H = 0.70 | permutation | 0.158 | 0.125–0.196 | inflated |
| fGn H = 0.85 | circular_shift | 0.052 | 0.035–0.079 | controlled |
| fGn H = 0.85 | phase_randomization | 0.058 | 0.039–0.085 | controlled |
| fGn H = 0.85 | permutation | 0.573 | 0.524–0.620 | inflated |
| fGn H = 0.95 | circular_shift | 0.060 | 0.041–0.088 | controlled |
| fGn H = 0.95 | phase_randomization | 0.052 | 0.035–0.079 | controlled |
| fGn H = 0.95 | permutation | 0.797 | 0.755–0.834 | inflated |
| GARCH(1,1) α = 0.10, β = 0.85 | circular_shift | 0.055 | 0.037–0.082 | controlled |
| GARCH(1,1) α = 0.10, β = 0.85 | phase_randomization | 0.062 | 0.043–0.091 | controlled |
| GARCH(1,1) α = 0.10, β = 0.85 | permutation | 0.043 | 0.027–0.067 | controlled |
| Random walk | circular_shift | 0.282 | 0.241–0.329 | inflated |
| Random walk | phase_randomization | 0.273 | 0.231–0.318 | inflated |
| Random walk | permutation | 0.885 | 0.850–0.913 | inflated |

### Power: a known edge when every channel has long memory

| Process | Null | True edge at true lag | Any other edge | Reverse edge |
| --- | --- | ---: | ---: | ---: |
| fGn H = 0.50 | circular_shift | 1.000 | 0.005 | 0.000 |
| fGn H = 0.50 | phase_randomization | 1.000 | 0.005 | 0.000 |
| fGn H = 0.70 | circular_shift | 0.990 | 0.020 | 0.005 |
| fGn H = 0.70 | phase_randomization | 0.995 | 0.010 | 0.000 |
| fGn H = 0.85 | circular_shift | 0.850 | 0.040 | 0.015 |
| fGn H = 0.85 | phase_randomization | 0.855 | 0.070 | 0.045 |
| fGn H = 0.95 | circular_shift | 0.500 | 0.135 | 0.105 |
| fGn H = 0.95 | phase_randomization | 0.480 | 0.080 | 0.060 |

### Preregistered hypotheses

| Id | Statement | Outcome |
| --- | --- | --- |
| H1 | The circular-shift null is controlled on every stationary process, including fractional Gaussian noise at H = 0.95 and GARCH(1,1) noise. | **supported** |
| H2 | The phase-randomization null is controlled on every stationary Gaussian process, which is fractional Gaussian noise at every H. | **supported** |
| H3 | The permutation null is inflated on every long-memory process, which is fractional Gaussian noise with H > 0.5. | **supported** |
| H4 | The circular-shift null is inflated on the random walk, because its stationarity assumption fails there. | **supported** |
| H5 | The phase-randomization null is inflated on the random walk, because its stationarity assumption fails there. | **supported** |
| H6 | Long memory costs power, not size: with the circular-shift null, the lag-2 edge at coupling 0.3 is recovered less often at the largest H than at H = 0.5. | **supported** |

6 of 6 hypotheses were supported, including the two (H4, H5) that predicted
the method's own failure.

## Reading the result

- **Stationarity is the load-bearing assumption, not short memory.** The
  circular-shift null holds 0.048–0.060 from white noise through H = 0.95,
  where the autocorrelation at lag 20 is still 0.63. It holds at 0.055 on
  GARCH(1,1), whose volatility clusters. On a random walk it reports a false
  edge in 28% of systems. The earlier qualitative caveat ("not a good fit for
  strongly nonstationary series") now has a number.
- **Independence is load-bearing.** The permutation null is inflated at every
  long-memory level: 0.158 at H = 0.70, 0.573 at 0.85, 0.797 at 0.95. It is
  fine on GARCH (0.043), which is uncorrelated, and it is the worst null on a
  random walk (0.885).
- **Gaussianity was not load-bearing here (exploratory).** Phase
  randomization Gaussianizes each series, yet it measured 0.062 on GARCH(1,1)
  noise. For independent channels the variance of a lagged cross-correlation
  factorizes, so heavy tails in each channel do not widen the null much at 512
  samples. This was not a preregistered hypothesis and one GARCH setting is
  not a theorem.
- **Long memory costs power, not size.** The lag-2 edge at coupling 0.3 is
  recovered in 100% of trials at H = 0.50 and 50% at H = 0.95 under the
  default null, while spurious edges rise to 13.5% and reverse edges to 10.5%.
  Phase randomization has the same power (48%) with fewer spurious (8.0%) and
  reverse (6.0%) edges in this run; the difference is within the intervals and
  is recorded as an observation, not a result.

## What this does and does not establish

It establishes properties of the implemented test on simulated processes with
a known truth: a family-wise error near nominal for the stationary nulls on
every stationary process tried, a measured failure on a nonstationary one, and
a measured power cost of long memory. It is evidence about the operators. It
is not evidence that DRR is useful on a real problem, and it does not touch the
external QBO comparison, whose reviewed claim status remains `not_supported`.
Only Gaussian long memory, one GARCH parameterization and one sample length
were tried, and the transfer-entropy backend was not measured.

## Open questions, in falsifiable form

1. Differencing the random walk should restore the circular-shift size to
   near 0.05. Untested.
2. A Granger-style design that conditions on each target's own past should
   remove most reverse edges under strong memory without losing the true edge.
   Untested.
3. The circular-shift seam should matter more as series shorten; the size at
   128 and 256 samples under H = 0.95 is unmeasured.
4. An amplitude-adjusted Fourier surrogate that keeps the marginal as well as
   the spectrum would separate the Gaussianity question from the spectral one.
   Not implemented.

## Provenance and external references

No external mathematics source was used in this study; every generator and
every rule is in this repository. The [R.A.I.N. Lab mathematics
scout](https://github.com/topherchris420/lop-nur-twin/blob/main/docs/MATH_PORTFOLIO.md)
remains available for lexical scouting of the
[OpenAI `math` manuscript catalogue](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/CONTENTS.md)
at its pinned revision, under the search phrase *dependent random process
correlation spectral bounds*. A matched title there is a reference to read, not
a theorem to transfer, and nothing from it was installed in this runtime. A
formal proof establishes only what was formalized; this simulation tests only
its own implementation and scenarios; external and physical claims need
separate evidence. Do not change a control policy, a registry verdict, a
preregistered criterion, or the meaning of an evidence class on the strength
of a matched phrase or of this study alone.
