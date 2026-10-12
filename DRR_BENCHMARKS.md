# DRR benchmarks

Every number in this file is produced by code in this repository and recorded
in a committed artifact. There are three:

| Artifact | What it measures | Regenerate |
| --- | --- | --- |
| [`results/expected/rooting_calibration_study.json`](results/expected/rooting_calibration_study.json) | Size, power, and reference distributions of the DRR operators on simulated systems with a known truth | `python scripts/run_calibration_study.py --output-dir results/expected` |
| [`results/expected/dependent_noise_null_study.json`](results/expected/dependent_noise_null_study.json) | Family-wise error of each surrogate null under long memory, volatility clustering, and a random walk, power under long memory, and six preregistered hypotheses | `python scripts/run_dependent_noise_study.py --output-dir results/expected` |
| [`results/expected/qbo_structural_change_benchmark.json`](results/expected/qbo_structural_change_benchmark.json) | The preregistered external comparison on NOAA CPC QBO zonal wind | `python scripts/run_structural_change_benchmark.py --output-dir results/expected` |

`tests/test_calibration_study.py`, `tests/test_dependent_noise_study.py`, and
`tests/test_structural_change_benchmark.py` rerun parts of all three and fail if
a committed number no longer reproduces.

## 1. Does the rooting test keep its promised false-alarm rate?

Three independent AR(1) channels have no directed relationship, so any edge the
test reports is a family-wise false alarm. 400 simulated systems per row, 512
samples, lags 1–4, 99 surrogates, alpha = 0.05, max-statistic correction.

| AR(1) coefficient | Surrogate null | Family-wise error | 95% interval |
| ---: | --- | ---: | --- |
| 0.00 | circular_shift | 0.062 | 0.043–0.091 |
| 0.50 | circular_shift | 0.040 | 0.025–0.064 |
| 0.90 | circular_shift | 0.048 | 0.031–0.073 |
| 0.97 | circular_shift | 0.062 | 0.043–0.091 |
| 0.00 | permutation | 0.050 | 0.033–0.076 |
| 0.50 | permutation | 0.233 | 0.194–0.276 |
| 0.90 | permutation | 0.850 | 0.812–0.882 |
| 0.97 | permutation | 0.973 | 0.951–0.985 |

**Reading.** The default circular-shift null stays near 5% across the whole range
of autocorrelation. The legacy permutation null is valid only for white noise:
shuffling time points erases each series' memory, so the null is too narrow and
false alarms reach 97%. Use `surrogate_method="permutation"` only for series you
know to be serially independent.

**Caveat.** When a series is short relative to its autocorrelation time, the
circular-shift test runs slightly above nominal. At 256 samples and AR 0.9
(effective sample size about 13) it measured about 0.06 over 3,200 trials.

**History.** Before this fix, the circular-shift surrogates split their slack
with a multinomial draw that placed every surrogate near evenly spaced offsets.
The surrogates were near copies of one another, and the same test rejected 14%
of true nulls on white noise and 19–30% on red noise.

## 2. Does it find real edges?

An AR(0.5) source drives a target at lag 2, and a third channel is an
independent distractor. Coupling is the correlation between the target and the
lagged source. 200 trials per row.

| Coupling | True edge at true lag | True edge at any lag | Any other edge |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.000 | 0.005 | 0.025 |
| 0.10 | 0.105 | 0.105 | 0.065 |
| 0.15 | 0.315 | 0.325 | 0.040 |
| 0.20 | 0.660 | 0.660 | 0.025 |
| 0.30 | 0.995 | 0.995 | 0.035 |
| 0.50 | 1.000 | 1.000 | 0.000 |

**Reading.** With 512 samples, a coupling of 0.3 is found almost every time and
0.2 about two times in three. "Any other edge" is the rate of spurious extra
edges when a real one exists.

### Memory costs power and adds spurious edges

Coupling 0.3 at lag 2 as above, but every channel is AR(1) with the coefficient
shown. 200 trials per row. "Reverse edge" is the target reported as leading
its own source.

| AR(1) coefficient | True edge at true lag | Any other edge | Reverse edge |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.000 | 0.000 | 0.000 |
| 0.50 | 0.995 | 0.010 | 0.000 |
| 0.80 | 0.760 | 0.130 | 0.095 |
| 0.90 | 0.380 | 0.185 | 0.155 |

**Reading.** The same 0.3 coupling that is found 99.5% of the time at AR 0.5 is
found 38% of the time at AR 0.9. Surrogates that preserve strong memory give a
wider null. Spurious edges rise to 18.5%, and most of them (15.5 points) are
the reverse edge: the target appears to lead its own source because the
source's past is visible in both. Lagged correlation measures lead–lag, not
causation. Conditioning on each target's own past (a Granger-style
design) would address this, and DRR does not currently do it.

## 3. What does a resonance-depth value mean?

Depth (`drr_composite_v2`) is a descriptive score, not a test. Median and 5–95%
range over 200 draws, window 256, tones at 0.0625 cycles per sample in
unit-variance white noise.

| Signal | Parameter | 5% | Median | 95% |
| --- | --- | ---: | ---: | ---: |
| ar1_noise | AR(1) coefficient 0.00 | 0.122 | 0.144 | 0.227 |
| ar1_noise | AR(1) coefficient 0.50 | 0.148 | 0.226 | 0.327 |
| ar1_noise | AR(1) coefficient 0.90 | 0.286 | 0.447 | 0.608 |
| tone_in_white_noise | amplitude 0.25 (SNR -15.1 dB) | 0.126 | 0.153 | 0.269 |
| tone_in_white_noise | amplitude 0.50 (SNR -9.0 dB) | 0.193 | 0.356 | 0.459 |
| tone_in_white_noise | amplitude 1.00 (SNR -3.0 dB) | 0.533 | 0.580 | 0.641 |

**Reading.** Depth separates a clear tone from white noise. It does not separate
red noise from a weak resonance: AR(1) noise at 0.9 scores higher than a tone at
−9 dB. A Fourier surrogate of a tone, which keeps the power spectrum and
randomizes the phases, lowers depth by only about 0.005. Depth is carried almost
entirely by the spectrum. Compare a depth value against a surrogate or
red-noise reference, not against a fixed cutoff.

## 4. The preregistered external comparison

The question, frozen before the run: does full DRR flag the 2015–2016 and
2019–2020 disruptions of the quasi-biennial oscillation that rolling volatility,
rolling correlation, and a VAR residual miss, without a holdout false-alarm rate
above 0.05? Each detector's threshold was set for a 5% false-alarm rate on
months before 2011 and then applied unchanged to the 2011–2026 holdout.

| Model | Role | Holdout false-alarm rate |
| --- | --- | ---: |
| rolling_volatility | conventional | 0.572 |
| rolling_correlation | conventional | 0.461 |
| var_residual | conventional | 0.072 |
| drr_spectral | ablation | 0.272 |
| drr_full | DRR | 0.272 |

The reviewed `claim_status` is `not_supported`, with reason code
`drr_false_alarm_rate`. Full DRR's holdout false-alarm rate exceeds the
preregistered tolerance, and it has no disruption hit that a calibrated
conventional detector misses. The result is confined to equatorial
stratospheric wind. It neither supports nor refutes DRR in another domain. The
[protocol](docs/external-evidence.md) and [memo](DRR_VALIDATION_REPORT.md) have
the details.

## 5. Which null assumptions survive long memory?

Each surrogate null is an assumption about a series. The circular-shift null
assumes stationarity; phase randomization assumes a stationary linear Gaussian
process; permutation assumes serial independence. The processes below break
those assumptions by degrees: fractional Gaussian noise (fGn) with Hurst
exponent H has autocorrelation that decays like k^(2H−2) and never sums to a
finite total, GARCH(1,1) is uncorrelated but its volatility clusters, and the
random walk is not stationary. Three independent channels, 512 samples, lags
1–4, 99 surrogates, alpha = 0.05, 400 systems per row. *Controlled* means the
Wilson 95% lower bound is at most 0.05; *inflated* means it is above. The rule
was fixed before the run.

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

**Reading.** Stationarity is the assumption that matters, not short memory.
The default null holds 0.048–0.060 through H = 0.95, where the lag-20
autocorrelation is still 0.63, and 0.055 on GARCH(1,1). Phase randomization
matches it, including on GARCH (0.062) even though it Gaussianizes each series.
The permutation null is inflated at every long-memory level and reaches 0.797.
On a random walk every null fails: 0.282, 0.273, and 0.885. Difference or
detrend nonstationary series before rooting.

### Long memory costs power, not size

Channel 0 drives channel 1 at lag 2 with coupling 0.3, a third channel is an
independent distractor, and all three are fGn at the H shown. 200 trials per
row. The size rows above at the same H are the negative control.

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

**Reading.** The same 0.3 coupling that is always found on white noise is found
half the time at H = 0.95, and spurious edges rise with it, most of them the
reverse edge. Phase randomization reports fewer spurious edges at H = 0.95 in
this run; the gap is within the intervals and is an observation, not a
preregistered result.

**Hypotheses.** Six were fixed before the run and 6 were supported, including
the two that predicted the method's own failure on a random walk. The
statements, rules, and outcomes are in [`docs/MATH_RESEARCH.md`](docs/MATH_RESEARCH.md)
and in the artifact.

## Withdrawn material

Earlier versions of this file carried tables of AUROC, lead time, and error
rates for the 2007–2008, March 2020, and spring 2023 financial stress episodes,
along with institution-tier and rate-shock findings. No code or data in this
repository produced those numbers, so they were removed in favor of the measured
results above. They remain in the git history. No supported claim about
financial stress episodes currently exists.
