# DRR validation report

This is the review memo for the evidence in this repository. It states what is
measured, where the measurement lives, and what the measurement does and does
not support. The machine-readable records are
[`results/expected/rooting_calibration_study.json`](results/expected/rooting_calibration_study.json)
and the preregistered
[`results/expected/qbo_structural_change_benchmark.json`](results/expected/qbo_structural_change_benchmark.json),
whose reviewed `claim_status` is `not_supported`. The tables are in
[`DRR_BENCHMARKS.md`](DRR_BENCHMARKS.md).

## Research question

What does DRR demonstrate that conventional analytics do not, and how strong is
the evidence?

## Short answer

- **The operators do what their definitions say.** Specification tests,
  invariance properties, and a JavaScript port that reproduces the Python
  results to 1e-9 check the implementation, not usefulness.
- **The rooting test is calibrated.** On independent series its family-wise
  false-alarm rate stays near the nominal 5% from white noise to AR(1) at 0.97,
  and it recovers a 0.3 lead–lag coupling almost every time at 512 samples.
- **Resonance depth is descriptive.** It separates a clear tone from white
  noise. It does not separate red noise from a weak resonance, and it is carried
  almost entirely by the power spectrum.
- **The one external test did not support the claim.** On NOAA stratospheric
  wind data, full DRR's holdout false-alarm rate was 0.272 against a
  preregistered 0.05. Nothing in this repository yet shows DRR adding
  information over conventional detectors on real data.

## 1. Implementation checks

| Property | Test |
| --- | --- |
| Depth is unchanged when the time unit changes (Hz ↔ cycles per month) | `tests/test_depth_invariants.py` |
| A clean tone scores the same wherever it falls on the FFT grid | `tests/test_depth_invariants.py` |
| Depth ignores gain and offset | `tests/test_depth_invariants.py` |
| Circular-shift surrogates are uniform over every valid shift configuration | `tests/test_calibration_study.py` |
| Committed reproduction and quickstart outputs match a fresh run | `tests/test_depth_invariants.py` |
| The browser engine matches the Python operators | `tests/test_web_engine_parity.py` |
| The preregistered artifact matches a fresh run and its own decision rule | `tests/test_structural_change_benchmark.py` |

## 2. Calibration (simulation with a known truth)

Measured by `python scripts/run_calibration_study.py`. Section numbers refer to
[`DRR_BENCHMARKS.md`](DRR_BENCHMARKS.md).

- **Size (§1).** Circular-shift family-wise error is 0.040–0.062 across AR(1)
  coefficients 0 to 0.97. The permutation null reaches 0.973 at 0.97 and is
  valid only for serially independent data.
- **Power (§2).** At 512 samples and lag 2, the exact edge is recovered in
  66% of trials at coupling 0.2 and 99.5% at 0.3.
- **Memory (§2).** When every channel is AR(1) at 0.9, the same 0.3 coupling
  is recovered in 38% of trials, and a reverse edge (the target appearing to
  lead its source) is reported in 15.5%. This is the documented boundary
  between lead–lag and causation.
- **Depth reference (§3).** White noise has a median depth of 0.14. AR(1) noise
  at 0.9 has 0.45, above a tone at −9 dB (0.36).

## 3. Placebo checks, corrected

`drr_framework.sensitivity_tests.run_placebo_and_null_tests` now uses a valid
Fourier surrogate. The surrogate is real-valued, keeps the power spectrum
exactly, and randomizes the phases. The earlier version drew phases without
Hermitian symmetry, kept only the real part of the result, and reported the
surrogate's own depth under the name `depth_reduction_from_shuffle`.

| Earlier statement | Measured now |
| --- | --- |
| "In pure Gaussian i.i.d. noise, DRR returns $D_R < 0.25$ with 0 false positive edges." | Depth of white noise has median 0.14 and 95th percentile 0.23, so it holds for the v2 score. Before the unit fix it failed at low sampling rates (mean 0.37 at 1 Hz). False edges occur at about the nominal 5% rate, as a calibrated test must. The claim of zero was never a property of the test. |
| "Phase-shuffled signals show a significant drop in phase coherence, confirming that $D_R$ isolates phase organization rather than static power spectral density." | Not supported. Phase randomization lowers the depth of a tone by about 0.005. Depth is carried by the spectrum. |

## 4. External evidence

The preregistered QBO comparison asks whether full DRR flags repeated,
published disruptions that rolling volatility, rolling correlation, and a VAR
residual miss, without exceeding a 0.05 holdout false-alarm rate. It does not:
the rate is 0.272, and there are no unique full-DRR detections. The status is
`not_supported`, confined to equatorial stratospheric zonal wind. See
[`docs/external-evidence.md`](docs/external-evidence.md).

## 5. Withdrawn statements

The previous version of this memo reported incremental early-warning lead time
of "+6 to +10 days" ahead of rolling-volatility and VAR models across the
2007–2008, March 2020, and spring 2023 financial stress episodes. It also
reported institution-tier sensitivities and a "+185 bps" regime transition. No
code or data in this repository produced those figures. They are withdrawn.
Any future claim about financial stress episodes needs its own preregistered
protocol, like the QBO study.

## 6. Deliverables

| File | Role |
| --- | --- |
| `src/drr_framework/calibration.py`, `scripts/run_calibration_study.py` | Monte Carlo size, power, memory, and depth study |
| `src/drr_framework/external_benchmark/`, `scripts/run_structural_change_benchmark.py` | Preregistered external comparison |
| `src/drr_framework/sensitivity_tests.py` | Parameter sweeps and placebo checks |
| `src/drr_framework/evidence_card.py`, `scripts/run_drr_validation.py` | Evidence cards whose statistics come from the measured run |
| `assets/web-demo/drr-engine.js`, `index.html` | Live browser lab running the tested port |
| `DRR_BENCHMARKS.md`, `DRR_LIMITATIONS.md` | Tables and known limits |
