# Phase and harmonic mathematics for DRR: performance and limitations

**Record:** [`results/expected/resonance_math_benchmark.json`](../results/expected/resonance_math_benchmark.json)
with a [rendered companion](../results/expected/resonance_math_benchmark.md).
**Definitions and proofs:** [`docs/mathematical-extensions.md`](mathematical-extensions.md).
**Status of the features:** experimental, off by default.

Every number below is read from the committed artifact.
`tests/test_resonance_math_benchmark.py` recomputes every rate, interval and
hypothesis outcome from one character per trial, and reruns a prefix of
committed rows exactly. Each table row quoted here is a line of the rendered
report; the same test file checks that.

## The answer in brief

On simulated channel pairs with a known relationship, held out from
development and tested against nine hypotheses fixed before the held-out run:

- **The new phase-synchrony test kept its false-relationship rate controlled
  on every stationary unrelated condition** (0.003–0.058 at a nominal 0.05,
  400 pairs per row). This includes coherent oscillators at one frequency,
  where existing DRR rooting reported a false relationship in 14.5% of pairs.
- **It detects relationships existing DRR cannot see.** On 1:2 and 2:3
  harmonic coupling it detected 35.5% and 28.0% of pairs; DRR rooting detected
  5.0% and 2.5%, which is chance.
- **It is the weaker test for ordinary 1:1 coupling.** Its mean detection rate
  over six 1:1 conditions was 0.263, against 0.441 for DRR rooting. The
  preregistered hypothesis that it would come within 0.10 (H9) is
  **not supported**.
- **Combining the two with a Bonferroni split did not help.** The union lost
  power on 1:1 coupling (mean 0.237), kept only part of the harmonic gain
  (0.180 and 0.225), and inherited DRR rooting's inflation on coherent
  oscillators (0.092).
- **Textbook thresholds fail on oscillations.** Pearson correlation with a
  Fisher z test reported false relationships in up to 89.5% of independent
  pairs, Welch coherence with its analytic threshold in up to 61.8%, and the
  phase-locking value with a Rayleigh threshold in up to 99.8%.
- **The data-quality gate abstained on 100% of random walks and 88.5% of
  shared frequency sweeps,** the two nonstationarities it targets, and on
  0.5–9% of stationary conditions. It does not see a shared amplitude trend,
  which inflated DRR rooting (0.150) but not the phase test (0.037).
- **Three new facts about existing DRR rooting:** its null is inflated on
  coherent oscillators (0.145), on independent channels sampled with a common
  jittered clock and interpolated (0.072), and on a shared amplitude trend
  (0.150). On the held-out 1:1 couplings without delay at 250 Hz it named the
  true direction in only 40.5–51.8% of its detections.

Eight of nine hypotheses were supported, including two that predicted
failures: of existing DRR rooting (H6) and of the new test without its gate
(H7). The one that was not supported (H9) was about the new test's power.

## Heilmeier catechism

**What are we trying to do?** Tell from two recorded rhythms whether one keeps
time with the other, including when one runs at a simple multiple of the
other's pace. Do it without being fooled by rhythms that merely look alike,
and say "cannot tell" when the record cannot support an answer.

**How is it done today, and what are the limits?** DRR asks whether one
series' recent values track another's: lagged correlation or transfer entropy
against circular-shift surrogates. That cannot see a 1:2 relationship, whose
waveforms are uncorrelated. Conventional practice adds correlation, coherence
or a phase-locking value with textbook thresholds. Those thresholds assume
independent samples, which oscillations violate; the held-out rows above show
by how much. DRR's own null had been measured on noise processes but not on
narrowband oscillators, and it inflates on the coherent ones. Nothing checked
stationarity automatically, although the dependent-noise study had shown that
every null fails on a random walk.

**What is new, and why should it work?** A phase test whose statistic sees
`n:m` locking, and whose randomization over every circular shift is made exact
by a small phase ramp that removes the seam a shift creates (Propositions 1 and
2 of the definitions page). Pairs with too little phase variability are labelled
inconclusive. A gate tests for random-walk scaling and frequency drift before
any relationship is declared. Each result carries an evidence record that
separates proven statements from measurements and from untested hypotheses.
The exactness argument gave a reason to expect calibration. The development
runs, and then the held-out run, measured it.

**Who cares?** Anyone using DRR on oscillatory records: physics and sensing
laboratories, physiological rhythms, anything where two clocks may or may not
be coupled. Detecting harmonic relationships is new; not reporting false ones
on coherent or trending oscillators protects every downstream conclusion.

**What are the risks?** That stationarity fails in ways the gate cannot see;
the shared amplitude trend is one. That the narrowband assumption does not
hold. That exactness costs resolution: p-values cannot fall much below the
phase-coherence time divided by the record length. That everything here is
synthetic. That the ratio is chosen from the Welch peak, which only
approximates the condition under which the test is exact.

**What does it cost?** No new dependency. Per pair, the phase test took a
median 8 ms, against 175 ms for DRR rooting with 99 surrogates and 132 ms for
the transfer-entropy backend (medians over the held-out conditions, most of
them 2048 samples, on the generating machine with four workers). The whole
benchmark took 831 seconds there.

**How long will it take?** This slice is complete. Each next step in the
recommendations below needs its own preregistered study, like this one.

**What are the mid-term and final exams?** The mid-term was this held-out
run: eight of nine preregistered hypotheses supported, one not. The final exam
is a preregistered study on real data in a domain adapter, with negative
controls, of the kind the QBO comparison set up for structural change. It has
not been run.

## Held-out results

### False relationships on stationary unrelated pairs

400 pairs per row. Bold marks inflation: the Wilson 95% lower bound exceeds 0.05.

| Condition | `drr_rooting` | `drr_rooting_te` | `pearson_analytic` | `coherence_analytic` | `coherence_shift` | `plv_rayleigh` | `phase_sync` | `oriented_area` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent, similar frequencies | 0.048 | 0.037 | **0.595** | **0.128** | 0.043 | **0.970** | 0.058 | 0.065 |
| Independent, identical frequency, coherent | **0.145** | **0.077** | **0.895** | **0.618** | 0.068 | **0.998** | 0.040 | 0.070 |
| Independent, exact 1:2 frequencies | 0.043 | 0.058 | 0.062 | **0.077** | 0.048 | 0.000 | 0.052 | 0.065 |
| Independent, exact 2:3 frequencies | 0.045 | 0.060 | **0.185** | 0.048 | 0.048 | **0.158** | 0.050 | 0.050 |
| Independent AR(1), coefficient 0.95 | 0.058 | 0.037 | **0.627** | **0.072** | 0.040 | **0.955** | 0.003 | 0.050 |
| Phase-randomized coupled 1:1 (negative control) | 0.065 | 0.037 | **0.600** | **0.220** | 0.068 | **0.980** | 0.045 | **0.087** |
| Independent, 15% missing | 0.035 | 0.045 | **0.618** | **0.120** | 0.040 | **0.975** | 0.035 | 0.045 |
| Independent, irregular timestamps | **0.072** | 0.040 | **0.645** | **0.105** | 0.060 | **0.990** | 0.052 | 0.040 |
| Independent, 50 Hz, 512 samples | 0.065 | 0.065 | **0.425** | **0.077** | 0.028 | **0.920** | 0.030 | 0.050 |

Welch coherence against every circular shift (`coherence_shift`) was
controlled on every row here, as was the phase test. The statistic matters less
than the null: the same coherence with its textbook threshold was inflated on
eight of nine rows.

### Detection on related pairs

200 pairs per row.

| Condition | `drr_rooting` | `drr_rooting_te` | `coherence_shift` | `phase_sync` | `phase_sync_1to1` | `oriented_area` | `drr_plus_phase` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled 1:1 | 0.610 | 0.125 | 0.375 | 0.315 | 0.320 | 0.045 | 0.300 |
| Coupled 1:2 (harmonic) | 0.050 | 0.045 | 0.070 | 0.355 | 0.020 | 0.050 | 0.180 |
| Coupled 2:3 (harmonic) | 0.025 | 0.045 | 0.045 | 0.280 | 0.030 | 0.050 | 0.225 |
| Coupled 1:1, 0.15 s delay | 0.115 | 0.020 | 0.110 | 0.050 | 0.050 | 0.045 | 0.030 |
| Coupled 1:1, amplitude modulated | 0.280 | 0.105 | 0.205 | 0.215 | 0.245 | 0.020 | 0.195 |
| Coupled 1:1, 15% missing | 0.605 | 0.070 | 0.400 | 0.340 | 0.345 | 0.035 | 0.345 |
| Coupled 1:1, irregular timestamps | 0.590 | 0.055 | 0.410 | 0.365 | 0.365 | 0.025 | 0.350 |
| Coupled 1:1, 50 Hz, 512 samples | 0.445 | 0.120 | 0.285 | 0.290 | 0.335 | 0.020 | 0.205 |

The held-out couplings are harder than the development ones (more phase
diffusion, more noise, shorter records): the phase test detected the
development 1:2 coupling in 86% of pairs and the held-out one in 35.5%. On the
held-out 1:1 coupling, 60.5% of the phase test's trials ended *inconclusive*:
not significant, with too few effective phase samples to count as evidence of
no relationship. It did not call those pairs unrelated.

### Nonstationary pairs and the gate

| Condition | Related | `drr_rooting` | `phase_sync` | `drr_plus_phase` | Gate abstains | `phase_sync_gated` | `drr_plus_phase_gated` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled 1:1, shared amplitude trend | yes | 0.580 | 0.295 | 0.375 | 0.070 | 0.285 | 0.360 |
| Coupled 1:1, swept 6-8 Hz | yes | 0.995 | 0.845 | 0.905 | 0.715 | 0.260 | 0.245 |
| Independent, shared amplitude trend | no | **0.150** | 0.037 | **0.072** | 0.072 | 0.037 | 0.068 |
| Independent random walks | no | **0.188** | 0.000 | **0.155** | 1.000 | 0.000 | 0.000 |
| Independent, shared 6-8 Hz sweep | no | **0.098** | **0.077** | **0.080** | 0.885 | 0.010 | 0.005 |

The gate abstains on a coupled sweep too (71.5%). That is the intended
behavior: under drift no shift-based null can separate coupling from a shared
sweep, so the honest answer is no answer.

### Preregistered hypotheses

| Id | Statement | Rule | Outcome |
| --- | --- | --- | --- |
| H1 | The phase-synchrony test is controlled on every stationary unrelated held-out condition, including the phase-randomized negative control and independent oscillators at identical or harmonically related frequencies. | phase_sync is controlled on every stationary unrelated condition | **supported** |
| H2 | Each conventional baseline with a textbook threshold (Pearson correlation, Welch coherence, Rayleigh phase-locking value) is inflated on at least one stationary unrelated held-out condition. | each of pearson_analytic, coherence_analytic, plv_rayleigh is inflated on at least one stationary unrelated condition | **supported** |
| H3 | On every held-out harmonic coupling (1:2 and 2:3), the phase-synchrony test detects the relationship at least 0.25 more often than existing DRR rooting. | phase_sync detection rate minus drr_rooting detection rate >= 0.25 on every harmonic condition | **supported** |
| H4 | The ratio search carries H3: with the ratio fixed at 1:1 the phase test detects each held-out harmonic coupling in at most 15% of trials. | phase_sync_1to1 detection rate <= 0.15 on every harmonic condition | **supported** |
| H5 | The seam correction is load-bearing: without it the phase test is inflated on the held-out coherent identical-frequency condition, where the corrected test is controlled. | phase_sync_no_seam is inflated and phase_sync controlled on the coherent condition | **supported** |
| H6 | Existing DRR rooting is inflated on the held-out coherent identical-frequency condition. | drr_rooting is inflated on the coherent condition | **supported** |
| H7 | Without the gate, the phase-synchrony test is inflated on the held-out shared-sweep condition, because its stationarity assumption fails there. | phase_sync is inflated on common_chirp | **supported** |
| H8 | The data-quality gate abstains in at least 80% of trials on the held-out random walks and shared sweep, and in at most 10% of trials on every stationary held-out condition. | gate abstention >= 0.8 on random_walk and common_chirp and <= 0.1 on every stationary condition | **supported** |
| H9 | On the stationary held-out 1:1 couplings, the phase-synchrony test's mean detection rate is no more than 0.10 below existing DRR rooting's. | mean phase_sync detection rate on the stationary 1:1 couplings is at least the mean drr_rooting rate minus 0.10 | **not_supported** |

## What did not work, or did not improve

- **Power on 1:1 coupling (H9).** The phase test ignores amplitude and gives
  up resolution for exactness; lagged correlation uses both. For ordinary 1:1
  coupling DRR rooting remains the more powerful test.
- **The Bonferroni union.** Each test gets `alpha / 2`. The phase test's
  attainable p-values are coarse, so many of its detections fall between
  0.025 and 0.05 and are lost. A single null for both statistics would avoid
  the split; it does not exist yet.
- **The seam correction costs power.** Without it the test detected more
  often (0.440 against 0.315 on the 1:1 coupling) but reported false
  relationships in 17.0% of coherent unrelated pairs. Trimming filter edges
  and band-pass filtering are also load-bearing for calibration; the ablation
  table in the artifact has every row.
- **The gate's blind spots.** It sees random-walk scaling and frequency drift.
  It did not see a shared amplitude trend, and it abstained on up to 9% of
  stationary held-out pairs, because the drift statistic spreads wider than the
  unit spread its Brownian approximation assumes (1.1–1.3 in development).
- **Geometry adds nothing new.** The oriented area is provably a function of
  the lag-one cross-covariance. It had almost no power on the in-phase-locked
  couplings (2.0–5.0%). Its sign followed the phase lag, not the coupling:
  every development detection (17 of 17) named the driven channel as the
  leader, and on the held-out 1:1 coupling 0 of 9 named the driver.
- **Recurrence and linear-recurrence features are fragile.** Recurrence
  determinism of white noise is 0.57 with a one-sample embedding delay (so
  the default refuses it), and the AR(2) frequency of a unit tone moves by
  about 20% with noise of standard deviation 0.1. They are reported as
  descriptions, and the benchmark does not use them to declare relationships.
- **Direction.** On the held-out 1:1 couplings without delay DRR rooting named
  the true direction in 40.5–51.8% of detections (73.9% with a 0.15 s delay),
  against 86–95% in development at 100 Hz. At 250 Hz its lags of 1–4 samples
  cover a small fraction of a 7 Hz cycle, and a lead of `d` samples cannot be
  told from a lag of `P - d`. Direction claims from lagged correlation on
  oscillations depend on the sampling rate.

## Limitations

- Every result is synthetic: phase oscillators with Wiener phase noise,
  Kuramoto coupling, white measurement noise, AR(1) noise and random walks. No
  result transfers to a real domain without a study of that domain.
- The phase test is undirected and statistical. A synchronized pair is
  consistent with coupling, a common driver, or a shared nonstationarity the
  gate missed. It is never evidence of a mechanism.
- Exactness holds for independent, identically distributed phase increments.
  For band-pass filtered phases it is approximate; its size was measured on
  the conditions here and nowhere else.
- The inconclusive threshold (20 effective phase samples) and the gate's
  thresholds are judgments, set on development data.
- Linear interpolation of gaps and jittered timestamps is a modelling choice.
  It recorded no inflation for the phase test here, and slight inflation for
  DRR rooting with a common jittered clock.

## How the work was done

The development conditions were run repeatedly while the methods and
thresholds were set. Three changes came from those runs. A segment-based drift
check abstained on up to 33% of stationary development pairs and was replaced by
the quadratic-phase test. The default recurrence delay was floored at two
samples. The pass band was kept at the Morlet value (1/6) after a comparison
with 1/4 showed no consistent gain. One threshold choice, a smaller resolution
floor for the old drift check, was made knowing the held-out sampling rates
and sweep widths, though not any held-out result; that check was then replaced
by the quadratic-phase test.

The held-out conditions, methods, thresholds and hypotheses were committed in
`1f4f876` with their SHA-256 fingerprint before the held-out conditions were
first run. The artifact records that its configuration still matches that
fingerprint. The held-out split was first run at `1f4f876`, and its outcomes
were recorded in `600cbbe`. After that run, the phase test gained a check that
refuses a constant channel, the report began listing methods in a fixed order,
and the documented argument for how the ratio is chosen and two literature
references were corrected. The artifact was then regenerated with the final
code, under the same Python and NumPy versions, so that it stores only the
per-trial codes: the stored rates, which the tests already recomputed from those
codes, had pushed the file past the 500 KB limit of the repository's pre-commit
large-file check. All 207,000 per-trial codes, every
other field of every row, every hypothesis outcome and every integrity check
matched the first run exactly. Only the wall-clock times differ; the figures
above are from the regenerated run.

## Recommendations for R.A.I.N., Anna, and CIRCLE

This repository describes R.A.I.N. only through its mathematics scout (see
[`MATH_RESEARCH.md`](MATH_RESEARCH.md)), and it says nothing about Anna or
CIRCLE. The recommendations are therefore limited to interface contracts any
consumer can adopt; their owners should check them before anything is built.

1. **Exchange evidence records, not scores.** The unit of exchange should be
   the record (`drr-math-evidence-v1`): definitions, provenance hashes,
   assumption checks, statements and conclusion travel together.
2. **Respect the statement classes.** A `proven` statement can be cited as
   mathematics with its reference. An `empirical` statement must be cited
   with its artifact and scope. A `hypothesis` must not be asserted. A matched
   reference from the R.A.I.N. scout enters as a hypothesis until a proof or a
   measurement exists.
3. **Never turn `inconclusive` into "no relationship".** A consumer that
   aggregates DRR results should carry the three outcomes, and abstentions,
   separately.
4. **Treat phase synchronization as undirected and non-causal,** and treat
   lagged-correlation direction on oscillations as dependent on the sampling
   rate.
5. **Reuse the preregistration pattern.** A configuration with frozen
   hypotheses, a SHA-256 fingerprint committed before the held-out run, and
   rows stored one character per trial lets any laboratory audit a claim.
6. **Do not reimplement these operators elsewhere.** Call DRR, and keep the
   version identifiers (`drr_phase_sync_v1`, `drr_data_quality_v1`) with
   every result.

## Files

| File | Role |
| --- | --- |
| `src/drr_framework/phase_sync.py` | `n:m` phase synchronization, seam-corrected exact test, phase dynamics |
| `src/drr_framework/data_quality.py` | DFA, frequency-drift test, regularization, the gate |
| `src/drr_framework/recurrence.py` | delay embedding, recurrence quantification, AR(2) recurrences |
| `src/drr_framework/trajectory_geometry.py` | oriented area, its identity, cyclic order |
| `src/drr_framework/math_evidence.py` | evidence records, statement classes, feature definitions |
| `src/drr_framework/math_features.py` | the opt-in orchestrator with negative controls |
| `src/drr_framework/resonance_math_benchmark.py`, `scripts/run_resonance_math_benchmark.py` | the preregistered benchmark |
| `src/drr_framework/analysis.py` | `analyze_system(mathematical_features=...)`; embedding delegates to `delay_embedding` |
| `src/drr_framework/csv_analysis.py` | `drr-analyze --mathematical-features` |
| `src/drr_framework/__init__.py` | lazy exports of the new entry points |
| `tests/test_phase_sync.py`, `tests/test_data_quality.py`, `tests/test_recurrence_geometry.py`, `tests/test_math_evidence.py`, `tests/test_resonance_math_benchmark.py` | unit, integration, regression and deterministic-replay tests |
| `tests/test_depth_invariants.py` | compares edges one at a time; pytest 9.2 rejects `approx` on a list of dicts |
| `results/expected/resonance_math_benchmark.{json,md}` | the artifact |
| `docs/mathematical-extensions.md`, this report, and updates to the README, benchmarks, limitations, validation report, API, architecture, reproducibility and changelog | documentation |

## Reproduce

```bash
python scripts/run_resonance_math_benchmark.py --output-dir results/expected --workers 4
```

Rows are identical for any number of workers. `--trials-scale 0.1` gives a
quick look; the committed run uses 1.
