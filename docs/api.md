# API Documentation

The public API is exported from `drr_framework`. The examples below assume:

```python
from drr_framework import DynamicResonanceRooting
```

## Core Orchestrator

### `DynamicResonanceRooting`

Coordinates resonance detection, rooting analysis, resonance-depth scoring,
QBist belief updates, and optional state-space diagnostics.

Important methods:

- `time_delay_embedding(data)`: build Takens-style delayed coordinates for a
  one-dimensional signal.
- `detect_resonances(data, method="fft", peak_height_ratio=0.1)`: detect
  dominant frequencies per dimension. `method` accepts `"fft"`, `"welch"`,
  `"wavelet"`, or `"markov"`.
- `calculate_resonance_depths(window_size=100)`: compute scalar depth values and
  store component details.
- `analyze_influence_network(... rooting_method="lagged_correlation",
  rooting_max_lag=None, rooting_n_surrogates=25, rooting_random_state=0,
  rooting_alpha=0.05, rooting_surrogate_method="circular_shift",
  rooting_correction="max_statistic")`: build a directed NetworkX graph from
  significant rooting edges only. The full rooting payload remains in
  `self.rooting_results`.
- `analyze_system(data, multivariate=False, window_size=100, state_space=True,
  method="fft", peak_height_ratio=0.1, rooting_method="lagged_correlation",
  rooting_max_lag=None, rooting_n_surrogates=25, rooting_random_state=0,
  rooting_alpha=0.05, rooting_surrogate_method="circular_shift",
  rooting_correction="max_statistic")`: run the end-to-end DRR workflow with
  the chosen spectral method and rooting configuration. For multivariate
  inputs, the returned `rooting_analysis` includes `score_matrix`, the
  compatibility alias `transfer_entropy`, raw and adjusted p-values,
  `candidate_edges`, `significant_edges`, `minimum_attainable_p_value`, and
  `inference_available`. If rooting fails, the facade stores a structured error
  record instead of omitting the section.

```python
from drr_framework import DynamicResonanceRooting

drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=200.0)
results = drr.analyze_system(
    data,
    multivariate=True,
    window_size=256,
    rooting_method="lagged_correlation",
    rooting_n_surrogates=25,
    rooting_random_state=42,
)
print(results["rooting_analysis"]["candidate_edges"])
print(results["rooting_analysis"]["significant_edges"])
```

## Resonance Modules

### `ResonanceDetector`

```python
from drr_framework import ResonanceDetector

result = ResonanceDetector().detect(
    signal,
    method="welch",
    sampling_rate=200.0,
    peak_height_ratio=0.2,
)
```

Returns method name, dominant frequencies, peak magnitudes, confidence estimates,
noise floor, frequency axis, and spectrum.

Supported methods:

- `"fft"`: single FFT magnitude spectrum.
- `"welch"`: Welch-averaged power spectral density (lower variance).
- `"wavelet"`: Morlet continuous wavelet transform. Peaks are selected from the
  time-averaged global wavelet spectrum, and the result additionally includes a
  `scalogram` (frequency x time power map) and `time_axis` for nonstationary
  signals. For very long inputs the scalogram is omitted (`None`) to bound
  memory; the spectrum and peaks are still returned.
- `"markov"`: KMeans state clustering with transition-matrix persistence
  analysis (returns a different, state-based schema).

### `DepthCalculator`

```python
from drr_framework import DepthCalculator

depth = DepthCalculator().calculate(
    signal,
    window_size=256,
    sampling_rate=200.0,
    resonance_frequencies=[12.5],
)
```

Returns `resonance_depth`, component scores, confidence interval, selected target
frequency, and legacy rolling standard deviation.

### `RootingAnalyzer`

```python
from drr_framework import RootingAnalyzer

rooting = RootingAnalyzer().analyze(
    multivariate_data,
    max_lag=4,
    n_surrogates=25,
    random_state=42,
    method="lagged_correlation",
    surrogate_method="circular_shift",
    correction="max_statistic",
)
```

Returns the canonical `score_matrix` plus the legacy `transfer_entropy` alias,
effective lags, raw and adjusted p-values, the exploratory `candidate_edges`
list, selected `significant_edges`, the selected correction rule, surrogate
method, inference flag, and the minimum attainable p-value. With
`n_surrogates=0`, off-diagonal p-values are `NaN`, `inference_available` is
`False`, and `significant_edges` is empty.

Rooting graphs use `significant_edges` only. `candidate_edges` are retained for
inspection, but they do not enter the NetworkX graph.

Three nulls are available, each named for the assumption it makes about a
series. `DRR_BENCHMARKS.md` reports the measured family-wise error of each.

| `surrogate_method` | Assumes | Keeps | Use when |
| --- | --- | --- | --- |
| `circular_shift` (default) | stationarity | each series' joint distribution, except at one seam | the series are stationary, whatever their memory or marginal |
| `phase_randomization` | a stationary linear Gaussian process | each series' periodogram, hence its sample autocorrelation at every lag | a Gaussian linear null is the right comparison and the seam is a concern |
| `permutation` | serial independence | each series' marginal distribution | the series are white noise; on AR(1) at 0.9 it reports a false edge 85% of the time |

All three assume at least stationarity. On a random walk every one is
inflated (family-wise error 0.27–0.89 at a nominal 0.05 in
`results/expected/dependent_noise_null_study.json`), so difference or detrend
such series first.

### Calibration

```python
from drr_framework.calibration import rooting_size, run_calibration_study

row = rooting_size(
    n_trials=200, n_samples=512, n_variables=3, ar_coefficient=0.9,
    max_lag=4, n_surrogates=99, alpha=0.05,
    surrogate_method="circular_shift", seed=1,
)
row["family_wise_error"]  # {"rate", "count", "trials", "ci95"}
```

`run_calibration_study()` runs the full size, power, memory, and depth study
behind `results/expected/rooting_calibration_study.json`. Use `rooting_size` and
`rooting_power` to check a configuration of your own (sample length, lags,
surrogate count) before trusting its p-values.

### Dependent-noise nulls

```python
from drr_framework.dependent_noise import make_null_process, null_false_alarms

draw = make_null_process({"family": "fractional_gaussian_noise", "hurst": 0.9}, 512, 3)
row = null_false_alarms(
    draw, n_trials=200, max_lag=4, n_surrogates=99, alpha=0.05,
    surrogate_method="circular_shift", seed=1,
)
row["family_wise_error"]  # {"rate", "count", "trials", "ci95"}
```

`drr_framework.dependent_noise` holds exact simulators for fractional Gaussian
noise (`simulate_fractional_gaussian_noise`, Davies–Harte embedding, any Hurst
exponent in (0, 1)), GARCH(1,1) noise (`simulate_garch`), random walks
(`simulate_random_walk`), and a lag-coupled long-memory triple
(`simulate_long_memory_pair`). `null_false_alarms` measures the family-wise
error of any null on any draw function; `coupled_edge_recovery` measures power.
`run_dependent_noise_study()` runs the preregistered study behind
`results/expected/dependent_noise_null_study.json`, whose hypotheses and decision
rule are fixed in `DEFAULT_CONFIG` before the rows are seen.

## Experimental mathematical features

Off by default and experimental. Definitions, assumptions and proofs are in
[mathematical-extensions.md](mathematical-extensions.md); measured behavior is in
[resonance-math-report.md](resonance-math-report.md).

```python
from drr_framework import DynamicResonanceRooting, analyze_mathematical_features

results = DynamicResonanceRooting(sampling_rate=250.0).analyze_system(
    data, multivariate=True, mathematical_features=True
)
features = results["mathematical_features"]
features["evidence_record"]["conclusion"]  # {"status": ..., "reasons": [...]}
```

`mathematical_features` also accepts a mapping of keyword arguments for
`analyze_mathematical_features(data, sampling_rate, *, column_names=None,
timestamps=None, phase_options=None, quality_thresholds=None,
negative_controls=19, recurrence_points=1000, random_state=0, data_source=None)`.
That function interpolates `NaN` gaps or irregular `timestamps` onto a regular
grid (recorded), runs the data-quality gate, tests every channel pair for phase
synchronization, runs the same test on phase-randomized copies as negative
controls, describes each channel with recurrence quantification and a
second-order linear recurrence, and returns a deterministic evidence record.
Every other result of `analyze_system` is unchanged. On the command line,
`drr-analyze ... --mathematical-features` adds the same section to the report.

### `PhaseSynchronyAnalyzer`

```python
from drr_framework.phase_sync import PhaseSynchronyAnalyzer

result = PhaseSynchronyAnalyzer(sampling_rate=250.0).analyze(data, column_names=("a", "b"))
pair = result["pairs"][0]
pair["status"]          # synchronized | not_detected | inconclusive | not_tested
pair["selected_ratio"]  # [n, m] with f_b / f_a close to n / m
pair["plv"], pair["adjusted_p_value"], pair["effective_samples"]
```

The p-value counts every circular shift of the seam-corrected phasor, so it is
deterministic and its smallest value is `1 / n_shifts`. Holm's procedure adjusts
across pairs. A pair that is not significant with fewer than
`min_effective_samples` (20) effective phase samples is `inconclusive`, never
evidence of no relationship. The module also exposes
`band_limited_analytic_signal`, `instantaneous_frequency`, `phase_locking_value`,
`farey_ratio_candidates`, `phase_shift_test`, `effective_phase_samples`,
`phase_difference_dynamics`, `windowed_phase_locking`, and `single_change_point`.

### Data quality and abstention

```python
from drr_framework.data_quality import assess_series_quality, regularize_series

values, report = regularize_series(raw_with_nans, timestamps=times, sampling_rate=250.0)
quality = assess_series_quality(values, 250.0, regularization=report)
quality["abstain"], quality["reasons"], quality["warnings"]
```

`detrended_fluctuation` returns the DFA exponent and `frequency_drift` the
quadratic-phase drift test. The thresholds are in `DEFAULT_THRESHOLDS`.

### Recurrence, geometry and evidence

- `recurrence.recurrence_quantification(x, ...)`: determinism, laminarity, mean
  diagonal length, recurrence period at a fixed recurrence rate.
- `recurrence.second_order_recurrence(x, sampling_rate)`: AR(2) roots, frequency,
  damping time, quality factor.
- `trajectory_geometry.oriented_area_matrix(data)`: equal to the antisymmetrized
  lag-one cross-covariance, so no information beyond second-order statistics.
- `math_evidence.MathematicalEvidenceRecord` and `MathematicalStatement`: typed
  records whose statements are `proven`, `empirical`, or `hypothesis`.

### Benchmark

`drr_framework.resonance_math_benchmark.run_benchmark(config=None, workers=1)`
runs the preregistered study behind `results/expected/resonance_math_benchmark.json`.
`run_condition` and `run_trial` rerun any row or trial with its seed schedule.

## Dataset Adapters

- `PolicyResonanceDataset`: prepares tabular policy observables.
- `SupervisoryPanelDataset`: prepares institution/date/metric panels.
- `load_policy_dataset`, `load_policy_dataset_from_sql`
- `load_supervisory_panel`, `load_supervisory_panel_from_sql`

These adapters preserve metadata such as source, transform, standardization, and
row counts so downstream reports carry provenance.

## Reporting

- `write_analysis_report`: writes JSON and Markdown reports.
- `write_tableau_artifacts`: exports tidy CSV artifacts for dashboards.
- `serialize_analysis_results`: converts NumPy, dataclass, and NetworkX objects
  into JSON-safe structures.
- `render_markdown_report`: renders caveated audience-specific summaries.

## State-Space Diagnostics

- `fit_resonance_state_space`
- `kalman_filter`
- `impulse_response`
- `analyze_resonance_state_space` — now runs a smoother by default
  (`smooth=True`, `smoother_method="koopman"`)

These functions expose transition, measurement, covariance, likelihood,
innovation, stability, and impulse-response diagnostics.

### Smoothers, simulation smoothers, and fast likelihoods

Ported from the New York Fed's
[`StateSpaceRoutines.jl`](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl)
as dependency-free NumPy, these estimate the latent resonance state *given the
whole sample*, recover the structural shocks that drove it, and provide fast /
nonlinear likelihood evaluation.

- `kalman_smoother(system, observations, method="koopman")` — dispatches to the
  disturbance (`"koopman"`) or Rauch–Tung–Striebel (`"hamilton"`) smoother.
- `koopman_smoother`, `hamilton_smoother` — return a `SmootherResult` with
  `smoothed_states`, `smoothed_covariances`, `smoothed_shocks`, and
  `smoothed_observables`.
- `durbin_koopman_smoother`, `carter_kohn_smoother` — draw posterior state and
  shock paths (`SimulationSmootherResult`), with `posterior_mean()` and
  `posterior_band(lower, upper)` for credible bands.
- `chandrasekhar_recursion(system, observations)` — exact log-likelihood of a
  stable time-invariant system, propagating small fixed-size matrices instead
  of the full covariance. `stationary_initialization(system)` returns the
  unconditional mean/covariance it uses.

```python
from drr_framework import kalman_smoother, durbin_koopman_smoother

smoothed = kalman_smoother(system, observations, method="koopman")
print(smoothed.smoothed_shocks)          # E[eps_t | y_1:T]

draws = durbin_koopman_smoother(system, observations, n_draws=500, random_state=0)
band = draws.posterior_band(5.0, 95.0)   # 90% credible band per state
```

### Nonlinear systems: tempered particle filter

For nonlinear resonance systems (chaotic flows, stochastic volatility), the
tempered particle filter (Herbst & Schorfheide, 2019) estimates the
log-likelihood when the Kalman filter no longer applies.

- `NonlinearStateSpaceModel(transition, measurement, shock_cov, measurement_cov, ...)`
  — the nonlinear model interface; `transition` and `measurement` are vectorised
  over the particle axis.
- `tempered_particle_filter(model, observations, n_particles=1000, r_star=2.0, ...)`
  — returns a `ParticleFilterResult` with per-period `log_likelihood`,
  `filtered_states`, effective sample sizes, tempering stages, and MH
  acceptance rates.
- `linear_gaussian_model(system)` — wraps a linear `StateSpaceSystem` as a
  `NonlinearStateSpaceModel` (useful for validation: the filter reproduces the
  Kalman log-likelihood up to Monte Carlo error).

```python
from drr_framework import NonlinearStateSpaceModel, tempered_particle_filter

model = NonlinearStateSpaceModel(
    transition=phi, measurement=psi,
    shock_cov=Q, measurement_cov=E,
    n_states=2, n_shocks=2, n_observables=1,
)
result = tempered_particle_filter(model, observations, n_particles=2000, random_state=0)
print(result.total_log_likelihood)
```

## Validation Readiness

- `generate_coupled_oscillator`
- `run_reproduction_experiment`
- `build_model_risk_card`
- `build_validation_readiness_packet`
- `run_event_backtest`
- `create_shadow_review_record`
- `append_shadow_review_record`
- `explain_supervisory_signal`

These APIs support reproducible smoke benchmarks and model-risk review packets.
