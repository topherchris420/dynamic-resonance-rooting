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
- `analyze_influence_network()`: build a directed NetworkX graph from rooting
  scores for multivariate data.
- `analyze_system(data, multivariate=False, window_size=100, state_space=True,
  method="fft", peak_height_ratio=0.1)`: run the end-to-end DRR workflow with
  the chosen spectral method.

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
)
```

Returns directed score matrices, effective lags, p-values, edge threshold, and
significant-edge records.

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
