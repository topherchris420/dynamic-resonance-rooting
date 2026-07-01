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
  dominant frequencies per dimension.
- `calculate_resonance_depths(window_size=100)`: compute scalar depth values and
  store component details.
- `analyze_influence_network()`: build a directed NetworkX graph from rooting
  scores for multivariate data.
- `analyze_system(data, multivariate=False, window_size=100, state_space=True)`: run
  the end-to-end DRR workflow.

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
- `analyze_resonance_state_space`

These functions expose transition, measurement, covariance, likelihood,
innovation, stability, and impulse-response diagnostics.

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
