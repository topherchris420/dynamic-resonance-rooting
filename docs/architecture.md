# Architecture

Dynamic Resonance Rooting (DRR) is organized as a research pipeline: prepare a
time-indexed signal matrix, detect resonance structure, estimate directional
rooting relationships, compute resonance-depth metrics, attach state-space and
validation-readiness diagnostics, and export reviewer-readable artifacts.

## Overall DRR Workflow

```mermaid
flowchart LR
    A[Time-series or panel data] --> B[Dataset adapters]
    B --> C[DynamicResonanceRooting]
    C --> D[Resonance detection]
    C --> E[Rooting analysis]
    C --> F[Resonance depth]
    C --> G[State-space diagnostics]
    D --> H[Analysis report]
    E --> H
    F --> H
    G --> H
    H --> I[JSON, Markdown, CSV, plots]
```

## Data Pipeline

```mermaid
flowchart TD
    Raw[Raw observables] --> Validate[Column and type validation]
    Validate --> Transform[Optional transform: level, diff, pct_change, log_diff]
    Transform --> Impute[Optional interpolation]
    Impute --> Standardize[Optional standardization]
    Standardize --> Matrix[DRR numeric matrix]
    Matrix --> Metadata[Provenance metadata]
    Matrix --> Analysis[DRR analysis]
    Metadata --> Reports[Reports and exports]
    Analysis --> Reports
```

## Resonance Detection Process

```mermaid
flowchart TD
    Signal[Signal window] --> Center[Mean-center signal]
    Center --> Method{Detection method}
    Method --> FFT[FFT spectrum]
    Method --> Welch[Welch PSD]
    Method --> Wavelet[Morlet wavelet scalogram]
    Method --> Markov[Discrete Markov states]
    FFT --> Peaks[Peak selection]
    Welch --> Peaks
    Wavelet --> Global[Global wavelet spectrum]
    Global --> Peaks
    Markov --> Persistent[Persistent state detection]
    Peaks --> Confidence[Noise-floor confidence estimate]
    Persistent --> Result[Resonance result]
    Confidence --> Result
```

## Rooting Analysis

```mermaid
flowchart TD
    Matrix[Multivariate time series] --> Pairwise[Pairwise source-target scan]
    Pairwise --> LagSearch[Search lags 1..max_lag]
    LagSearch --> Backend{Backend}
    Backend --> Corr[Lagged-correlation fallback]
    Backend --> TE[Transfer entropy if pyinform is available]
    Corr --> Scores[Directed score matrix]
    TE --> Scores
    Scores --> Surrogates[Optional surrogate p-values]
    Surrogates --> Threshold[Edge threshold and alpha filter]
    Threshold --> Edges[Significant rooting edges]
```

## Phase Transition Detection

DRR currently exposes phase-transition evidence through changes in resonance
depth, spectral concentration, temporal persistence, rooting-edge structure, and
state-space stability diagnostics. The diagram below describes the intended
review pattern without claiming a validated universal phase-transition detector.

```mermaid
flowchart LR
    Windows[Rolling analysis windows] --> Metrics[Resonance-depth and component metrics]
    Windows --> Roots[Rooting-edge topology]
    Windows --> State[State-space stability diagnostics]
    Metrics --> Drift[Metric drift or abrupt change]
    Roots --> Drift
    State --> Drift
    Drift --> Review[Analyst/researcher review]
    Review --> Caveat[Candidate transition evidence, not causal proof]
```

## Package Architecture

```mermaid
flowchart TD
    API[src/drr_framework/__init__.py] --> Analysis[analysis.py]
    API --> Modules[modules.py]
    API --> Datasets[datasets.py]
    API --> Reporting[reporting.py]
    API --> StateSpace[state_space.py]
    API --> Validation[validation.py]
    API --> Supervision[supervision.py]
    Analysis --> Modules
    Analysis --> StateSpace
    Analysis --> Reporting
    Datasets --> Reporting
    Validation --> Modules
    Examples[examples/] --> API
    Tests[tests/] --> API
    Docs[docs/] --> API
```

## Design Boundaries

- DRR outputs are diagnostics for research and review, not causal proof.
- Transfer entropy is optional; lagged correlation remains the deterministic
  fallback and is reported explicitly.
- State-space diagnostics are Python-native and inspired by modeling discipline,
  not a copy of external DSGE implementations.
- Supervisory workflows are validation-ready artifacts, not validated
  supervisory methodology.
