# DRR Quantitative Research Lab Architecture & Integration Guide

## Overview

The **DRR Quant Research Lab** is a quantitative finance application layer built around Dynamic Resonance Rooting (DRR). Its purpose is to test whether DRR produces useful, reproducible structural representations of financial markets and whether those representations contain incremental information beyond conventional financial features out of sample.

The central scientific question is:
> *Does DRR describe market dynamical structure in a way that improves prediction, regime inference, risk management, or portfolio decisions out of sample?*

This is a falsifiable research project—a null result is scientifically acceptable.

---

## Architectural Principles

1. **DRR is the Primary Scientific Framework**: External quantitative packages are optional adapters around DRR.
2. **Clean Conceptual Data Flow**:
```text
Market Data (OpenBB / Qlib / DataFrame)
        │
        ▼
   DRR Framework
        │
        ▼
Structural Market State (MarketResonanceState)
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Qlib  Riskfolio      VectorBT
ML    Portfolio      Validation
```
3. **Strict Non-Monolithic Optional Dependencies**: The base `drr-framework` package remains lightweight. External quant dependencies (`openbb`, `riskfolio-lib`, `pyqlib`, `vectorbt`) are optional extras (`drr-framework[quant]`).
4. **Point-in-Time Anti-Leakage Invariants**: Signals, features, thresholds, and portfolio decisions dated at time $t$ rely strictly on market information available on or before time $t$.

---

## Component Roles & Adapters

### 1. Financial Data Infrastructure: OpenBB & Qlib Adapters
- `OpenBBMarketDataProvider` (`drr_framework.finance.data.openbb_adapter`): Interoperates with OpenBB Platform API to retrieve multi-asset historical daily prices.
- `QlibMarketDataProvider` (`drr_framework.finance.data.qlib_adapter`): Loads price panels directly from Microsoft Qlib data engine.
- `DataFrameMarketDataProvider` (`drr_framework.finance.data.base`): Provides offline, deterministic, synthetic, or CSV-based data for offline CI and reproducible research.

### 2. Primary Quant Research Platform: Microsoft Qlib Adapter
- `DRRQlibFeatureProvider` (`drr_framework.finance.qlib.feature_provider`): Combines conventional factors (returns, realized volatility, momentum, cross-asset correlation) with rolling DRR structural features.
- `QlibDatasetAdapter` (`drr_framework.finance.qlib.dataset`): Converts feature matrices and forward target labels into Qlib `DatasetH` structures.
- `QlibDRRMatchedExperiment` (`drr_framework.finance.qlib.experiment`): Executes matched out-of-sample experiments:
  - **CONTROL**: Qlib model + conventional financial features.
  - **EXPERIMENT**: Same Qlib model + conventional features + DRR features.
  - Calculates Information Coefficient (IC), Rank IC, ICIR, Rank ICIR, and runs feature ablation studies.

### 3. Portfolio & Risk Engine: Riskfolio-Lib Adapter
- `RiskfolioAllocator` (`drr_framework.finance.portfolio.riskfolio_adapter`): Interfaces with Riskfolio-Lib for Mean-Variance (MV) and Conditional Value at Risk (CVaR 95%) optimization, with a robust SciPy SLSQP fallback.
- `DRRRegimePortfolioPolicy` (`drr_framework.finance.portfolio.policy`): Evaluates expanding percentile, rolling percentile, and z-score thresholds on trailing DRR mean depth to dynamically select standard vs. high-resonance risk policies without lookahead bias.
- Centralized daily compounding risk-free rate conversion: $r_{\text{daily}} = (1 + r_{\text{annual}})^{1/252} - 1$.

### 4. High-Speed Robustness & Validation Engine: VectorBT Adapter
- `VectorBTAdapter` (`drr_framework.finance.validation.vectorbt_adapter`): Reconstructs backtests independently using VectorBT to validate native walk-forward calculations and executes parameter robustness sweeps across lookback windows, depth windows, and regime percentiles.

---

## Anti-Leakage & Statistical Rigor

- **Lookahead Verification Utility**: `assert_no_lookahead_leakage` (`drr_framework.finance.validation.leakage`) verifies the invariant that truncating a dataset after date $t$ produces identical outputs for all dates $\le t$.
- **Statistical Significance**: Computes Pearson/Spearman correlations with forward realized volatility and drawdowns, Newey-West / HAC robust standard errors (`calculate_hac_standard_errors`), Benjamini-Hochberg False Discovery Rate corrections (`benjamini_hochberg_fdr`), and date-shuffled negative controls (`run_negative_controls`).

---

## Reproduction Commands

To run the full Quant Macro Lab walk-forward evaluation offline:
```bash
python examples/quant_macro_lab.py --offline --start 2015-01-01 --end 2023-12-31
```

To run the Primary Qlib Matched Experiment:
```bash
python examples/qlib_drr_experiment.py --offline --model linear
```

To run Riskfolio portfolio optimization:
```bash
python examples/drr_riskfolio_portfolio.py --offline
```

To run VectorBT parameter robustness sweeps:
```bash
python examples/drr_vectorbt_robustness.py --offline
```

## Conventional challenge models (Phase 1)

A novel method is more informative when its behavior can be compared against
transparent conventional alternatives under the same information set and
evaluation design.

`drr_framework.supervisory.challenge_models` adds a fixed-lag VAR and a pooled
panel logit as **conventional challenge models**. They are research implementations
alongside DRR, not replacements for analyst judgment or evidence of model validity.
They may disagree with DRR or exhibit stronger or weaker detection performance.

### Installation and protected contracts

```bash
python -m pip install -e ".[dev,econometrics]"
python -m pytest
python -m ruff check .
python -m black --check .
```

Statsmodels is an optional dependency. Python 3.8 resolves `statsmodels>=0.14,<0.14.2`;
Python 3.9 and later resolve `statsmodels>=0.14.4,<0.15`. Each fitted result records
the actual package version. Missing dependencies withhold estimation with the
import error; there is no substitute estimator. CI installs this extra across the
existing Python 3.8–3.12 matrix. The existing base installation still imports and
runs its original models without statsmodels.

`run_baselines()` and its `BaselineResult` semantics, `run_event_backtest()`,
`matched_evaluation()`, `walk_forward_validate()`, the DRR state-space estimator,
rooting/topology outputs, ingestion/vintage rules, and analyst/evidence history
are unchanged. No NIST integration, filter packet orchestration, or demonstration
notebook is introduced in Phase 1.

### Information set and public API

The entry points are `walk_forward_var()` and `walk_forward_panel_logit()`.
Both accept the existing vintage store and semantic registry, a frozen
configuration, and explicit `ChallengeReview(as_of, reporting_period)` records.
Review times and reporting quarters must increase. Variables, institution scope,
lags, history length, numerical tolerances and warning thresholds are fixed in
the configuration; its `registered_at` must precede the first review.
Registration is a caller declaration, not proof of an external pre-registration.

At each review the code reconstructs sources strictly **before** its timestamp
(one microsecond earlier than the existing inclusive store cutoff). It retains
the latest filing vintage then known, subject to the existing semantic registry.
It fits only on earlier target quarters and scores the designated current quarter.
This can use revisions already known at that review; later revisions cannot enter
earlier fits. Date-only cutoffs mean midnight UTC.

The VAR consumes the canonical `time × metrics` regulatory representation.
The panel path creates a raw `SupervisoryPanelDataset` from those immutable
observations using its direct constructor. It deliberately does not invoke the
legacy panel adapter's full-sample standardization, interpolation or row cleaning.
Missing quarters are materialized. Lags use exact calendar-quarter keys, so a
missing quarter cannot become a longer, undocumented lag.

Semantic validation is restricted to the requested snapshot dates. Unrelated
legacy observations do not invalidate an otherwise valid current window. The
original store still controls as-of selection and validates revision and lineage
references, including ancestors outside the model window.

Every `ConventionalChallengeResult` exposes:

- `config`, `review`, `method`, `institution_id`, `status`, `score`, `flagged`,
  `withholding_reason`, `limitations`, and a deterministic `result_id`;
- `information_set`: exact source observations and IDs, filing vintages,
  semantic definitions, training quarters, lag-history start, threshold provenance,
  eligible training labels and their IDs, and active policy break records;
- `fit`: method-specific fitted coefficients, normalization, diagnostics,
  package/version, and forecast information or probability inputs.

The audit and fit payloads are stored as immutable canonical JSON strings;
accessors return fresh copies. `flagged` is **None** for an unavailable result,
not a negative analytical observation. The evidence class is
`CONVENTIONAL_ECONOMETRIC_EVIDENCE`, separate from outcome analysis and human judgment.

This interface example assumes a locally supplied store and registry with
historically verified definitions for the chosen form, codes and dates:

```python
from drr_framework.supervisory import (
    ChallengeReview, VARChallengeConfig, walk_forward_var,
)

spec = VARChallengeConfig(
    institution_id="YOUR_INSTITUTION_ID",
    form="FR Y-9C",
    variables=("BHCK0081", "BHCK2170", "BHCK3210"),
    registered_at="2025-12-01T00:00:00Z",
    training_periods=24,
    lag_order=1,
    warning_threshold=3.0,
)
results = walk_forward_var(
    store, registry, config=spec,
    reviews=(ChallengeReview("2026-05-15T12:00:00Z", "2026-03-31"),),
    policy_context=policy_context,
)
```

The bundled limited registry does not by itself supply this complete historical
coverage. Absent observations, unspecified perimeters, or unsupported historical
definitions yield explicit withholding.

### VAR meaning and diagnostics

`VARChallengeConfig.training_periods` is the exact number of training **target**
quarters. A VAR(p) additionally requires p initial lag quarters; the current
quarter is separate. Training-only means and population standard deviations
scale the variables. Statsmodels estimates an intercept and the pre-specified p
lag matrices by OLS; it performs no automatic lag selection or regularization.

Coefficient rows are the intercept followed by lag-major/variable-major predictors;
columns follow `variables`. Coefficients operate on training-standardized values.
The recorded mean and scale reconstruct raw-unit forecasts and coefficients.
The one-step forecast, observed current vector, errors, and lag inputs are retained.

The scalar warning score is
`sqrt(mean((current - forecast)**2 / training_residual_variance))`, with residual
variance transformed back to original units. It measures unusual forecast errors;
it is not an event probability. The threshold is fixed before evaluation.
Diagnostics include design rank/condition, residual degrees of freedom, residual
means, RMSE, covariance, covariance condition number, and linear dynamic stability.
The full training residual array remains inspectable. Stationarity and economic
appropriateness are assumptions requiring review; a stability diagnostic cannot
establish either economic validity or detection usefulness.

### Panel logit meaning and labels

`PanelLogitConfig` specifies a fixed institution tuple, form, `LaggedFeature(metric,
lag)` tuple, event definition, training-period count, minimum observations,
institutions, positive labels and negative labels, optimizer budget, numerical
tolerances, and probability threshold. Predictors are raw lagged levels with
centering/scaling fitted only on eligible training rows. An intercept and common
slopes form a pooled binary logit, estimated by statsmodels' Newton optimizer.
Institution identifiers and peer aggregates are never explanatory variables.

`BinaryEventLabel(institution_id, reporting_period, value, available_as_of,
definition, source, event_at=None)` is supplied independently from features.
Labels mean an explicitly observed event or non-event in that quarter, **not**
an automatically expanded future lead-window target. `value` accepts `1`, `0`,
or `None`. Zero requires an actually ascertained non-event under the supplied
definition. Omitted, unknown, or not-yet-available labels are excluded and counted
as unknown. Labels for the current target quarter are never fitted, even if known.
Positive-label counts concern institution-quarters; several may describe a single
shared economic episode. A positive label's optional `event_at` supplies actual
timing for subsequent lead-window evaluation.

Training uses labels whose availability is strictly earlier than the review;
conflicting label vintages at the same availability time raise an error. The code
reports the configured sample size, labeled/usable observations, usable firms,
positive and negative counts, event fraction, unknown-label count, missing cells
by variable, convergence, optimizer iterations and information-matrix conditioning.
For mixed lag lengths, `missing_cells` and `missing_cells_by_metric` count only
source cells referenced by the fixed training and scoring designs, including the
configured training rows without labels. Missing rectangular padding is recorded
as `unused_missing_cells` and does not withhold a fit. Required missing cells
still withhold the complete primary specification; no window is shortened.
An institution with no eligible labels is counted through the configured versus
usable scope; it is never silently assigned synthetic non-events.

This model does not incorporate institution effects, serial dependence, dependence
between firms, rare-event bias corrections or class weighting. Its common slopes
and conditional probabilities have those limitations. Positive/non-event minimums
are gates, not evidence that a sample is adequate. Probit, feature selection,
coefficient tests and post-selection inference are outside this specification.

### Withholding and statistical inference

The primary specification withholds if its fixed input window has missing or
non-finite values, insufficient history, an unknown perimeter, a definition/unit/
perimeter change, or an active policy/reporting breakpoint known at the review.
This includes current-quarter comparability metadata when available for panel
scoring. It also withholds for unidentified or materially ill-conditioned designs,
singular/nearly degenerate VAR residual covariance, insufficient residual degrees
of freedom, too few eligible labels/firms, separation, solver non-convergence,
material numerical warnings, or an unusable logit information matrix.

It never silently shortens the window, removes a variable or institution with
missing required features, compresses time, fills zeros, balances classes,
interpolates a break or substitutes a regularized fit. Unlabeled target rows are
excluded only under the explicit label-availability rule and remain counted.
Any alternative sensitivity specification must be separately configured,
identified, and evaluated. It cannot replace the primary result without disclosure.

Coefficients are descriptive fitted parameters; no coefficient p-values or
standard errors are reported. `calculate_hac_standard_errors()` remains an OLS/
Newey–West helper and is not applied to logit or general VAR coefficients. No new
multiplicity routine is introduced: any separately justified family of
pre-specified hypothesis tests must use the existing `benjamini_hochberg_fdr()`.
Tests verify that routine against known adjusted values; these models do not
manufacture a hypothesis-test family merely to produce significance markers.

### Outcome analysis and comparison with DRR

`evaluate_challenge_results(results, labels, design=ChallengeEvaluationDesign(...),
evaluation_as_of=...)` performs a **detection performance comparison under a
pre-registered evaluation design**. Outcome-label availability at the evaluation
cutoff is separate from training-label availability at each historical review.
The output is `EVENT_DETECTION_EVALUATION`, not binary-classification metrology.

The evaluation grid must contain every reporting quarter, including withheld
results. The lead window counts calendar quarters; actual warning/event timestamps
also determine whether an alert can receive credit. Missing positive-event timing
withholds detection metrics. An alert after an event never detects that event.

The adapter uses the unchanged `run_event_backtest()` matching rule for each
alert, supplying only known events that were not already over when the alert
became available. It never exposes the backtester's implicit negative-label
metrics as estimates over unknown outcomes. False-positive alerts are counted
only for fully observed future windows; unresolved/right-censored alerts are
reported separately. Known positive matches remain visible even near the end of
the record. `confirmed_false_positive_alerts` retains the count from fully observed
windows; the total `false_positive_alerts` and precision are withheld if any
alert's outcome remains indeterminate.

Reports include event count, detected/missed events, alert count/burden, scored and
unscored observations, unscored event rows, events with no scored lead window,
unknown outcomes, false-positive alerts where estimable, indeterminate alerts,
lead-time distributions in quarters and days, and right-censored reviews/alerts.
`unscored_event_count` describes an unscored event row, matching the legacy field;
`events_without_scored_lead_window` separately identifies absent warning coverage.
A configured minimum event count withholds rate comparisons while retaining
descriptive counts. A three-event result remains a three-event result.

Optional `drr_rows` must match the complete institution/review grid, with explicit
`drr_alert` booleans (or None for unavailable) and the same source `input_hash`.
Both methods' detection counts remain separate, and disagreement dates are listed.
`matched_evaluation(events=None)` supplies matched alert burden and incremental
alerts only when both complete score series are available. An unavailable matched
row withholds that comparison; it is never dropped or changed to False. The
caller must still substantiate that DRR used the declared information set and
training/tuning procedure; an input hash alone cannot attest to that procedure.

### Limits and review still required

Synthetic tests check known VAR/logit structure recovery, out-of-sample score
construction, future-input invariance, withholding, numerical failure handling,
determinism, outcome timing, unknown labels, censoring and disagreement. Synthetic
recovery does not establish real-world supervisory usefulness. No conventional
model is required to outperform or agree with DRR.

Actual use remains subject to independent conceptual-soundness review,
implementation verification, outcomes analysis on appropriate independent labels,
an ongoing-monitoring design, governance review and review of the intended use.
Monitoring should track missingness, withheld coverage, class prevalence, design
conditioning, coefficient drift, convergence, event counts and lead times, and
alert burden on a fixed outcome design. It must not equate stable numerics with
economic validity or convert probabilities into supervisory decisions.

Statistical API references:
[statsmodels VAR fitting](https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.vector_ar.var_model.VAR.fit.html),
[logit fitting](https://www.statsmodels.org/v0.14.4/generated/statsmodels.discrete.discrete_model.Logit.fit.html),
and [numerical pitfalls](https://www.statsmodels.org/v0.14.4/pitfalls.html).

## Supplementary binary metrology (Phase 2)

`supervisory.contingency.evaluate_contingency()` adds NIST Contingency-based
supplementary detection metrology for explicitly labeled classification targets.
It reports counts, MCC, precision/recall, F1/F2, average precision, and threshold
sensitivity. Pre-specified, tuning-derived, and retrospective analyses carry
different recorded protocols; unknown outcomes never become negatives.

This API does not call or change the event backtester, its lead-window metrics,
or the Phase 1 challenge scorers. The optional extra requires Python 3.12;
dependency and mathematical unavailability are explicit. See the
[metrology contract, examples, numerical conventions, and review limitations](contingency-metrology.md).
