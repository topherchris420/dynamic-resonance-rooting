# A falsifiable financial experiment

The study tests an incremental claim: **do DRR features improve forecasts and
cost-adjusted decisions relative to conventional alternatives?** It does not
equate a profitable crypto backtest with novel predictive information.

Read the [fixed protocol](../experiments/financial_holdout/protocol.json),
[amendment history](../experiments/financial_holdout/amendments.md),
[all validation trials](../experiments/financial_holdout/selection/trials.json),
[selection lock](../experiments/financial_holdout/selection/selection.json) and
[complete result](../results/expected/financial_holdout/report.md).

**Observed outcome: NOT SUPPORTED.** Validation chose ridge + DRR. On the holdout,
its MSE was 0.00084385 versus 0.00084675 for zero return and 0.00084450 for ridge.
Those small improvements did not meet the registered 1% minimum against every
baseline or survive the required significance tests. The primary portfolio's
positive net result, cross-year consistency and roughly 90% interval coverage
did not override the failed gates. The nonlinear DRR candidate was worse than
the zero-return baseline on MSE. All of these outcomes remain in the report.

## What is tested

| Choice | Fixed design |
| --- | --- |
| Universe | BTCUSDT and ETHUSDT, spot daily OHLCV, one exchange |
| Target | One-day simple open-to-open return, with a full intervening day after the feature bar closes |
| Development | 2018–2020, with purged boundaries |
| Selection | 2021–2022; final entry 2022-12-21 after label-maturity purge |
| Evaluation | 2023-01-01 through 2025-12-31; 1,096 dates, two correlated assets |
| Refitting | Quarterly, fixed procedure, only matured prior labels; last 1,095 fit rows plus separate 180-row calibration sample |
| Baselines | Zero return, historical mean, 20-day momentum, scaled ridge, histogram gradient boosting |
| Candidates | Ridge + DRR and gradient boosting + DRR, with identical tuning budgets to their conventional counterparts |
| Controls | Spectral-only, rooting-only and independent-noise augmentation |
| Selection metric | Equal-asset mean squared error, never holdout P&L |
| Execution | Long/cash, maximum 50% per asset, no borrowing; fixed forecast threshold |
| Cost scenarios | 0, 15 and 30 bps per unit of one-way turnover; entries, drift rebalancing and terminal exits included |
| Uncertainty | Held-out residual prediction intervals; dependence-aware paired bootstrap comparisons |
| Multiplicity | Holm adjustment over all 14 registered comparisons at each block length; all block lengths must pass |
| Regimes | Trailing volatility/trend labels with a development-only threshold; calendar-year diagnostics |

Daily features for date *t* use completed bars through *t*. Execution is at the
open of *t+2*, exit at the open of *t+3*. The unused intervening day avoids assuming
that a closing observation can be processed and filled at the simultaneous next
daily open. Daily-open fills remain a coarse hypothetical execution model.

Before each forecast quarter, labels must exit strictly before its first feature
date minus seven days. The latest 180 eligible rows calibrate intervals. Fitting
also excludes labels within seven days of the first calibration feature date.
Ridge standardization is fitted only on this fit subset; gradient boosting has
random early-stopping splits disabled. Earlier holdout observations may enter
later scheduled fits after they mature. Model choices, grids and thresholds stay
frozen. This is a locked **prequential evaluation**, not a model trained once and
held fixed for three years.

The 16 DRR features call the existing `DepthCalculator` and `RootingAnalyzer`.
The latter uses descriptive max-over-lag absolute correlations, not transfer
entropy or inferred causal edges. Depth components are computed within each
trailing window; no whole-series filtering, smoothing or normalization is used.

## Reproduce from the pinned snapshot

Use Python 3.12 for numerical reproduction. The general library remains tested
on its existing Python matrix; exact research replay has a separate environment.

```bash
python -m pip install -e ".[dev]" -c experiments/financial_holdout/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/test_financial_holdout.py -q
python scripts/run_financial_holdout.py select \
  --output-dir /tmp/drr-selection \
  --verify-against experiments/financial_holdout/selection
python scripts/run_financial_holdout.py evaluate \
  --selection-dir /tmp/drr-selection \
  --output-dir /tmp/drr-results \
  --verify-against results/expected/financial_holdout
```

Choose unused output directories. The runner refuses to overwrite evidence and
rejects changed protocol, market-data, source-code or selection checksums. It
always writes the full scientific outcome. A `NOT SUPPORTED` result is a successful
execution; CI checks reproducibility and integrity, not whether DRR wins.

To independently re-fetch the exact source vintage into a new directory:

```bash
python scripts/fetch_financial_holdout_data.py \
  --output-dir /tmp/drr-market-data \
  --expected-manifest experiments/financial_holdout/data/manifest.json
```

The fetcher checks upstream ZIP checksums and then the committed normalized
snapshot. If the provider revises history, verification fails rather than silently
changing the experiment. Original archives use milliseconds before 2025 and
microseconds from 2025; the loader checks exact daily bar boundaries in either
format. No network access is needed for analysis or CI replay.

## How to audit the claim

The initial protocol has its own published commit, before selection or test
scoring. An explicit, pre-scoring amendment fixes the cross-stage label purge.
A subsequent commit freezes code, data and validation selection before holdout
evaluation. These commits are retained by a merge commit, not collapsed into a
squash. The hashes establish reproducibility and reviewable chronology; they
cannot establish that historical events were unknown to the researcher.

`predictions.csv.gz` includes feature, entry, exit, fit and calibration dates;
actual targets; every model's prediction; interval bounds; year; and regime.
`daily_portfolios.csv.gz` records all positions' aggregate exposures, gross/net
returns, turnover and costs. `summary.json` exposes each decision gate and every
comparison, including unfavorable outcomes. Gzip payloads use a fixed timestamp.

The intervals use scaled absolute residuals from a disjoint historical sample,
with the finite-sample upper order statistic. They quantify realized predictive
error rather than a Bayesian parameter posterior. Serial dependence and drift
break the usual exchangeability guarantee: coverage is measured, not asserted.
Circular moving-block draws keep both assets and all models together on each
sampled date; 7-, 30- and 90-day blocks expose sensitivity to persistence. The
bootstrap is approximate, and severe nonstationarity can invalidate its nominal
error rates. The reported 95% intervals are pointwise; the inferential gate uses
Holm-adjusted p-values, not those intervals alone.

All model grids and validation trials are disclosed. The holdout family covers
two candidates against five forecast baselines, plus each candidate against cash
and the validation-selected baseline economically. Ablations, cost sweeps and
regime tables are descriptive, not additional discovery claims. The method does
not correct for unknown prior experiments, changing universes, or future reuse of
this holdout. Any subsequent tuning needs a fresh test period.

The success criterion intentionally requires forecast improvement, corrected
evidence, positive stressed performance, cross-year consistency and minimally
adequate interval coverage together. Failure of one criterion is published as
`NOT SUPPORTED`, without converting an attractive secondary result into a win.

## What the study cannot establish

The archive is a current historical snapshot, not an original point-in-time
data vintage. These two assets were chosen retrospectively; no full survivorship
claim is made. The universe is small and strongly related. Execution has no
order-book depth, measured spread, funding, account-specific fee tier, market
impact, capacity, exchange credit or stablecoin risk model. No real orders are
placed. Results do not demonstrate a production trading edge, nor do they
generalize automatically beyond these assets, venue, dates and target.

Data and derived results carry [separate attribution and non-commercial terms](../experiments/financial_holdout/DATA_LICENSE.md).

## Method references

- [Binance public-data schema, checksums and correction history](https://github.com/binance/binance-public-data).
- [scikit-learn histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
- [scikit-learn temporal cross-validation and gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [Holm (1979), A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733).
