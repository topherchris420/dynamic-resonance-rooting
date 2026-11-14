# Can DRR detect relationship breakdown earlier?

This experiment tests **shorter detection delay after a change begins**. It does
not test anticipation of a future change without a precursor. The primary claim
is confined to controlled simulations. Market warning usefulness and risk
attribution are separate, explicitly descriptive results.

Read the [registered protocol](../experiments/lead_lag_breakdown/protocol.json),
[calibration lock](../experiments/lead_lag_breakdown/calibration/lock.json), and
[complete results](../results/expected/lead_lag_breakdown/report.md).

## Fixed comparisons

| Component | Design |
| --- | --- |
| Primary candidate | Existing DRR depth components plus directed max-absolute lag correlations and selected lags |
| Ablations | Rooting only; spectral/depth only |
| Baselines | Signed lag-correlation change, VAR(3) residual-energy CUSUM, Student-t Bayesian online change-point detection |
| Known changes | Coupling disappears, reverses sign, moves from lag 1 to lag 3, or gradually weakens |
| Nulls | Unchanged coupling; volatility-only shock with fixed latent coupling |
| Confounding challenge | Unobserved common driver changes observed dependence while latent coupling stays fixed |
| Calibration | 64 independent paths per null; fixed 5% pathwise false-alarm budget |
| Holdout | 96 separate paired seeds per scenario; no scenario labels passed to detectors |
| Scoring | First alarm; misses and pre-change alarms receive the full delay penalty |
| Inference | Replicate-level paired bootstrap; Holm over all 15 registered comparisons |
| External data | New BTCUSDT/ETHUSDT hourly data from 2026, separate from the prior daily-return study |
| Market calibration/test | February–March / April–September; monthly fit on preceding 256 hours |
| External target | Following 24-hour VAR error relative to zero-return error |
| Risk controls | Always-long and constant exposure matched to each warning strategy; costs 0/15/30 bps |

Matched calibration budgets are not necessarily identical realized false-alarm
rates. Every held-out null rate and its one-sided Wilson interval is reported.
The claim is rejected if the fixed false-alarm comparability gate fails. Every
miss remains in the primary delay comparison. Common-driver alarms are a
specificity challenge, not automatically false alarms: observable relationships
can change even when the latent structural coefficient does not.

DRR rooting uses **absolute** correlations. A persistent sign reversal can leave
its fingerprint unchanged after the transition. This limitation is tested, not
removed by silently substituting signed correlations into DRR. The conventional
signed-correlation comparator uses exactly the same distance normalization.

The CUSUM monitors standardized VAR prediction-error energy. The BOCPD baseline
uses the normal/inverse-gamma Student-t update and a fixed hazard. It reports
posterior mass on short run lengths (0–8), because the reset probability alone
with a constant hazard is uninformative. Run-length state is initialized on fit
residuals before monitoring; no artificial certainty at startup is scored.

## Chronology and claim boundaries

The protocol was published in commit `705cdb6a96864140869e2f94d300b82790b22626`
before scoring. Calibration, source hashes and the new data vintage are committed
separately before held-out evaluation. Merge commits preserve this chronology.
Development tests use different seeds from both registered calibration and test
ranges. Public seeds and public historical markets do not constitute independent
blinding; detector inputs nevertheless contain no change-point labels.

Market calibration computes features and labels only through March. Each later
month refits on 256 already observed returns; neither detector thresholds nor
hyperparameters are selected on the market holdout. Score timestamps without a
complete next-24-hour label inside their month remain in the warning/risk ledger
but are excluded from the association calculation. The final observed hourly
open lacks a subsequent open, so that final holding interval is omitted and
reported. No missing prices are imputed or silently dropped.

Real break times are unknown. The external target measures whether an existing
forecast relationship performs worse after warnings. It does not define truth
using DRR's own scores, and it does not claim market change-point accuracy.
Circular-block intervals are descriptive, approximate and withheld for sparse
warning/control samples. A simulation success cannot substitute for external
validation, and a favorable market association cannot rescue a failed simulation
claim.

Risk warnings cause cash exposure with a full intervening hour before execution.
The constant matched-exposure control is calculated **ex post** to diagnose how
much lower risk comes from holding less of the assets. It is not a deployable
strategy. Risk tables do not establish timing skill or a trading edge; costs are
scenarios, and impact, capacity, exchange and stablecoin risks are not modeled.
HAR-RV/GARCH volatility forecasting remains a separate study.

## Reproduce offline

Use the same pinned Python 3.12 numerical environment as the financial holdout:

```bash
python -m pip install -e ".[dev]" -c experiments/financial_holdout/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/test_lead_lag_breakdown.py -q
python scripts/run_breakdown_study.py calibrate \
  --output-dir /tmp/breakdown-calibration \
  --verify-against experiments/lead_lag_breakdown/calibration
python scripts/run_breakdown_study.py evaluate \
  --calibration-dir /tmp/breakdown-calibration \
  --output-dir /tmp/breakdown-results \
  --verify-against results/expected/lead_lag_breakdown
```

Choose unused output directories. Changed data, protocol, source or calibration
artifacts fail validation. Scientific failure remains a successful execution;
CI checks reproducibility, not whether DRR wins. All episode/score/market/risk
ledgers are deterministic gzip CSV. Data acquisition is separately reproducible:

```bash
python scripts/fetch_breakdown_data.py --output-dir /tmp/breakdown-market \
  --expected-manifest experiments/lead_lag_breakdown/data/manifest.json
```

Data and market-derived artifacts under `experiments/lead_lag_breakdown/` and
`results/expected/lead_lag_breakdown/` carry Binance Vision attribution,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the
[source dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
They are not relicensed under the repository's MIT software license. Sources are
current historical archives, not original point-in-time vintages. The manifest
records every original URL and upstream-verified ZIP checksum.

References: [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742);
[Page, Continuous Inspection Schemes](https://doi.org/10.1093/biomet/41.1-2.100).
