# External evidence

Specification tests show that DRR implements its definitions. They do not show
that those definitions outperform established methods on a real problem. This
page is the external comparison.

The chain is fixed:

```text
public dataset → preregistered hypotheses → conventional models
    → DRR → ablations → temporal holdout → uncertainty → artifact
```

## Dataset

The series is the NOAA Climate Prediction Center QBO zonal-wind index at 30 mb
and 50 mb, original-data section, retrieved 2026-09-22. The snapshots are
vendored:

- `src/drr_framework/external_benchmark/data/qbo.u30.index`
- `src/drr_framework/external_benchmark/data/qbo.u50.index`

Both files are U.S. government work. The loader keeps the original-data section,
drops the CPC missing value `-999.90`, and requires a gap-free overlap. The
preregistration records the SHA-256 of each snapshot, the first month
`1979-01`, the last finite month `2026-08`, and 572 finite months. A changed
file fails the checksum check.

## Preregistration

`src/drr_framework/external_benchmark/data/preregistration.json` freezes the
plan. The disruption windows come from the papers, not from the scores:

| Id | Window | Anchors |
| --- | --- | --- |
| `qbo-2015-2016` | 2016-01 through 2016-04 | Osprey et al. 2016; Newman et al. 2016 |
| `qbo-2019-2020` | 2019-12 through 2020-03 | Anstey et al. 2021 |

The holdout starts in 2011-01. Every earlier scored month is a non-event.
Newman et al. describe the 2015–2016 anomaly as unprecedented in the observed
record, and both windows fall after the holdout cut.

The false-alarm tolerance is 0.05. Each scored model receives the least strict
threshold whose estimation alarm rate is at most that tolerance. The trailing
window is 84 months, three cycles of the roughly 28-month QBO period cited by
Newman et al. Rooting uses `RootingAnalyzer` lagged correlation with a maximum
lag of 12 and no surrogates. The VAR is order 1 with an intercept, fit only on
earlier months.

## Models

| Model | Role |
| --- | --- |
| Rolling volatility | Conventional. Mean trailing standard deviation of the two levels. |
| Rolling absolute correlation | Conventional. Drop below the estimation-period median correlation. |
| VAR residual | Conventional. One-step residual energy. |
| Regularized logistic | Conventional. Withheld on this dataset: the estimation window has no positive labels. |
| Spectral ablation | `DepthCalculator` spectral concentration, scored as one minus the mean of the two levels. |
| Rooting ablation | Absolute deviation of the dominant directed lag from the estimation median lag. |
| Full DRR | Maximum of the estimation-standardized spectral and rooting scores, with its own threshold. |

## Hypotheses and decision

- **H1.** Full DRR's false-alarm rate on holdout months outside the disruption windows is at most 0.05.
- **H2.** At least two disruption windows contain a full DRR alarm and contain no alarm from a conventional model that calibrated inside the same tolerance.
- **H3.** No single ablation that itself stays inside the tolerance hits every window counted by H2.

Support requires H1, H2, and H3, and a circular block bootstrap of the
estimation months whose 90% interval for the unique-hit count stays at or
above 1. The interval does not rescue a point estimate that fails. One unique
detection is inconclusive. A window without finite scores is inconclusive.

The block length is 28 months, the seed is `20260922`, and there are 400
replicates. One month draw is applied to every model. Holdout scores stay fixed.

## Reviewed run

The reviewed artifact is
[`results/expected/qbo_structural_change_benchmark.json`](../results/expected/qbo_structural_change_benchmark.json).
Its `claim_status` is `not_supported` and its reason is `drr_false_alarm_rate`.

Full DRR calibrates at an estimation false-alarm rate of 0.05 and then alarms
on more than a quarter of the quiet holdout months. Inside the published
windows it misses 2016 and flags 2019. The rooting ablation does not calibrate:
the lag score takes two values, and the higher value is common in the
estimation period, so no threshold meets the 5% budget. Rolling volatility,
rolling correlation, and the VAR residual also land above the holdout
tolerance. The logistic arm is withheld. The precise rates, thresholds, and
alarm months are in the JSON.

That is an atmospheric-adapter result. It does not establish a result for
physics, sensing, macro policy, supervision, or financial markets, and a later
result in one of those adapters would not change this run.

## Reproduce

From the repository root:

```bash
python scripts/run_structural_change_benchmark.py --output-dir results/expected
```

The Markdown report is rendered from the JSON. Edit the preregistration, the
snapshots, or the decision rule and the checksums and the claim move together.
Do not retune the windows or the tolerance after seeing the scores and leave
the old preregistration id in place.
