# QBO structural-change benchmark

Claim status: **not_supported**

The preregistered claim is not supported. The full DRR holdout false-alarm rate exceeds the preregistered tolerance. Domain: equatorial stratospheric zonal wind. The result stays inside that domain.

Reason code: `drr_false_alarm_rate`.

This is one domain adapter, equatorial stratospheric zonal wind. A result here does not transfer to physics, sensing, macro policy, supervision, financial markets, or another adapter.

## Preregistered question

Does full DRR flag repeated, published QBO disruptions that rolling volatility, rolling correlation, and a VAR residual miss, without a holdout false-alarm rate above the preregistered tolerance, after spectral-only and rooting-only ablations?

## Hypotheses

- **H1.** On holdout months outside the disruption windows, the full DRR false-alarm rate is at most 0.05.
- **H2.** At least two disruption windows contain a full DRR alarm and contain no alarm from any conventional model that calibrated inside the same tolerance.
- **H3.** No single ablation that itself stays inside the 0.05 tolerance hits every disruption window counted by H2.

## Models

| Model | Role | Status | Estimation FPR | Holdout FPR | Windows |
| --- | --- | --- | ---: | ---: | --- |
| rolling_volatility | conventional | calibrated | 0.050 | 0.572 | qbo-2015-2016 hit, qbo-2019-2020 hit |
| rolling_correlation | conventional | calibrated | 0.050 | 0.461 | qbo-2015-2016 hit, qbo-2019-2020 hit |
| var_residual | conventional | calibrated | 0.050 | 0.072 | qbo-2015-2016 hit, qbo-2019-2020 miss |
| regularized_logistic | conventional | withheld | — | — | — |
| drr_spectral | ablation | calibrated | 0.050 | 0.272 | qbo-2015-2016 miss, qbo-2019-2020 hit |
| drr_rooting | ablation | uncalibrated | — | — | qbo-2015-2016 miss, qbo-2019-2020 miss |
| drr_full | drr | calibrated | 0.050 | 0.272 | qbo-2015-2016 miss, qbo-2019-2020 hit |

## Unique detections and uncertainty

Unique full-DRR detections: none.

Bootstrap interval for the unique-hit count: 0.000 to 0.000 (400 circular-block replicates).

One circular-block draw resamples estimation months and is applied to every model. Holdout scores stay fixed. A replicate that cannot calibrate full DRR counts as no unique hits and a false-alarm rate of 1.

## Dataset

NOAA CPC QBO 30 mb and 50 mb zonal wind index, original-data section, retrieved 2026-09-22. Finite months 1979-01 through 2026-08 (572 months). Holdout starts 2011-01. False-alarm tolerance 0.05.

Checksums of the vendored CPC snapshots are in the preregistration. The JSON artifact is the machine-readable record.
