# Lead–lag breakdown study

**Registered simulation verdict: NOT SUPPORTED.**

Market findings below are descriptive. They cannot establish an earlier true break time, causality or a trading edge.

## Fixed success gates

| Gate | Passed |
| --- | --- |
| heldout false alarm budget comparable | False |
| pooled delay gain at least 8 and holm significant vs every baseline | False |
| no change type more than 8 observations worse | False |

## Every simulation outcome

First alarm only; early alarms and misses receive the full 256-observation delay penalty. The common-driver scenario changes observed dependence without changing latent coupling: its alarms are a confounding diagnostic, not proven false alarms. Delay columns are meaningful for the four true-change scenarios only.

| Scenario | Method | N | Any alarm | Upper 95% | Early alarm | Hit | No alarm | Penalized delay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unchanged | drr | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| unchanged | rooting_only | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| unchanged | spectral_only | 96 | 7.29% | 12.92% | 2.08% | 5.21% | 92.71% | 250.53 |
| unchanged | signed_correlation | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| unchanged | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| unchanged | var_bocpd | 96 | 6.25% | 11.63% | 4.17% | 2.08% | 93.75% | 253.48 |
| volatility_only | drr | 96 | 1.04% | 4.53% | 0.00% | 1.04% | 98.96% | 253.41 |
| volatility_only | rooting_only | 96 | 1.04% | 4.53% | 0.00% | 1.04% | 98.96% | 253.41 |
| volatility_only | spectral_only | 96 | 9.38% | 15.44% | 2.08% | 7.29% | 90.62% | 245.43 |
| volatility_only | signed_correlation | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| volatility_only | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| volatility_only | var_bocpd | 96 | 8.33% | 14.19% | 4.17% | 4.17% | 91.67% | 247.62 |
| drop | drr | 96 | 66.67% | 74.03% | 0.00% | 66.67% | 33.33% | 143.33 |
| drop | rooting_only | 96 | 83.33% | 88.66% | 0.00% | 83.33% | 16.67% | 109.83 |
| drop | spectral_only | 96 | 5.21% | 10.31% | 2.08% | 3.12% | 94.79% | 254.55 |
| drop | signed_correlation | 96 | 57.29% | 65.28% | 0.00% | 57.29% | 42.71% | 163.59 |
| drop | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| drop | var_bocpd | 96 | 7.29% | 12.92% | 4.17% | 3.12% | 92.71% | 249.97 |
| reverse | drr | 96 | 20.83% | 28.40% | 0.00% | 20.83% | 79.17% | 207.79 |
| reverse | rooting_only | 96 | 39.58% | 47.97% | 0.00% | 39.58% | 60.42% | 164.77 |
| reverse | spectral_only | 96 | 3.12% | 7.56% | 2.08% | 1.04% | 96.88% | 253.57 |
| reverse | signed_correlation | 96 | 95.83% | 98.12% | 0.00% | 95.83% | 4.17% | 49.79 |
| reverse | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| reverse | var_bocpd | 96 | 9.38% | 15.44% | 4.17% | 5.21% | 90.62% | 246.53 |
| lag_shift | drr | 96 | 36.46% | 44.81% | 0.00% | 36.46% | 63.54% | 201.55 |
| lag_shift | rooting_only | 96 | 36.46% | 44.81% | 0.00% | 36.46% | 63.54% | 190.22 |
| lag_shift | spectral_only | 96 | 4.17% | 8.96% | 2.08% | 2.08% | 95.83% | 253.90 |
| lag_shift | signed_correlation | 96 | 63.54% | 71.15% | 0.00% | 63.54% | 36.46% | 156.28 |
| lag_shift | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| lag_shift | var_bocpd | 96 | 8.33% | 14.19% | 4.17% | 4.17% | 91.67% | 248.54 |
| gradual_drop | drr | 96 | 64.58% | 72.11% | 0.00% | 64.58% | 35.42% | 174.60 |
| gradual_drop | rooting_only | 96 | 83.33% | 88.66% | 0.00% | 83.33% | 16.67% | 139.08 |
| gradual_drop | spectral_only | 96 | 5.21% | 10.31% | 2.08% | 3.12% | 94.79% | 254.55 |
| gradual_drop | signed_correlation | 96 | 55.21% | 63.30% | 0.00% | 55.21% | 44.79% | 190.95 |
| gradual_drop | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| gradual_drop | var_bocpd | 96 | 6.25% | 11.63% | 4.17% | 2.08% | 93.75% | 252.56 |
| common_driver | drr | 96 | 62.50% | 70.18% | 0.00% | 62.50% | 37.50% | 157.54 |
| common_driver | rooting_only | 96 | 72.92% | 79.67% | 0.00% | 72.92% | 27.08% | 141.27 |
| common_driver | spectral_only | 96 | 9.38% | 15.44% | 2.08% | 7.29% | 90.62% | 245.09 |
| common_driver | signed_correlation | 96 | 44.79% | 53.17% | 0.00% | 44.79% | 55.21% | 184.22 |
| common_driver | var_cusum | 96 | 0.00% | 2.74% | 0.00% | 0.00% | 100.00% | 256.00 |
| common_driver | var_bocpd | 96 | 8.33% | 14.19% | 4.17% | 4.17% | 91.67% | 248.29 |

## All 15 corrected comparisons

Positive gains favor DRR. Confidence intervals are pointwise. Holm correction includes all 15 comparisons. Resampling units are independent simulation replicates, paired across scenarios and methods.

| Test | Delay gain | CI low | CI high | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| drop:drr>signed_correlation | 20.26 | 4.09 | 36.65 | 0.0076 | 0.0304 |
| reverse:drr>signed_correlation | -158.00 | -177.26 | -137.76 | 1.0000 | 1.0000 |
| lag_shift:drr>signed_correlation | -45.27 | -67.26 | -23.23 | 1.0000 | 1.0000 |
| gradual_drop:drr>signed_correlation | 16.34 | 5.12 | 27.42 | 0.0012 | 0.0060 |
| pooled:drr>signed_correlation | -41.67 | -51.86 | -31.86 | 1.0000 | 1.0000 |
| drop:drr>var_cusum | 112.67 | 94.37 | 130.66 | 0.0001 | 0.0015 |
| reverse:drr>var_cusum | 48.21 | 29.37 | 68.21 | 0.0001 | 0.0015 |
| lag_shift:drr>var_cusum | 54.45 | 37.90 | 72.00 | 0.0001 | 0.0015 |
| gradual_drop:drr>var_cusum | 81.40 | 67.30 | 95.70 | 0.0001 | 0.0015 |
| pooled:drr>var_cusum | 74.18 | 62.25 | 86.32 | 0.0001 | 0.0015 |
| drop:drr>var_bocpd | 106.64 | 86.88 | 126.17 | 0.0001 | 0.0015 |
| reverse:drr>var_bocpd | 38.74 | 17.81 | 59.83 | 0.0004 | 0.0024 |
| lag_shift:drr>var_bocpd | 46.99 | 29.12 | 65.90 | 0.0001 | 0.0015 |
| gradual_drop:drr>var_bocpd | 77.96 | 62.19 | 93.76 | 0.0001 | 0.0015 |
| pooled:drr>var_bocpd | 67.58 | 55.10 | 80.49 | 0.0001 | 0.0015 |

## Frozen thresholds

| Method | Simulation path threshold | Market warning threshold |
| --- | ---: | ---: |
| drr | 3.835039 | 2.594138 |
| rooting_only | 7.070008 | 3.007820 |
| signed_correlation | 4.601942 | 2.558611 |
| spectral_only | 2.860599 | 2.947889 |
| var_bocpd | 0.962329 | 0.329557 |
| var_cusum | 3067.819265 | 352.528385 |

## Held-out 2026 market validation

BTCUSDT/ETHUSDT hourly bars. Thresholds use February–March; evaluation uses April–September. Each month refits on only its preceding 256 return observations. Warnings use no future labels. The next-24-hour target is frozen VAR error relative to zero-return error, in fit-standardized units. A positive warned-minus-unwarned difference indicates deterioration associated with warnings, not an independently known change point. Confidence intervals are descriptive circular-block estimates under dependence assumptions; intervals are withheld for sparse warnings.

| Method | Eligible | Warnings | Warning rate | Error-ratio difference | 95% interval | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| drr | 531 | 2 | 0.38% | 0.01498 | n/a | INSUFFICIENT WARNINGS OR CONTROLS |
| rooting_only | 531 | 7 | 1.32% | 0.00921 | n/a | INSUFFICIENT WARNINGS OR CONTROLS |
| spectral_only | 531 | 1 | 0.19% | -0.00676 | n/a | INSUFFICIENT WARNINGS OR CONTROLS |
| signed_correlation | 531 | 29 | 5.46% | -0.04053 | [-0.09006, 0.00368] | descriptive |
| var_cusum | 531 | 107 | 20.15% | -0.03192 | [-0.06179, -0.00367] | descriptive |
| var_bocpd | 531 | 26 | 4.90% | -0.01395 | [-0.04739, 0.02274] | descriptive |

Scored timestamps: 549; excluded incomplete 24-hour labels: 18. Warnings without complete labels remain in the risk simulation. The last hourly holding lacks a next open and is explicitly omitted.

## Risk attribution: all costs and exposure controls

After warnings, cash begins two hourly opens later and lasts 24 hours. Constant matched-exposure controls use the strategy's realized average exposure: they are ex-post attribution controls, not deployable strategies. Lower losses alone do not establish useful warning timing. No market impact, account-specific fees, funding, capacity or stablecoin risk is estimated. Costs are hypothetical. No inferential risk claim is made.

| Strategy | One-way bps | Net growth | Max drawdown | Worst-5% mean hourly loss | Exposure | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always_long | 0 | 25.56% | -32.92% | 1.0727% | 100.00% | 5.61 |
| always_long | 15 | 24.51% | -33.03% | 1.0729% | 100.00% | 5.61 |
| always_long | 30 | 23.46% | -33.14% | 1.0731% | 100.00% | 5.61 |
| drr | 0 | 16.82% | -32.92% | 1.0699% | 98.91% | 9.51 |
| drr | 15 | 15.17% | -33.03% | 1.0703% | 98.91% | 9.51 |
| drr | 30 | 13.53% | -33.14% | 1.0712% | 98.91% | 9.50 |
| drr_matched_exposure | 0 | 25.31% | -32.61% | 1.0609% | 98.91% | 5.55 |
| drr_matched_exposure | 15 | 24.27% | -32.72% | 1.0611% | 98.91% | 5.55 |
| drr_matched_exposure | 30 | 23.24% | -32.83% | 1.0613% | 98.91% | 5.55 |
| rooting_only | 0 | 15.79% | -32.92% | 1.0691% | 97.24% | 11.45 |
| rooting_only | 15 | 13.82% | -33.03% | 1.0700% | 97.24% | 11.45 |
| rooting_only | 30 | 11.88% | -33.14% | 1.0708% | 97.24% | 11.44 |
| rooting_only_matched_exposure | 0 | 24.93% | -32.14% | 1.0431% | 97.24% | 5.47 |
| rooting_only_matched_exposure | 15 | 23.91% | -32.25% | 1.0433% | 97.24% | 5.47 |
| rooting_only_matched_exposure | 30 | 22.90% | -32.36% | 1.0435% | 97.24% | 5.47 |
| spectral_only | 0 | 14.88% | -32.92% | 1.0706% | 99.45% | 7.54 |
| spectral_only | 15 | 13.59% | -33.03% | 1.0708% | 99.45% | 7.53 |
| spectral_only | 30 | 12.31% | -33.14% | 1.0710% | 99.45% | 7.53 |
| spectral_only_matched_exposure | 0 | 25.43% | -32.76% | 1.0668% | 99.45% | 5.58 |
| spectral_only_matched_exposure | 15 | 24.39% | -32.87% | 1.0670% | 99.45% | 5.58 |
| spectral_only_matched_exposure | 30 | 23.35% | -32.98% | 1.0672% | 99.45% | 5.58 |
| signed_correlation | 0 | 4.64% | -35.18% | 1.0270% | 90.16% | 27.16 |
| signed_correlation | 15 | 0.46% | -36.26% | 1.0272% | 90.16% | 27.14 |
| signed_correlation | 30 | -3.55% | -37.32% | 1.0275% | 90.16% | 27.12 |
| signed_correlation_matched_exposure | 0 | 23.29% | -30.12% | 0.9671% | 90.16% | 5.22 |
| signed_correlation_matched_exposure | 15 | 22.33% | -30.23% | 0.9673% | 90.16% | 5.21 |
| signed_correlation_matched_exposure | 30 | 21.38% | -30.34% | 0.9675% | 90.16% | 5.21 |
| var_cusum | 0 | 25.15% | -28.13% | 0.8910% | 78.68% | 8.68 |
| var_cusum | 15 | 23.53% | -28.19% | 0.8912% | 78.68% | 8.68 |
| var_cusum | 30 | 21.93% | -28.25% | 0.8917% | 78.68% | 8.67 |
| var_cusum_matched_exposure | 0 | 20.56% | -26.74% | 0.8440% | 78.68% | 5.05 |
| var_cusum_matched_exposure | 15 | 19.65% | -26.86% | 0.8442% | 78.68% | 5.04 |
| var_cusum_matched_exposure | 30 | 18.75% | -26.98% | 0.8444% | 78.68% | 5.04 |
| var_bocpd | 0 | 14.83% | -32.32% | 1.0006% | 86.20% | 51.06 |
| var_bocpd | 15 | 6.36% | -34.43% | 1.0013% | 86.20% | 51.02 |
| var_bocpd | 30 | -1.48% | -36.46% | 1.0029% | 86.20% | 50.98 |
| var_bocpd_matched_exposure | 0 | 22.36% | -28.97% | 0.9246% | 86.20% | 5.13 |
| var_bocpd_matched_exposure | 15 | 21.42% | -29.08% | 0.9248% | 86.20% | 5.13 |
| var_bocpd_matched_exposure | 30 | 20.49% | -29.19% | 0.9250% | 86.20% | 5.13 |

## Scope and provenance

Only lead-lag breakdown and descriptive risk attribution are implemented here. HAR-RV/GARCH volatility forecasting is a separate future study. Simulation results cannot establish real-market early detection, causality or trading edge. Historical 2026 data were not used in the earlier 2018-2025 daily-return study, but are public and not independently blinded.

All raw score traces, episode outcomes, market scores/labels and hourly risk paths are included as deterministic compressed CSV. The separate calibration lock pins code, protocol, input data and calibration evidence. Market data and derived results: Binance Vision, CC BY-NC-SA 4.0 and its dataset terms. Software remains MIT. Calibration budgets do not guarantee equal realized false-alarm rates under distribution shift. The simulation family is deliberately limited and public historical data are not independently blinded.
