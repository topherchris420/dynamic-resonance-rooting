# Financial holdout: delayed daily return prediction

**Registered conclusion: NOT SUPPORTED.**

Validation selected **ridge_drr** as the primary candidate and **zero** as the economic comparator. The test covers 1096 entry dates, 2023-01-01 through 2025-12-31, on BTCUSDT and ETHUSDT. Positive returns alone are not evidence of incremental forecast value.

## Decision gates

| Fixed criterion | Passed |
| --- | --- |
| at least one percent mse improvement against every baseline | False |
| all primary tests holm significant at every block length | False |
| positive cumulative return at stress cost | True |
| beats selected baseline in at least two test years | True |
| at least 85 percent coverage on each asset | True |

## Every forecasting result

| Model | MSE | MAE | 90% interval coverage | Mean width | Interval score |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 0.00084675 | 0.019956 | 90.05% | 0.1010 | 0.1434 |
| historical_mean | 0.00084442 | 0.020009 | 90.05% | 0.1015 | 0.1430 |
| momentum20 | 0.00088337 | 0.020758 | 90.01% | 0.1024 | 0.1443 |
| ridge | 0.00084450 | 0.020000 | 90.19% | 0.1015 | 0.1429 |
| hgb | 0.00089963 | 0.021165 | 90.05% | 0.1059 | 0.1456 |
| ridge_drr | 0.00084385 | 0.020000 | 89.96% | 0.1014 | 0.1428 |
| hgb_drr | 0.00089248 | 0.021057 | 90.24% | 0.1046 | 0.1450 |
| hgb_spectral | 0.00090260 | 0.021257 | 90.15% | 0.1056 | 0.1465 |
| hgb_rooting | 0.00089025 | 0.020989 | 90.47% | 0.1053 | 0.1447 |
| hgb_noise | 0.00090108 | 0.021017 | 90.19% | 0.1048 | 0.1472 |

## All cost scenarios

Long/cash positions, fixed 30 bps forecast threshold, full entry/exit costs and drift-aware rebalancing. Costs are hypothetical one-way scenarios; 15 bps consists of 10 bps fees plus 5 bps slippage. Cash earns zero in the quote currency. Sharpe uses 365 days and is descriptive.

| Model | Cost bps | Total net return | CAGR | Sharpe | Max drawdown | Exposure | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 0 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| zero | 15 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| zero | 30 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| historical_mean | 0 | 13.88% | 4.42% | 0.454 | -13.95% | 12.55% | 2.05 |
| historical_mean | 15 | 13.53% | 4.32% | 0.444 | -14.03% | 12.55% | 2.05 |
| historical_mean | 30 | 13.18% | 4.21% | 0.435 | -14.11% | 12.55% | 2.05 |
| momentum20 | 0 | 113.02% | 28.64% | 0.996 | -33.90% | 34.81% | 83.01 |
| momentum20 | 15 | 88.08% | 23.41% | 0.855 | -36.28% | 34.81% | 82.97 |
| momentum20 | 30 | 66.06% | 18.40% | 0.715 | -38.57% | 34.81% | 82.93 |
| ridge | 0 | 74.54% | 20.38% | 1.424 | -12.19% | 11.45% | 89.98 |
| ridge | 15 | 52.50% | 15.09% | 1.096 | -12.46% | 11.45% | 89.95 |
| ridge | 30 | 33.25% | 10.03% | 0.766 | -14.61% | 11.45% | 89.91 |
| hgb | 0 | 119.72% | 29.97% | 0.971 | -26.03% | 44.57% | 406.92 |
| hgb | 15 | 19.34% | 6.06% | 0.343 | -39.02% | 44.57% | 406.70 |
| hgb | 30 | -35.18% | -13.45% | -0.285 | -50.51% | 44.57% | 406.49 |
| ridge_drr | 0 | 113.60% | 28.76% | 1.925 | -5.83% | 12.82% | 98.02 |
| ridge_drr | 15 | 84.40% | 22.60% | 1.571 | -7.31% | 12.82% | 97.97 |
| ridge_drr | 30 | 59.19% | 16.75% | 1.213 | -10.00% | 12.82% | 97.93 |
| hgb_drr | 0 | 200.39% | 44.24% | 1.296 | -21.40% | 42.02% | 368.00 |
| hgb_drr | 15 | 72.96% | 20.02% | 0.726 | -29.15% | 42.02% | 367.81 |
| hgb_drr | 30 | -0.42% | -0.14% | 0.156 | -42.19% | 42.02% | 367.62 |
| hgb_spectral | 0 | 154.94% | 36.57% | 1.115 | -23.88% | 44.48% | 376.21 |
| hgb_spectral | 15 | 45.00% | 13.17% | 0.541 | -29.16% | 44.48% | 376.01 |
| hgb_spectral | 30 | -17.54% | -6.22% | -0.034 | -47.94% | 44.48% | 375.82 |
| hgb_rooting | 0 | 156.64% | 36.87% | 1.169 | -32.03% | 41.83% | 403.03 |
| hgb_rooting | 15 | 40.21% | 11.91% | 0.518 | -46.13% | 41.83% | 402.81 |
| hgb_rooting | 30 | -23.40% | -8.50% | -0.132 | -57.45% | 41.83% | 402.59 |
| hgb_noise | 0 | 222.94% | 47.76% | 1.390 | -32.20% | 44.98% | 474.80 |
| hgb_noise | 15 | 58.42% | 16.56% | 0.641 | -39.65% | 44.98% | 474.53 |
| hgb_noise | 30 | -22.29% | -8.05% | -0.108 | -49.91% | 44.98% | 474.28 |
| cash | 0 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| cash | 15 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| cash | 30 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | 0.00 |
| equal_weight_long | 0 | 283.18% | 56.42% | 1.118 | -48.01% | 100.00% | 9.68 |
| equal_weight_long | 15 | 277.66% | 55.66% | 1.109 | -48.09% | 100.00% | 9.68 |
| equal_weight_long | 30 | 272.21% | 54.91% | 1.100 | -48.17% | 100.00% | 9.68 |

## Complete multiplicity family

Positive differences favor the candidate. Loss differences are in squared-return units; net differences are daily portfolio returns. Every row is retained. The intervals are pointwise 95% bootstrap intervals; only the Holm-adjusted p-values control the registered 14-test family (approximately, subject to bootstrap assumptions). All three block lengths must pass; none is selected after seeing the results.

| Comparison | Block days | Mean difference | CI low | CI high | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mse:ridge_drr>zero | 7 | 0.00000290 | -0.00000259 | 0.00000863 | 0.1526 | 1.0000 |
| mse:ridge_drr>historical_mean | 7 | 0.00000057 | -0.00000197 | 0.00000305 | 0.3250 | 1.0000 |
| mse:ridge_drr>momentum20 | 7 | 0.00003952 | 0.00002148 | 0.00005705 | 0.0001 | 0.0014 |
| mse:ridge_drr>ridge | 7 | 0.00000065 | -0.00000054 | 0.00000181 | 0.1424 | 1.0000 |
| mse:ridge_drr>hgb | 7 | 0.00005577 | 0.00003332 | 0.00007796 | 0.0001 | 0.0014 |
| net:ridge_drr>cash | 7 | 0.00058328 | 0.00020512 | 0.00100090 | 0.0040 | 0.0480 |
| net:ridge_drr>zero | 7 | 0.00058328 | 0.00020512 | 0.00100090 | 0.0040 | 0.0480 |
| mse:hgb_drr>zero | 7 | -0.00004573 | -0.00006741 | -0.00002368 | 1.0000 | 1.0000 |
| mse:hgb_drr>historical_mean | 7 | -0.00004805 | -0.00006847 | -0.00002747 | 1.0000 | 1.0000 |
| mse:hgb_drr>momentum20 | 7 | -0.00000911 | -0.00003517 | 0.00001621 | 0.7580 | 1.0000 |
| mse:hgb_drr>ridge | 7 | -0.00004798 | -0.00006837 | -0.00002754 | 1.0000 | 1.0000 |
| mse:hgb_drr>hgb | 7 | 0.00000715 | -0.00001060 | 0.00002479 | 0.2083 | 1.0000 |
| net:hgb_drr>cash | 7 | 0.00064153 | -0.00035634 | 0.00165008 | 0.1083 | 1.0000 |
| net:hgb_drr>zero | 7 | 0.00064153 | -0.00035634 | 0.00165008 | 0.1083 | 1.0000 |
| mse:ridge_drr>zero | 30 | 0.00000290 | -0.00000283 | 0.00000937 | 0.1717 | 1.0000 |
| mse:ridge_drr>historical_mean | 30 | 0.00000057 | -0.00000170 | 0.00000284 | 0.3027 | 1.0000 |
| mse:ridge_drr>momentum20 | 30 | 0.00003952 | 0.00002542 | 0.00005408 | 0.0001 | 0.0014 |
| mse:ridge_drr>ridge | 30 | 0.00000065 | -0.00000068 | 0.00000199 | 0.1742 | 1.0000 |
| mse:ridge_drr>hgb | 30 | 0.00005577 | 0.00003421 | 0.00007668 | 0.0001 | 0.0014 |
| net:ridge_drr>cash | 30 | 0.00058328 | 0.00017371 | 0.00109985 | 0.0162 | 0.1944 |
| net:ridge_drr>zero | 30 | 0.00058328 | 0.00017371 | 0.00109985 | 0.0162 | 0.1944 |
| mse:hgb_drr>zero | 30 | -0.00004573 | -0.00006654 | -0.00002493 | 1.0000 | 1.0000 |
| mse:hgb_drr>historical_mean | 30 | -0.00004805 | -0.00006663 | -0.00002946 | 1.0000 | 1.0000 |
| mse:hgb_drr>momentum20 | 30 | -0.00000911 | -0.00003112 | 0.00001260 | 0.8006 | 1.0000 |
| mse:hgb_drr>ridge | 30 | -0.00004798 | -0.00006670 | -0.00002937 | 1.0000 | 1.0000 |
| mse:hgb_drr>hgb | 30 | 0.00000715 | -0.00001095 | 0.00002554 | 0.2224 | 1.0000 |
| net:hgb_drr>cash | 30 | 0.00064153 | -0.00036920 | 0.00175904 | 0.1248 | 1.0000 |
| net:hgb_drr>zero | 30 | 0.00064153 | -0.00036920 | 0.00175904 | 0.1248 | 1.0000 |
| mse:ridge_drr>zero | 90 | 0.00000290 | -0.00000255 | 0.00000865 | 0.1613 | 1.0000 |
| mse:ridge_drr>historical_mean | 90 | 0.00000057 | -0.00000169 | 0.00000279 | 0.3016 | 1.0000 |
| mse:ridge_drr>momentum20 | 90 | 0.00003952 | 0.00002580 | 0.00005399 | 0.0001 | 0.0014 |
| mse:ridge_drr>ridge | 90 | 0.00000065 | -0.00000089 | 0.00000223 | 0.2005 | 1.0000 |
| mse:ridge_drr>hgb | 90 | 0.00005577 | 0.00003721 | 0.00007389 | 0.0001 | 0.0014 |
| net:ridge_drr>cash | 90 | 0.00058328 | 0.00015105 | 0.00111431 | 0.0174 | 0.2088 |
| net:ridge_drr>zero | 90 | 0.00058328 | 0.00015105 | 0.00111431 | 0.0174 | 0.2088 |
| mse:hgb_drr>zero | 90 | -0.00004573 | -0.00006486 | -0.00002664 | 1.0000 | 1.0000 |
| mse:hgb_drr>historical_mean | 90 | -0.00004805 | -0.00006461 | -0.00003134 | 1.0000 | 1.0000 |
| mse:hgb_drr>momentum20 | 90 | -0.00000911 | -0.00003171 | 0.00001481 | 0.7795 | 1.0000 |
| mse:hgb_drr>ridge | 90 | -0.00004798 | -0.00006484 | -0.00003087 | 1.0000 | 1.0000 |
| mse:hgb_drr>hgb | 90 | 0.00000715 | -0.00001139 | 0.00002775 | 0.2313 | 1.0000 |
| net:hgb_drr>cash | 90 | 0.00064153 | -0.00033384 | 0.00167442 | 0.1112 | 1.0000 |
| net:hgb_drr>zero | 90 | 0.00064153 | -0.00033384 | 0.00167442 | 0.1112 | 1.0000 |

## Regimes and predictive uncertainty

The high-volatility cutoff is 0.045976 daily return standard deviation, estimated only on development data. Trends use trailing 60-day momentum. No regime is defined from future test outcomes. The following complete per-asset table reports every model, calendar year and volatility/trend stratum. These are descriptive comparisons.

| Model | Group | Stratum | N | MSE | Coverage | Width | Interval score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zero | symbol | BTCUSDT | 1096 | 0.00059667 | 90.42% | 0.0876 | 0.1225 |
| zero | symbol | ETHUSDT | 1096 | 0.00109683 | 89.69% | 0.1144 | 0.1643 |
| zero | year | 2023/BTCUSDT | 365 | 0.00053843 | 90.41% | 0.0858 | 0.1279 |
| zero | year | 2023/ETHUSDT | 365 | 0.00060403 | 90.41% | 0.0860 | 0.1230 |
| zero | year | 2024/BTCUSDT | 366 | 0.00077625 | 90.98% | 0.1019 | 0.1353 |
| zero | year | 2024/ETHUSDT | 366 | 0.00115137 | 89.07% | 0.1182 | 0.1685 |
| zero | year | 2025/BTCUSDT | 365 | 0.00047483 | 89.86% | 0.0750 | 0.1044 |
| zero | year | 2025/ETHUSDT | 365 | 0.00153493 | 89.59% | 0.1389 | 0.2013 |
| zero | regime | high_vol_down/BTCUSDT | 23 | 0.00102103 | 100.00% | 0.1443 | 0.1443 |
| zero | regime | high_vol_down/ETHUSDT | 23 | 0.00126194 | 100.00% | 0.2170 | 0.2170 |
| zero | regime | high_vol_up/BTCUSDT | 5 | 0.00095983 | 100.00% | 0.1842 | 0.1842 |
| zero | regime | high_vol_up/ETHUSDT | 5 | 0.00099377 | 100.00% | 0.1557 | 0.1557 |
| zero | regime | low_vol_down/BTCUSDT | 383 | 0.00057277 | 91.64% | 0.0867 | 0.1194 |
| zero | regime | low_vol_down/ETHUSDT | 383 | 0.00122675 | 89.82% | 0.1158 | 0.1706 |
| zero | regime | low_vol_up/BTCUSDT | 685 | 0.00059313 | 89.34% | 0.0855 | 0.1232 |
| zero | regime | low_vol_up/ETHUSDT | 685 | 0.00101940 | 89.20% | 0.1099 | 0.1590 |
| historical_mean | symbol | BTCUSDT | 1096 | 0.00059316 | 90.15% | 0.0870 | 0.1224 |
| historical_mean | symbol | ETHUSDT | 1096 | 0.00109568 | 89.96% | 0.1160 | 0.1637 |
| historical_mean | year | 2023/BTCUSDT | 365 | 0.00053087 | 89.86% | 0.0859 | 0.1272 |
| historical_mean | year | 2023/ETHUSDT | 365 | 0.00060216 | 90.14% | 0.0903 | 0.1224 |
| historical_mean | year | 2024/BTCUSDT | 366 | 0.00077197 | 90.44% | 0.0995 | 0.1351 |
| historical_mean | year | 2024/ETHUSDT | 366 | 0.00114880 | 90.16% | 0.1187 | 0.1673 |
| historical_mean | year | 2025/BTCUSDT | 365 | 0.00047616 | 90.14% | 0.0756 | 0.1049 |
| historical_mean | year | 2025/ETHUSDT | 365 | 0.00153594 | 89.59% | 0.1389 | 0.2014 |
| historical_mean | regime | high_vol_down/BTCUSDT | 23 | 0.00101796 | 100.00% | 0.1451 | 0.1451 |
| historical_mean | regime | high_vol_down/ETHUSDT | 23 | 0.00126718 | 100.00% | 0.2191 | 0.2191 |
| historical_mean | regime | high_vol_up/BTCUSDT | 5 | 0.00092055 | 100.00% | 0.1696 | 0.1696 |
| historical_mean | regime | high_vol_up/ETHUSDT | 5 | 0.00096843 | 100.00% | 0.1685 | 0.1685 |
| historical_mean | regime | low_vol_down/BTCUSDT | 383 | 0.00057180 | 91.38% | 0.0869 | 0.1191 |
| historical_mean | regime | low_vol_down/ETHUSDT | 383 | 0.00123260 | 89.82% | 0.1161 | 0.1704 |
| historical_mean | regime | low_vol_up/BTCUSDT | 685 | 0.00058845 | 89.05% | 0.0845 | 0.1231 |
| historical_mean | regime | low_vol_up/ETHUSDT | 685 | 0.00101430 | 89.64% | 0.1121 | 0.1580 |
| momentum20 | symbol | BTCUSDT | 1096 | 0.00062381 | 89.96% | 0.0885 | 0.1239 |
| momentum20 | symbol | ETHUSDT | 1096 | 0.00114293 | 90.05% | 0.1162 | 0.1647 |
| momentum20 | year | 2023/BTCUSDT | 365 | 0.00056644 | 90.68% | 0.0874 | 0.1301 |
| momentum20 | year | 2023/ETHUSDT | 365 | 0.00062991 | 90.41% | 0.0912 | 0.1253 |
| momentum20 | year | 2024/BTCUSDT | 366 | 0.00081350 | 89.89% | 0.1020 | 0.1370 |
| momentum20 | year | 2024/ETHUSDT | 366 | 0.00121668 | 90.16% | 0.1167 | 0.1691 |
| momentum20 | year | 2025/BTCUSDT | 365 | 0.00049096 | 89.32% | 0.0761 | 0.1047 |
| momentum20 | year | 2025/ETHUSDT | 365 | 0.00158200 | 89.59% | 0.1406 | 0.1998 |
| momentum20 | regime | high_vol_down/BTCUSDT | 23 | 0.00106875 | 100.00% | 0.1504 | 0.1504 |
| momentum20 | regime | high_vol_down/ETHUSDT | 23 | 0.00138113 | 100.00% | 0.2158 | 0.2158 |
| momentum20 | regime | high_vol_up/BTCUSDT | 5 | 0.00095522 | 100.00% | 0.1728 | 0.1728 |
| momentum20 | regime | high_vol_up/ETHUSDT | 5 | 0.00108578 | 100.00% | 0.1728 | 0.1728 |
| momentum20 | regime | low_vol_down/BTCUSDT | 383 | 0.00060940 | 91.64% | 0.0879 | 0.1198 |
| momentum20 | regime | low_vol_down/ETHUSDT | 383 | 0.00127562 | 89.56% | 0.1154 | 0.1699 |
| momentum20 | regime | low_vol_up/BTCUSDT | 685 | 0.00061450 | 88.61% | 0.0862 | 0.1250 |
| momentum20 | regime | low_vol_up/ETHUSDT | 685 | 0.00106117 | 89.93% | 0.1129 | 0.1601 |
| ridge | symbol | BTCUSDT | 1096 | 0.00059398 | 90.33% | 0.0878 | 0.1224 |
| ridge | symbol | ETHUSDT | 1096 | 0.00109502 | 90.05% | 0.1152 | 0.1633 |
| ridge | year | 2023/BTCUSDT | 365 | 0.00053172 | 90.96% | 0.0871 | 0.1276 |
| ridge | year | 2023/ETHUSDT | 365 | 0.00060118 | 90.41% | 0.0896 | 0.1221 |
| ridge | year | 2024/BTCUSDT | 366 | 0.00077277 | 90.44% | 0.1012 | 0.1350 |
| ridge | year | 2024/ETHUSDT | 366 | 0.00114858 | 90.16% | 0.1169 | 0.1667 |
| ridge | year | 2025/BTCUSDT | 365 | 0.00047696 | 89.59% | 0.0750 | 0.1047 |
| ridge | year | 2025/ETHUSDT | 365 | 0.00153516 | 89.59% | 0.1391 | 0.2011 |
| ridge | regime | high_vol_down/BTCUSDT | 23 | 0.00101584 | 100.00% | 0.1434 | 0.1434 |
| ridge | regime | high_vol_down/ETHUSDT | 23 | 0.00125620 | 100.00% | 0.2177 | 0.2177 |
| ridge | regime | high_vol_up/BTCUSDT | 5 | 0.00093128 | 100.00% | 0.1764 | 0.1764 |
| ridge | regime | high_vol_up/ETHUSDT | 5 | 0.00099985 | 100.00% | 0.1672 | 0.1672 |
| ridge | regime | low_vol_down/BTCUSDT | 383 | 0.00057454 | 91.12% | 0.0871 | 0.1195 |
| ridge | regime | low_vol_down/ETHUSDT | 383 | 0.00123358 | 89.82% | 0.1153 | 0.1702 |
| ridge | regime | low_vol_up/BTCUSDT | 685 | 0.00058822 | 89.49% | 0.0857 | 0.1230 |
| ridge | regime | low_vol_up/ETHUSDT | 685 | 0.00101284 | 89.78% | 0.1114 | 0.1576 |
| hgb | symbol | BTCUSDT | 1096 | 0.00061909 | 90.51% | 0.0915 | 0.1235 |
| hgb | symbol | ETHUSDT | 1096 | 0.00118016 | 89.60% | 0.1204 | 0.1677 |
| hgb | year | 2023/BTCUSDT | 365 | 0.00055742 | 89.86% | 0.0896 | 0.1269 |
| hgb | year | 2023/ETHUSDT | 365 | 0.00069628 | 89.59% | 0.0967 | 0.1311 |
| hgb | year | 2024/BTCUSDT | 366 | 0.00078045 | 90.71% | 0.1038 | 0.1337 |
| hgb | year | 2024/ETHUSDT | 366 | 0.00121164 | 90.16% | 0.1226 | 0.1687 |
| hgb | year | 2025/BTCUSDT | 365 | 0.00051897 | 90.96% | 0.0810 | 0.1099 |
| hgb | year | 2025/ETHUSDT | 365 | 0.00163248 | 89.04% | 0.1420 | 0.2034 |
| hgb | regime | high_vol_down/BTCUSDT | 23 | 0.00107014 | 95.65% | 0.1531 | 0.1659 |
| hgb | regime | high_vol_down/ETHUSDT | 23 | 0.00142103 | 100.00% | 0.2094 | 0.2094 |
| hgb | regime | high_vol_up/BTCUSDT | 5 | 0.00086434 | 100.00% | 0.1820 | 0.1820 |
| hgb | regime | high_vol_up/ETHUSDT | 5 | 0.00108249 | 100.00% | 0.1943 | 0.1943 |
| hgb | regime | low_vol_down/BTCUSDT | 383 | 0.00059530 | 90.08% | 0.0890 | 0.1187 |
| hgb | regime | low_vol_down/ETHUSDT | 383 | 0.00135639 | 89.30% | 0.1186 | 0.1779 |
| hgb | regime | low_vol_up/BTCUSDT | 685 | 0.00061546 | 90.51% | 0.0901 | 0.1243 |
| hgb | regime | low_vol_up/ETHUSDT | 685 | 0.00107426 | 89.34% | 0.1179 | 0.1605 |
| ridge_drr | symbol | BTCUSDT | 1096 | 0.00059330 | 90.24% | 0.0879 | 0.1225 |
| ridge_drr | symbol | ETHUSDT | 1096 | 0.00109441 | 89.69% | 0.1149 | 0.1631 |
| ridge_drr | year | 2023/BTCUSDT | 365 | 0.00052952 | 90.68% | 0.0866 | 0.1271 |
| ridge_drr | year | 2023/ETHUSDT | 365 | 0.00060030 | 89.32% | 0.0884 | 0.1215 |
| ridge_drr | year | 2024/BTCUSDT | 366 | 0.00077171 | 90.44% | 0.1018 | 0.1352 |
| ridge_drr | year | 2024/ETHUSDT | 366 | 0.00114660 | 90.16% | 0.1172 | 0.1663 |
| ridge_drr | year | 2025/BTCUSDT | 365 | 0.00047816 | 89.59% | 0.0754 | 0.1051 |
| ridge_drr | year | 2025/ETHUSDT | 365 | 0.00153619 | 89.59% | 0.1390 | 0.2014 |
| ridge_drr | regime | high_vol_down/BTCUSDT | 23 | 0.00101518 | 100.00% | 0.1443 | 0.1443 |
| ridge_drr | regime | high_vol_down/ETHUSDT | 23 | 0.00125565 | 100.00% | 0.2174 | 0.2174 |
| ridge_drr | regime | high_vol_up/BTCUSDT | 5 | 0.00093944 | 100.00% | 0.1766 | 0.1766 |
| ridge_drr | regime | high_vol_up/ETHUSDT | 5 | 0.00100296 | 100.00% | 0.1703 | 0.1703 |
| ridge_drr | regime | low_vol_down/BTCUSDT | 383 | 0.00057402 | 91.12% | 0.0876 | 0.1196 |
| ridge_drr | regime | low_vol_down/ETHUSDT | 383 | 0.00123357 | 89.56% | 0.1149 | 0.1701 |
| ridge_drr | regime | low_vol_up/BTCUSDT | 685 | 0.00058738 | 89.34% | 0.0855 | 0.1229 |
| ridge_drr | regime | low_vol_up/ETHUSDT | 685 | 0.00101186 | 89.34% | 0.1110 | 0.1572 |
| hgb_drr | symbol | BTCUSDT | 1096 | 0.00062094 | 90.42% | 0.0906 | 0.1239 |
| hgb_drr | symbol | ETHUSDT | 1096 | 0.00116401 | 90.05% | 0.1186 | 0.1662 |
| hgb_drr | year | 2023/BTCUSDT | 365 | 0.00055495 | 90.96% | 0.0871 | 0.1257 |
| hgb_drr | year | 2023/ETHUSDT | 365 | 0.00067305 | 90.41% | 0.0940 | 0.1260 |
| hgb_drr | year | 2024/BTCUSDT | 366 | 0.00079052 | 89.62% | 0.1025 | 0.1349 |
| hgb_drr | year | 2024/ETHUSDT | 366 | 0.00122880 | 90.16% | 0.1235 | 0.1689 |
| hgb_drr | year | 2025/BTCUSDT | 365 | 0.00051689 | 90.68% | 0.0821 | 0.1110 |
| hgb_drr | year | 2025/ETHUSDT | 365 | 0.00159001 | 89.59% | 0.1383 | 0.2036 |
| hgb_drr | regime | high_vol_down/BTCUSDT | 23 | 0.00109534 | 95.65% | 0.1527 | 0.1626 |
| hgb_drr | regime | high_vol_down/ETHUSDT | 23 | 0.00130206 | 100.00% | 0.2095 | 0.2095 |
| hgb_drr | regime | high_vol_up/BTCUSDT | 5 | 0.00104384 | 100.00% | 0.1805 | 0.1805 |
| hgb_drr | regime | high_vol_up/ETHUSDT | 5 | 0.00149392 | 100.00% | 0.1942 | 0.1942 |
| hgb_drr | regime | low_vol_down/BTCUSDT | 383 | 0.00059777 | 90.34% | 0.0876 | 0.1205 |
| hgb_drr | regime | low_vol_down/ETHUSDT | 383 | 0.00131846 | 89.56% | 0.1177 | 0.1761 |
| hgb_drr | regime | low_vol_up/BTCUSDT | 685 | 0.00061488 | 90.22% | 0.0894 | 0.1240 |
| hgb_drr | regime | low_vol_up/ETHUSDT | 685 | 0.00107061 | 89.93% | 0.1156 | 0.1590 |
| hgb_spectral | symbol | BTCUSDT | 1096 | 0.00062398 | 90.60% | 0.0914 | 0.1249 |
| hgb_spectral | symbol | ETHUSDT | 1096 | 0.00118121 | 89.69% | 0.1198 | 0.1681 |
| hgb_spectral | year | 2023/BTCUSDT | 365 | 0.00055695 | 90.41% | 0.0887 | 0.1280 |
| hgb_spectral | year | 2023/ETHUSDT | 365 | 0.00071387 | 89.04% | 0.0956 | 0.1315 |
| hgb_spectral | year | 2024/BTCUSDT | 366 | 0.00079599 | 90.16% | 0.1050 | 0.1354 |
| hgb_spectral | year | 2024/ETHUSDT | 366 | 0.00122421 | 90.44% | 0.1258 | 0.1688 |
| hgb_spectral | year | 2025/BTCUSDT | 365 | 0.00051853 | 91.23% | 0.0805 | 0.1112 |
| hgb_spectral | year | 2025/ETHUSDT | 365 | 0.00160545 | 89.59% | 0.1379 | 0.2039 |
| hgb_spectral | regime | high_vol_down/BTCUSDT | 23 | 0.00106589 | 95.65% | 0.1517 | 0.1607 |
| hgb_spectral | regime | high_vol_down/ETHUSDT | 23 | 0.00130720 | 100.00% | 0.2079 | 0.2079 |
| hgb_spectral | regime | high_vol_up/BTCUSDT | 5 | 0.00116436 | 100.00% | 0.1812 | 0.1812 |
| hgb_spectral | regime | high_vol_up/ETHUSDT | 5 | 0.00140517 | 100.00% | 0.1953 | 0.1953 |
| hgb_spectral | regime | low_vol_down/BTCUSDT | 383 | 0.00060136 | 90.86% | 0.0890 | 0.1207 |
| hgb_spectral | regime | low_vol_down/ETHUSDT | 383 | 0.00134936 | 89.03% | 0.1188 | 0.1788 |
| hgb_spectral | regime | low_vol_up/BTCUSDT | 685 | 0.00061785 | 90.22% | 0.0901 | 0.1256 |
| hgb_spectral | regime | low_vol_up/ETHUSDT | 685 | 0.00108133 | 89.64% | 0.1168 | 0.1605 |
| hgb_rooting | symbol | BTCUSDT | 1096 | 0.00061414 | 90.78% | 0.0910 | 0.1222 |
| hgb_rooting | symbol | ETHUSDT | 1096 | 0.00116636 | 90.15% | 0.1197 | 0.1672 |
| hgb_rooting | year | 2023/BTCUSDT | 365 | 0.00055311 | 90.41% | 0.0878 | 0.1246 |
| hgb_rooting | year | 2023/ETHUSDT | 365 | 0.00064859 | 90.14% | 0.0951 | 0.1260 |
| hgb_rooting | year | 2024/BTCUSDT | 366 | 0.00077360 | 90.44% | 0.1035 | 0.1331 |
| hgb_rooting | year | 2024/ETHUSDT | 366 | 0.00120864 | 90.98% | 0.1245 | 0.1707 |
| hgb_rooting | year | 2025/BTCUSDT | 365 | 0.00051526 | 91.51% | 0.0817 | 0.1088 |
| hgb_rooting | year | 2025/ETHUSDT | 365 | 0.00164173 | 89.32% | 0.1394 | 0.2047 |
| hgb_rooting | regime | high_vol_down/BTCUSDT | 23 | 0.00107744 | 95.65% | 0.1521 | 0.1670 |
| hgb_rooting | regime | high_vol_down/ETHUSDT | 23 | 0.00141727 | 100.00% | 0.2086 | 0.2086 |
| hgb_rooting | regime | high_vol_up/BTCUSDT | 5 | 0.00103623 | 100.00% | 0.1783 | 0.1783 |
| hgb_rooting | regime | high_vol_up/ETHUSDT | 5 | 0.00121458 | 100.00% | 0.1962 | 0.1962 |
| hgb_rooting | regime | low_vol_down/BTCUSDT | 383 | 0.00059977 | 89.82% | 0.0891 | 0.1197 |
| hgb_rooting | regime | low_vol_down/ETHUSDT | 383 | 0.00136400 | 89.56% | 0.1176 | 0.1790 |
| hgb_rooting | regime | low_vol_up/BTCUSDT | 685 | 0.00060354 | 91.09% | 0.0894 | 0.1216 |
| hgb_rooting | regime | low_vol_up/ETHUSDT | 685 | 0.00104707 | 90.07% | 0.1173 | 0.1589 |
| hgb_noise | symbol | BTCUSDT | 1096 | 0.00062312 | 90.60% | 0.0903 | 0.1250 |
| hgb_noise | symbol | ETHUSDT | 1096 | 0.00117904 | 89.78% | 0.1193 | 0.1695 |
| hgb_noise | year | 2023/BTCUSDT | 365 | 0.00056175 | 89.86% | 0.0852 | 0.1286 |
| hgb_noise | year | 2023/ETHUSDT | 365 | 0.00068907 | 89.59% | 0.0926 | 0.1311 |
| hgb_noise | year | 2024/BTCUSDT | 366 | 0.00077426 | 90.44% | 0.1055 | 0.1357 |
| hgb_noise | year | 2024/ETHUSDT | 366 | 0.00119546 | 90.16% | 0.1223 | 0.1699 |
| hgb_noise | year | 2025/BTCUSDT | 365 | 0.00053293 | 91.51% | 0.0803 | 0.1106 |
| hgb_noise | year | 2025/ETHUSDT | 365 | 0.00165255 | 89.59% | 0.1432 | 0.2075 |
| hgb_noise | regime | high_vol_down/BTCUSDT | 23 | 0.00110182 | 95.65% | 0.1550 | 0.1623 |
| hgb_noise | regime | high_vol_down/ETHUSDT | 23 | 0.00134332 | 100.00% | 0.2161 | 0.2161 |
| hgb_noise | regime | high_vol_up/BTCUSDT | 5 | 0.00082695 | 100.00% | 0.2012 | 0.2012 |
| hgb_noise | regime | high_vol_up/ETHUSDT | 5 | 0.00110809 | 100.00% | 0.1866 | 0.1866 |
| hgb_noise | regime | low_vol_down/BTCUSDT | 383 | 0.00060574 | 89.82% | 0.0870 | 0.1198 |
| hgb_noise | regime | low_vol_down/ETHUSDT | 383 | 0.00134446 | 88.51% | 0.1176 | 0.1779 |
| hgb_noise | regime | low_vol_up/BTCUSDT | 685 | 0.00061528 | 90.80% | 0.0892 | 0.1261 |
| hgb_noise | regime | low_vol_up/ETHUSDT | 685 | 0.00108156 | 90.07% | 0.1166 | 0.1631 |

## Economic regime breakdown

Base-cost returns are partitioned from the continuous portfolio path, retaining actual boundary turnover. Noncontiguous regime growth is a conditional descriptive calculation, not a separately tradable strategy. Detailed costs, annualized statistics and break-even costs are in summary.json.

| Model | Group | Stratum | Days | Mean daily net | Conditional growth | Exposure |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| zero | year | 2023 | 365 | 0.000000 | 0.00% | 0.00% |
| zero | year | 2024 | 366 | 0.000000 | 0.00% | 0.00% |
| zero | year | 2025 | 365 | 0.000000 | 0.00% | 0.00% |
| zero | regime | high_vol_down | 23 | 0.000000 | 0.00% | 0.00% |
| zero | regime | high_vol_up | 5 | 0.000000 | 0.00% | 0.00% |
| zero | regime | low_vol_down | 383 | 0.000000 | 0.00% | 0.00% |
| zero | regime | low_vol_up | 685 | 0.000000 | 0.00% | 0.00% |
| historical_mean | year | 2023 | 365 | 0.000398 | 13.62% | 37.67% |
| historical_mean | year | 2024 | 366 | -0.000002 | -0.07% | 0.00% |
| historical_mean | year | 2025 | 365 | 0.000000 | 0.00% | 0.00% |
| historical_mean | regime | high_vol_down | 23 | 0.000000 | 0.00% | 0.00% |
| historical_mean | regime | high_vol_up | 5 | 0.000000 | 0.00% | 0.00% |
| historical_mean | regime | low_vol_down | 383 | 0.000013 | 0.07% | 11.10% |
| historical_mean | regime | low_vol_up | 685 | 0.000203 | 13.45% | 13.87% |
| momentum20 | year | 2023 | 365 | 0.000892 | 33.25% | 39.45% |
| momentum20 | year | 2024 | 366 | 0.000369 | 7.47% | 40.85% |
| momentum20 | year | 2025 | 365 | 0.000828 | 31.34% | 24.11% |
| momentum20 | regime | high_vol_down | 23 | -0.000011 | -0.03% | 2.17% |
| momentum20 | regime | high_vol_up | 5 | -0.007369 | -3.68% | 30.00% |
| momentum20 | regime | low_vol_down | 383 | -0.000243 | -10.64% | 14.23% |
| momentum20 | regime | low_vol_up | 685 | 0.001303 | 118.58% | 47.45% |
| ridge | year | 2023 | 365 | 0.000727 | 28.54% | 22.74% |
| ridge | year | 2024 | 366 | 0.000440 | 15.90% | 11.34% |
| ridge | year | 2025 | 365 | 0.000065 | 2.37% | 0.27% |
| ridge | regime | high_vol_down | 23 | -0.000473 | -1.12% | 6.52% |
| ridge | regime | high_vol_up | 5 | -0.001456 | -0.77% | 40.00% |
| ridge | regime | low_vol_down | 383 | -0.000099 | -3.77% | 1.04% |
| ridge | regime | low_vol_up | 685 | 0.000739 | 61.52% | 17.23% |
| hgb | year | 2023 | 365 | 0.000807 | 27.07% | 62.60% |
| hgb | year | 2024 | 366 | 0.001144 | 44.15% | 40.98% |
| hgb | year | 2025 | 365 | -0.001041 | -34.85% | 30.14% |
| hgb | regime | high_vol_down | 23 | -0.002085 | -5.17% | 45.65% |
| hgb | regime | high_vol_up | 5 | 0.004577 | 2.29% | 10.00% |
| hgb | regime | low_vol_down | 383 | -0.000861 | -31.75% | 41.12% |
| hgb | regime | low_vol_up | 685 | 0.001004 | 80.28% | 46.72% |
| ridge_drr | year | 2023 | 365 | 0.000836 | 33.36% | 26.71% |
| ridge_drr | year | 2024 | 366 | 0.000849 | 35.08% | 11.48% |
| ridge_drr | year | 2025 | 365 | 0.000065 | 2.37% | 0.27% |
| ridge_drr | regime | high_vol_down | 23 | -0.000473 | -1.12% | 6.52% |
| ridge_drr | regime | high_vol_up | 5 | -0.005221 | -2.60% | 30.00% |
| ridge_drr | regime | low_vol_down | 383 | -0.000013 | -0.76% | 4.70% |
| ridge_drr | regime | low_vol_up | 685 | 0.000995 | 92.94% | 17.45% |
| hgb_drr | year | 2023 | 365 | 0.001256 | 50.84% | 59.45% |
| hgb_drr | year | 2024 | 366 | 0.001027 | 37.04% | 36.34% |
| hgb_drr | year | 2025 | 365 | -0.000360 | -16.33% | 30.27% |
| hgb_drr | regime | high_vol_down | 23 | -0.000498 | -1.68% | 41.30% |
| hgb_drr | regime | high_vol_up | 5 | -0.004917 | -2.46% | 10.00% |
| hgb_drr | regime | low_vol_down | 383 | -0.000714 | -27.50% | 37.73% |
| hgb_drr | regime | low_vol_up | 685 | 0.001478 | 148.76% | 44.67% |
| hgb_spectral | year | 2023 | 365 | 0.000811 | 27.67% | 63.15% |
| hgb_spectral | year | 2024 | 366 | 0.001078 | 39.30% | 39.89% |
| hgb_spectral | year | 2025 | 365 | -0.000438 | -18.47% | 30.41% |
| hgb_spectral | regime | high_vol_down | 23 | -0.000543 | -1.76% | 34.78% |
| hgb_spectral | regime | high_vol_up | 5 | -0.000147 | -0.07% | 0.00% |
| hgb_spectral | regime | low_vol_down | 383 | -0.000471 | -20.66% | 41.25% |
| hgb_spectral | regime | low_vol_up | 685 | 0.001057 | 86.17% | 46.93% |
| hgb_rooting | year | 2023 | 365 | 0.001262 | 51.56% | 53.01% |
| hgb_rooting | year | 2024 | 366 | 0.001365 | 56.18% | 40.03% |
| hgb_rooting | year | 2025 | 365 | -0.001312 | -40.77% | 32.47% |
| hgb_rooting | regime | high_vol_down | 23 | -0.000813 | -2.38% | 43.48% |
| hgb_rooting | regime | high_vol_up | 5 | -0.004917 | -2.46% | 10.00% |
| hgb_rooting | regime | low_vol_down | 383 | -0.000975 | -34.19% | 38.38% |
| hgb_rooting | regime | low_vol_up | 685 | 0.001311 | 123.73% | 43.94% |
| hgb_noise | year | 2023 | 365 | 0.000963 | 35.28% | 57.12% |
| hgb_noise | year | 2024 | 366 | 0.001688 | 74.79% | 43.72% |
| hgb_noise | year | 2025 | 365 | -0.000985 | -33.00% | 34.11% |
| hgb_noise | regime | high_vol_down | 23 | -0.003126 | -7.58% | 50.00% |
| hgb_noise | regime | high_vol_up | 5 | 0.007804 | 3.89% | 30.00% |
| hgb_noise | regime | low_vol_down | 383 | -0.000727 | -27.50% | 41.64% |
| hgb_noise | regime | low_vol_up | 685 | 0.001344 | 127.57% | 46.79% |
| cash | year | 2023 | 365 | 0.000000 | 0.00% | 0.00% |
| cash | year | 2024 | 366 | 0.000000 | 0.00% | 0.00% |
| cash | year | 2025 | 365 | 0.000000 | 0.00% | 0.00% |
| cash | regime | high_vol_down | 23 | 0.000000 | 0.00% | 0.00% |
| cash | regime | high_vol_up | 5 | 0.000000 | 0.00% | 0.00% |
| cash | regime | low_vol_down | 383 | 0.000000 | 0.00% | 0.00% |
| cash | regime | low_vol_up | 685 | 0.000000 | 0.00% | 0.00% |
| equal_weight_long | year | 2023 | 365 | 0.002441 | 121.84% | 100.00% |
| equal_weight_long | year | 2024 | 366 | 0.002068 | 82.56% | 100.00% |
| equal_weight_long | year | 2025 | 365 | 0.000232 | -6.75% | 100.00% |
| equal_weight_long | regime | high_vol_down | 23 | -0.000181 | -1.63% | 100.00% |
| equal_weight_long | regime | high_vol_up | 5 | 0.009767 | 4.76% | 100.00% |
| equal_weight_long | regime | low_vol_down | 383 | 0.000051 | -12.45% | 100.00% |
| equal_weight_long | regime | low_vol_up | 685 | 0.002436 | 318.62% | 100.00% |

## Limits and reproduction

- This is a preregistered analysis of already historical data, not a blind live trial. No claim of independence from public knowledge of 2023-2025 events. Reusing the holdout for methodological tuning invalidates confirmatory language; corrections must be logged.
- Historical archive downloaded now, not an original point-in-time vintage. Revisions and retrospective universe selection remain limitations.
- Daily open prices with hypothetical proportional costs are execution proxies. No order-book, spread, market impact, capacity, exchange failure, stablecoin depeg or fill guarantee is estimated. Fee assumptions are scenarios, not a verified account fee tier.
- Bootstrap inference is approximate under weak dependence and locally stable distributions; block-length and regime reports expose sensitivity but cannot eliminate nonstationarity. Correction covers this registered family, not undisclosed prior research or alternate universes.
- These are empirical time-series prediction intervals, not distribution-free guarantees under serial dependence or regime shifts. Publish coverage, width and interval score for each asset, model, calendar year and ex-ante regime.
- Two correlated crypto assets on one venue do not establish generalization to equities, futures, other venues or future regimes.
- A negative study does not prove no predictive information exists; its power and target are limited.
- The DRR lag-correlation and depth scores are descriptive features, not inferred causal edges or evidence of a new financial mechanism.

Every validation trial is in `experiments/financial_holdout/selection/trials.json`. The separate selection lock pins the protocol, source files and data. `predictions.csv.gz` contains targets, forecasts, bounds and fit/calibration timestamps; `daily_portfolios.csv.gz` contains every daily cost scenario. `provenance.json` pins the published artifacts and `environment.json` records dependencies.

Data attribution: Binance Vision, normalized from public daily spot klines. Data and derived results: CC BY-NC-SA 4.0 and Binance Vision Dataset Terms. No affiliation or endorsement is implied.
