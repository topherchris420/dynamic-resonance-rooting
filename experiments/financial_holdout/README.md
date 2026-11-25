# DRR financial holdout study

The machine-readable [protocol](protocol.json) fixes the target, timing, data,
feature sets, model grids, selection procedure, costs, inference family and
success criteria **before any holdout scoring**. Protocol registration,
implementation/selection and results occupy separate commits so the chronology
can be inspected. See the [amendment and execution history](amendments.md).

**Outcome: NOT SUPPORTED.** The selected ridge + DRR model did not establish the
registered minimum forecast advantage or corrected significance against every
baseline. Positive simulated net returns do not override those failed criteria.

- [Complete report, including unfavorable results](../../results/expected/financial_holdout/report.md)
- [Method and reproduction commands](../../docs/financial-holdout.md)
- [Every validation trial](selection/trials.json) and [frozen selection](selection/selection.json)
- [Original source checksums](data/manifest.json) and [data license](DATA_LICENSE.md)

The study asks whether DRR contributes useful incremental information to delayed
daily return prediction. A negative result is a completed experiment, not a failed
test suite. It is historical research on BTCUSDT and ETHUSDT, not evidence of
deployable alpha or broad financial-market performance.

The raw-data source is [Binance Vision](https://data.binance.vision/). Its
[archive documentation](https://github.com/binance/binance-public-data) describes
checksums, timestamp units and historical corrections. Data and derived results
must retain the source's [dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) attribution;
they are not relicensed under this repository's MIT software license.
