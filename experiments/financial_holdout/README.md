# DRR financial holdout study

The machine-readable [protocol](protocol.json) fixes the target, timing, data,
feature sets, model grids, selection procedure, costs, inference family and
success criteria **before any holdout scoring**. This first commit contains no
experiment results. Implementation, data provenance and results follow in
separate commits so the chronology can be inspected.

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
