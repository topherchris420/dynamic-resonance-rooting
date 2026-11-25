# Dataset and derived-result attribution

The normalized market bars in `data/bars.csv` are derived from **Binance Vision**
spot daily klines for BTCUSDT and ETHUSDT. `data/manifest.json` identifies every
original download URL, upstream-verified SHA-256, byte size and record count.
Normalization selects daily UTC dates and OHLCV columns, restricts the registered
date range, sorts date/symbol and writes an LF-terminated CSV. No price adjustment,
imputation or interpolation is performed.

These data and derived experiment results (including selection outputs and
`results/expected/financial_holdout/`) are provided under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), subject to the
[Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
They are **not** relicensed under the repository's MIT software license.
The source terms permit non-commercial research; commercial use requires the
appropriate separate source license. No affiliation, sponsorship or endorsement
by Binance or any trading firm is implied.

The experiment implementation remains MIT-licensed. Replacing the input data
does not transfer the published empirical findings to the replacement source.
