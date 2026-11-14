# Lead–lag breakdown experiment

The [protocol](protocol.json) was published as commit
`705cdb6a96864140869e2f94d300b82790b22626` before scoring.
The next commit seals the [calibration lock](calibration/lock.json), all source
hashes, calibration evidence and the checksum-verified 2026 hourly market data
before held-out evaluation. Held-out results follow in a separate commit.

See the [method and replay guide](../../docs/lead-lag-breakdown.md).
Simulation detection delay is the primary endpoint. Market warning association
and risk attribution are descriptive; real change times are unknown. Public
simulation seeds and historical data are not independently blinded.

The common-driver challenge tests confounding sensitivity. It is not labeled as
unchanged observable dependence. BOCPD state is warmed on fit residuals before
monitoring. Market timestamps without mature 24-hour labels remain in risk
calculations; one terminal hourly holding without a next open is excluded.
These implementation details were settled before calibration and test scoring.

Data attribution: **Binance Vision**. Normalization preserves original OHLCV
decimal values and UTC hourly boundaries, sorting time/symbol into deterministic
gzip CSV. `data/manifest.json` records original archive URLs, upstream verified
SHA-256 checksums, retrieval time and normalization. Data and market-derived
artifacts are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
under the [source dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md),
not the repository's MIT software license. No endorsement is implied.
