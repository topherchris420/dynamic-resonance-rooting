# Protocol amendments and execution history

1. Initial protocol published as commit
   `835292aa6b02a66e97992652d8a8e640ec07b877`, before market-data scoring.
2. **Before selection or holdout scoring:** make the cross-stage label purge
   explicit. The last nominal validation labels would otherwise mature after the
   first holdout forecast. Require validation exits strictly before the first
   test feature date minus seven days; use the equivalent cutoff for the
   development-only regime threshold. This shortens validation to entry dates
   through 2022-12-21. No outcomes were consulted for this correction. All
   remaining target, grids, costs, hypotheses and success gates are unchanged.

The data fetch initially downloaded one January 2018 BTC archive as a transport
probe; no returns, candidate scores or test outcomes were inspected. Historical
data are public and already in the past: registration before this analysis is
not a prospective or independently blinded live experiment.

3. Code, 194 checksum-verified source archives normalized to 5,846 daily asset
   bars, all 29 validation trials and the selection lock were published in
   commit `b8e6a2ff14752c04b5639d2272883ace2a150eb1` before the first holdout
   evaluation. Validation selected `ridge_drr` and baseline `zero`.
4. The first completed holdout evaluation returned **NOT SUPPORTED**. The
   minimum improvement and corrected-significance gates failed. No parameters,
   costs, target, features or decision criteria were changed in response.
   Subsequent runs replay the same frozen computation for verification.
