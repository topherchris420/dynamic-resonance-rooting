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
