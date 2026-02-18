## 2025-02-18 - [Boundary Conditions in Optimization]
**Learning:** When replacing high-level abstractions like Set lookups with low-level array indexing, precise boundary conditions are critical. In `reactionDiffusionWorker.js`, simply checking `V > threshold` was insufficient because the original logic implicitly excluded boundary indices (0 and size-1) by only populating the `active` set from inner loops.
**Action:** Always verify the implicit constraints of the code being replaced, not just the explicit logic.
