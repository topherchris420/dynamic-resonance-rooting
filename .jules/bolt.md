## 2025-05-22 - [JS Worker Simulation Optimization]
**Learning:** The simulation workers (`reactionDiffusionWorker.js`) rely heavily on nested loops over 3D grids stored as 1D arrays. Inlining helper functions (like the Laplacian stencil) and replacing `Set<String>` lookups with direct array index checks provided a ~3x speedup.
**Action:** When optimizing other workers (`structuralWorker.js`, `modalWorker.js`), prioritize inlining math operations and using direct array access over object/Map lookups.
