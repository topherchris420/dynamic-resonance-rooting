## 2025-05-22 - [JS Worker Simulation Optimization]
**Learning:** The simulation workers (`reactionDiffusionWorker.js`) rely heavily on nested loops over 3D grids stored as 1D arrays. Inlining helper functions (like the Laplacian stencil) and replacing `Set<String>` lookups with direct array index checks provided a ~3x speedup.
**Action:** When optimizing other workers (`structuralWorker.js`, `modalWorker.js`), prioritize inlining math operations and using direct array access over object/Map lookups.

## 2025-05-22 - [Python Path for Tests]
**Learning:** The project structure requires setting `export PYTHONPATH=.` to run tests locally, as the package is not installed in editable mode by default and `setup.py` doesn't handle this automatically for `pytest`.
**Action:** Always export `PYTHONPATH=.` before running `pytest` in this repository.

## 2025-05-22 - [Logic Duplication in Resonance Detection]
**Learning:** `ResonanceDetector` in `modules.py` and `DynamicResonanceRooting.detect_resonances` in `analysis.py` contain duplicate resonance detection logic. `analysis.py` was already optimized with `rfft`, while `modules.py` was using slower `fft`.
**Action:** When optimizing, check for duplicated logic across modules, as one implementation might be more optimized than the other. Future refactoring should consolidate these.
