# DRR Audience Guide

Dynamic Resonance Rooting is intended to be a cross-domain diagnostic toolkit
for complex adaptive systems. The same core methods can serve physicists and
policy analysts, but each audience needs a different entry point, vocabulary,
and validation standard.

## Physics Lab

Use the Physics Lab workflow when the system is a physical, simulated, or
engineered dynamical process.

Typical questions:

- Which oscillatory modes dominate the system?
- Are modes persistent or transient?
- Which variables appear to lead or respond under lagged coupling?
- How does the fitted transition system respond to perturbations?
- Does the inferred state-space model look stable?

Primary APIs:

- `DynamicResonanceRooting.analyze_system`
- `ResonanceDetector`
- `RootingAnalyzer`
- `DepthCalculator`
- `analyze_resonance_state_space`
- `write_analysis_report(..., audience="physics")`

Run:

```bash
python examples/physics_lab.py
```

Expected outputs:

- Dominant frequencies and resonance-depth components.
- Directed rooting edges with lag and p-value metadata.
- State-space transition and measurement diagnostics.
- Impulse-response paths for perturbation analysis.
- JSON and Markdown reports under `results/physics_lab/`.

Validation expectations:

- Compare detected modes against controlled synthetic or experimental truth.
- Run perturbation sweeps and verify response decay or amplification.
- Check sensitivity to sampling rate, window size, and noise.
- Treat DRR outputs as diagnostics until calibrated against the physical system.

## Policy Lab

Use the Policy Lab workflow when the system is a set of macro, financial,
institutional, or governance observables.

Typical questions:

- Which observables share persistent cyclical structure?
- Which variables lead others under lagged diagnostic scoring?
- Are fitted transition dynamics stable across windows or regimes?
- How do shocks propagate through the fitted observable system?
- Do results remain stable across data vintages, transformations, or subsamples?

Primary APIs:

- `PolicyResonanceDataset`
- `load_policy_dataset`
- `DynamicResonanceRooting.analyze_system`
- `analyze_resonance_state_space`
- `write_analysis_report(..., audience="policy")`

Run:

```bash
python examples/policy_lab.py
```

Expected outputs:

- DRR-ready matrix built from labeled tabular observables.
- State-space diagnostics in language familiar to DSGE and policy workflows.
- Lagged directed relationships among observables.
- Markdown report with explicit policy caveats.
- JSON artifact suitable for reproducible review.

Validation expectations:

- Use documented transformations such as levels, differences, percent changes,
  or log differences.
- Repeat results across data vintages, regimes, and rolling windows.
- Avoid policy-prescriptive language unless separately supported by a validated
  model and institutional review.
- Treat DRR as a research diagnostic, not a forecast or causal policy engine.

## Shared Research Standard

Both audiences should expect:

- Deterministic examples that run from a clean checkout.
- JSON artifacts for downstream review.
- Markdown summaries that state caveats clearly.
- Tests for new workflow surfaces.
- Explicit separation between diagnostic evidence and domain interpretation.
