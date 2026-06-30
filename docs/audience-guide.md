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

## Supervisory Analytics Lab

Use the Supervisory Analytics Lab when the system is an institution-level panel
of financial, risk, governance, or controls observables. This workflow is built
for analysts who need auditable diagnostics, peer-group comparisons, exports
that can move into SQL or Tableau review loops, and validation-ready evidence
for model-risk review.

Typical questions:

- Which institution metrics show persistent cyclical or stress-linked patterns?
- Which metrics appear to lead other metrics under lagged diagnostic scoring?
- Does one institution differ from its peer-group aggregate under the same setup?
- Are fitted transition dynamics stable enough to monitor across rolling windows?
- Which outputs should be routed for source-data review rather than interpreted as findings?

Primary APIs:

- `SupervisoryPanelDataset`
- `load_supervisory_panel`
- `load_supervisory_panel_from_sql`
- `DynamicResonanceRooting.analyze_system`
- `write_analysis_report(..., audience="supervision")`
- `write_tableau_artifacts`
- `build_supervisory_alignment_metadata`
- `build_model_risk_card`
- `build_validation_readiness_packet`
- `run_event_backtest`
- `create_shadow_review_record` and `append_shadow_review_record`
- `explain_supervisory_signal`

Run:

```bash
python examples/supervisory_policy_lab.py
```

Fed Supervisory Alignment Pack:

- Institution presets: community banks, regional banks and foreign banks under $100B, large banks and large FBOs, G-SIBs, and financial market utilities.
- Risk domains: capital, liquidity, asset quality, earnings, market risk, operational resilience, governance and controls, and interconnectedness.
- Data lineage fields: reporting form, MDRM codes, FFIEC source, NIC identifier, data vintage, peer group, and review owner.
- Report checklist: materiality, proportionality, timeliness, source-data lineage, peer-group fit, and diagnostic boundary.
- Reference basis: Federal Reserve supervision and regulation, supervisory operating principles, the Supervision and Regulation Report, MDRM, NIC, and SR 11-7.

Validation Readiness Pack:

- Model-risk card with intended use, prohibited uses, owners, assumptions, limitations, data lineage, and validation status.
- SR 11-7-style checklist covering conceptual soundness, implementation verification, outcomes analysis, ongoing monitoring, governance, and independent validation.
- Event backtest helper for alert counts, detected events, missed events, false positives, precision, recall, and lead time.
- Shadow-mode JSONL review log for analyst feedback that does not affect supervisory decisions.
- Explainability summary for dominant modes, lead-lag edges, state-space stability, lineage, caveats, and analyst review questions.

Expected outputs:

- Institution-level and peer-group DRR reports with supervisory caveats.
- Supervisory Alignment report section with segment preset, risk domains, source lineage, review checklist, and official-reference links.
- Validation Readiness report section with model card, SR 11-7-style checklist, outcomes snapshot, shadow-mode snapshot, and validation-reference links.
- Tidy panel CSV shaped for Tableau, with institution/date/metric/value fields.
- Summary, resonance-map, rooting-edge, and state-space diagnostic CSV exports.
- JSON artifacts suitable for reproducible review or handoff to model-risk staff.

Validation expectations:

- Document the institution population, peer-group definition, metric lineage, and transformations.
- Re-run outputs across rolling windows, data vintages, and alternative peer groups.
- Treat directed edges as monitoring signals for analyst review, not as causal findings.
- Keep examination ratings, enforcement decisions, policy recommendations, MRAs, and MRIAs outside DRR output.
## Shared Research Standard

All workflows should expect:

- Deterministic examples that run from a clean checkout.
- JSON artifacts for downstream review.
- Markdown summaries that state caveats clearly.
- Tests for new workflow surfaces.
- Explicit separation between diagnostic evidence and domain interpretation.
