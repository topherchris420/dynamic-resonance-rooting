# Examples

## Quickstart Resonance Export

```bash
python examples/quickstart_resonance_export.py
```

Demonstrates how to load the checked-in sample dataset, detect resonance,
compute resonance-depth metrics, save a visualization, and export JSON/CSV
summary artifacts.

## Physics Lab

```bash
python examples/physics_lab.py
```

Uses a deterministic coupled oscillator and writes a physics-facing report under
`results/physics_lab/`.

## Policy Lab

```bash
python examples/policy_lab.py
```

Builds synthetic quarterly policy observables, applies a documented
transformation, and writes a caveated policy-facing report.

## Supervisory Analytics Lab

```bash
python examples/supervisory_policy_lab.py
```

Builds an institution panel, compares institution and peer-group diagnostics,
writes validation-readiness packets, and exports Tableau-ready CSVs.

## Reproduction Harness

```bash
python -m drr_framework.experiments --output-dir results/reproduction
```

Regenerates the compact JSON and CSV benchmark artifacts used by the README.
