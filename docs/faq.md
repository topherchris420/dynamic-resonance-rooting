# Frequently Asked Questions

## Does DRR prove causality?

No. Rooting edges are directed diagnostic signals based on lagged structure or
transfer entropy when available. They should be interpreted as leads for review,
not as causal proof.

## Is DRR validated supervisory methodology?

No. The repository includes validation-readiness artifacts to support model-risk
review. A specific intended use would still require independent validation,
governance approval, outcomes analysis, and ongoing monitoring.

## Why use a `src/` layout?

The `src/` layout makes local development behave more like installed-package
usage. It reduces accidental imports from the repository root and improves
packaging reliability for collaborators.

## Why include synthetic datasets?

Synthetic examples make the basic resonance and rooting behavior reproducible
without requiring private, licensed, or large external datasets.

## What happens if `pyinform` is unavailable?

`RootingAnalyzer` reports a deterministic lagged-correlation fallback. The
returned `method` field states which backend was used.

## Can I use this in a regulated, clinical, defense, or infrastructure setting?

Only after independent validation and review for the exact intended use. The
repository is a research implementation and should not be treated as a validated
operational decision system.
