# LFBO workbench implementation record

Baseline: main `07440f94f806e1d8af3038278f6db722aae936a8`, 150 tests passing;
repository Ruff and Black checks passing on Python 3.12.

The workbench extends `PolicyResonanceDataset` in the canonical `datasets.py`.
The existing multivariate facade expects `(time, variables)` and uses dimension
names internally. Regulatory observations retain their semantic labels and source IDs.
Legacy policy interpolation is retrospective; it is not used by the new workbench.

## Authoritative reference review, September 12, 2026

- [SR 26-2 letter](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm),
  April 17, 2026: supersedes SR 11-7 and SR 21-8.
- [Full guidance, pages 1–12](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf):
  risk-based tailoring, complexity and input quality; exposure and purpose jointly
  determine materiality; effective challenge; development and use boundaries;
  conceptual soundness, outcomes analysis and monitoring; governance and third parties.
  Simple arithmetic and deterministic rules are outside its model definition.
  Generative/agentic AI is outside this guidance's scope. It is nonbinding guidance,
  expected most relevant to banking organizations above $30 billion in assets.
- [MDRM dictionary](https://www.federalreserve.gov/apps/mdrm/data-dictionary) and official
  CSV archive downloaded. Exact source rows and SHA-256 digests are bundled.
  Initial verified coverage is deliberately scoped to three FR Y-9C Schedule HC
  items in the March 2026 form. Other periods/forms require verified definitions.
  The architecture accepts FR Y-9C, FFIEC 002 and FR Y-15 independently. It never
  treats equal suffixes or similar labels as equivalent metrics.

All development tests and demonstration records are separate from independent
validation evidence. No government approval or validated supervisory methodology
is claimed.

## Typed judgment overlay

The judgment provider runs only after `EvidenceEntry` creation, falsification,
`ReviewState` construction, and `AttentionBudget.select`. It is not part of
`DynamicResonanceRooting.analyze_system`, baselines, statistical tests, or
walk-forward reconstruction. Question set `evidence-packet-v1` and policy `v1`
are versioned. Remote use is optional, off by default, and limited to minimized
public or synthetic evidence packets.

Policy `v2` keeps that decision rule and emits `EVIDENCE_REVIEWABLE` where policy
`v1` emitted `READY`. The new token names readiness for ordinary human review of
the packet. Stored `v1` records remain readable and are labeled as such.
