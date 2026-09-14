# LFBO Supervisory Research Workbench

The LFBO workbench reduces the mechanical work around public regulatory-data
monitoring while keeping interpretation and supervisory judgment with people. Its
governing sequence is:

> machine organizes → machine tests → machine prioritizes → analyst investigates →
> analyst interprets

It is not an automated supervision system. Outputs are research diagnostics and may
not be represented as supervisory ratings, findings, MRAs, MRIAs, enforcement
recommendations, legal conclusions, governance assessments, or proof of causation.

## Analyst workflow

1. Ingest a locally obtained filing with an explicit reporting period, filing and
   amendment dates, ingestion time, availability time, source URL, and SHA-256.
2. Resolve every selected column against the exact form, MDRM code, definition
   version, effective period, and verification cutoff.
3. Reconstruct what the local system knew at the review cutoff. Later amendments and
   downloads are excluded.
4. Reconcile missing quarters, revisions, unit/definition/perimeter breaks, totals,
   ratios, carry-forwards, unexpected zeros, and discontinuities. Exceptions are
   surfaced; they are never repaired silently.
5. Measure transparent changes and compare the institution with an explicit,
   versioned peer cohort.
6. Run simple trailing baselines. Optional DRR analysis is withheld when inputs are
   incomplete, too short, or cross an active comparability breakpoint.
7. Challenge flagged results across windows, transformations, scaling, lags, methods,
   correction rules, variable omissions, and data-vintage alternatives.
8. Write content-addressed evidence and apply the configured attention budget.
9. Record analyst dispositions separately from immutable analytical claims.
10. Generate a concise Morning Brief and institution brief containing only new,
    changed, weakened, or unresolved material items.

Disabling DRR does not disable ingestion, reconstruction, reconciliation, change
detection, peer analysis, baselines, evidence, monitoring deltas, or briefs.

### Multiple observational perspectives

Institutional analysis, quantitative models, expert judgment, local measurements,
distribution-sensitive indicators, and lived material experience can all describe the
same complex system at different scales or horizons. The perspective layer records
each claim with its population, geography, dimension, method, evidence, limitations,
and outside scope. It does not score one way of knowing against another. A supported
aggregate assessment and a supported local material assessment can coexist, with their
divergence documented and attributed. Institutional expertise is not discounted, and
lived experience is not presumed more accurate simply because it is local; each record
is judged against its own evidence and scope. The question is what each instrument
reveals and what remains outside its field of view.

## Public data and semantic scope

`SemanticRegistry` isolates forms and versions. A matching suffix is never enough to
equate `RCON`/`RCFD` with `BHCK`/`BHCF`/`BHDM`/`BHFN`. Production use requires a
`VERIFIED` definition with:

- exact form, MDRM code, schedule, line, definition, unit, and frequency;
- effective dates and a version identifier;
- an authoritative Federal Reserve or FFIEC HTTPS source;
- verification date/known-as-of time and a SHA-256 of the source artifact.

The package has architecture for FR Y-9C, FFIEC 002, and FR Y-15, but the bundled
snapshot intentionally contains only three verified March 2026 FR Y-9C Schedule HC
items. An absent, ambiguous, future-known, synthetic, or unverified definition is
blocked by default. Synthetic definitions require both the `SYN_` namespace and an
explicit `allow_synthetic=True` configuration.

## Filing input contract

`ingest_wide_csv` accepts a local wide CSV only when its bytes match the
`FilingContext.source_hash`. Institution identifiers are read as strings and financial
columns must be exact MDRM codes. A batch is atomic: duplicate or conflicting facts
reject the ingestion rather than selecting an arbitrary record.

```python
from drr_framework.supervisory import (
    FilingContext,
    bundled_registry,
    ingest_wide_csv,
)

context = FilingContext(
    form="FR Y-9C",
    reporting_period="2026-03-31",
    original_filing_date="2026-05-01T00:00:00Z",
    ingestion_date="2026-05-02T14:00:00Z",
    available_as_of="2026-05-02T14:00:00Z",
    source_vintage="2026Q1-download-1",
    source="https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload",
    source_hash="<sha256 of the local filing>",
    definition_version="2026-03-31",
)
store = ingest_wide_csv(
    "filing.csv",
    bundled_registry(),
    context,
    metric_columns=("BHCK0081", "BHCK2170", "BHCK3210"),
)
```

The source hash proves which bytes were ingested; the observation ID hashes the fully
normalized record. Those are different identities and are recorded separately.

## Point-in-time reconstruction

`VintageStore` is append-only. Each `RegulatoryObservation` records the reporting
period, original filing date, optional amendment date, ingestion date, source vintage,
effective availability time, semantic version, source, provenance, and optional
calculation lineage.

`store.as_of(t)` selects only records whose `available_as_of` is no later than `t`, then
selects the latest filing known at that cutoff. Re-ingesting an older original filing
cannot displace a known amendment. Conflicting values with the same filing timestamp
are blocked for reconciliation. `compare_vintages` and `filing_revision_history`
expose the history explicitly.

The workbench's `observation_revisions` audit uses the **exact observation IDs in the
two review states**, rather than substituting whatever a later store reconstructs at
the old cutoff. It retains both complete source records, changed fields, numeric and
percent revisions when comparable, and concurrent changes to the corresponding
metric's signals. These are same-period filing revisions, not quarterly movements.
Concurrent signal transitions are descriptive and do not attribute causation.

An absent historical source is `source_unavailable`; null values are `missing_value`;
unit, definition, perimeter, or provenance changes are `incomparable`. None of these
produces a numeric difference. A zero prior value has no percent revision. A raw
revision outside floating-point range is `numeric_overflow`; an overflowing percent
revision is withheld with an explicit limitation. The Filing revisions panel, Morning
Brief, and `filing-revisions.json` expose the same audit,
which is covered by the monitoring payload's passport hash. Initial reviews have no
revision comparison. Observations outside the intersection of the two review scopes
are not described as filing revisions.

Date-only cutoffs mean midnight UTC. Callers that need end-of-day scope should pass an
explicit timestamp. A later local download cannot prove what a different system knew
historically; the walk-forward report states that limitation.

## Missingness and derived values

`RegulatoryAnalysisDataset` has rows as quarters and columns as financial metrics,
which is the orientation consumed by `DynamicResonanceRooting.analyze_system`. Missing
quarters are materialized and missing values remain `NaN`; `to_drr_input()` refuses an
incomplete matrix.

Derived and imputed records require `CalculationLineage`: formula, exact input IDs,
metrics and periods, software version, parameters, unit, and timestamp. The vintage
store rejects future inputs. Imputation is additionally restricted to trailing,
institution-local observations with matching form, metric, unit, definition, and
perimeter. The historical enum value `IMPUTED_CAUSAL` means a forward-safe imputation
path; it is not a claim of economic causality.

## Changes, peers, and comparability

`detect_material_changes` reports the raw and percent QoQ change, YoY change, rolling
mean/median/volatility, robust z-score, historical percentile, trend slope,
acceleration, persistence, reversal, materiality, and missingness-adjusted confidence.
No composite score is shown without its components.

Peer groups specify membership, exclusions, filters, source, effective date, and
known-as-of date. Comparisons require the exact form, code, version, unit, and period;
the target is excluded. Results distinguish firm-specific, peer-cluster, broad-common,
and indeterminate patterns. These are descriptive contexts, not causal attribution.

Policy/reporting events can insert explicit comparability breakpoints. Baselines and
DRR are withheld across a break in their active window instead of treating a
definition change as economic movement.

## Baselines, DRR, and falsification

Simple competitors include robust z-scores, rolling percentiles, EWMA, CUSUM, rolling
volatility, a conventional mean-shift heuristic, lagged correlations, and an AR(1)
forecast residual. Matched evaluation compares baseline-only with baseline-plus-DRR
on identical review dates. Without independent event labels, it reports alert burden
and explicitly leaves precision/recall unestimated.

`LFBORegimeAnalyzer` maps DRR dimensions back to metric labels and exposes significant
directional lead-lag edges, adjusted p-values, rooting backend, resonance, state-space
stability, structural surprise, and topology drift. It does not label distress or
supervisory deficiencies.

Falsification performs real reruns. Failed or inapplicable specifications count
against survival rather than disappearing from the denominator. Reports use
`robust`, `mixed`, `fragile`, or `not supported` and preserve contradictory evidence
and individual failure cases.

## Evidence and review state

Every selected signal points to an immutable `EvidenceEntry` containing source facts,
calculation, historical and peer context, baseline and DRR evidence, robustness,
contradictory evidence, policy context, limitations, and the explicit decision
boundary. Its ID is a SHA-256 of canonical content. Analyst reviews are append-only,
timestamped records stored separately, so a disposition never rewrites the claim.

`ReviewState` compares the last reviewed state with the current state: new/revised
observations, appearing/disappearing/strengthening/weakening signals, data exceptions,
policy events, entity relationships, DRR relationships, and failed robustness results.
`AttentionBudget(top_n=5)` ranks only unresolved changes and can truthfully return
“Nothing material changed.” Blocked data never produces that message.

The live server keeps analytical evidence separate from current human activity:

| Surface | Content |
| --- | --- |
| `GET /api/snapshot` | The original canonical analytical snapshot; unchanged by reviews |
| `GET /api/review-activity` | Current dispositions and attention queue, tied to an analysis ID and a separate review cutoff |
| `POST /api/review` | An append-only disposition for evidence in the current snapshot |

Resolving an item (`useful`, `explained`, `noisy`, or `dismissed`) removes it from
active triage and promotes the next eligible deferred candidate. Reopening it
(`unresolved` or `investigate`) restores its original score and rank among that run's
candidates. Scores and analytical signals are not recalculated from the analyst's
opinion. Refresh the page after recording a disposition; the queue also reconstructs
from persisted reviews on restart. The current activity cutoff is displayed separately
from the historical analytical cutoff. Historical exports continue to include only
reviews available at the analytical cutoff.

## Analysis passport

Each run produces a content-addressed `AnalysisPassport` with software version, Git
commit and dirty-state flag, source-code SHA-256, analytical cutoff, filing vintages,
semantic registry version, institution and peer scope, variables, transformations,
DRR/baseline/robustness configuration, random seeds, upstream source hashes (or an
explicit canonical synthetic-record snapshot hash), and output hashes. Hashes provide
integrity and reproducibility evidence; they are not digital signatures. The
`monitoring` output hash is explicitly the stable ID of the complete monitoring
payload before the passport field is appended; the passport records that scope and
the verified prior review-state ID.

`verify_monitoring_snapshot(snapshot)` checks the passport ID, review-state ID, shared
analytical cutoff, and monitoring output hash. Serving and export reject inconsistent
snapshots; export also checks its supplied state/passport before creating files. This
detects content inconsistency, not malicious recomputation of every hash.

## Local operation

```bash
# Deterministic end-to-end lab and exported static workstation
python examples/lfbo_monitoring_workbench.py

# Interactive local review queue
drr-monitor --demo --serve --port 8765 --role analyst

# Public normalized observations, without DRR
drr-monitor --input observations.jsonl --as-of 2026-08-10T00:00:00Z \
  --registry registry.json --cohort cohort.json --no-drr
```

Exports include the snapshot, review state, Morning Brief, institution briefs, static
HTML, evidence JSON/JSONL/Markdown, analyst reviews, audit events, dependency inventory,
perspectives, and passport. A supplied perspective inventory can be loaded with
`--perspectives perspectives.json`; its evidence IDs may refer to source observation
IDs or existing ledger entries. Artifacts are idempotent: an attempt to overwrite the same path with
different content fails.

## Model-risk reference basis

The workbench is aligned with selected principles in the Federal Reserve's Revised
Guidance on Model Risk Management, issued through SR 26-2 on April 17, 2026. The
metadata covers intended and reasonably foreseeable use, materiality, complexity,
data quality, conceptual soundness, implementation verification, outcomes analysis,
monitoring, limitations, change control, governance, independent review, and use
boundaries. SR 11-7 and SR 21-8 are recorded only as superseded historical references.

This is a validation-readiness architecture, not a statement of SR 26-2 compliance or
independent validation. The guidance is nonbinding, risk-based, and outside the scope
of generative/agentic AI; deploying an LLM beside this deterministic workbench would
require a separate risk assessment and evidence boundary.

## Performance and incremental recomputation

Semantic metadata and immutable DRR results are content-keyed. When a new quarter is
added, unchanged institution/form/config inputs reuse cached DRR results inside the
workbench process. Panel calculations use NumPy/pandas operations, and the heavier DRR
path is short-history gated. Correctness, as-of integrity, and comparability controls
take precedence over speed.

Run `python scripts/benchmark_lfbo.py` to record cold and warm timings for the
deterministic five-institution/480-observation DRR path on the target environment. The
benchmark checks that cached and uncached monitoring state IDs agree; it deliberately
does not impose a hardware-dependent timing threshold.

See [Security Posture](security-posture.md) before exposing the local interface or
admitting confidential data.
