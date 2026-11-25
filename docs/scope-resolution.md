# Scope resolution without invented agreement

`DRR_ScopeResolver` preserves observations across MACRO, MESO, MICRO, and LOCAL
scales. Each record carries a source, confidence supplied by the caller, indicators,
and evidence provenance. The resolver organizes those statements; it does not verify
their truth or compare their underlying populations and measurement methods.

```python
from drr_framework import DRR_ScopeResolver, ObservationalPerspective

resolver = DRR_ScopeResolver([
    ObservationalPerspective(
        source_id="aggregate-fixture", scale="MACRO", confidence_score=0.95,
        indicators={"funding": "stable", "concentration": "fragile"},
        evidence_provenance="synthetic-example-a",
    ),
    ObservationalPerspective(
        source_id="local-fixture", scale="LOCAL", confidence_score=0.8,
        indicators={"cash_flow": "declining"},
        evidence_provenance="synthetic-example-b", dissent_logged=True,
    ),
])
result = resolver.generate_drr_conclusion()
print(result["status"])           # divergent_scopes_preserved
print(result["scope_coverage"])   # MESO and MICRO remain missing
print(result["divergence_pairs"]) # references individual indicator_assessments
```

The examples are synthetic. No conclusion about an actual institution follows.

## Interpretation contract

| Status | Meaning |
| --- | --- |
| `no_perspectives` | No observations were supplied |
| `indeterminate` | Missing scopes, empty records, or unclassified indicators prevent cross-scope agreement from being established |
| `divergent_scopes_preserved` | Recognized opposing descriptions occur across supplied scales |
| `divergent_perspectives_preserved` | Recognized opposing descriptions occur within a supplied scale, including within one record |
| `concordant_consensus` | Legacy API label retained for textual concordance: at least two scales, all indicators classified, no opposing descriptors |

`concordant_consensus` **does not mean verified consensus**. Its conclusion and
`comparison_basis` explicitly state that limitation. Callers must handle the
`indeterminate` and `no_perspectives` states; absence of detected divergence is not
evidence of agreement.

Classification accepts only exact descriptors from the module's two vocabulary sets,
with case and trailing sentence punctuation normalized. Examples include `stable`,
`robust`, `declining`, and `fragile`. Numeric values, compound statements, negation
(`not stable`), qualifications (`possibly strong`), and unfamiliar descriptions remain
unclassified. Even a recognized descriptor is not a measure of economic desirability:
growth in a risk exposure and growth in a resource can mean very different things.

Each indicator is retained in `indicator_assessments`. `divergence_pairs` references
those assessment indices and records whether the pair crosses scales. Aggregating
sentiment within a scale first could erase its disagreement, so the resolver never
does that. Conclusions name only supplied scales. Missing `macro_state` and
`local_state` values are `undetermined`, with missing scales listed explicitly in
`scope_coverage`.

Confidence is descriptive metadata supplied by the caller. It neither validates the
source nor selects a winning perspective. Repeated source IDs retain all records;
`dissent_records` preserves each dissent flag and the compatibility summary reports
whether any record under that source logs dissent.

Registration and returned views use independent deep copies, including nested
indicator structures. Mutating the caller's inputs or an exported result cannot alter
the resolver's registered evidence. Each call's pair indices refer to that call's
assessment list; they are not global evidence identifiers.

For attributed, evidence-backed comparisons with explicit population, geography,
dimension, horizon, method, and availability time, use `ScopedPerspective`,
`DocumentedDisagreement`, and `PerspectiveInventory` in the
[LFBO workbench](lfbo-workbench.md). The lexical resolver does not automatically
create documented disagreements or institutional findings.
