# Typed judgment overlay

Typed judgment is an optional characterization of an evidence packet the analyst is
about to read. It sits after measurement, statistical testing, falsification, and
deterministic attention selection. It does not replace those steps, and it does not
decide what an institution's evidence means.

```text
MEASURE → TEST → PRIORITIZE → JUDGE → INTERPRET
```

DRR measures and tests. The attention budget prioritizes. A judgment provider, when
enabled, answers a fixed set of questions about the supplied packet. The analyst
interprets.

## Configuration

Remote judgment is off by default.

```python
WorkbenchConfig(
    judgment_enabled=False,          # default; no judgment call
    judgment_provider="typesafe",    # typesafe, mock, or disabled
    judgment_model="jev-latest",     # alias sent only when typesafe is enabled
    judgment_timeout_seconds=10.0,
    judgment_policy_version="v1",
)
```

```bash
# local-only
drr-monitor --demo

# explicit remote judgment; requires TYPESAFE_API_KEY and Python 3.10+
drr-monitor --demo --judgment --judgment-provider typesafe --judgment-model jev-latest

# offline stand-in used by tests
drr-monitor --demo --judgment --judgment-provider mock
```

`TYPESAFE_API_KEY` is read from the environment by the TypeSafe SDK. Do not put it
in a config file, passport, evidence record, or audit event. The optional extra is:

```bash
pip install "drr-framework[judgment]"
```

The published `typesafe-sdk` release used here is 0.7.x and requires Python 3.10 or
newer. On older interpreters the workbench stays in local-only mode. The provider
does not retry a failed call and does not fall back to another model.

## Question set `evidence-packet-v1`

| Id | Primitive | Question |
| --- | --- | --- |
| `evidence_adequacy` | Choice: adequate, limited, insufficient | Given only the supplied evidence packet, is there enough evidence to meaningfully evaluate the stated claim? |
| `scope_overreach` | Noul | The stated claim materially goes beyond what the supplied evidence establishes. |
| `contradiction_material` | Noul | The supplied contradictory evidence is materially important to interpreting this claim. |
| `review_complexity` | Choice: routine, moderate, complex | How complex is this evidence packet for a human analyst to review? |
| `limitations_material` | Noul | The documented limitations materially constrain interpretation of this claim. |
| `additional_review_needed` | Noul | The evidence contains enough ambiguity, disagreement, limitation, or uncertainty that additional human scrutiny is warranted. |

Choice and Score answers retain `probabilities` and `confidence`. Noul answers retain
`noul` only. Model confidence is not data confidence and is not a p-value.

## Policy `v1`

A Noul is material when it is strictly greater than 0.5.

- provider failure, disabled provider, or an unusable answer → `JUDGMENT_UNAVAILABLE`
- `evidence_adequacy == insufficient` → `INSUFFICIENT_EVIDENCE`
- limited adequacy, or any material Noul → `REVIEW_CAREFULLY`
- adequate evidence and no material Noul → `READY`

`READY` means the packet is coherent enough for ordinary human review under this
policy. It does not mean a supervisory conclusion is correct.

## What is sent

The state builder reads one `EvidenceEntry` and keeps the claim, source facts,
calculation provenance, historical and peer context, baseline and DRR evidence,
robustness, contradictory evidence, limitations, policy context, and the decision
boundary. It records `evidence_id`, `analysis_id` when the analytical passport
already exists, `as_of`, source ids, and source vintages. The canonical JSON is
SHA-256 hashed with the same `stable_id` function as other workbench records.

Analyst rationale, reviewer identity, credentials, the rest of the ledger, and
unrelated evidence are not fields of that state. If the packet exceeds 48,000
characters, named sections are removed in a fixed order and `truncation.truncated`
is true. A packet that is still too large is not sent.

## Example artifact

The following record is synthetic. It is not a live model response. Timestamps are
audit fields; `judgment_id` is the SHA-256 of the remaining content.

```json
{
  "evidence_id": "abc123",
  "policy_outcome": "REVIEW_CAREFULLY",
  "policy_version": "v1",
  "review_complexity": "moderate",
  "warnings": ["additional_review_needed=0.7200 exceeds 0.5"],
  "judgment_result": {
    "schema_version": "judgment-result-v1",
    "provider": "typesafe",
    "model": "jev-latest",
    "provider_model_version": "jev-1.13.0",
    "question_set_version": "evidence-packet-v1",
    "provider_status": "completed",
    "state_truncated": false,
    "answers": [
      {
        "question_id": "evidence_adequacy",
        "primitive": "choice",
        "choice": "adequate",
        "probabilities": {"adequate": 0.81, "limited": 0.14, "insufficient": 0.05},
        "confidence": 0.72
      },
      {
        "question_id": "review_complexity",
        "primitive": "choice",
        "choice": "moderate",
        "probabilities": {"routine": 0.18, "moderate": 0.7, "complex": 0.12},
        "confidence": 0.55
      },
      {"question_id": "scope_overreach", "primitive": "noul", "noul": 0.12},
      {"question_id": "contradiction_material", "primitive": "noul", "noul": 0.34},
      {"question_id": "limitations_material", "primitive": "noul", "noul": 0.67},
      {"question_id": "additional_review_needed", "primitive": "noul", "noul": 0.72}
    ],
    "limitations": []
  }
}
```

The workstation shows those answers beside materiality, data confidence, and
robustness, and beside the human disposition. It does not collapse them into one
score.
