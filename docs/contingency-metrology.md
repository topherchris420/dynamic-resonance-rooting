# Supplementary binary metrology

Phase 2 adds **NIST Contingency-based supplementary detection metrology** in
`drr_framework.supervisory.contingency`. Its evidence class is
`BINARY_CLASSIFICATION_METROLOGY`. It is descriptive research output, separate
from `EVENT_DETECTION_EVALUATION` and `CONVENTIONAL_ECONOMETRIC_EVIDENCE`.

`run_event_backtest()` asks whether alerts precede events inside a lead window.
`evaluate_contingency()` asks how scalar predictions classify independently
labeled institution-quarter targets. An earlier alert can detect an event in the
former while being a false positive for its own quarter in the latter. Counts,
precision, and recall therefore have different denominators and meanings.
These outputs are never combined into a composite score. Event counts, alert
burden, lead times, unknown outcome windows, and right censoring retain their
existing Phase 1 contracts. No event list is converted into period negatives.

## Installation and reviewed API

On **Python 3.12 or later**:

```sh
python -m pip install -e ".[contingency]"
# Include Phase 1 fitting when needed:
python -m pip install -e ".[econometrics,contingency]"
```

The optional extra pins `contingency-tools==0.2.3` and adds
`beartype>=0.21,<0.23`. Inspection of the released wheel on September 15, 2026
identified two packaging discrepancies: its metadata declares Python >=3.11,
but its source uses Python 3.12 type aliases and generic-function syntax; it also
imports `beartype` without declaring that dependency. Python 3.11 grammar parsing
rejects the released source, and a fresh import without `beartype` fails.
The integration accounts for these discrepancies without modifying or copying
the upstream package. Local verification used Python 3.12.14 and beartype 0.22.9.

The release requires NumPy >=2.3, SciPy >=1.16, and jaxtyping >=0.3. NumPy and
SciPy already belong to DRR's base dependency set, but installing this extra can
upgrade them. The NIST wheel is about 322 KiB; jaxtyping, its transitive helper,
and beartype add runtime dependencies. No NIST plotting extra is installed.
There is no network call or NIST import when DRR is imported.

DRR's base contract remains Python >=3.8. On Python 3.8–3.11 the extra's markers
install no NIST backend, and evaluation explicitly reports its unavailability.
Missing dependencies or an unreviewed NIST version also withhold evaluation;
there is no fallback implementation. The CI matrix retains Python 3.8–3.12.
Numerical NIST tests explicitly skip on 3.8–3.11 with the syntax limitation
reported by `pytest -ra`; Python 3.12 installs the backend and runs those tests.
Schema, temporal-boundary, and dependency-withholding tests run across the matrix.

The reviewed public interface is `from contingency import Contingent`, its binary
constructor, `TP`, `FP`, `FN`, `TN`, `mcc`, `precision`, `recall`, `F`, `F2`, and
`expected("aps")`. The documented `from_scalar()` constructor min-max rescales
scores and exposes normalized weights. We instead threshold original scalar
scores explicitly and pass the resulting boolean matrix to the public binary
constructor. Registered thresholds retain their original units; ties use
`score >= threshold`. No private NIST function or locally copied metric is used.

Primary references reviewed:

- [Official installation instructions](https://pages.nist.gov/Contingency/getting-started/01-installation/)
- [Tutorial and scalar-score families](https://pages.nist.gov/Contingency/getting-started/02-tutorial/)
- [Public Contingent API and numerical conventions](https://pages.nist.gov/Contingency/api/contingent/)
- [Vectorization, memory, and subsampling](https://pages.nist.gov/Contingency/getting-started/03-performance/)
- [Release 0.2.3 metadata and artifacts](https://pypi.org/project/contingency-tools/0.2.3/)
- [NIST source repository](https://github.com/usnistgov/Contingency)

## Input and result contract

All four records are frozen: `ContingencyScore`, `ContingencyDesign`,
`ContingencyThresholdResult`, and `ContingencyResult`.

Each score records an institution, explicit target quarter, availability timestamp,
original scalar value (or `None` with a reason), source artifact ID, and score
family. One score per institution/target quarter is permitted. All scores must
belong to the declared family and be available by the evaluation cutoff.
Finite numbers are required; NaN and infinity are rejected. Unavailable scores
are supplied as `None`, never zero.

Labels reuse Phase 1's `BinaryEventLabel` contract. The latest label vintage
available at the evaluation cutoff is selected. A zero must be independently
ascertained under the declared definition. `None`, missing labels, and labels
that arrive later remain unknown. Conflicting vintages and incompatible known
definitions raise `ValueError`; future label revisions do not alter earlier
results. Each complete case pairs a score with its exact institution/target key.

The design requires a population description, score family, event definition,
registration timestamp, and threshold protocol. To permit classification it also
requires `label_semantics="explicit_institution_quarter"` and a nonempty
`negative_label_definition` explaining ascertainment. `event_inventory_only`,
`unspecified`, or an absent negative rule withholds classification metrics.
These declarations expose assumptions; they cannot prove the label design sound.

Results retain the full supplied population, selected labels and their sources,
complete-case keys, hashes, library versions, threshold results, and limitations.
`information_set` returns a fresh decoded copy; `result_id` hashes canonical
content. Ordering scores or label vintages differently does not change the result.

| Field | Meaning |
| --- | --- |
| `observation_count` | All supplied score records, including unavailable scores |
| `positive_label_count`, `negative_label_count` | Known labels across that full population; negative count is withheld if semantics are invalid |
| `unlabeled_observation_count` | Missing, explicitly unknown, or not-yet-available outcomes |
| `unscored_observation_count` | Records whose scores are unavailable |
| `evaluated_observation_count` | Scored, explicitly labeled cases used in threshold metrics |
| `evaluated_positive_count`, `evaluated_negative_count` | Class counts within those evaluated cases |
| `threshold_results` | Raw threshold, four counts, five metrics, and per-metric withholding reasons |
| `average_precision` | NIST average precision over the complete distinct-score family |
| `selected_threshold` | Declared threshold, earlier-history choice, or `None` for sensitivity |
| `prospective` | Whether the declared threshold protocol and all score/target timestamps meet the timing condition below |

Unscored and unlabeled counts can overlap; adding them is not an exclusion count.
Metrics on complete cases describe that subset, not the original population.
Missingness can bias apparent performance. `available_with_exclusions` makes a
restricted denominator visible; the population counts remain explicit even when
`partially_withheld` also applies.

## Threshold integrity

| Mode | Evaluation thresholds | Integrity checks |
| --- | --- | --- |
| `pre_specified` | Exactly one declared threshold | Protocol registered before every evaluation score |
| `tuning_derived` | One threshold selected from declared candidates | Separate earlier target block, scores and labels known before tuning cutoff, cutoff before every evaluation score |
| `retrospective_sensitivity` | All declared candidate thresholds | Explicitly retrospective; selects no threshold and makes no prospective claim |

Tuning accepts separate `tuning_scores` and `tuning_labels`. Its protocol must be
registered before selection. Tuning labels use a **strictly before** availability
cutoff; a label arriving exactly at the selection timestamp is excluded. Tuning
and evaluation target-period blocks must be strictly separated, and source score
artifacts cannot overlap. The objective is declared `mcc`, `f1`, or `f2`;
maximization breaks exact ties at the highest threshold. Undefined candidates
are retained in the audit but cannot win. No valid tuning history or objective
withholds selection, with no threshold fallback. Changing evaluation labels,
including making all of them unknown, cannot change the tuning choice.

`prospective=True` additionally requires every score to precede the **start** of
its target quarter. Pre-specifying a threshold alone does not establish this.
In particular, a Phase 1 score computed after a reporting quarter is not a
prospective classifier for that same quarter. Timestamp checks cannot establish
that a caller fitted a model without leakage; score construction still needs
independent review. A withheld result's timing flag is not a performance claim.

Retrospective sensitivity reports min/max values and defined-threshold counts for
each metric. It never chooses or recommends the best holdout threshold. Its
results cannot be relabeled as prospective or out-of-sample threshold selection.
Average precision is descriptive ranking metrology, not a threshold selector.

## Numerical conventions

The integration exposes NIST counts and metrics where their mathematical
denominators are defined. The release fills undefined precision/recall with 1
and some undefined MCC/F-scores with 0. DRR instead reports `None` plus a specific
reason for zero-denominator cases; it does not present those fill conventions as
measured performance. Valid NIST metric values are otherwise unchanged.

MCC requires nonzero true and predicted class marginals. Precision requires
predicted positives; recall and average precision require observed positives.
F1/F2 are undefined only when both observed and predicted positive counts are
zero; a failure to find known positives yields a defined zero F-score.
Single-class populations can still have descriptive counts and some defined
metrics; no missing class is manufactured.

Average precision uses every distinct **raw** score and an explicit all-negative
endpoint with the public `expected("aps")` method. There is no min-max scaling,
threshold subsampling, epsilon perturbation of ties, or trapezoidal precision-
recall approximation. Constant scores produce prevalence as average precision
when positives exist. Extreme finite scores do not require subtracting extrema.

The default `maximum_prediction_cells=2_000_000` bounds each dense threshold
matrix. An oversized declared family withholds threshold evaluation. An oversized
average-precision family withholds AP separately while preserving available
fixed-threshold results. The exact limit and reasons are recorded; no approximate
alternative is silently substituted. Numerical warnings, non-finite values,
and arithmetic failures are surfaced; unexpected implementation errors propagate.

## Minimal synthetic example

```python
from drr_framework.supervisory import (
    BinaryEventLabel, ContingencyScore, ContingencyDesign, evaluate_contingency,
)

definition = "Synthetic quarter outcome; explicitly drawn 0/1 labels"
values = (0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0)
truth = (1, 0, 1, 1, 0, 1, 0, 0)
scores = tuple(
    ContingencyScore(f"SYN_{i}", "2021-03-31", "2020-12-30",
                     value, f"synthetic-score-{i}", "toy-v1")
    for i, value in enumerate(values)
)
labels = tuple(
    BinaryEventLabel(f"SYN_{i}", "2021-03-31", value, "2021-04-01",
                     definition, "Synthetic example")
    for i, value in enumerate(truth)
)
design = ContingencyDesign(
    observation_population="Eight synthetic institutions in 2021Q1",
    event_definition=definition,
    score_family="toy-v1",
    registered_at="2020-12-01",
    thresholds=(0.5,),
    label_semantics="explicit_institution_quarter",
    negative_label_definition="Zero is an explicit synthetic non-event draw",
)
result = evaluate_contingency(scores, labels, design=design,
                              evaluation_as_of="2021-04-30")
row = result.threshold_results[0]
assert (row.true_positives, row.false_positives,
        row.false_negatives, row.true_negatives) == (2, 1, 2, 3)
```

This example has precision 2/3, recall 1/2, F1 4/7, F2 10/19, and MCC
`4 / sqrt(240)`. Average precision is `(1 + 2/3 + 3/4 + 4/6) / 4`.
These are arithmetic checks, not evidence of empirical usefulness.

To adapt a Phase 1 result, use
`ContingencyScore.from_challenge(result, target_period=...)`. The target is
explicit: the adapter preserves the score, missing state, and source result ID,
and does not shift labels or construct lead-window outcomes. Changing the target
horizon does not establish that the original model is calibrated for that horizon.
Use the returned score family's ID in the design. The existing Phase 1 scoring
and event evaluation APIs remain unchanged.

## Verification and limits before actual use

Tests independently calculate contingency counts, MCC, F-scores, precision,
recall, and average precision; cover raw ties/extreme scores and undefined
denominators; and test determinism, reconstruction, missingness, timing,
tuning isolation, memory limits, dependency failures, and unchanged Phase 1
scores and lead-window event-backtesting behavior.

These checks establish implementation behavior for the tested cases. They do not
establish population representativeness, independent economic episodes, causal
relationships, validity of the model, or supervisory usefulness. No inference,
confidence intervals, ratings, or policy conclusions are generated.

Independent conceptual-soundness review must assess the unit of observation,
negative ascertainment, target horizon, censoring and missingness, threshold
protocol, event prevalence, and dependence. Implementation review should
reconstruct inputs, cutoff selection, score provenance, and package conventions.
Outcomes analysis needs an independently reviewed inventory and genuinely held-out
data. Monitoring should track coverage, label revisions, class prevalence,
undefined metrics, candidate/selected thresholds, and score-family changes.
Governance and intended-use review remain necessary before actual use. Using
NIST software establishes neither approval nor the appropriateness of DRR for
any supervisory use.
