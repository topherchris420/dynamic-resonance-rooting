# Mathematical research agenda: Dynamic Resonance Rooting

**Status:** candidate research protocol, not an implemented algorithmic improvement or a new empirical result.

## Why this exists

[OpenAI's `math`](https://github.com/openai/math) is a mathematics manuscript collection with some accompanying Lean proof artifacts, not an installable mathematics or inference engine. Its own [README](https://github.com/openai/math/blob/main/README.md) warns that some manuscripts do not have formalizations and some unformalized results may have issues.

The current R.A.I.N. Lab at [Lop Nur Twin](https://github.com/topherchris420/lop-nur-twin) already holds a commit-pinned, read-only lexical index derived from that collection. **Use one shared substrate, not eight forked copies.** Its new [portfolio scout](https://github.com/topherchris420/lop-nur-twin/blob/main/docs/MATH_PORTFOLIO.md) can return candidate titles, matched terms, upstream commit and index hash, without upgrading source material to proven product behavior.

The upstream catalogue was inspected at commit [`fd4aeeb2ee4f`](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb) on 9 October 2026. **The bundled R.A.I.N. index may pin a different commit**; its reported provenance, not the inspection date, controls the scouting results.

## The question

**Which statistical assumptions keep lag-search false alarms controlled under long-memory or dependent noise?**

Candidate search terms: `dependent random process correlation spectral bounds`

From a checkout of [R.A.I.N. Lab](https://github.com/topherchris420/lop-nur-twin):

```sh
npm run rain:math:verify
npm run rain:math:portfolio -- --project dynamic-resonance-rooting --json
```

An empty result is acceptable. Search hits are lexical references; they are not recommendations to transfer a theorem, and a source's Lean listing is not proof of a new experimental finding in this repository.

## Assumptions that must be mapped before importing a mathematical idea

1. A surrogate distribution respects the tested process' serial structure.
2. A discovered directed edge is an estimate of lead-lag, not a causal arrow.
3. All thresholds, lag windows and multiple-test corrections are fixed before the holdout.

## Baseline already available

`drr-reproduce --no-artifacts` for known synthetic fixtures, and `python scripts/run_structural_change_benchmark.py --output-dir results/expected` for the pinned public-data comparison.

## Next falsifiable experiment (not yet executed)

Compare the shipped surrogate method against at least one conservative null on seeded independent and lag-coupled AR processes at varied persistence. Lock a false-alarm tolerance first, then evaluate previously unseen seeds and the already documented external QBO benchmark separately.

**Negative control:** Run uncoupled processes with the same autocorrelation as coupled ones; count false edges and structural alarms.

**Completion artifact:** A preregistered benchmark, per-seed JSON outputs, calibration plots, negative controls, a failure report and no retroactive editing of the original result.

## Authority, scope and refusal

The openai/math catalog is mathematical literature, not evidence that DRR's surrogate model meets its error budget. Its previously published QBO external claim remains not supported.

Do not change a control policy, a research registry verdict, an experiment's preregistered criteria, the deployed product, or the meaning of an evidence class on the strength of a matched mathematical phrase. A formal proof establishes only the theorem and assumptions actually formalized; a software simulation tests its own implementation and scenarios; external and physical claims require separate evidence.

**Current verdict:** `NOT TESTED` for this proposed mathematical extension. This document is a reproducible research question and a reference-workflow entry, not a passing benchmark.

See also [the R.A.I.N. mathematics workflow](https://github.com/topherchris420/lop-nur-twin/blob/main/docs/MATH_PORTFOLIO.md) and the [OpenAI mathematics catalogue](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/CONTENTS.md).
