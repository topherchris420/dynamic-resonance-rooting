# Evidence-Grade Rooting Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DRR's directed-rooting output statistically explicit, multiple-test aware, configurable through the public facade, and incapable of silently presenting untested effects as significant.

**Architecture:** Keep `RootingAnalyzer` as the numerical boundary, add a deterministic surrogate-inference layer inside it, and expose that configuration through `DynamicResonanceRooting`. Preserve existing dictionary consumers with a compatibility alias while moving graph construction to the canonical backend-neutral contract.

**Tech Stack:** Python 3.8+, NumPy, NetworkX, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-03-evidence-grade-rooting-design.md`

## Global Constraints

- Preserve FFT, Welch, wavelet, Markov, resonance-depth, and state-space behavior.
- Add no dependency.
- Preserve the `transfer_entropy` result key as an alias to `score_matrix`.
- Never emit a numeric significance p-value when `n_surrogates == 0`.
- Use deterministic NumPy random generation for fixed seeds.
- Keep the default high-level estimator `lagged_correlation` and surrogate count `25`.
- Build influence graphs from statistically significant edges only.
- Follow Python 3.8-compatible syntax in `src/drr_framework/modules.py` and `src/drr_framework/analysis.py`.

---

### Task 1: Statistical rooting result contract

**Files:**
- Modify: `src/drr_framework/modules.py:337-520`
- Modify: `tests/test_research_grade_outputs.py`
- Modify: `tests/test_drr_framework.py`

**Interfaces:**
- Consumes: existing `_lagged_correlation_scores`, `_transfer_entropy_scores`, and `_significant_edges` behavior.
- Produces: `RootingAnalyzer.analyze(..., surrogate_method: str = "circular_shift", correction: str = "max_statistic") -> dict`, canonical `score_matrix`, raw `p_values`, `adjusted_p_values`, `candidate_edges`, `significant_edges`, and inference metadata.

- [ ] **Step 1: Add failing disabled-inference and compatibility tests**

```python
def test_rooting_without_surrogates_does_not_fabricate_significance():
    rng = np.random.default_rng(11)
    source = rng.normal(size=200)
    target = np.roll(source, 2)
    result = RootingAnalyzer().analyze(np.column_stack([source, target]), n_surrogates=0)
    assert result["inference_available"] is False
    assert np.isnan(result["p_values"][0, 1])
    assert np.isnan(result["adjusted_p_values"][0, 1])
    assert result["significant_edges"] == []
    assert result["candidate_edges"]
    assert result["score_matrix"] is result["transfer_entropy"]
```

- [ ] **Step 2: Run the new test and confirm it fails because the existing implementation fabricates p-values or lacks the new fields**

Run: `python -m pytest tests/test_research_grade_outputs.py::test_rooting_without_surrogates_does_not_fabricate_significance -q`

- [ ] **Step 3: Add failing corrected-inference tests**

Use a deterministic three-series fixture with `source`, `np.roll(source, 2) + noise`, and an independent series. Assert shapes, `adjusted_p_values >= p_values` for finite off-diagonal cells, effective lag `2`, circular-shift metadata, and recovery of `dim_0 -> dim_1` with `n_surrogates=49`.

- [ ] **Step 4: Run the corrected-inference tests and confirm they fail on missing parameters/fields**

Run: `python -m pytest tests/test_research_grade_outputs.py -q`

- [ ] **Step 5: Implement the minimal statistical layer**

Add parameter validation for `surrogate_method in {"circular_shift", "permutation"}` and `correction in {"max_statistic", "none"}`. Add `_surrogate_statistics(...) -> Tuple[np.ndarray, np.ndarray]` that returns raw and max-statistic adjusted p-values. For circular shifts, draw an independent offset for each variable from valid offsets greater than `max_lag` and less than `n_samples - max_lag`; fall back to any non-zero offset for short inputs. For zero surrogates, fill off-diagonal cells with `NaN` and diagonals with `1.0`.

Build `candidate_edges` from the effect threshold. Build `significant_edges` from candidates using adjusted p-values for `max_statistic` and raw p-values for `none`. Include both p-values in each significant edge. Return inference metadata and alias the same array object under `score_matrix` and `transfer_entropy`.

- [ ] **Step 6: Run focused tests until green**

Run: `python -m pytest tests/test_research_grade_outputs.py tests/test_drr_framework.py -q`

- [ ] **Step 7: Refactor names and docstrings while focused tests remain green**

Replace backend-specific local names with `scores`/`score_matrix`, document the circular-shift null and max-statistic correction, and keep private helpers focused.

### Task 2: Public configuration and truthful graph behavior

**Files:**
- Modify: `src/drr_framework/analysis.py:186-312`
- Modify: `tests/test_drr_framework.py`

**Interfaces:**
- Consumes: Task 1's `RootingAnalyzer.analyze` signature and result fields.
- Produces: `analyze_influence_network(...rooting options...)` and `analyze_system(...rooting options...)`, significant-only graph construction, and structured `rooting_analysis` failures.

- [ ] **Step 1: Add a failing public-configuration propagation test**

Create a real multivariate lag fixture and call `analyze_system(..., state_space=False, rooting_method="lagged_correlation", rooting_max_lag=3, rooting_n_surrogates=19, rooting_random_state=17, rooting_alpha=0.10, rooting_surrogate_method="circular_shift", rooting_correction="max_statistic")`. Assert the result records method, surrogate count, alpha, null method, and correction.

- [ ] **Step 2: Add a failing no-promotion graph test**

Patch only the analyzer boundary to return a real schema with a candidate edge and no significant edges, call `analyze_influence_network`, and assert the returned graph has zero edges. This catches the current fallback promotion branch.

- [ ] **Step 3: Add a failing structured-error test**

Make the analyzer boundary raise `ValueError("rooting failed")`, run `analyze_system` with state-space disabled, and assert `results["rooting_analysis"] == {"error": "rooting failed", "method": "lagged_correlation"}`.

- [ ] **Step 4: Run the three tests and verify their expected failures**

Run: `python -m pytest tests/test_drr_framework.py -q`

- [ ] **Step 5: Implement minimal parameter propagation and graph behavior**

Add the seven keyword parameters from the spec to `analyze_system`; add corresponding explicit keyword-only parameters to `analyze_influence_network`. Resolve `rooting_max_lag=None` from `tau`, forward all values, use `score_matrix`, add only `significant_edges` to the graph, and store structured error metadata before returning `None` on direct failure.

- [ ] **Step 6: Run core and downstream consumer tests**

Run: `python -m pytest tests/test_drr_framework.py tests/test_audience_workflows.py tests/test_supervisory_workflows.py tests/test_validation_readiness.py tests/test_quickstart_example.py -q`

### Task 3: Documentation and end-to-end validation

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/architecture.md`
- Modify: `docs/reproducibility.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the final public signatures and result schema from Tasks 1 and 2.
- Produces: accurate user guidance for backend selection, candidate versus significant edges, surrogate resolution, and reproducible configuration.

- [ ] **Step 1: Update the quick-start and API documentation**

Show a multivariate call that explicitly sets `rooting_method`, `rooting_n_surrogates`, and `rooting_random_state`. Document `score_matrix`, the legacy alias, raw and adjusted p-values, candidate edges, significant edges, and the zero-surrogate semantics.

- [ ] **Step 2: Update architecture and reproducibility guidance**

Describe circular-shift surrogates, max-statistic correction, minimum attainable p-value, and why nonstationary series may require a different null model. State that graph edges are significant-only by default.

- [ ] **Step 3: Record the change in the changelog**

Add an unreleased entry describing evidence-grade rooting inference and the intentionally stricter edge selection.

- [ ] **Step 4: Run the complete verification matrix**

Run, in order:

```text
python -m compileall src examples scripts tests
python -m pytest --cov=drr_framework --cov-report=term-missing
python -m ruff check .
python -m black --check .
python -m mypy src/drr_framework
python -m build
python -m drr_framework.experiments --no-save
```

If the experiments CLI has no `--no-save` option, run its documented deterministic command with a temporary output directory instead. Record exact outputs and distinguish pre-existing failures from regressions.

- [ ] **Step 5: Inspect the final diff for compatibility and scope**

Confirm no dependency changed, no unrelated refactor entered the diff, the compatibility alias remains, every new option is documented, no generated output artifact is tracked, and no secret or unsafe file operation was introduced.
