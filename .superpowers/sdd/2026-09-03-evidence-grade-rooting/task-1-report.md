# Task 1 Report: Statistical Rooting Result Contract

## Summary

Implemented the evidence-grade rooting contract in `RootingAnalyzer` with:

- canonical `score_matrix` plus legacy `transfer_entropy` alias to the same array
- raw `p_values` and max-statistic `adjusted_p_values`
- explicit `candidate_edges` versus statistically filtered `significant_edges`
- `surrogate_method`, `correction`, `n_surrogates`, `minimum_attainable_p_value`, and `inference_available` metadata
- deterministic circular-shift and legacy permutation surrogate generation
- validation for supported surrogate methods and correction modes

## RED Cycle

### RED 1

Command:

```powershell
python -m pytest tests/test_research_grade_outputs.py::test_rooting_without_surrogates_does_not_fabricate_significance -q
```

Output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
_______ test_rooting_without_surrogates_does_not_fabricate_significance _______

>       assert result["inference_available"] is False
E       KeyError: 'inference_available'

tests\test_research_grade_outputs.py:86: KeyError
=========================== short test summary info ============================
FAILED tests/test_research_grade_outputs.py::test_rooting_without_surrogates_does_not_fabricate_significance
1 failed in 49.04s
```

### RED 2

Command:

```powershell
python -m pytest tests/test_research_grade_outputs.py -q
```

Output:

```text
FF.                                                                   [100%]
================================== FAILURES ===================================
_______ test_rooting_without_surrogates_does_not_fabricate_significance _______

>       assert result["inference_available"] is False
E       KeyError: 'inference_available'

tests\test_research_grade_outputs.py:86: KeyError
_________ test_rooting_analyzer_reports_corrected_surrogate_inference _________

>       result = RootingAnalyzer().analyze(
            ...
            surrogate_method="circular_shift",
            correction="max_statistic",
        )
E       TypeError: RootingAnalyzer.analyze() got an unexpected keyword argument 'surrogate_method'

tests\test_research_grade_outputs.py:103: TypeError
=========================== short test summary info ============================
FAILED tests/test_research_grade_outputs.py::test_rooting_without_surrogates_does_not_fabricate_significance
FAILED tests/test_research_grade_outputs.py::test_rooting_analyzer_reports_corrected_surrogate_inference
2 failed, 4 passed in 32.58s
```

### RED 3

Command:

```powershell
python -m pytest tests/test_drr_framework.py::test_rooting_analyzer_validates_inference_configuration -q
```

Output:

```text
F                                                                        [100%]
================================== FAILURES ===================================
___________ test_rooting_analyzer_validates_inference_configuration ___________

>           analyzer.analyze(data, surrogate_method="bootstrap")
E           TypeError: RootingAnalyzer.analyze() got an unexpected keyword argument 'surrogate_method'

tests\test_drr_framework.py:78: TypeError
=========================== short test summary info ============================
FAILED tests/test_drr_framework.py::test_rooting_analyzer_validates_inference_configuration
1 failed in 63.84s (0:01:03)
```

## GREEN Cycle

### Targeted GREEN Checks

Commands and outputs:

```powershell
python -m pytest tests/test_research_grade_outputs.py::test_rooting_without_surrogates_does_not_fabricate_significance -q
```

```text
.                                                                        [100%]
1 passed in 31.50s
```

```powershell
python -m pytest tests/test_research_grade_outputs.py::test_rooting_analyzer_reports_corrected_surrogate_inference -q
```

```text
.                                                                        [100%]
1 passed in 35.49s
```

```powershell
python -m pytest tests/test_drr_framework.py::test_rooting_analyzer_validates_inference_configuration -q
```

```text
.                                                                        [100%]
1 passed in 31.88s
```

### Focused Suite

Command:

```powershell
python -m pytest tests/test_research_grade_outputs.py tests/test_drr_framework.py -q
```

Output:

```text
27 passed in 42.02s
```

### Additional Verification

Command:

```powershell
python -m compileall src/drr_framework/modules.py
```

Output: command succeeded with exit code `0` and no emitted diagnostics.

Command:

```powershell
python -m ruff check src/drr_framework/modules.py tests/test_research_grade_outputs.py tests/test_drr_framework.py
```

Output:

```text
All checks passed!
```

## Changed Files

- `src/drr_framework/modules.py`
- `tests/test_research_grade_outputs.py`
- `tests/test_drr_framework.py`

## Self-Review

- Kept the public compatibility alias by returning the exact same `numpy.ndarray` object under both `score_matrix` and `transfer_entropy`.
- Preserved Python 3.8 compatibility by staying within existing typing/style constraints.
- Kept scope limited to the statistical rooting layer: no facade, graph, or unrelated benchmark changes.
- Used the same observed-score threshold for exploratory `candidate_edges`, while reserving `significant_edges` for inferentially supported links only.
- Computed both raw and family-wise-error-controlled adjusted p-values on every surrogate run so later callers can choose correction policy without recomputation.

## Concerns

- The circular-shift null follows the brief exactly, but like any shift-based surrogate it assumes reasonably stationary series.
- I did not wait on a slower targeted `mypy` run after the parent override that Task 1 should stop at the focused statistical validation gate; final validation can own broader static checks.
