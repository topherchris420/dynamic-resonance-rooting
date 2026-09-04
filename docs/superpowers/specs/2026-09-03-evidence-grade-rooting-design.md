# Evidence-Grade Rooting Inference Design

## Purpose

Strengthen the statistical meaning and public configurability of DRR's directed-rooting analysis without changing its spectral, depth, or state-space algorithms. The improvement must make a clear distinction between large observed effects and effects that survive a stated surrogate test.

## Current problems

`RootingAnalyzer.analyze` uses a backend-specific result name (`transfer_entropy`) even when the selected backend is lagged correlation. When surrogate testing is disabled, it converts an effect-size threshold into synthetic zero p-values. The high-level `DynamicResonanceRooting` facade hard-codes the rooting backend, lag, surrogate count, alpha, and seed. It also promotes threshold-only edges into the influence graph when no statistically significant edge exists and can silently omit rooting output after an exception.

Together these behaviors make it difficult for callers and generated reports to tell which estimator ran, which inferential procedure was used, and whether an edge is merely a candidate or statistically supported.

## Selected design

### Backend-neutral result contract

`RootingAnalyzer.analyze` will return `score_matrix` as the canonical score field. The existing `transfer_entropy` field will remain as an alias to the same array for backwards compatibility. The `method` field will continue to report the effective backend, including fallback from unavailable transfer entropy.

### Surrogate null model

The default tested null will independently circular-shift each variable by a random non-zero offset outside the searched lag band. This preserves each variable's marginal distribution and periodic autocorrelation while breaking the observed cross-series alignment. `surrogate_method="permutation"` remains available for callers that explicitly want the historical value-permutation null.

When `n_surrogates == 0`, off-diagonal p-values and adjusted p-values will be `NaN`, `inference_available` will be false, and `significant_edges` will be empty. No synthetic p-value will be created.

### Multiple-testing control

Each surrogate produces both a per-edge score matrix and the maximum off-diagonal score across the family. Raw empirical p-values use per-edge exceedances. Adjusted empirical p-values compare every observed edge against the surrogate maximum distribution, providing max-statistic family-wise error-rate control. `correction="max_statistic"` is the default; `correction="none"` selects raw p-values.

The result will report the effective correction, surrogate method, surrogate count, and minimum attainable p-value `1 / (n_surrogates + 1)`.

### Candidate and significant edges

`candidate_edges` will contain edges above the existing effect-size threshold and will not claim significance. `significant_edges` will be the subset whose selected p-value is at most `alpha`. Each significant edge will include `p_value`, `adjusted_p_value`, and `lag`.

The influence graph will include significant edges only. An empty edge set is a valid analysis result, not a reason to promote candidates.

### Public orchestration

`DynamicResonanceRooting.analyze_system` and `analyze_influence_network` will accept and forward:

- `rooting_method="lagged_correlation"`
- `rooting_max_lag=None`, resolving to `max(1, tau)`
- `rooting_n_surrogates=25`
- `rooting_random_state=0`
- `rooting_alpha=0.05`
- `rooting_surrogate_method="circular_shift"`
- `rooting_correction="max_statistic"`

Existing callers that do not pass these parameters keep deterministic, tested rooting with the same estimator and sample count. The statistical selection becomes stricter and explicitly reported.

If rooting fails inside `analyze_system`, the returned payload will contain `rooting_analysis={"error": ..., "method": requested_method}` instead of silently omitting the section. Direct calls to `analyze_influence_network` will continue returning `None` on failure for compatibility, while storing the structured error in `self.rooting_results`.

## Validation

Tests will prove:

1. Disabled surrogate inference produces no fabricated significance.
2. Circular-shift surrogates are deterministic for a fixed seed and preserve each series' values/autocorrelation structure.
3. Raw and max-statistic adjusted p-value matrices have valid shapes and ordering (`adjusted >= raw`).
4. A strong injected lagged direction is recovered with the expected lag.
5. Independent autocorrelated null series do not produce promoted graph edges under the deterministic fixture.
6. High-level configuration reaches `RootingAnalyzer` and the actual method/correction appear in results.
7. Rooting failures produce structured error metadata.
8. Existing compatibility keys and consumers still work.

The final gate is the repository's complete test, Ruff, Black, mypy, compile, build, and relevant benchmark/example commands. No new dependency is required.

## Risks and mitigations

- Stricter inference may yield fewer graph edges. This is intentional; `candidate_edges` retains exploratory effect-size output without calling it significant.
- Circular shifts are inappropriate for strongly nonstationary series. The result records the null model, and callers can explicitly select permutation while future work can add block or phase-randomized surrogates.
- Twenty-five surrogates imply a minimum attainable p-value of about 0.0385. The result exposes this resolution so callers can increase the count for stronger evidence.
- Transfer entropy remains optional. The requested and effective backend are both visible through configuration and the existing `method` result field.

## Out of scope

This change does not fix the separate supervisory-array transpose, rewrite unsupported historical benchmark documents, optimize state-space serialization, or harden the UCI ZIP downloader. Those remain follow-up work.
