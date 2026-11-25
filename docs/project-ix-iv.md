# Project IX/IV: computational geometry and resonant intelligence

**Status:** exploratory implementation; no performance claim, no production backend changes.
**Source:** OpenAI, *The matrix multiplication exponent is nine fourths*, November 24, 2026,
Theorem 1.1 and Corollaries 1.2–1.3. The supplied preprint claims a matching lower bound
for general exact arithmetic circuits. The claim is not independently verified here.
The lower bound does not supply a practical algorithm, predict floating-point wall time,
or apply directly to restricted sparse or factorized workloads.

## Study protocol

Before collecting performance results, fix the following synthetic study:
- Sizes 64, 128, 256, 512; seeds 19, 73, 127; seven timed repetitions per method,
  alternating method order after an untimed correctness evaluation.
- Dense square: NumPy/BLAS versus one Strassen level with BLAS leaves.
- DRR-like Gram operator: explicit F F^T times eight vectors versus F(F^T X),
  with feature count floor(sqrt(n)). Factors are supplied, not inferred.
- Sparse transition: dense matrix versus SciPy CSR, with 2% expected nonzero entries.
  Compare repeated actions and one-shot setup plus median action separately.
- Float64 accuracy: finite outputs, scaled Frobenius error <= 1e-10 and
  elementwise rtol 1e-9, atol 1e-10. A row passes the speed gate only if it is
  accurate and the candidate is at least 1.2x faster.
- Record all rows, raw times, setup costs, environment and operator storage estimates.
  Storage estimates exclude input vectors, outputs, temporaries and allocator overhead;
  they are not peak-memory measurements.
- Do not interpret an isolated row as a general speedup. Require accuracy in every
  row, consistent improvement across seeds and sizes relevant to an actual DRR
  workload, and independent end-to-end replication before changing defaults.

Run from an installed checkout (set thread counts before Python starts):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m drr_framework.project_ix_iv --output results/project-ix-iv.json
python -m pytest tests/test_project_ix_iv.py
```

Repeat on independent seeds 211, 307, 401 and the deployment hardware; retain all
negative results. These fixtures are synthetic, not measurements of the existing
resonance kernel, Koopman smoother or real biosignals.

## R.A.I.N. research interface

The JSON report is suitable for ingestion by an external research agent. The
integer-only rectangular_rank_bound function encodes the preprint's claimed
lower bound max(ab, bc, ca, ceil((abc)^(3/4))). It is a necessary condition only,
not a correctness verifier or a predictor of runtime. A future exact algorithm
search must independently verify every tensor coefficient, and a near-9/4
algorithm must be sourced from an upper-bound construction before implementation.
No autonomous agent integration or new asymptotic algorithm is claimed here.
