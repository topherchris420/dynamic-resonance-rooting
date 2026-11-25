"""Project IX/IV: opt-in structured linear algebra experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
from functools import partial
from math import isqrt
from operator import matmul
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import scipy
from scipy import sparse


def rectangular_rank_bound(a: int, b: int, c: int) -> int:
    """Claimed exact tensor-rank bound, conditional on the Nov. 24 preprint."""
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (a, b, c)):
        raise ValueError("dimensions must be positive integers")
    power = (a * b * c) ** 3
    root = isqrt(isqrt(power))
    ceiling = root + int(root**4 < power)
    return max(a * b, b * c, c * a, ceiling)


def strassen_step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One level of Strassen with BLAS leaves; no claim of exponent 9/4."""
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("equal square matrices required")
    n = a.shape[0]
    if n < 2 or n % 2:
        return a @ b
    h = n // 2
    a11, a12, a21, a22 = a[:h, :h], a[:h, h:], a[h:, :h], a[h:, h:]
    b11, b12, b21, b22 = b[:h, :h], b[:h, h:], b[h:, :h], b[h:, h:]
    m1 = (a11 + a22) @ (b11 + b22)
    m2 = (a21 + a22) @ b11
    m3 = a11 @ (b12 - b22)
    m4 = a22 @ (b21 - b11)
    m5 = (a11 + a12) @ b22
    m6 = (a21 - a11) @ (b11 + b12)
    m7 = (a12 - a22) @ (b21 + b22)
    return np.block([[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]])


def gram_action(features: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply F F.T without materializing the dense Gram matrix."""
    if features.ndim != 2 or vectors.ndim != 2 or features.shape[0] != vectors.shape[0]:
        raise ValueError("incompatible two-dimensional inputs")
    return features @ (features.T @ vectors)


def _time_pair(
    baseline: Callable[[], np.ndarray], candidate: Callable[[], np.ndarray], repeats: int
) -> dict:
    reference, actual = baseline(), candidate()
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    error = float(np.linalg.norm(actual - reference) / max(1.0, np.linalg.norm(reference)))
    elementwise_close = bool(np.allclose(actual, reference, rtol=1e-9, atol=1e-10))
    accurate = finite and error <= 1e-10 and elementwise_close
    timings: dict = {"baseline": [], "candidate": []}
    for i in range(repeats):
        order = [("baseline", baseline), ("candidate", candidate)]
        if i % 2:
            order.reverse()
        for name, operation in order:
            start = perf_counter()
            operation()
            timings[name].append(perf_counter() - start)
    base, trial = median(timings["baseline"]), median(timings["candidate"])
    return {
        "baseline_seconds": base,
        "candidate_seconds": trial,
        "raw_seconds": timings,
        "speedup": base / max(trial, 1e-15),
        "relative_frobenius_error": error,
        "accurate": accurate,
        "passes_reused_operation_gate": accurate and base >= 1.2 * trial,
    }


def benchmark(sizes=(64, 128, 256, 512), seeds=(19, 73, 127), repeats: int = 7) -> dict:
    """Run fixed-seed paired comparisons on synthetic DRR-like workloads."""
    if repeats < 3 or not sizes or not seeds or any(n < 2 or n > 2048 for n in sizes):
        raise ValueError("require repeats >= 3, nonempty seeds, and sizes in [2, 2048]")
    rows = []
    for seed in seeds:
        for n in sizes:
            rng = np.random.default_rng(seed)
            a, b = rng.standard_normal((2, n, n))
            f = rng.standard_normal((n, max(1, isqrt(n))))
            x = rng.standard_normal((n, 8))
            w = rng.standard_normal((n, n))
            w[rng.random((n, n)) >= 0.02] = 0.0
            for case in ("dense", "gram_action", "sparse_transition"):
                start = perf_counter()
                if case == "dense":
                    base = partial(matmul, a, b)
                elif case == "gram_action":
                    gram = f @ f.T
                    base = partial(matmul, gram, x)
                else:
                    base = partial(matmul, w, x)
                base_setup = perf_counter() - start
                start = perf_counter()
                if case == "dense":
                    trial = partial(strassen_step, a, b)
                    storage: Optional[dict] = None
                elif case == "gram_action":
                    trial = partial(gram_action, f, x)
                    storage = {"baseline": int(gram.nbytes), "candidate": int(f.nbytes)}
                else:
                    csr = sparse.csr_matrix(w)
                    trial = partial(matmul, csr, x)
                    storage = {
                        "baseline": int(w.nbytes),
                        "candidate": int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes),
                    }
                candidate_setup = perf_counter() - start
                result = _time_pair(base, trial, repeats)
                baseline_total = base_setup + result["baseline_seconds"]
                candidate_total = candidate_setup + result["candidate_seconds"]
                result.update(
                    {
                        "case": case,
                        "n": n,
                        "seed": seed,
                        "baseline_setup_seconds": base_setup,
                        "candidate_setup_seconds": candidate_setup,
                        "operator_storage_bytes": storage,
                        "passes_one_shot_gate": result["accurate"]
                        and baseline_total >= 1.2 * candidate_total,
                    }
                )
                rows.append(result)
    return {
        "study": "Project IX/IV",
        "status": "exploratory_synthetic_benchmark",
        "theory": "conditional lower bound only; no 9/4 algorithm implemented",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "repeats": repeats,
        "rows": rows,
        "all_accurate": all(row["accurate"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 73, 127])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(tuple(args.sizes), tuple(args.seeds), args.repeats)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if not report["all_accurate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
