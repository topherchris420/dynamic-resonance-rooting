# DRR-Mellin: multiscale spectral diagnostics

**Status:** exploratory research extension, opt-in Python API. No change to default analysis, financial models, or inference thresholds.

## Mathematical source

Jimmie Adriazola, *The Alpöge Proof of the Riemann Hypothesis*, 287-page manuscript supplied by the project owner (November 2026), Section 2, equations (2.1)–(2.5) and Theorem 2.4. This is a conventional reconstruction of the formal argument, and its introduction explicitly notes missing conventional proofs of some internal statements. DRR uses its analytic definitions and window-kernel ideas, not the later arithmetic-geometric proof. No theorem certification or proof claim is made here.

For the von Mangoldt function Λ and Euler's constant γ:

    G(x) = sum_{n<=x} Λ(n)(x/n - 1) - x log(x) + (1+γ)x.

The residue at a nontrivial zero ρ is proportional to -1/[ρ(ρ-1)] with the zero's multiplicity. A finite reconstruction uses conjugate pairs from the existing ten-ordinate fixture, the constant log(2π), and a truncated trivial-zero contribution. Known zeros are inputs, not outputs inferred from prime data.

For a nontrivial Dirichlet character, the corresponding expression uses Re χ(n) and adds x Re[L'(1,χ)/L(1,χ)]. The caller must provide a valid periodic table and that independently evaluated constant; this API does not certify the character or compute the constant.

## Reproduce

```bash
python -m pip install -e .
python -m pytest tests/test_mellin.py -q
python -c "import json; from drr_framework.mellin import benchmark; print(json.dumps(benchmark(), indent=2))"
```

Fixed half-integer grid from 10.5 to 1000.5 (991 points). RMSE of (reconstruction minus exact smoothed sum)/sqrt(x):

| Positive zeros | Normalized RMSE |
| ---: | ---: |
| 0 | 0.0082000001 |
| 1 | 0.0048502467 |
| 4 | 0.0023624416 |
| 10 | 0.0011550711 |

Approximately 85.9% lower error than the zero-mode baseline on this grid. This is a descriptive reconstruction, not a forecast or direct comparison with the unsmoothed Chebyshev benchmark.

## Finite-data operators

- `smoothed_prime_sum`: bounded sieve (x <= 1,000,000), cumulative weighted sums.
- `smoothed_zero_sum`: truncated known-zero residue expansion; finite approximation, especially near x=1.
- `log_spectrum`: one-sided FFT amplitudes of x^(-beta)y on a uniformly log-spaced grid; frequency in radians per natural-log unit. This approximates spectral behavior along a Mellin vertical line, not analytic continuation.
- `fit_log_mode`: single conjugate-mode least-squares scan over supplied growth and angular-frequency candidates, with a relative squared residual. No confidence interval, discovery p-value, off-grid guarantee, or multi-mode separation.
- `window_excess`: sampled one-sided bound violations and maximum normalized value. A finite sample cannot establish the theorem's continuum or arbitrarily large window hypotheses.
- `gamma_window_mass`: normalized Gamma(k+1, rate σ−1) mass in a log-scale interval, inspired by the concentration kernel in the proof. This models the bounding kernel, not a proof of the theorem.

```python
import numpy as np
from drr_framework.mellin import fit_log_mode, log_spectrum

u = np.arange(256) * (2 * np.pi / 256)
x = np.exp(u)
y = np.sqrt(x) * np.cos(14 * u + 0.4)
frequency, amplitude = log_spectrum(x, y)
print(fit_log_mode(x, y, [0.4, 0.5, 0.6], [13, 14, 15]))
```

The noiseless, on-grid fixture recovers beta=0.5 and omega=14. This is a specification test, not evidence of robustness. Next research step: preregister noisy and off-grid trials, compare to ordinary log-coordinate regression/FFT and multi-mode baselines, quantify false discoveries and coverage, and report negative outcomes. No biological, financial, or consciousness implications follow from the arithmetic result.
