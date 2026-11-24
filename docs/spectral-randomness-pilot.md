# Spectral structure, oscillations, and apparent randomness

Status: exploratory synthetic pilot, 2026-11-24; not a preregistered confirmation or a new theorem about natural systems.

## Mathematical question

Which structural properties survive a transformation from a signal to its power spectrum, and which require phase-sensitive measurements? For a finite real sequence, the DFT fixes the sequence only when both amplitudes and phases are retained. Circular autocorrelation depends on squared Fourier magnitudes alone. Keeping the magnitudes while randomizing positive-frequency phases (and maintaining conjugate symmetry, DC and Nyquist) preserves the periodogram and circular autocorrelation exactly up to rounding, but can destroy higher-order phase relationships. Thus no statistic depending exclusively on those magnitudes can distinguish such matched pairs.

For bins a=7, b=11, c=18=a+b, define phase closure C=|mean(exp(i*(phi_a+phi_b-phi_c)))| over M independent blocks. This is an unweighted phase-only statistic related to bispectral methods, not amplitude-weighted bicoherence. Closure is invariant under circular shifts of each block because a+b-c=0. Under independent uniform closure phases, E[C^2]=1/M; under independent identically distributed closure phases with mean phasor mu, E[C^2]=1/M+(1-1/M)|mu|^2. For noiseless wrapped Gaussian closure jitter sigma, |mu|=exp(-sigma^2/2). Additive noise attenuates coherence further.

## Reproduction

From the repository root:

```bash
python scripts/run_spectral_randomness_pilot.py
```

NumPy only; seed 20261124; 300 trials per condition; each trial contains 32 independent 128-sample standard Gaussian blocks. Alternative: add three on-grid complex Fourier coefficients of magnitude 32 with independent uniform base phases and third phase equal to their sum plus Gaussian jitter. Use 199 independently generated uniform closure-null samples and the rank p-value (1 + exceedances)/200, reject at 0.05. The simplified closure-null simulation is equivalent for this statistic to drawing independent uniform phases at the three distinct bins conditional on magnitudes. A separate full-spectrum phase randomization verifies matched magnitudes after inverse and forward FFT. No frequency search or multiple comparisons are performed.

## Observed local pilot

| Condition | Rejections / 300 | Rate | Mean C |
| --- | ---: | ---: | ---: |
| Independent Gaussian noise | 17 | 0.0567 | 0.1647 |
| Jitter 0 rad | 300 | 1.0000 | 0.9065 |
| Jitter 0.5 rad | 300 | 1.0000 | 0.8054 |
| Jitter 1 rad | 300 | 1.0000 | 0.5638 |
| Jitter 2 rad | 35 | 0.1167 | 0.1950 |

Maximum absolute FFT magnitude discrepancy between original and randomized inverse/forward reconstructions: 3.552713678800501e-14. Null rejection rate is compatible with nominal 5% at this sample size; high-jitter power is weak. The pilot does not establish calibration on real time series, causality, chaos detection, prediction, or an advantage over established bicoherence. A coherent harmonic from a single nonsinusoidal source can give the same closure.

## Next confirmatory study (proposed, not executed)

Freeze independent seeds, on/off-grid tones, SNR, record lengths, and a grid of triads before testing; use a max-statistic or multiplicity correction for searched triads. Include independent AR/long-memory processes, non-Gaussian innovations, nonsinusoidal single oscillators, deterministic chaotic systems, transient chirps and random walks. Match spectra using phase surrogates only where justified and compare amplitude-adjusted surrogates with explicit null assumptions. Benchmark DRR resonance depth, Welch entropy, weighted bicoherence, phase closure, and out-of-sample predictive skill; report uncertainty and abstain on nonstationary data or inadequately calibrated nulls. Distinguish ensemble/block structure from single-record identifiability. Publish negative outcomes.

Arithmetic extension: the explicit formula for the Chebyshev function relates prime-power counting deviations to zeta zeros via terms x^rho/rho. On the critical line, log(x) is the oscillation coordinate. The Montgomery-Odlyzko connection to random-matrix local spacing statistics motivates a separate study against Poisson and GUE controls using independently specified unfolding and finite-size corrections. No claim of a proof of the Riemann hypothesis, no claim that primes are physically oscillating, and no transfer of this synthetic result to arithmetic.

## Riemann hypothesis: status as of 2026-11-24

On November 20, 2026, Levent Alpöge announced a Claude-assisted proof of the Riemann hypothesis in Lean. On November 21, the Lean FRO and Mathlib Initiative issued a joint statement certifying the theorem statement and proof: independent compilation, theorem comparison and multiple proof checkers supported the result. Broader mathematical exposition and interpretation remain underway; formal verification and Millennium Prize recognition are distinct. DRR does not reproduce the formal proof or claim new arithmetic results. The proposed arithmetic experiment uses the classical explicit formula and numerical zeros; its statistical conclusions must be independently validated and do not follow from RH alone.

## References

- Theiler et al., Testing for nonlinearity in time series: the method of surrogate data (1992), https://doi.org/10.1016/0167-2789(92)90102-S
- Schreiber and Schmitz, Surrogate time series (2000), https://doi.org/10.1016/S0167-2789(00)00043-9
- Odlyzko, On the distribution of spacings between zeros of the zeta function (1987), https://www.jstor.org/stable/2007890
- Lean FRO and Mathlib Initiative, Lean verification of the Riemann Hypothesis (2026-11-21), https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
- Clay Mathematics Institute, Riemann Hypothesis, https://www.claymath.org/millennium/Riemann-Hypothesis/
