# Dynamic Resonance Rooting (DRR) Framework: Technical Assessment & Limitations

## 1. Executive Summary & Overview

This document provides a rigorous technical assessment of the **Dynamic Resonance Rooting (DRR) Framework** (v4.3.0). It documents the current mathematical assumptions, signal-processing techniques, resonance and rooting metrics, state-space diagnostics, regime-detection logic, banking adapters, stress-testing utilities, and existing benchmarks. It explicitly evaluates how DRR compares against conventional econometric, quantitative finance, and supervisory techniques, distinguishing genuinely novel combinations from standard established tools.

---

## 2. Technical Architecture & Component Analysis

### 2.1 Mathematical Assumptions & Embedding
* **Time-Delay Embedding**: Given a 1D observable $x(t)$, phase-space reconstruction is formed as $\mathbf{X}_t = [x(t), x(t-\tau), \dots, x(t-(m-1)\tau)]^T$ where $m$ is the embedding dimension and $\tau$ is the time delay.
* **Stationarity & Linearity Assumptions**: Core spectral methods (FFT, Welch) assume linear, weakly stationary processes within the observation window. The Morlet Wavelet path relaxes stationarity by computing localized time-frequency scalograms $\text{Scalogram}(f, t) = |(x * \psi_{f,\sigma})(t)|^2$.
* **Markov Discretization**: In Markov-based resonance detection, continuous phase space is partitioned into $K$ discrete states via $K$-means clustering, assuming a first-order Markov process with stationary transition probabilities $P_{ij} = P(S_{t+1}=j \mid S_t=i)$.

### 2.2 Signal-Processing & Resonance Detection
DRR supports four resonance detection backends:
1. **FFT (`rfft`)**: Computes real-valued discrete Fourier transform. Identifies peaks exceeding `peak_height_ratio * max_power`.
2. **Welch PSD (`welch`)**: Estimates power spectral density via averaged modified periodograms over overlapping segments ($n_{perseg} \le 256$, 50% overlap).
3. **Morlet Wavelet (`wavelet`)**: Computes continuous wavelet transform (CWT) over logarithmically spaced frequencies using $L_1$-normalized analytic Morlet wavelets with wavenumber $\omega_0 = 6.0$. Parabolic interpolation in log-frequency space refines peak locations.
4. **Markov Clustering (`markov`)**: Clusters multidimensional data into $K$ states, constructs transition matrix $P$, and identifies self-transition probabilities $P_{ii} > 0.9$ as resonant persistence states.

### 2.3 Resonance Metrics (Resonance Depth)
The composite **Resonance Depth** $D_R \in [0, 1]$ integrates four normalized sub-metrics:
$$D_R = 0.35 \cdot S_{\text{conc}} + 0.25 \cdot T_{\text{pers}} + 0.25 \cdot \Phi_{\text{coh}} + 0.15 \cdot A_{\text{stab}}$$
* **Target frequency ($f_0$)**: The Welch peak refined to sub-bin precision by Gaussian (log-parabolic) interpolation (`drr_composite_v2`). Without refinement, a clean tone's phase coherence varied from 0.76 to 0.996 depending only on its position on the FFT grid.
* **Spectral Concentration ($S_{\text{conc}}$)**: Ratio of power within a $\pm 1$ bin band around target frequency $f_0$ to total power $\sum P(f)$.
* **Temporal Persistence ($T_{\text{pers}}$)**: Mean exponential decay score $\exp(-((f_k - f_0)/\delta)^2)$ of target frequency stability across sub-windows, where $\delta = f_s / n_{\text{seg}}$ is one bin of the segment spectrum. The v1 score floored $\delta$ at 0.5 in the caller's units, so series sampled monthly or at 1 Hz always looked persistent.
* **Phase Coherence ($\Phi_{\text{coh}}$)**: Circular mean $\left| \frac{1}{N} \sum_{t=1}^N e^{i (\theta(t) - 2\pi f_0 t)} \right|$ where $\theta(t) = \text{angle}(\text{hilbert}(x(t)))$.
* **Amplitude Stability ($A_{\text{stab}}$)**: Clamped coefficient of variation $1 - \frac{\sigma(A)}{\mu(A)}$ where $A(t) = |\text{hilbert}(x(t))|$.

### 2.4 Epistemic Belief Updating (QBism Agent)
* **Agent Model**: Updates prior belief $b_0 = 0.5$ using a sigmoid likelihood mapping: $L(D_R) = \frac{1}{1 + e^{-D_R}}$.
* **Update Rule**: $b_{k+1} = \text{clip}\left(b_k + \eta (L(D_{R, k}) - b_k), 0.01, 0.99\right)$ with learning rate $\eta = 0.1$.
* **System Rooting Decision**: A system is flagged as "rooted" if final belief $b_{final} > 0.65$.

### 2.5 Rooting & Directed Dependency Analysis
* **Lagged Correlation**: Evaluates absolute cross-correlation $| \text{corr}(x_{i, t-\ell}, x_{j, t}) |$ across lags $\ell \in [1, \text{max\_lag}]$.
* **Transfer Entropy (TE)**: Computes discrete transfer entropy $TE_{i \to j}^{(k)} = \sum P(y_{t}, y_{t-1}^{(k)}, x_{t-\ell}) \log \frac{P(y_{t} \mid y_{t-1}^{(k)}, x_{t-\ell})}{P(y_{t} \mid y_{t-1}^{(k)})}$. Uses PyInform backend when available, defaulting to lagged correlation fallback.
* **Surrogate Significance Testing**: Each surrogate circularly shifts every series by an offset drawn uniformly over all configurations whose pairwise separation exceeds the maximum lag. This preserves each series' autocorrelation and destroys cross-dependence. Raw $p = \frac{\sum \mathbb{I}(S_{\text{surr}} \ge S_{\text{obs}}) + 1}{N_{\text{surrogates}} + 1}$; the default max-statistic adjustment compares each edge with the largest surrogate score over all edges and lags, which controls the family-wise error rate. The legacy permutation null is available but is invalid for autocorrelated series (family-wise error 0.85 at AR 0.9; see `DRR_BENCHMARKS.md`).

### 2.6 State-Space Diagnostics & Filtering
* **Linear State Space**: Implements Kalman filtering, Chandrasekhar recursions, Hamilton (RTS) backward smoothing, Koopman disturbance smoothing, and Durbin-Koopman / Carter-Kohn simulation smoothing.
* **Nonlinear Filtering**: Implements Herbst-Schorfheide tempered particle filtering for nonlinear observation equations.

### 2.7 Banking Adapters & Stress Testing
* **Regulatory Backend (`layer1_regulatory_backend.py`)**: Integrates Call Report (FFIEC 002 / FR Y-9C) bank balance sheet series (MDRM codes for cash, unencumbered liquid assets, core vs. wholesale deposits, AOCI).
* **Supervisory Alignment (`supervision.py`)**: Maps risk signals to Fed supervisory frameworks (CAMELS, LFI, RFI, ROCA, G-SIB surcharges).
* **Stress Module**: Simulates 200 bps interest rate shocks under Full AOCI vs. Opt-Out accounting rules to measure balance sheet capital and liquidity impacts.
* **Portfolio Mapping & Validation Infrastructure**: Provides `AutomatedSchemaMapper`, `MDRMCrosswalk`, and `validate_portfolio_dataset` to eliminate manual per-MDRM coding and tedious spreadsheet mapping across multi-form portfolio datasets (FR Y-9C, FFIEC 031/041/051, FFIEC 002, FR Y-15), automatically producing point-in-time registry patches and pre-ingestion audit reports.

---

## 3. Comparison with Established Financial-Stability Methods

| Conventional Method | DRR Approach | Novelty vs. Conventional Overlap |
| :--- | :--- | :--- |
| **Correlation Analysis** (Pearson/Spearman) | Time-delay embedding + directed lagged correlation / Transfer Entropy | **Novel combination**: Embeds scalar series into phase space before measuring non-reciprocal directed dependency with surrogate significance. |
| **Volatility / GARCH Models** | Rolling amplitude stability $A_{\text{stab}}$ + Spectral Concentration | **Conventional overlap**: GARCH explicitly models conditional variance dynamics; DRR treats volatility as a component of amplitude envelope stability. |
| **VAR / VECM** | State-Space Kalman / Chandrasekhar + Influence Network (DiGraph) | **Conventional overlap**: Linear state-space fitting uses standard Kalman filtering equations ported from Julia state-space packages. |
| **Granger Causality** | Transfer Entropy / Permutation-tested Lagged Correlation | **Novel combination**: Non-parametric TE captures non-linear informational flows that Granger causality (linear VAR F-test) misses. |
| **Spectral Analysis** (FFT/Welch) | Morlet Scalograms + Composite 4-component Resonance Depth | **Novel combination**: Combines standard spectral power with phase circular variance and Hilbert envelope stability into a single scalar $[0,1]$ metric. |
| **Early Warning Systems (EWS)** | QBism agent belief updating over composite depth metrics | **Novel combination**: Uses QBism subjective probability updating to aggregate resonance depth across multivariate dimensions into a system rooting score. |
| **Bank Liquidity Metrics** (LCR, NSFR) | "Dash for Cash" loop tracking (Treasury shock $\to$ Liquidity drain lag) | **Domain adaptation**: Applies lag-directed rooting to high-frequency supervisory balance-sheet series. |
| **Systemic Risk Indicators** (SRISK, CoVaR) | State-space impulse responses + Influence Graph centrality | **Conventional overlap**: SRISK models capital shortfall conditional on systemic drop; DRR models modal resonance and directional lead-lag. |

---

## 4. External comparative evidence

Specification tests show that the implementation follows its definitions. The
preregistered comparison against conventional detectors is the NOAA CPC QBO
study in `results/expected/qbo_structural_change_benchmark.json`. The reviewed
`claim_status` is `not_supported`: full DRR's holdout false-alarm rate exceeds
the frozen tolerance. That statement is about equatorial stratospheric wind.
It does not license a performance claim for another adapter. The calibration
study in `results/expected/rooting_calibration_study.json` measures the
operators on simulated systems with a known truth. It supports statements about
the tests' size and power, not about real-world usefulness.

## 5. Known Technical & Methodological Limitations

1. **Parameter Sensitivity**: Resonance Depth and Rooting edge detection are sensitive to the choice of embedding dimension $m$, delay $\tau$, window size $W$, and FFT segment length $n_{perseg}$.
2. **Discretization Artifacts in Transfer Entropy**: Discretizing continuous data into $N_{\text{bins}} = 10$ uniform bins creates boundary effects and information loss in high-volatility financial regimes.
3. **Linearity Bottleneck in Composite Depth**: Hilbert transform phase unwrapping assumes single-component narrow-band signals. Multi-component broad-band financial signals cause Hilbert phase distortion.
4. **Computational Complexity of Particle Filter**: Tempered particle filtering scales exponentially with state space dimension ($O(N_{\text{particles}} \cdot d^2)$), constraining real-time supervisory applications.
5. **Heuristic Thresholding**: The QBism belief threshold ($b > 0.65$) and the candidate-edge gate ($\mu + \sigma$ of the score matrix) are heuristics. The surrogate test that follows the gate is calibrated; the gate itself only removes candidates, so it can cost power but cannot add false alarms.
6. **Portfolio Scale & Manual Mapping Mitigation**: Broad portfolio deployment across hundreds of institutions and thousands of MDRM series historically required manual schema definition. The framework mitigates this via `AutomatedSchemaMapper` bulk expansion, `MDRMCrosswalk` cross-form translations, and pre-ingestion `validate_portfolio_dataset` auto-patching, though unmapped non-MDRM fields still require explicit synthetic (`SYN_`) namespace scoping to preserve strict auditability.
7. **Depth Is Not a Test**: Resonance depth has no null distribution of its own. AR(1) noise at 0.9 has a median depth of 0.45, higher than a tone at −9 dB (0.36). A Fourier surrogate that keeps the power spectrum lowers a tone's depth by only about 0.005. Compare depth against a surrogate or red-noise reference, not a fixed cutoff.
8. **Lead–Lag Under Memory**: When every channel is strongly autocorrelated, lagged correlation reports the target as leading its own source, and power falls (see the memory table in `DRR_BENCHMARKS.md`). DRR does not condition on each target's own past, as a Granger-style test would.
9. **Short Series With Long Memory**: When the series is short relative to its autocorrelation time (effective sample size near 10), the circular-shift test runs slightly above nominal, at about 0.06 for a nominal 0.05.
10. **Transfer-Entropy Surrogates**: The transfer-entropy backend is scored one surrogate at a time through `pyinform` and has not been through the calibration study. Its size is unmeasured.
