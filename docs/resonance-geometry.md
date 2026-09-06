# Dynamic Resonance Geometry

## Status and architecture/novelty audit

This is a **new DRR extension**, not a claim of academic novelty. The audit was
performed before implementation. Existing DRR computes univariate FFT, Welch,
wavelet, and Markov resonance; component depth; directed lagged-association or
transfer-entropy scores with surrogate inference; and a fixed VAR(1)-style
state-space representation. It did not retain a frequency-resolved pairwise
field, collective resonance eigenmodes, temporal graph distances, a causal
fingerprint innovation score, or a structural Gram kernel.

The external projects remain boundary systems. Qlib supplies data/model
workflows, Riskfolio-Lib supplies portfolio optimizers, OpenBB supplies data,
and VectorBT supplies parameterized backtest validation through the existing
adapters. None defines the mathematics in this module, and no external source
code is vendored or copied. Ordinary coherence, eigenspectrum entropy,
Jensen--Shannon divergence, ridge regression, Mahalanobis distance, and Gram
kernels are established concepts. DRR's contribution here is their explicit,
typed composition around resonance and rooting state. The initial online
literature/API search was unavailable (HTTP 401 in the execution environment),
so stronger novelty language is deliberately rejected pending a reproducible
literature review.

Relevant conceptual references include Welch's spectral estimator (IEEE,
1967), magnitude-squared coherence as documented by
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html),
effective rank (Roy and Vetterli, 2007), and Jensen--Shannon divergence (Lin,
1991). These references motivate definitions; the implementation was derived
independently.

## Multiscale Cross-Resonance Tensor

For Welch cross-spectrum \(S_{ij}(f)\), the tensor stores

\[
C_{ij}(f)=\frac{|S_{ij}(f)|^2}{S_{ii}(f)S_{jj}(f)},\qquad
\phi_{ij}(f)=\arg S_{ij}(f),\qquad
w_{ij}(f)=\frac{\sqrt{S_{ii}(f)S_{jj}(f)}}
{\sum_g\sqrt{S_{ii}(g)S_{jj}(g)}}.
\]

Unlike one correlation coefficient, this representation distinguishes
frequency, phase, and the frequency bins where both components carry energy.
Arrays use `(frequency, source, target)` ordering. Band coupling is a weighted
average of coherence matrices, projected only to remove negative numerical
eigenvalues and normalized to unit trace. Full tensor storage is
\(O(FN^2)\); callers can immediately aggregate selected bands.

## Collective resonance modes

For nonnegative coupling eigenvalues \(\lambda_k\), let
\(p_k=\lambda_k/\sum_j\lambda_j\). DRR reports dominant share \(p_1\),
entropy \(H=-\sum p_k\log p_k\), effective rank \(\exp(H)\), and participation
ratio \(1/\sum p_k^2\). Effective rank near one means concentration in one
mode; a larger value means distributed modes. Neither implies benefit or harm.

## Dynamic rooting topology

Adjacency entries are nonnegative directed-dependence scores. Their semantics
remain those of the effective rooting backend: lagged association is never
causal, and transfer entropy is directed information dependence absent an
identification design. The root distribution is normalized outgoing strength,
\(q_i=\sum_j A_{ij}/\sum_{kl}A_{kl}\). Root Migration is

\[
\operatorname{RMI}_t=\operatorname{JS}_2(q_t,q_{t-1})\in[0,1].
\]

Topology drift compares the complete vectorized adjacency geometry with
\(1-\langle A_t,A_{t-1}\rangle/(\|A_t\|_F\|A_{t-1}\|_F)\). It is scale
invariant and bounded on nonnegative graphs; two empty graphs have distance
zero and one empty graph versus a nonempty graph has distance one.

## Structural Resonance Surprise

At every time \(t\), DRR fits a ridge VAR(1) only to pairs ending before
\(t\), predicts \(z_t\), estimates covariance from those historical residuals,
and reports

\[
S_t=\sqrt{e_t^\top(\widehat\Sigma_{e,t}+\epsilon sI)^{-1}e_t}.
\]

Signed feature contributions are \(e_i[\Sigma^{-1}e]_i\) and may be aggregated
into caller-declared depth, cross-resonance, rooting, topology, and state-space
groups. Individual signed contributions need not be positive, but their sum is
the nonnegative squared score. This measures unexpected structural evolution,
not crash probability or causality.

## Resonance Risk Kernel

Rows of structural loading matrix \(F\) are normalized and
\(K=FF^\top\). Thus \(K\) is symmetric PSD and dimensionless. Given a PSD
return covariance \(\Sigma\), volatility matrix
\(D=\operatorname{diag}(\sqrt{\Sigma_{ii}})\), and \(\lambda\ge0\),

\[
\Sigma_{DRR}=\Sigma+\lambda D K D
\]

has return-variance units and remains PSD. The kernel encodes shared structural
exposure, not expected return. Lambda must be selected on training/validation
data and compared with ordinary shrinkage under matched optimization inputs.

## Causality, uncertainty, and falsification

Rolling geometry uses trailing windows and is prefix invariant. This first
slice does not yet estimate confidence intervals, conditioned switching state
space, shock maps, or empirical financial lift. Claims for Qlib incrementality,
portfolio improvement, and practical application therefore remain completely
unproven. Required follow-up controls are independently phase-randomized,
circular-shifted, block-shuffled, and covariance-matched systems. Financial
reports must publish negative outcomes and distinguish exploratory hypotheses
from predeclared confirmation.
