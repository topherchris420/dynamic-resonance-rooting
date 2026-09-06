# Dynamic Resonance Geometry: Research Note

## Motivation

Univariate resonance depth and directed rooting answer different questions.
The proposed geometry retains pair, frequency, phase, collective-mode, and
temporal-topology information in one falsifiable framework.

## Existing DRR

The established package provides resonance detectors, depth decomposition,
surrogate-tested rooting, and conservative linear state-space analysis. The new
API is additive and does not alter those interfaces.

## Problem formulation

For multivariate observations, estimate a real-valued cross-resonance field,
derive a PSD band coupling, and follow both its modes and rooting adjacency
through causal trailing windows. Precise equations and edge cases are in
`docs/resonance-geometry.md`.

## Multivariate resonance geometry

The first implementation exposes Welch coherence, cross-spectral phase, joint
power weights, dominant frequency, effective rank, participation ratio, and
synchronization entropy. Synthetic phase-locked and separated-frequency tests
are mathematical controls, not evidence from markets.

## Dynamic rooting topology

Root Migration measures movement of normalized outgoing influence via bounded
Jensen--Shannon divergence. Topology drift separately measures angular change
of the complete weighted adjacency. Both inherit the conservative semantics of
the chosen rooting backend.

## Structural surprise

An expanding ridge VAR predicts each fingerprint from past fingerprints only.
A regularized historical innovation covariance yields a decomposable
Mahalanobis score. Prefix invariance is tested directly.

## Resonance-conditioned dynamics

Not implemented in this slice. A future implementation must cluster only
training fingerprints, shrink regime transitions to a global model, report raw
instability, and keep stabilized counterfactual matrices separate.

## Portfolio risk kernel

A normalized Gram matrix of asset structural loadings is PSD by construction.
Adding its volatility-scaled form to a PSD covariance preserves units and PSD.
This is a risk representation, not an expected-return model.

## Synthetic validation

Current invariant tests recover injected sinusoid frequency and phase, reject
broad coupling for frequency-separated signals, establish effective-rank
limits, order graph perturbations, detect a predeclared abrupt fingerprint
shift, and verify prefix and PSD properties.

## Financial validation

No empirical financial result is asserted. Qlib information lift, Riskfolio
covariance comparisons, OpenBB datasets, and VectorBT parameter landscapes
remain independent follow-up experiments.

## Limitations

Welch estimates are finite-sample objects; coherence bias depends on segment
count, windowing, and stationarity. Pairwise coherence does not establish a
multivariate causal model. Numerical PSD projection can mask estimator
inconsistency and is recorded in the algorithm definition. Surprise calibration
can be poor in short or nonstationary histories.

## Falsification criteria

Reject practical claims if phase randomization preserves cross-resonance,
prefix extension changes history, random features match DRR lift, the kernel
adds nothing beyond covariance shrinkage, or results occupy isolated parameter
points. Negative results are valid scientific outcomes.
