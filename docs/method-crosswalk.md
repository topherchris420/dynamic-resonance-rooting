# DRR Method Crosswalk

This crosswalk maps DRR vocabulary to terms more familiar to physicists and
policy analysts. It is intentionally conservative: the same calculation can
support different interpretations, but interpretation belongs to the domain.

| DRR Concept | Physics Reading | Policy / State-Space Reading |
| --- | --- | --- |
| Resonance detection | Dominant mode or oscillatory frequency | Cyclical component or recurring observable pattern |
| Resonance depth | Persistence and coherence of a mode | Stability of a diagnostic pattern across the sample window |
| Phase coherence | Phase alignment across a signal window | Timing consistency in transformed observables |
| Rooting edge | Lagged coupling from one variable to another | Lead-lag diagnostic among observables |
| Effective lag | Delay between source and response | Number of periods by which one series leads another |
| Surrogate p-value | Evidence against random coupling under permutation | Diagnostic significance under a simple null model |
| Transition matrix `TTT` | Linearized state evolution | VAR-style fitted transition among observables |
| Measurement matrix `ZZ` | Sensor or observable loading | Mapping from latent state vector to observed series |
| Shock covariance `QQ` | Process-noise energy | Innovation covariance in the fitted transition system |
| Measurement covariance `EE` | Sensor noise | Observation or measurement uncertainty |
| Kalman innovations | Model residuals after prediction | One-step-ahead forecast errors in a state-space diagnostic |
| Log likelihood | Fit of filtered model to observed trajectory | Model-comparison diagnostic, not a policy objective |
| Spectral radius | Linear stability of transition dynamics | Stability check for fitted observable transition |
| Impulse response | Response to a controlled perturbation | Shock-propagation path for scenario design |

## Boundary Language

Recommended:

- "DRR detects diagnostic lead-lag structure."
- "The fitted transition system is stable under this sample and transformation."
- "Impulse responses describe propagation through the fitted diagnostic model."
- "Results should be re-run across windows, transformations, and data vintages."

Avoid:

- "DRR proves causality."
- "DRR predicts policy."
- "This edge means the source variable controls the target variable."
- "The report is sufficient for operational decision-making."

## Reviewer Checklist

- Are input transformations documented?
- Are sampling rate and window size explicit?
- Are uncertainty and caveats included in the report?
- Are synthetic or holdout validations available?
- Are domain conclusions separated from algorithmic diagnostics?
