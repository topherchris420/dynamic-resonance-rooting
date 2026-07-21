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
| Smoothed state | Best retrospective estimate of a mode given the whole record | Two-sided estimate of a latent series using all observations |
| Smoothed shock `eps_t` | Structural disturbance that drove the trajectory | Retrospective innovation attributed to each period |
| Simulation-smoother draw | One plausible latent path consistent with the data | Posterior sample used to build credible bands |
| Particle-filter likelihood | Monte Carlo fit of a nonlinear model to the trajectory | Model-comparison diagnostic for nonlinear dynamics |
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

## Supervisory Analytics Language

| DRR / Policy Term | Supervisory Analytics Reading |
| --- | --- |
| Policy observable | Institution metric or peer-group aggregate |
| Data vintage | Reporting period, extract date, or supervisory data cut |
| Rooting edge | Lead-lag monitoring signal for analyst review |
| State-space stability | Stability of the fitted metric-transition diagnostic under the sample |
| Impulse response | Scenario path through fitted monitoring metrics, not a stress-test result |
| Tableau export | Tidy artifact for dashboards, review packets, and repeatable analyst workflows |
| Supervisory alignment metadata | Segment, risk-domain, data-lineage, and checklist context for review packets |
| Caveat | Boundary between diagnostic evidence and examination or policy judgment |

Fed alignment pack mapping:

| Pack Field | Supervisory Purpose |
| --- | --- |
| Institution preset | Keeps review context aligned to Fed supervision segments such as community bank, LFBO, large FBO, G-SIB, or FMU |
| Risk domain | Maps metrics to review language such as capital, liquidity, asset quality, operational resilience, or governance and controls |
| Data lineage | Records reporting form, MDRM code, FFIEC source, NIC identifier, data vintage, and review owner |
| Checklist | Forces materiality, proportionality, timeliness, peer-group fit, and source-data checks into the report |
| Reference basis | Links the output to official Fed supervision resources without implying Fed validation of DRR |

Supervisory outputs should be written as diagnostics for review: they can help
prioritize questions, compare institutions with peers, and identify places where
metric movement deserves source-data review. They should not be written as
ratings, findings, enforcement recommendations, MRAs, MRIAs, policy decisions,
or causal claims.
## Reviewer Checklist

- Are input transformations documented?
- Are sampling rate and window size explicit?
- Are uncertainty and caveats included in the report?
- Are synthetic or holdout validations available?
- Are domain conclusions separated from algorithmic diagnostics?
