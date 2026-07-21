"""Tempered particle filter for nonlinear Dynamic Resonance Rooting models.

Linear-Gaussian resonance systems are handled exactly by the Kalman filter in
:mod:`drr_framework.state_space`. Many interesting resonance systems -- coupled
nonlinear oscillators, Lorenz/Rossler flows, stochastic volatility -- are *not*
linear, and their likelihood cannot be evaluated in closed form. This module
provides a dependency-free NumPy port of the tempered particle filter from the
New York Fed's ``StateSpaceRoutines.jl`` (Herbst & Schorfheide, 2019,
*Tempered Particle Filtering*, Journal of Econometrics).

The nonlinear state-space model is::

    s_t = Phi(s_{t-1}, eps_t),   eps_t ~ N(0, shock_cov)
    y_t = Psi(s_t) + u_t,        u_t  ~ N(0, measurement_cov)

``Phi`` and ``Psi`` are arbitrary (possibly nonlinear) callables. The Gaussian
measurement error keeps the tempering weights available in closed form, which is
what makes the filter efficient. Each period the filter runs three moves:

* **Correction** -- particles are weighted by a *tempered* measurement density
  ``p(y_t | s_t)^{phi}`` whose inverse-temperature ``phi`` is raised adaptively
  from 0 to 1 so that the particle inefficiency stays near a target ``r_star``.
* **Selection** -- particles are resampled (systematic or multinomial).
* **Mutation** -- a random-walk Metropolis-Hastings sweep rejuvenates the
  particles, with the proposal scale ``c`` adapted toward a target acceptance
  rate.

The per-period increments telescope to the standard bootstrap particle-filter
likelihood estimate, so on a linear-Gaussian system the returned log-likelihood
matches the exact Kalman value up to Monte Carlo error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .state_space import StateSpaceSystem, _as_2d_data, _safe_inverse_and_logdet, _symmetrize_psd

SeedLike = Union[int, np.random.Generator, None]

TransitionFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
MeasurementFn = Callable[[np.ndarray], np.ndarray]


def _as_generator(random_state: SeedLike) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable ``log(sum(exp(values)))``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    ceiling = float(np.max(finite))
    return ceiling + float(np.log(np.sum(np.exp(values - ceiling))))


@dataclass(frozen=True)
class NonlinearStateSpaceModel:
    """A nonlinear state-space model for the tempered particle filter.

    Parameters
    ----------
    transition:
        ``Phi(states_prev, shocks) -> states``. Must be vectorised over the
        leading (particle) axis: given ``(M, n_states)`` previous states and
        ``(M, n_shocks)`` shocks it returns ``(M, n_states)`` states.
    measurement:
        ``Psi(states) -> observables``. Vectorised: ``(M, n_states)`` in,
        ``(M, n_observables)`` out.
    shock_cov:
        ``(n_shocks, n_shocks)`` covariance of the Gaussian state shocks.
    measurement_cov:
        ``(n_observables, n_observables)`` covariance of the Gaussian
        measurement error.
    """

    transition: TransitionFn
    measurement: MeasurementFn
    shock_cov: np.ndarray
    measurement_cov: np.ndarray
    n_states: int
    n_shocks: int
    n_observables: int

    def __post_init__(self) -> None:
        shock_cov = _symmetrize_psd(
            np.atleast_2d(np.asarray(self.shock_cov, dtype=float)), jitter=0.0
        )
        measurement_cov = _symmetrize_psd(
            np.atleast_2d(np.asarray(self.measurement_cov, dtype=float)), jitter=0.0
        )
        if shock_cov.shape != (self.n_shocks, self.n_shocks):
            raise ValueError("shock_cov shape does not match n_shocks")
        if measurement_cov.shape != (self.n_observables, self.n_observables):
            raise ValueError("measurement_cov shape does not match n_observables")
        object.__setattr__(self, "shock_cov", shock_cov)
        object.__setattr__(self, "measurement_cov", measurement_cov)

    def sample_shocks(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(np.zeros(self.n_shocks), self.shock_cov, size=size)


def linear_gaussian_model(system: StateSpaceSystem) -> NonlinearStateSpaceModel:
    """Wrap a linear :class:`StateSpaceSystem` as a nonlinear model.

    Useful for validation -- running the tempered particle filter on the result
    should reproduce the exact Kalman log-likelihood up to Monte Carlo error.
    """
    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    ZZ = system["ZZ"]
    DD = system["DD"]

    def transition(states_prev: np.ndarray, shocks: np.ndarray) -> np.ndarray:
        return states_prev @ TTT.T + shocks @ RRR.T + CCC

    def measurement(states: np.ndarray) -> np.ndarray:
        return states @ ZZ.T + DD

    return NonlinearStateSpaceModel(
        transition=transition,
        measurement=measurement,
        shock_cov=system["QQ"],
        measurement_cov=system["EE"],
        n_states=system.n_states,
        n_shocks=system.n_shocks,
        n_observables=system.n_observables,
    )


@dataclass(frozen=True)
class ParticleFilterResult:
    """Output of the tempered particle filter.

    Attributes
    ----------
    log_likelihood:
        ``(T,)`` per-period conditional log-likelihood increments.
    filtered_states:
        ``(T, n_states)`` particle-mean filtered state ``E[s_t | y_{1:t}]``.
    filtered_state_std:
        ``(T, n_states)`` particle standard deviation of the filtered state.
    effective_sample_sizes:
        ``(T,)`` mean effective sample size across tempering stages.
    tempering_stages:
        ``(T,)`` number of tempering iterations used each period.
    acceptance_rates:
        ``(T,)`` mean Metropolis-Hastings acceptance rate each period.
    n_particles:
        Number of particles used.
    """

    log_likelihood: np.ndarray
    filtered_states: np.ndarray
    filtered_state_std: np.ndarray
    effective_sample_sizes: np.ndarray
    tempering_stages: np.ndarray
    acceptance_rates: np.ndarray
    n_particles: int

    @property
    def total_log_likelihood(self) -> float:
        return float(np.sum(self.log_likelihood))

    def summary(self) -> Dict[str, object]:
        return {
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": float(np.mean(self.log_likelihood)),
            "mean_effective_sample_size": float(np.mean(self.effective_sample_sizes)),
            "mean_tempering_stages": float(np.mean(self.tempering_stages)),
            "mean_acceptance_rate": float(np.mean(self.acceptance_rates)),
            "n_particles": self.n_particles,
        }


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _resample_indices(weights: np.ndarray, method: str, rng: np.random.Generator) -> np.ndarray:
    if method == "systematic":
        return _systematic_resample(weights, rng)
    if method == "multinomial":
        return rng.choice(weights.shape[0], size=weights.shape[0], p=weights)
    raise ValueError("resampling must be 'systematic' or 'multinomial'")


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return np.full(log_weights.shape[0], 1.0 / log_weights.shape[0])
    return weights / total


def _inefficiency(quad: np.ndarray, phi: float, phi_prev: float) -> float:
    weights = _normalized_weights(-0.5 * (phi - phi_prev) * quad)
    return float(quad.shape[0] * np.sum(weights**2))


def _next_phi(quad: np.ndarray, phi_prev: float, r_star: float) -> float:
    """Adaptively pick the next inverse-temperature to hit ``r_star``."""
    if _inefficiency(quad, 1.0, phi_prev) <= r_star:
        return 1.0
    low, high = phi_prev, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if _inefficiency(quad, mid, phi_prev) < r_star:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _adapt_scale(scale: float, acceptance_rate: float, target: float) -> float:
    """Multiplicatively nudge the RWMH scale toward the target acceptance rate."""
    adjustment = 0.95 + 0.10 / (1.0 + np.exp(-16.0 * (acceptance_rate - target)))
    return float(np.clip(scale * adjustment, 1e-3, 1e3))


def _measurement_quadratic(
    model: NonlinearStateSpaceModel,
    states: np.ndarray,
    observed: np.ndarray,
    y_obs: np.ndarray,
    measurement_inverse: np.ndarray,
) -> np.ndarray:
    """Mahalanobis measurement error ``e' E^{-1} e`` for each particle state."""
    errors = y_obs - model.measurement(states)[:, observed]
    return np.einsum("ij,jk,ik->i", errors, measurement_inverse, errors)


def tempered_particle_filter(
    model: NonlinearStateSpaceModel,
    observations: np.ndarray,
    n_particles: int = 1000,
    r_star: float = 2.0,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
    initial_particles: Optional[np.ndarray] = None,
    n_mutation: int = 1,
    c_init: float = 0.3,
    target_acceptance: float = 0.25,
    resampling: str = "systematic",
    n_presample: int = 0,
    adaptive: bool = True,
    random_state: SeedLike = None,
) -> ParticleFilterResult:
    """Estimate the log-likelihood of a nonlinear state-space model.

    Parameters
    ----------
    model:
        The :class:`NonlinearStateSpaceModel` to filter.
    observations:
        ``(T, n_observables)`` data. ``NaN`` marks missing observations.
    n_particles:
        Number of particles ``M``.
    r_star:
        Target inefficiency ratio (``> 1``) controlling the tempering schedule.
        Larger values take fewer, coarser tempering steps.
    initial_state, initial_covariance:
        Gaussian prior ``N(initial_state, initial_covariance)`` used to draw the
        initial particle cloud when ``initial_particles`` is not supplied.
    initial_particles:
        Optional ``(M, n_states)`` array of starting particles.
    n_mutation:
        Number of random-walk Metropolis-Hastings sweeps per tempering stage.
    c_init:
        Initial RWMH proposal scale.
    target_acceptance:
        Target MH acceptance rate used to adapt ``c``.
    resampling:
        ``"systematic"`` (default) or ``"multinomial"``.
    n_presample:
        Number of initial periods whose likelihood is excluded from the total.
    adaptive:
        When ``True`` (default) the tempering schedule is chosen adaptively.
        When ``False`` a single correction step (``phi = 1``) is used, giving a
        standard bootstrap particle filter.
    random_state:
        Seed or :class:`numpy.random.Generator`.

    Returns
    -------
    ParticleFilterResult
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2")
    if r_star <= 1.0:
        raise ValueError("r_star must be greater than 1")
    y = _as_2d_data(observations, name="observations", allow_nan=True)
    if y.shape[1] != model.n_observables:
        raise ValueError("observations column count must match model observables")

    rng = _as_generator(random_state)
    n_obs = y.shape[0]

    if initial_particles is not None:
        particles = np.asarray(initial_particles, dtype=float)
        if particles.shape != (n_particles, model.n_states):
            raise ValueError("initial_particles must have shape (n_particles, n_states)")
    else:
        mean = (
            np.asarray(initial_state, dtype=float).reshape(-1)
            if initial_state is not None
            else np.zeros(model.n_states)
        )
        covariance = (
            np.asarray(initial_covariance, dtype=float)
            if initial_covariance is not None
            else np.eye(model.n_states)
        )
        particles = rng.multivariate_normal(
            mean, _symmetrize_psd(covariance, jitter=0.0), size=n_particles
        )

    shock_inverse, _ = _safe_inverse_and_logdet(model.shock_cov)
    shock_factor = np.linalg.cholesky(_symmetrize_psd(model.shock_cov))

    log_likelihood = np.zeros(n_obs)
    filtered_states = np.zeros((n_obs, model.n_states))
    filtered_state_std = np.zeros((n_obs, model.n_states))
    effective_sample_sizes = np.zeros(n_obs)
    tempering_stages = np.zeros(n_obs, dtype=int)
    acceptance_rates = np.zeros(n_obs)

    scale = float(c_init)

    for t in range(n_obs):
        observed = np.isfinite(y[t])
        n_observed = int(np.sum(observed))

        shocks = model.sample_shocks(n_particles, rng)
        states = model.transition(particles, shocks)

        if n_observed == 0:
            particles = states
            filtered_states[t] = states.mean(axis=0)
            filtered_state_std[t] = states.std(axis=0)
            effective_sample_sizes[t] = float(n_particles)
            tempering_stages[t] = 0
            acceptance_rates[t] = np.nan
            continue

        y_obs = y[t, observed]
        measurement_cov_obs = model.measurement_cov[np.ix_(observed, observed)]
        measurement_inverse, measurement_logdet = _safe_inverse_and_logdet(measurement_cov_obs)
        log_norm_const = -0.5 * (n_observed * np.log(2 * np.pi) + measurement_logdet)

        ancestors = particles
        quad = _measurement_quadratic(model, states, observed, y_obs, measurement_inverse)

        phi_prev = 0.0
        stage_ess: List[float] = []
        stage_accept: List[float] = []
        n_stage = 0

        while phi_prev < 1.0:
            phi = 1.0 if not adaptive else _next_phi(quad, phi_prev, r_star)
            n_stage += 1

            if phi_prev == 0.0:
                log_weights = log_norm_const + 0.5 * n_observed * np.log(phi) - 0.5 * phi * quad
            else:
                log_weights = (
                    0.5 * n_observed * (np.log(phi) - np.log(phi_prev))
                    - 0.5 * (phi - phi_prev) * quad
                )

            log_likelihood[t] += _logsumexp(log_weights) - np.log(n_particles)

            weights = _normalized_weights(log_weights)
            stage_ess.append(float(1.0 / np.sum(weights**2)))

            index = _resample_indices(weights, resampling, rng)
            ancestors = ancestors[index]
            shocks = shocks[index]
            states = states[index]
            quad = quad[index]

            accepted_total = 0
            for _ in range(n_mutation):
                proposal = shocks + scale * (rng.standard_normal(shocks.shape) @ shock_factor.T)
                proposed_states = model.transition(ancestors, proposal)
                proposed_quad = _measurement_quadratic(
                    model, proposed_states, observed, y_obs, measurement_inverse
                )

                current_target = -0.5 * phi * quad - 0.5 * np.einsum(
                    "ij,jk,ik->i", shocks, shock_inverse, shocks
                )
                proposed_target = -0.5 * phi * proposed_quad - 0.5 * np.einsum(
                    "ij,jk,ik->i", proposal, shock_inverse, proposal
                )
                accept = np.log(rng.random(n_particles)) < (proposed_target - current_target)
                shocks[accept] = proposal[accept]
                states[accept] = proposed_states[accept]
                quad[accept] = proposed_quad[accept]
                accepted_total += int(np.sum(accept))

            stage_accept.append(accepted_total / (n_particles * n_mutation))
            phi_prev = phi

        particles = states
        filtered_states[t] = states.mean(axis=0)
        filtered_state_std[t] = states.std(axis=0)
        effective_sample_sizes[t] = float(np.mean(stage_ess))
        tempering_stages[t] = n_stage
        mean_accept = float(np.mean(stage_accept)) if stage_accept else np.nan
        acceptance_rates[t] = mean_accept
        if np.isfinite(mean_accept):
            scale = _adapt_scale(scale, mean_accept, target_acceptance)

    if n_presample > 0:
        log_likelihood[:n_presample] = 0.0

    return ParticleFilterResult(
        log_likelihood=log_likelihood,
        filtered_states=filtered_states,
        filtered_state_std=filtered_state_std,
        effective_sample_sizes=effective_sample_sizes,
        tempering_stages=tempering_stages,
        acceptance_rates=acceptance_rates,
        n_particles=n_particles,
    )
