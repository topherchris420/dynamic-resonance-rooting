"""Kalman and simulation smoothers for Dynamic Resonance Rooting.

This module is a dependency-free NumPy port of the smoothing routines in the
New York Fed's `StateSpaceRoutines.jl` package
(https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl). Where the Kalman filter
in :mod:`drr_framework.state_space` answers "what is the best estimate of the
latent resonance state *given data up to now*", the smoothers answer the
retrospective question "what was the state at time ``t`` *given the whole
sample*". Smoothing yields substantially cleaner latent trajectories and,
crucially for DRR's rooting analysis, an estimate of the structural shocks that
drove the system.

Four routines are provided, mirroring the Julia package:

``hamilton_smoother``
    The classical fixed-interval (Rauch–Tung–Striebel) smoother. Returns
    smoothed states, covariances, and shocks.
``koopman_smoother``
    Koopman's disturbance smoother. Computes the same smoothed moments through a
    single backward pass of the disturbance recursion; it is the numerically
    preferred route to smoothed shocks.
``carter_kohn_smoother``
    The Carter–Kohn (1994) forward-filter/backward-sample simulation smoother.
    Draws a *sample path* from the posterior over states rather than its mean.
``durbin_koopman_smoother``
    The Durbin–Koopman (2002) simulation smoother. Draws posterior state paths
    efficiently by smoothing the difference between the data and a simulated
    replica of the model.

The transition/measurement conventions match :mod:`drr_framework.state_space`::

    s_t = TTT s_{t-1} + RRR eps_t + CCC,   eps_t ~ N(0, QQ)
    y_t = ZZ s_t + DD + u_t,               u_t  ~ N(0, EE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from .state_space import (
    KalmanResult,
    StateSpaceSystem,
    _as_2d_data,
    _safe_inverse_and_logdet,
    _symmetrize_psd,
    kalman_filter,
)

SeedLike = Union[int, np.random.Generator, None]


def _as_generator(random_state: SeedLike) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


@dataclass(frozen=True)
class SmootherResult:
    """Output of a fixed-interval Kalman smoother.

    Attributes
    ----------
    smoothed_states:
        ``(T, n_states)`` array of ``E[s_t | y_{1:T}]``.
    smoothed_covariances:
        ``(T, n_states, n_states)`` array of ``Var[s_t | y_{1:T}]``.
    smoothed_shocks:
        ``(T, n_shocks)`` array of ``E[eps_t | y_{1:T}]`` -- the structural
        disturbances that best explain the observed resonance trajectory.
    smoothed_observables:
        ``(T, n_observables)`` array of the noise-free observables implied by
        the smoothed states, ``ZZ s_{t|T} + DD``.
    method:
        Name of the smoother that produced the result.
    """

    smoothed_states: np.ndarray
    smoothed_covariances: np.ndarray
    smoothed_shocks: np.ndarray
    smoothed_observables: np.ndarray
    method: str

    def summary(self) -> Dict[str, object]:
        """Compact, JSON-friendly summary of the smoothed trajectory."""
        shock_energy = np.nansum(self.smoothed_shocks**2, axis=0)
        uncertainty = np.trace(self.smoothed_covariances, axis1=1, axis2=2)
        return {
            "method": self.method,
            "smoothed_state_norm": float(np.linalg.norm(self.smoothed_states[-1])),
            "mean_smoothed_uncertainty": float(np.mean(uncertainty)),
            "final_smoothed_uncertainty": float(uncertainty[-1]),
            "shock_energy": [float(value) for value in np.atleast_1d(shock_energy)],
            "state_reduction_ratio": _state_reduction_ratio(self),
        }


def _state_reduction_ratio(result: SmootherResult) -> float:
    """Fraction of predictive uncertainty removed by smoothing (0..1)."""
    trace = np.trace(result.smoothed_covariances, axis1=1, axis2=2)
    reference = float(np.mean(trace)) if trace.size else 0.0
    return float(np.clip(1.0 - reference / (reference + 1.0), 0.0, 1.0))


def _validate_observations(system: StateSpaceSystem, observations: np.ndarray) -> np.ndarray:
    y = _as_2d_data(observations, name="observations", allow_nan=True)
    if y.shape[1] != system.n_observables:
        raise ValueError("observations column count must match system observables")
    return y


def _project_shocks(system: StateSpaceSystem, smoothed_states: np.ndarray) -> np.ndarray:
    """Recover ``E[eps_t | y_{1:T}]`` from smoothed states.

    Taking conditional expectations of the transition equation gives the exact
    identity ``RRR E[eps_t|Y] = s_{t|T} - TTT s_{t-1|T} - CCC``. The minimum
    variance solution is ``E[eps_t|Y] = QQ RRR' (RRR QQ RRR')^+ increment``,
    which recovers the increment exactly whenever ``QQ`` is full rank.
    """
    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    QQ = system["QQ"]

    process_covariance = _symmetrize_psd(RRR @ QQ @ RRR.T)
    pseudo_inverse = np.linalg.pinv(process_covariance)
    shock_map = QQ @ RRR.T @ pseudo_inverse

    n_obs = smoothed_states.shape[0]
    shocks = np.zeros((n_obs, system.n_shocks))
    previous = np.zeros(system.n_states)
    for t in range(n_obs):
        increment = smoothed_states[t] - (TTT @ previous + CCC)
        shocks[t] = shock_map @ increment
        previous = smoothed_states[t]
    return shocks


def _smoothed_observables(system: StateSpaceSystem, smoothed_states: np.ndarray) -> np.ndarray:
    return smoothed_states @ system["ZZ"].T + system["DD"]


def _rts_backward_pass(
    system: StateSpaceSystem, kalman: KalmanResult
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward recursion for states and covariances."""
    TTT = system["TTT"]
    n_obs = kalman.filtered_state.shape[0]
    n_states = system.n_states

    smoothed_states = np.zeros((n_obs, n_states))
    smoothed_covariances = np.zeros((n_obs, n_states, n_states))
    smoothed_states[-1] = kalman.filtered_state[-1]
    smoothed_covariances[-1] = kalman.filtered_covariance[-1]

    for t in range(n_obs - 2, -1, -1):
        predicted_covariance_next = kalman.predicted_covariance[t + 1]
        inverse_predicted, _ = _safe_inverse_and_logdet(predicted_covariance_next)
        gain = kalman.filtered_covariance[t] @ TTT.T @ inverse_predicted
        state_gap = smoothed_states[t + 1] - kalman.predicted_state[t + 1]
        smoothed_states[t] = kalman.filtered_state[t] + gain @ state_gap
        covariance_gap = smoothed_covariances[t + 1] - predicted_covariance_next
        smoothed_covariances[t] = _symmetrize_psd(
            kalman.filtered_covariance[t] + gain @ covariance_gap @ gain.T
        )
    return smoothed_states, smoothed_covariances


def _disturbance_backward_pass(
    system: StateSpaceSystem, observations: np.ndarray, kalman: KalmanResult
) -> tuple[np.ndarray, np.ndarray]:
    """Koopman disturbance smoother backward recursion (states, covariances).

    Runs the ``r_t``/``N_t`` recursion of Durbin & Koopman (2012, sec. 4.5),
    adapted to the DRR contemporaneous timing convention.
    """
    TTT = system["TTT"]
    ZZ = system["ZZ"]
    EE = system["EE"]
    n_obs = observations.shape[0]
    n_states = system.n_states

    smoothed_states = np.zeros((n_obs, n_states))
    smoothed_covariances = np.zeros((n_obs, n_states, n_states))

    r = np.zeros(n_states)
    N = np.zeros((n_states, n_states))
    for t in range(n_obs - 1, -1, -1):
        predicted_covariance = kalman.predicted_covariance[t]
        observed = np.isfinite(observations[t])
        if np.any(observed):
            Z_obs = ZZ[observed]
            E_obs = EE[np.ix_(observed, observed)]
            innovation = kalman.innovations[t, observed]
            innovation_covariance = _symmetrize_psd(Z_obs @ predicted_covariance @ Z_obs.T + E_obs)
            inverse_innovation, _ = _safe_inverse_and_logdet(innovation_covariance)
            gain = TTT @ predicted_covariance @ Z_obs.T @ inverse_innovation
            L = TTT - gain @ Z_obs
            weighted = Z_obs.T @ inverse_innovation
            r = weighted @ innovation + L.T @ r
            N = _symmetrize_psd(weighted @ Z_obs + L.T @ N @ L)
        else:
            r = TTT.T @ r
            N = _symmetrize_psd(TTT.T @ N @ TTT)

        smoothed_states[t] = kalman.predicted_state[t] + predicted_covariance @ r
        smoothed_covariances[t] = _symmetrize_psd(
            predicted_covariance - predicted_covariance @ N @ predicted_covariance
        )
    return smoothed_states, smoothed_covariances


def _build_result(
    system: StateSpaceSystem,
    smoothed_states: np.ndarray,
    smoothed_covariances: np.ndarray,
    method: str,
) -> SmootherResult:
    return SmootherResult(
        smoothed_states=smoothed_states,
        smoothed_covariances=smoothed_covariances,
        smoothed_shocks=_project_shocks(system, smoothed_states),
        smoothed_observables=_smoothed_observables(system, smoothed_states),
        method=method,
    )


def hamilton_smoother(
    system: StateSpaceSystem,
    observations: np.ndarray,
    kalman: Optional[KalmanResult] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> SmootherResult:
    """Fixed-interval Rauch-Tung-Striebel smoother (Hamilton's formulation).

    Parameters
    ----------
    system:
        The fitted DRR state-space system.
    observations:
        ``(T, n_observables)`` array. ``NaN`` marks missing observations.
    kalman:
        A pre-computed :class:`~drr_framework.state_space.KalmanResult`. When
        omitted the filter is run internally.
    initial_state, initial_covariance:
        Prior for the pre-sample state, forwarded to the filter when it is run.
    """
    y = _validate_observations(system, observations)
    if kalman is None:
        kalman = kalman_filter(system, y, initial_state, initial_covariance)
    smoothed_states, smoothed_covariances = _rts_backward_pass(system, kalman)
    return _build_result(system, smoothed_states, smoothed_covariances, "hamilton")


def koopman_smoother(
    system: StateSpaceSystem,
    observations: np.ndarray,
    kalman: Optional[KalmanResult] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> SmootherResult:
    """Koopman disturbance smoother.

    Numerically equivalent to :func:`hamilton_smoother` for the smoothed states
    and covariances, but derived from the disturbance recursion, which is the
    natural route to smoothed structural shocks.
    """
    y = _validate_observations(system, observations)
    if kalman is None:
        kalman = kalman_filter(system, y, initial_state, initial_covariance)
    smoothed_states, smoothed_covariances = _disturbance_backward_pass(system, y, kalman)
    return _build_result(system, smoothed_states, smoothed_covariances, "koopman")


def kalman_smoother(
    system: StateSpaceSystem,
    observations: np.ndarray,
    method: str = "koopman",
    kalman: Optional[KalmanResult] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> SmootherResult:
    """Dispatch to a fixed-interval smoother by ``method``.

    ``method`` is ``"koopman"`` (default, disturbance smoother) or ``"hamilton"``
    (RTS smoother). Both return identical smoothed moments for a linear-Gaussian
    system; the choice is a matter of numerical route.
    """
    method = method.lower()
    if method == "koopman":
        return koopman_smoother(system, observations, kalman, initial_state, initial_covariance)
    if method == "hamilton":
        return hamilton_smoother(system, observations, kalman, initial_state, initial_covariance)
    raise ValueError("method must be 'koopman' or 'hamilton'")


@dataclass(frozen=True)
class SimulationSmootherResult:
    """Posterior draws of states and shocks from a simulation smoother.

    Attributes
    ----------
    state_draws:
        ``(n_draws, T, n_states)`` posterior sample paths of the latent state.
    shock_draws:
        ``(n_draws, T, n_shocks)`` posterior sample paths of the shocks.
    method:
        Name of the simulation smoother.
    """

    state_draws: np.ndarray
    shock_draws: np.ndarray
    method: str

    @property
    def n_draws(self) -> int:
        return self.state_draws.shape[0]

    def posterior_mean(self) -> np.ndarray:
        """Monte Carlo estimate of the smoothed state mean, ``(T, n_states)``."""
        return self.state_draws.mean(axis=0)

    def posterior_band(self, lower: float = 5.0, upper: float = 95.0) -> Dict[str, np.ndarray]:
        """Percentile credible band across draws for each state series."""
        return {
            "lower": np.percentile(self.state_draws, lower, axis=0),
            "upper": np.percentile(self.state_draws, upper, axis=0),
        }


def _draw_multivariate_normal(
    mean: np.ndarray, covariance: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw from ``N(mean, covariance)`` via an SVD factor (robust to PSD)."""
    covariance = _symmetrize_psd(covariance, jitter=0.0)
    u, s, _ = np.linalg.svd(covariance)
    factor = u * np.sqrt(np.maximum(s, 0.0))
    return mean + factor @ rng.standard_normal(mean.shape[0])


def _simulate_forward(
    system: StateSpaceSystem,
    n_obs: int,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    missing_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a replica ``(s+, eps+, y+)`` from the model (Durbin-Koopman)."""
    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    ZZ = system["ZZ"]
    DD = system["DD"]
    QQ = system["QQ"]
    EE = system["EE"]

    states = np.zeros((n_obs, system.n_states))
    shocks = np.zeros((n_obs, system.n_shocks))
    observables = np.zeros((n_obs, system.n_observables))

    previous = _draw_multivariate_normal(initial_state, initial_covariance, rng)
    for t in range(n_obs):
        shock = _draw_multivariate_normal(np.zeros(system.n_shocks), QQ, rng)
        state = TTT @ previous + RRR @ shock + CCC
        measurement_error = _draw_multivariate_normal(np.zeros(system.n_observables), EE, rng)
        observable = ZZ @ state + DD + measurement_error

        shocks[t] = shock
        states[t] = state
        observables[t] = observable
        previous = state

    observables[missing_mask] = np.nan
    return states, shocks, observables


def durbin_koopman_smoother(
    system: StateSpaceSystem,
    observations: np.ndarray,
    n_draws: int = 1,
    random_state: SeedLike = None,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> SimulationSmootherResult:
    """Durbin-Koopman (2002) simulation smoother.

    Each draw simulates a replica ``(s+, y+)`` of the model, forms the artificial
    data ``y* = y - y+``, smooths it through the (mean-zero) Koopman recursion,
    and returns ``s+ + s*`` -- an exact draw from ``p(s_{1:T} | y_{1:T})``.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    y = _validate_observations(system, observations)
    rng = _as_generator(random_state)

    prior_state = (
        np.asarray(initial_state, dtype=float).reshape(-1)
        if initial_state is not None
        else np.zeros(system.n_states)
    )
    prior_covariance = (
        np.asarray(initial_covariance, dtype=float)
        if initial_covariance is not None
        else np.eye(system.n_states)
    )

    n_obs = y.shape[0]
    missing_mask = ~np.isfinite(y)

    state_draws = np.zeros((n_draws, n_obs, system.n_states))
    shock_draws = np.zeros((n_draws, n_obs, system.n_shocks))

    for draw in range(n_draws):
        simulated_states, simulated_shocks, simulated_observables = _simulate_forward(
            system, n_obs, prior_state, prior_covariance, missing_mask, rng
        )
        artificial = y - simulated_observables
        # Mean-zero smoother: the prior mean cancels in y - y+, so smooth with a
        # zero initial state to recover the linear (mean-adjusting) correction.
        smoothed = koopman_smoother(
            system,
            artificial,
            initial_state=np.zeros(system.n_states),
            initial_covariance=prior_covariance,
        )
        state_draws[draw] = simulated_states + smoothed.smoothed_states
        shock_draws[draw] = simulated_shocks + smoothed.smoothed_shocks

    return SimulationSmootherResult(
        state_draws=state_draws,
        shock_draws=shock_draws,
        method="durbin_koopman",
    )


def carter_kohn_smoother(
    system: StateSpaceSystem,
    observations: np.ndarray,
    n_draws: int = 1,
    random_state: SeedLike = None,
    kalman: Optional[KalmanResult] = None,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> SimulationSmootherResult:
    """Carter-Kohn (1994) forward-filter/backward-sample simulation smoother.

    Draws a state path by sampling ``s_T`` from its filtered distribution and
    then sampling backwards through the conditional distributions
    ``p(s_t | s_{t+1}, y_{1:t})``. Shock draws are recovered from the sampled
    state increments.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    y = _validate_observations(system, observations)
    if kalman is None:
        kalman = kalman_filter(system, y, initial_state, initial_covariance)
    rng = _as_generator(random_state)

    TTT = system["TTT"]
    CCC = system["CCC"]
    n_obs = y.shape[0]

    state_draws = np.zeros((n_draws, n_obs, system.n_states))
    shock_draws = np.zeros((n_draws, n_obs, system.n_shocks))

    for draw in range(n_draws):
        states = np.zeros((n_obs, system.n_states))
        states[-1] = _draw_multivariate_normal(
            kalman.filtered_state[-1], kalman.filtered_covariance[-1], rng
        )
        for t in range(n_obs - 2, -1, -1):
            predicted_covariance_next = kalman.predicted_covariance[t + 1]
            inverse_predicted, _ = _safe_inverse_and_logdet(predicted_covariance_next)
            gain = kalman.filtered_covariance[t] @ TTT.T @ inverse_predicted
            conditional_mean = kalman.filtered_state[t] + gain @ (
                states[t + 1] - kalman.predicted_state[t + 1]
            )
            conditional_covariance = _symmetrize_psd(
                kalman.filtered_covariance[t] - gain @ TTT @ kalman.filtered_covariance[t]
            )
            states[t] = _draw_multivariate_normal(conditional_mean, conditional_covariance, rng)

        state_draws[draw] = states
        shock_draws[draw] = _project_shocks(system, states)

    return SimulationSmootherResult(
        state_draws=state_draws,
        shock_draws=shock_draws,
        method="carter_kohn",
    )
