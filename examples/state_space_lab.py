"""State-Space Lab: filtering, smoothing, and particle filtering in DRR.

This example showcases the state-space routines DRR ports from the New York
Fed's ``StateSpaceRoutines.jl``:

1. Fit a linear-Gaussian state-space model to a resonance trajectory.
2. Filter it with the Kalman filter (real-time state estimates).
3. Smooth it with the Koopman disturbance smoother (retrospective states and
   the structural shocks that drove the system).
4. Draw posterior state paths with the Durbin-Koopman simulation smoother to
   get credible bands.
5. Evaluate the likelihood of a *nonlinear* variant with the tempered
   particle filter.

Run with::

    python examples/state_space_lab.py
"""

from __future__ import annotations

import numpy as np

from drr_framework import (
    NonlinearStateSpaceModel,
    chandrasekhar_recursion,
    durbin_koopman_smoother,
    fit_resonance_state_space,
    kalman_filter,
    kalman_smoother,
    stationary_initialization,
    tempered_particle_filter,
)


def generate_resonant_trajectory(n: int = 400, seed: int = 7) -> np.ndarray:
    """A damped two-mode oscillator observed with measurement noise."""
    rng = np.random.default_rng(seed)
    omega = 0.35
    rotation = np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]])
    transition = 0.98 * rotation
    state = np.array([1.0, 0.0])
    observations = np.zeros((n, 2))
    for t in range(n):
        state = transition @ state + rng.normal(scale=0.05, size=2)
        observations[t] = state + rng.normal(scale=0.15, size=2)
    return observations


def main() -> None:
    observations = generate_resonant_trajectory()

    # 1-2. Fit and filter.
    system = fit_resonance_state_space(observations, state_names=["mode_a", "mode_b"])
    filtered = kalman_filter(system, observations)
    print("=== Fit & Filter ===")
    print(f"spectral radius      : {system.diagnostics()['spectral_radius']:.3f}")
    print(f"filter log-likelihood: {filtered.total_log_likelihood:.2f}")

    # 3. Smooth (retrospective states + structural shocks).
    smoothed = kalman_smoother(system, observations, method="koopman", kalman=filtered)
    print("\n=== Koopman Smoother ===")
    for key, value in smoothed.summary().items():
        print(f"{key:24s}: {value}")

    # 4. Posterior draws -> credible band on the first latent mode.
    draws = durbin_koopman_smoother(system, observations, n_draws=500, random_state=1)
    band = draws.posterior_band(lower=5.0, upper=95.0)
    width = float(np.mean(band["upper"][:, 0] - band["lower"][:, 0]))
    print("\n=== Durbin-Koopman Simulation Smoother ===")
    print(f"posterior draws        : {draws.n_draws}")
    print(f"mean 90% band width (a): {width:.3f}")

    # 5. Fast exact likelihood via the Chandrasekhar recursions.
    stationary_mean, stationary_cov = stationary_initialization(system)
    chandrasekhar = chandrasekhar_recursion(system, observations)
    kalman_stationary = kalman_filter(
        system, observations, initial_state=stationary_mean, initial_covariance=stationary_cov
    )
    print("\n=== Chandrasekhar Recursion ===")
    print(f"chandrasekhar log-lh : {chandrasekhar.total_log_likelihood:.4f}")
    print(f"kalman (stationary)  : {kalman_stationary.total_log_likelihood:.4f}")

    # 6. Tempered particle filter on a nonlinear (energy) observation.
    def transition(states_prev: np.ndarray, shocks: np.ndarray) -> np.ndarray:
        return states_prev @ system["TTT"].T + shocks

    def measurement(states: np.ndarray) -> np.ndarray:
        # Observe the instantaneous energy of the oscillator (nonlinear).
        return np.sum(states**2, axis=1, keepdims=True)

    energy = np.sum(observations**2, axis=1, keepdims=True)
    nonlinear_model = NonlinearStateSpaceModel(
        transition=transition,
        measurement=measurement,
        shock_cov=system["QQ"],
        measurement_cov=np.array([[0.1]]),
        n_states=2,
        n_shocks=2,
        n_observables=1,
    )
    particle = tempered_particle_filter(
        nonlinear_model,
        energy,
        n_particles=2000,
        random_state=0,
        initial_state=np.zeros(2),
        initial_covariance=np.eye(2),
    )
    print("\n=== Tempered Particle Filter (nonlinear observation) ===")
    for key, value in particle.summary().items():
        print(f"{key:26s}: {value}")


if __name__ == "__main__":
    main()
