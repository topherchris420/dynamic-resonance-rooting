"""Tests for the tempered particle filter ported from StateSpaceRoutines.jl."""

import numpy as np
import pytest

from drr_framework.particle_filter import (
    NonlinearStateSpaceModel,
    ParticleFilterResult,
    linear_gaussian_model,
    tempered_particle_filter,
)
from drr_framework.state_space import Measurement, StateSpaceSystem, Transition, kalman_filter


def _linear_system_and_data(random_state=7, n=120):
    rng = np.random.default_rng(random_state)
    TTT = np.array([[0.6, 0.2], [0.0, 0.5]])
    QQ = np.array([[0.20, 0.05], [0.05, 0.15]])
    EE = np.eye(2) * 0.30
    system = StateSpaceSystem(
        transition=Transition(TTT, np.eye(2), np.zeros(2)),
        measurement=Measurement(np.eye(2), np.zeros(2), QQ, EE),
        state_names=["a", "b"],
        observable_names=["a", "b"],
        shock_names=["e0", "e1"],
    )
    state = np.zeros(2)
    chol_q = np.linalg.cholesky(QQ)
    chol_e = np.linalg.cholesky(EE)
    observations = np.zeros((n, 2))
    for t in range(n):
        state = TTT @ state + chol_q @ rng.standard_normal(2)
        observations[t] = state + chol_e @ rng.standard_normal(2)
    return system, observations


def test_tempered_particle_filter_matches_kalman_on_linear_gaussian():
    system, observations = _linear_system_and_data()
    kalman = kalman_filter(
        system, observations, initial_state=np.zeros(2), initial_covariance=np.eye(2)
    )
    model = linear_gaussian_model(system)

    result = tempered_particle_filter(
        model,
        observations,
        n_particles=5000,
        random_state=0,
        initial_state=np.zeros(2),
        initial_covariance=np.eye(2),
    )

    assert isinstance(result, ParticleFilterResult)
    # Particle-filter likelihood is a Monte Carlo estimate of the exact value.
    assert abs(result.total_log_likelihood - kalman.total_log_likelihood) < 4.0


def test_particle_filter_is_reproducible_with_seed():
    system, observations = _linear_system_and_data()
    model = linear_gaussian_model(system)

    first = tempered_particle_filter(model, observations, n_particles=500, random_state=42)
    second = tempered_particle_filter(model, observations, n_particles=500, random_state=42)

    assert first.total_log_likelihood == second.total_log_likelihood
    assert np.array_equal(first.filtered_states, second.filtered_states)


def test_particle_filter_result_shapes_and_summary():
    system, observations = _linear_system_and_data(n=60)
    model = linear_gaussian_model(system)
    result = tempered_particle_filter(model, observations, n_particles=800, random_state=1)

    assert result.log_likelihood.shape == (60,)
    assert result.filtered_states.shape == (60, 2)
    assert result.filtered_state_std.shape == (60, 2)
    assert np.all(result.effective_sample_sizes <= result.n_particles + 1e-9)
    assert np.all(result.tempering_stages >= 1)

    summary = result.summary()
    assert summary["n_particles"] == 800
    assert set(summary) >= {"total_log_likelihood", "mean_effective_sample_size"}


def test_particle_filter_on_nonlinear_model_runs():
    """A nonlinear (quadratic-measurement) oscillator should filter cleanly."""
    rng = np.random.default_rng(3)
    omega = 0.3
    TTT = np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]) * 0.99
    shock_cov = np.eye(2) * 0.02
    measurement_cov = np.array([[0.05]])

    def transition(states_prev, shocks):
        return states_prev @ TTT.T + shocks

    def measurement(states):
        # Nonlinear observation: squared radius of the oscillator.
        return np.sum(states**2, axis=1, keepdims=True)

    model = NonlinearStateSpaceModel(
        transition=transition,
        measurement=measurement,
        shock_cov=shock_cov,
        measurement_cov=measurement_cov,
        n_states=2,
        n_shocks=2,
        n_observables=1,
    )

    state = np.array([1.0, 0.0])
    chol_q = np.linalg.cholesky(shock_cov)
    observations = np.zeros((80, 1))
    for t in range(80):
        state = TTT @ state + chol_q @ rng.standard_normal(2)
        observations[t] = state @ state + np.sqrt(0.05) * rng.standard_normal()

    result = tempered_particle_filter(
        model,
        observations,
        n_particles=2000,
        random_state=0,
        initial_state=np.array([1.0, 0.0]),
        initial_covariance=np.eye(2) * 0.1,
    )

    assert np.isfinite(result.total_log_likelihood)
    assert result.filtered_states.shape == (80, 2)


def test_particle_filter_handles_missing_observations():
    system, observations = _linear_system_and_data(n=50)
    observations[10:15, :] = np.nan
    model = linear_gaussian_model(system)

    result = tempered_particle_filter(model, observations, n_particles=500, random_state=0)

    # Missing periods contribute zero to the likelihood.
    assert np.allclose(result.log_likelihood[10:15], 0.0)
    assert np.isfinite(result.total_log_likelihood)


def test_bootstrap_mode_runs_without_tempering():
    system, observations = _linear_system_and_data(n=40)
    model = linear_gaussian_model(system)

    result = tempered_particle_filter(
        model, observations, n_particles=1000, adaptive=False, random_state=0
    )

    assert np.all(result.tempering_stages == 1)


def test_particle_filter_validates_inputs():
    system, observations = _linear_system_and_data(n=20)
    model = linear_gaussian_model(system)

    with pytest.raises(ValueError):
        tempered_particle_filter(model, observations, n_particles=1)
    with pytest.raises(ValueError):
        tempered_particle_filter(model, observations, r_star=1.0)
    with pytest.raises(ValueError):
        tempered_particle_filter(model, observations[:, :1])
