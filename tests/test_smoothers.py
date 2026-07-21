"""Tests for the Kalman and simulation smoothers ported from StateSpaceRoutines.jl."""

import numpy as np
import pytest

from drr_framework.smoothers import (
    SimulationSmootherResult,
    SmootherResult,
    carter_kohn_smoother,
    durbin_koopman_smoother,
    hamilton_smoother,
    kalman_smoother,
    koopman_smoother,
)
from drr_framework.state_space import (
    Measurement,
    StateSpaceSystem,
    Transition,
    fit_resonance_state_space,
    kalman_filter,
)


def _stable_var(random_state=11, n=400):
    rng = np.random.default_rng(random_state)
    transition = np.array([[0.72, 0.12], [0.03, 0.48]])
    data = np.zeros((n, 2))
    shocks = rng.normal(scale=0.08, size=data.shape)
    for idx in range(1, len(data)):
        data[idx] = transition @ data[idx - 1] + shocks[idx]
    return data


def _noisy_system(random_state=5, n=250):
    """A system with genuine measurement noise, plus data simulated from it."""
    rng = np.random.default_rng(random_state)
    TTT = np.array([[0.6, 0.2], [0.0, 0.5]])
    QQ = np.array([[0.20, 0.05], [0.05, 0.15]])
    EE = np.eye(2) * 0.25
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


def test_hamilton_and_koopman_smoothers_agree():
    data = _stable_var()
    system = fit_resonance_state_space(data)
    kalman = kalman_filter(system, data)

    hamilton = hamilton_smoother(system, data, kalman=kalman)
    koopman = koopman_smoother(system, data, kalman=kalman)

    assert np.allclose(hamilton.smoothed_states, koopman.smoothed_states, atol=1e-6)
    assert np.allclose(hamilton.smoothed_covariances, koopman.smoothed_covariances, atol=1e-6)
    assert np.allclose(hamilton.smoothed_shocks, koopman.smoothed_shocks, atol=1e-6)


def test_smoothed_shocks_reconstruct_states():
    """Taking E[.|Y] of the transition equation is an exact linear identity."""
    system, observations = _noisy_system()
    result = koopman_smoother(system, observations)

    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    previous = np.zeros(system.n_states)
    for t in range(len(observations)):
        reconstructed = TTT @ previous + RRR @ result.smoothed_shocks[t] + CCC
        assert np.allclose(reconstructed, result.smoothed_states[t], atol=1e-6)
        previous = result.smoothed_states[t]


def test_smoother_reduces_uncertainty_relative_to_filter():
    system, observations = _noisy_system()
    kalman = kalman_filter(system, observations)
    smoothed = koopman_smoother(system, observations, kalman=kalman)

    filtered_trace = np.trace(kalman.filtered_covariance, axis1=1, axis2=2)
    smoothed_trace = np.trace(smoothed.smoothed_covariances, axis1=1, axis2=2)
    # Smoothing uses future information, so it never increases uncertainty.
    assert np.all(smoothed_trace <= filtered_trace + 1e-9)
    # Strictly lower somewhere before the final observation.
    assert np.any(smoothed_trace[:-1] < filtered_trace[:-1] - 1e-6)


def test_kalman_smoother_dispatch_and_summary():
    system, observations = _noisy_system()
    result = kalman_smoother(system, observations, method="koopman")

    assert isinstance(result, SmootherResult)
    assert result.smoothed_states.shape == observations.shape
    assert result.smoothed_observables.shape == observations.shape
    summary = result.summary()
    assert summary["method"] == "koopman"
    assert set(summary) >= {"smoothed_state_norm", "mean_smoothed_uncertainty", "shock_energy"}

    with pytest.raises(ValueError):
        kalman_smoother(system, observations, method="not-a-method")


def test_koopman_smoother_handles_missing_observations():
    system, observations = _noisy_system()
    observed = observations.copy()
    observed[40:55, 1] = np.nan

    result = koopman_smoother(system, observed)

    assert np.all(np.isfinite(result.smoothed_states))
    assert np.all(np.isfinite(result.smoothed_shocks))


def _assert_converges_to_smoothed_mean(draws, analytic):
    """The simulation-smoother draws are centred on the exact smoothed mean.

    Compares the Monte Carlo posterior mean to the analytic smoother mean in
    units of the Monte Carlo standard error, which is the statistically correct
    tolerance for a stochastic estimator.
    """
    error = draws.posterior_mean() - analytic.smoothed_states
    standard_error = draws.state_draws.std(axis=0) / np.sqrt(draws.n_draws)
    z_scores = np.abs(error) / (standard_error + 1e-12)
    # An unbiased estimator has E|z| ~ 0.8; almost every point stays within 4 SE.
    assert np.mean(z_scores) < 1.5
    assert np.mean(z_scores < 4.0) > 0.98


def test_durbin_koopman_posterior_mean_matches_analytic_smoother():
    system, observations = _noisy_system(n=150)
    analytic = koopman_smoother(system, observations)

    draws = durbin_koopman_smoother(system, observations, n_draws=300, random_state=1)

    assert isinstance(draws, SimulationSmootherResult)
    assert draws.state_draws.shape == (300, len(observations), system.n_states)
    _assert_converges_to_smoothed_mean(draws, analytic)


def test_carter_kohn_posterior_mean_matches_analytic_smoother():
    system, observations = _noisy_system(n=150)
    analytic = koopman_smoother(system, observations)

    draws = carter_kohn_smoother(system, observations, n_draws=300, random_state=2)

    assert draws.method == "carter_kohn"
    _assert_converges_to_smoothed_mean(draws, analytic)


def test_simulation_smoother_bands_are_ordered():
    system, observations = _noisy_system()
    draws = durbin_koopman_smoother(system, observations, n_draws=200, random_state=3)

    band = draws.posterior_band(lower=10.0, upper=90.0)
    assert np.all(band["lower"] <= band["upper"])


def test_simulation_smoothers_reject_bad_draw_count():
    system, observations = _noisy_system()
    with pytest.raises(ValueError):
        durbin_koopman_smoother(system, observations, n_draws=0)
    with pytest.raises(ValueError):
        carter_kohn_smoother(system, observations, n_draws=-1)
