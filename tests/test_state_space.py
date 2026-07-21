import numpy as np
import pytest

from drr_framework import DynamicResonanceRooting
from drr_framework.state_space import (
    Measurement,
    StateSpaceSystem,
    Transition,
    analyze_resonance_state_space,
    chandrasekhar_recursion,
    fit_resonance_state_space,
    impulse_response,
    kalman_filter,
    stationary_initialization,
)


def _known_system():
    TTT = np.array([[0.6, 0.2], [0.0, 0.5]])
    QQ = np.array([[0.20, 0.05], [0.05, 0.15]])
    EE = np.eye(2) * 0.30
    return StateSpaceSystem(
        transition=Transition(TTT, np.eye(2), np.array([0.1, -0.05])),
        measurement=Measurement(np.eye(2), np.zeros(2), QQ, EE),
        state_names=["a", "b"],
        observable_names=["a", "b"],
        shock_names=["e0", "e1"],
    )


def _simulate(system, n=200, random_state=3):
    rng = np.random.default_rng(random_state)
    TTT = system["TTT"]
    CCC = system["CCC"]
    chol_q = np.linalg.cholesky(system["QQ"])
    chol_e = np.linalg.cholesky(system["EE"])
    state = np.zeros(system.n_states)
    observations = np.zeros((n, system.n_observables))
    for t in range(n):
        state = TTT @ state + chol_q @ rng.standard_normal(system.n_shocks) + CCC
        observations[t] = (
            system["ZZ"] @ state + system["DD"] + chol_e @ rng.standard_normal(system.n_observables)
        )
    return observations


def _generate_stable_var(random_state=11):
    rng = np.random.default_rng(random_state)
    transition = np.array([[0.72, 0.12], [0.03, 0.48]])
    data = np.zeros((500, 2))
    shocks = rng.normal(scale=0.08, size=data.shape)
    for idx in range(1, len(data)):
        data[idx] = transition @ data[idx - 1] + shocks[idx]
    return transition, data


def test_fit_resonance_state_space_recovers_stable_transition_shape():
    _, data = _generate_stable_var()

    system = fit_resonance_state_space(
        data,
        state_names=["driver", "response"],
        observable_names=["driver", "response"],
    )

    assert system["TTT"].shape == (2, 2)
    assert system["RRR"].shape == (2, 2)
    assert system["ZZ"].shape == (2, 2)
    assert system.state_names == ("driver", "response")
    diagnostics = system.diagnostics()
    assert diagnostics["spectral_radius"] < 1.0
    assert diagnostics["is_stable"] is True


def test_kalman_filter_handles_missing_observations_and_reports_likelihood():
    _, data = _generate_stable_var()
    system = fit_resonance_state_space(data)
    observed = data.copy()
    observed[20:30, 1] = np.nan

    result = kalman_filter(system, observed)

    assert result.filtered_state.shape == data.shape
    assert result.predicted_covariance.shape == (len(data), 2, 2)
    assert np.isfinite(result.total_log_likelihood)
    assert np.isnan(result.innovations[20:30, 1]).all()
    assert np.isfinite(result.innovations[20:30, 0]).all()


def test_impulse_response_and_analysis_bundle_are_consistent():
    _, data = _generate_stable_var()
    analysis = analyze_resonance_state_space(data, impulse_horizon=8)

    response = impulse_response(analysis["system"], horizon=8, shock_index=0)

    assert response["state_response"].shape == (9, 2)
    assert response["observable_response"].shape == (9, 2)
    assert len(analysis["impulse_responses"]) == 2
    assert analysis["diagnostics"]["state_count"] == 2
    assert np.isfinite(analysis["diagnostics"]["total_log_likelihood"])


def test_stationary_initialization_solves_lyapunov_equation():
    system = _known_system()
    mean, covariance = stationary_initialization(system)

    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    QQ = system["QQ"]
    # mu = T mu + C  and  Sigma = T Sigma T' + R Q R'
    assert np.allclose(mean, TTT @ mean + CCC, atol=1e-9)
    assert np.allclose(covariance, TTT @ covariance @ TTT.T + RRR @ QQ @ RRR.T, atol=1e-8)


def test_chandrasekhar_matches_kalman_under_stationary_initialization():
    system = _known_system()
    observations = _simulate(system)
    mean, covariance = stationary_initialization(system)

    kalman = kalman_filter(system, observations, initial_state=mean, initial_covariance=covariance)
    chandrasekhar = chandrasekhar_recursion(system, observations)

    assert abs(chandrasekhar.total_log_likelihood - kalman.total_log_likelihood) < 1e-4
    assert np.allclose(chandrasekhar.log_likelihood, kalman.log_likelihood, atol=1e-4)


def test_chandrasekhar_rejects_unstable_system_and_missing_data():
    unstable = StateSpaceSystem(
        transition=Transition(np.array([[1.05, 0.0], [0.0, 0.4]]), np.eye(2), np.zeros(2)),
        measurement=Measurement(np.eye(2), np.zeros(2), np.eye(2), np.eye(2)),
        state_names=["a", "b"],
        observable_names=["a", "b"],
        shock_names=["e0", "e1"],
    )
    with pytest.raises(ValueError):
        chandrasekhar_recursion(unstable, np.ones((10, 2)))

    system = _known_system()
    observations = _simulate(system, n=20)
    observations[5, 0] = np.nan
    with pytest.raises(ValueError):
        chandrasekhar_recursion(system, observations)


def test_analyze_bundle_includes_smoother_by_default():
    data = _generate_stable_var()[1]
    analysis = analyze_resonance_state_space(data, smooth=True)

    assert "smoother" in analysis
    assert analysis["smoother"].smoothed_states.shape == data.shape
    assert "mean_smoothed_uncertainty" in analysis["diagnostics"]

    without = analyze_resonance_state_space(data, smooth=False)
    assert "smoother" not in without


def test_dynamic_resonance_rooting_attaches_state_space_diagnostics():
    _, data = _generate_stable_var()
    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=100.0)

    results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=128,
        state_space=True,
        state_space_horizon=6,
    )

    assert "state_space_analysis" in results
    diagnostics = results["state_space_analysis"]["diagnostics"]
    assert diagnostics["state_count"] == 2
    assert diagnostics["shock_count"] == 2
    assert np.isfinite(diagnostics["mean_log_likelihood"])
