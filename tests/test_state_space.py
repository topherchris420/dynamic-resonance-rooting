import numpy as np

from drr_framework import DynamicResonanceRooting
from drr_framework.state_space import (
    analyze_resonance_state_space,
    fit_resonance_state_space,
    impulse_response,
    kalman_filter,
)


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
