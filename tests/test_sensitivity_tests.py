"""Unit tests for sensitivity and null testing framework."""

import pytest
import numpy as np
from drr_framework.sensitivity_tests import (
    run_parameter_sensitivity_experiment,
    run_placebo_and_null_tests,
)


def test_run_parameter_sensitivity_experiment():
    data = np.random.normal(size=(150, 2))
    res = run_parameter_sensitivity_experiment(
        data, window_sizes=[32, 64], methods=["welch"], tau_values=[1]
    )
    assert "sensitivity_trials" in res
    assert "fragility_score" in res
    assert len(res["sensitivity_trials"]) == 2


def test_run_placebo_and_null_tests():
    res = run_placebo_and_null_tests(n_samples=200, random_state=42)
    assert "gaussian_noise" in res
    assert "phase_shuffled" in res
    assert "decorrelated_multivariate" in res


@pytest.mark.parametrize("n_samples", [255, 256])
def test_fourier_surrogate_keeps_the_power_spectrum_and_is_real(n_samples):
    from drr_framework.sensitivity_tests import fourier_surrogate

    rng = np.random.default_rng(0)
    series = rng.standard_normal(n_samples).cumsum()
    surrogate = fourier_surrogate(series, rng)
    assert surrogate.dtype == float and surrogate.shape == series.shape
    assert np.allclose(np.abs(np.fft.rfft(surrogate)), np.abs(np.fft.rfft(series)))
    assert not np.allclose(surrogate, series)


def test_phase_shuffle_reports_an_actual_reduction():
    res = run_placebo_and_null_tests(n_samples=500, random_state=42)["phase_shuffled"]
    assert res["depth_reduction_from_shuffle"] == pytest.approx(
        res["original_resonance_depth"] - res["resonance_depth"]
    )
    # A tone keeps its spectrum under phase randomization, so depth barely moves.
    assert abs(res["depth_reduction_from_shuffle"]) < 0.05
