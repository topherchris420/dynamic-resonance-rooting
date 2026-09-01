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
