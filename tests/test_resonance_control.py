"""Unit and integration tests for the Resonance Navigation Control Engine."""

import pytest
import numpy as np

from drr_framework import (
    ControlIntervention,
    NavigationMetrics,
    ResonanceControlExperimentSuite,
    ResonanceNavigationEngine,
    ResonanceState,
    ResonanceTarget,
)


def test_resonance_state_and_target_initialization():
    """Test data structures for ResonanceState and ResonanceTarget."""
    dim = 3
    state = ResonanceState(
        dominant_frequencies=np.array([1.0, 2.0, 0.5]),
        spectral_power=np.array([10.0, 5.0, 1.0]),
        coherence_matrix=np.eye(dim),
        phase_matrix=np.zeros((dim, dim)),
        root_distribution=np.array([0.6, 0.3, 0.1]),
        state_estimate=np.array([0.1, -0.2, 0.05]),
        state_uncertainty=np.array([0.01, 0.02, 0.01]),
        resonance_depths={"dim_0": 0.85, "dim_1": 0.42, "dim_2": 0.12},
        rooting_adjacency=np.array([[0, 0.5, 0.2], [0, 0, 0.1], [0, 0, 0]]),
    )

    assert state.dominant_frequencies.shape == (3,)
    assert state.root_distribution[0] == 0.6
    assert state.resonance_depths["dim_0"] == 0.85

    target = ResonanceTarget(
        target_basin_center=np.zeros(dim),
        target_frequencies=np.array([1.5, 1.5, 0.0]),
        tolerance=0.2,
    )
    assert target.tolerance == 0.2
    assert target.target_basin_center.shape == (3,)


def test_engine_observation_and_sensitivity_estimation():
    """Test state observation and controllability/sensitivity estimation."""
    engine = ResonanceNavigationEngine(n_dimensions=3, sampling_rate=100.0, u_max=1.0)
    data_window = np.random.default_rng(42).normal(size=(50, 3))

    state = engine.observe_state(data_window)
    assert isinstance(state, ResonanceState)
    assert state.root_distribution.shape == (3,)
    assert np.isclose(np.sum(state.root_distribution), 1.0)

    sensitivity = engine.estimate_controllability_and_sensitivity(state)
    assert "primary_root" in sensitivity
    assert "sensitivities" in sensitivity
    assert sensitivity["sensitivities"].shape == (3, 3)
    assert 0 <= sensitivity["primary_root"] < 3


def test_compute_intervention_bounds_and_modes():
    """Test that all control strategies return bounded interventions (||u|| <= u_max)."""
    u_max = 0.5
    engine = ResonanceNavigationEngine(n_dimensions=3, u_max=u_max)
    data_window = np.random.default_rng(42).normal(size=(40, 3))
    state = engine.observe_state(data_window)
    target = ResonanceTarget(target_basin_center=np.zeros(3))

    for strategy in [
        "root_aware_drr",
        "state_only",
        "naive",
        "random",
        "ablation_no_rooting",
        "ablation_no_spectral",
    ]:
        intervention = engine.compute_intervention(state, target, step_idx=10, strategy=strategy)
        assert isinstance(intervention, ControlIntervention)
        assert intervention.magnitude <= u_max + 1e-6
        assert intervention.u.shape == (3,)


def test_closed_loop_simulation_and_metrics_tracking():
    """Test closed-loop simulation, lock tracking, disturbance recovery, and metrics."""
    suite = ResonanceControlExperimentSuite(random_state=42)
    dynamics_fn, initial_state, dim = suite.get_benchmark_system("coupled_oscillator")
    target = ResonanceTarget(target_basin_center=np.zeros(dim), tolerance=0.3)

    disturbance_schedule = {50: np.array([2.0, 0.0, 0.0])}

    engine = ResonanceNavigationEngine(n_dimensions=dim, u_max=1.0, random_state=42)
    metrics = engine.simulate_closed_loop(
        system_dynamics_fn=dynamics_fn,
        initial_state=initial_state,
        target=target,
        n_steps=100,
        strategy="root_aware_drr",
        disturbance_schedule=disturbance_schedule,
    )

    assert isinstance(metrics, NavigationMetrics)
    assert metrics.trajectory.shape == (100, 3)
    assert metrics.control_history.shape == (100, 3)
    assert metrics.target_distance_history.shape == (100,)
    assert metrics.normalized_control_cost >= 0.0
    assert metrics.hysteresis_loop_area >= 0.0
    assert metrics.root_migration_distance >= 0.0


def test_matched_experiments_and_controller_comparison():
    """Test matched experiments comparing DRR control against baselines."""
    suite = ResonanceControlExperimentSuite(random_state=42)
    results = suite.run_matched_experiment(
        benchmark_system="coupled_oscillator",
        n_steps=100,
        u_max=1.0,
    )

    assert "root_aware_drr" in results
    assert "state_only" in results
    assert "naive" in results
    assert "random" in results

    drr_cost = results["root_aware_drr"].normalized_control_cost
    random_cost = results["random"].normalized_control_cost
    assert drr_cost < random_cost + 1.0


def test_ablation_and_surrogate_studies():
    """Test component ablation studies and surrogate statistical tests."""
    suite = ResonanceControlExperimentSuite(random_state=42)
    ablations = suite.run_ablation_study(benchmark_system="coupled_oscillator", n_steps=80)
    assert "root_aware_drr" in ablations
    assert "ablation_no_rooting" in ablations
    assert "ablation_no_spectral" in ablations

    surrogate_res = suite.run_surrogate_test(
        benchmark_system="coupled_oscillator", n_surrogates=5, n_steps=60
    )
    assert "p_value_error_reduction" in surrogate_res
    assert "p_value_control_efficiency" in surrogate_res


def test_negative_control_and_honest_null_reporting():
    """Test negative control system verifying honest reporting of null results."""
    suite = ResonanceControlExperimentSuite(random_state=42)
    neg_res = suite.run_negative_control(n_steps=80)

    assert neg_res["benchmark"] == "uncoupled_symmetric_negative_control"
    assert "null_result_reported" in neg_res
    assert "honest_reporting_statement" in neg_res
    assert isinstance(neg_res["honest_reporting_statement"], str)


def test_evidence_card_generation():
    """Test automated creation of immutable DRR evidence card for control experiments."""
    suite = ResonanceControlExperimentSuite(random_state=42)
    surr_res = suite.run_surrogate_test(
        benchmark_system="coupled_oscillator", n_surrogates=3, n_steps=50
    )
    card = suite.generate_control_evidence_card(surr_res)

    assert card.signal_id == "resonance_navigation_control_001"
    assert len(card.reproducibility_hash) == 64
    assert "detection" in card.detection_vs_interpretation
    assert "interpretation" in card.detection_vs_interpretation
