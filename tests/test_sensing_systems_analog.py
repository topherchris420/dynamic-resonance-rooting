"""Tests for Sensing Systems DRR Benchmark Analog (US 8,169,362 B2 inspiration)."""

import numpy as np
import pytest

from drr_framework import BenchmarkSystems, DynamicResonanceRooting, generate_micro_doppler_analog


def test_generator_output_shape_and_metadata():
    """Verify signal dimensions, time step spacing, and metadata attributes."""
    duration = 20.0
    sampling_rate = 50.0
    n_channels = 3
    lag = 2

    t, data, metadata = generate_micro_doppler_analog(
        duration=duration,
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        target_frequency_hz=0.3,
        clutter_frequency_hz=0.02,
        lag=lag,
        regime_change_time=10.0,
        noise_scale=0.05,
        random_state=42,
    )

    n_expected = int(duration * sampling_rate)
    assert t.shape == (n_expected,)
    assert data.shape == (n_expected, n_channels)
    assert np.isclose(t[1] - t[0], 1.0 / sampling_rate)

    assert metadata["benchmark_system"] == "micro_doppler_sensing_analog"
    assert metadata["sampling_rate_hz"] == sampling_rate
    assert metadata["duration_seconds"] == duration
    assert metadata["n_channels"] == n_channels
    assert metadata["target_frequency_hz"] == 0.3
    assert metadata["clutter_frequency_hz"] == 0.02
    assert metadata["inter_channel_lag_samples"] == lag
    assert metadata["regime_change_time_seconds"] == 10.0


def test_benchmark_systems_class_method():
    """Verify BenchmarkSystems.generate_micro_doppler_analog static method."""
    t1, d1, m1 = BenchmarkSystems.generate_micro_doppler_analog(random_state=123)
    t2, d2, m2 = generate_micro_doppler_analog(random_state=123)

    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(d1, d2)
    assert m1 == m2


def test_analyze_system_ground_truth_recovery():
    """Verify complete DRR system analysis on generated micro-Doppler signal."""
    sampling_rate = 100.0
    duration = 30.0
    t, data, metadata = generate_micro_doppler_analog(
        duration=duration,
        sampling_rate=sampling_rate,
        n_channels=3,
        target_frequency_hz=0.3,
        clutter_frequency_hz=0.02,
        lag=2,
        regime_change_time=15.0,
        noise_scale=0.02,
        random_state=42,
    )

    drr = DynamicResonanceRooting(embedding_dim=3, tau=1, sampling_rate=sampling_rate)
    results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=200,
        state_space=True,
        state_space_horizon=10,
        method="welch",
    )

    assert "resonances" in results
    assert "resonance_depths" in results
    assert "influence_network" in results
    assert "state_space_analysis" in results

    depths = results["resonance_depths"]
    assert len(depths) == 3
    for k, depth in depths.items():
        assert 0.0 <= depth <= 1.0


def test_generator_parameter_validation():
    """Verify ValueError exceptions on invalid generator parameters."""
    with pytest.raises(ValueError, match="duration must be positive"):
        generate_micro_doppler_analog(duration=-5.0)

    with pytest.raises(ValueError, match="sampling_rate must be positive"):
        generate_micro_doppler_analog(sampling_rate=0.0)

    with pytest.raises(ValueError, match="n_channels must be at least 1"):
        generate_micro_doppler_analog(n_channels=0)
