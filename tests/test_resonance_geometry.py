import numpy as np

from drr_framework import (
    DynamicResonanceGeometry,
    collective_modes,
    estimate_cross_resonance,
    resonance_regularized_covariance,
    resonance_risk_kernel,
    root_migration,
    structural_surprise,
    topology_drift,
)


def test_cross_resonance_recovers_frequency_phase_and_symmetry():
    sampling_rate = 100.0
    time = np.arange(2048) / sampling_rate
    offset = 0.7
    data = np.column_stack((np.sin(2 * np.pi * 7 * time), np.sin(2 * np.pi * 7 * time + offset)))
    tensor = estimate_cross_resonance(data, sampling_rate, nperseg=512)
    index = np.argmin(abs(tensor.frequencies - 7))
    assert tensor.coherence[index, 0, 1] > 0.99
    assert abs(tensor.dominant_frequency(0, 1) - 7) < 0.25
    assert abs(np.angle(np.exp(1j * (tensor.phase[index, 0, 1] - offset)))) < 0.05
    assert np.allclose(tensor.phase[:, 0, 1], -tensor.phase[:, 1, 0])


def test_frequency_separation_does_not_create_power_weighted_coupling():
    time = np.arange(2048) / 100
    data = np.column_stack((np.sin(2 * np.pi * 4 * time), np.sin(2 * np.pi * 17 * time)))
    tensor = estimate_cross_resonance(data, 100, nperseg=256)
    joint = tensor.coherence[:, 0, 1] * tensor.power_weight[:, 0, 1]
    assert joint.max() < 0.02


def test_collective_mode_effective_rank_limits():
    rank_one = collective_modes(np.ones((4, 4)))
    independent = collective_modes(np.eye(4))
    assert np.isclose(rank_one.effective_rank, 1)
    assert np.isclose(independent.effective_rank, 4)
    assert rank_one.dominant_mode_strength > independent.dominant_mode_strength


def test_root_migration_and_topology_drift_invariants():
    first = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], float)
    perturb = first.copy()
    perturb[1, 2] = 0.05
    rewire = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], float)
    assert root_migration(first, first) == 0
    assert 0 <= root_migration(first, rewire) <= 1
    assert root_migration(first, rewire) > root_migration(first, perturb)
    assert topology_drift(first, first) == 0
    assert topology_drift(first, rewire) > topology_drift(first, perturb)


def test_structural_surprise_is_prefix_invariant_and_detects_break():
    rng = np.random.default_rng(4)
    stable = np.zeros((90, 2))
    for index in range(1, len(stable)):
        stable[index] = 0.8 * stable[index - 1] + rng.normal(0, 0.05, 2)
    changed = stable.copy()
    changed[65:] += np.array([2.0, -2.0])
    prefix = structural_surprise(changed[:75], min_history=20)
    extended = structural_surprise(np.vstack((changed[:75], rng.normal(size=(10, 2)))), 20)
    assert np.allclose(prefix.score, extended.score[:75], equal_nan=True)
    assert prefix.score[65] > np.nanpercentile(prefix.score[20:60], 95)


def test_kernel_and_regularized_covariance_are_psd():
    rng = np.random.default_rng(8)
    loadings = rng.normal(size=(6, 3))
    kernel = resonance_risk_kernel(loadings)
    sample = rng.normal(size=(100, 6))
    covariance = np.cov(sample, rowvar=False)
    adjusted = resonance_regularized_covariance(covariance, kernel, 0.2)
    assert np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10
    assert np.linalg.eigvalsh(adjusted).min() > -1e-10


def test_rolling_geometry_is_prefix_invariant_and_serializable():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(80, 3))
    geometry = DynamicResonanceGeometry(nperseg=16)
    short = geometry.rolling(data[:60], 32)
    long = geometry.rolling(data, 32)
    short_values = [result.fingerprint.to_dict() for result in short]
    long_values = [result.fingerprint.to_dict() for result in long[: len(short)]]
    assert short_values == long_values
