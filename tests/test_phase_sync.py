"""Definitions, invariances, and the calibration property of the phase-synchrony test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drr_framework.modules import ResonanceDetector
from drr_framework.phase_sync import (
    DEFAULT_RELATIVE_BANDWIDTH,
    PhaseSynchronyAnalyzer,
    band_limited_analytic_signal,
    dominant_frequency,
    effective_phase_samples,
    farey_ratio_candidates,
    holm_adjust,
    instantaneous_frequency,
    phase_difference_dynamics,
    phase_locking_value,
    phase_shift_test,
    single_change_point,
    windowed_phase_locking,
)
from drr_framework.resonance_math_benchmark import simulate_condition

FS = 100.0


def _pair(seed: int, n: int = 2048, **fields):
    spec = {
        "family": "phase_oscillators",
        "sampling_rate": FS,
        "n_samples": n,
        "fx": 10.0,
        "fy": 10.2,
        "coupling": 0.0,
        "ratio": [1, 1],
        "phase_diffusion": 2.0,
        "delay_seconds": 0.0,
        "noise": 0.3,
        "am_depth": 0.0,
        "am_rate_hz": 0.0,
        "amplitude_ramp": False,
        "chirp_sweep_hz": 0.0,
        "common_chirp": False,
        "missing_fraction": 0.0,
        "timing_jitter": 0.0,
        "burn_in_seconds": 10.0,
    }
    spec.update(fields)
    data, _ = simulate_condition(spec, np.random.default_rng(seed))
    return data


# ------------------------------------------------------------------ the analytic band


def test_band_is_the_morlet_wavelet_of_the_resonance_detector():
    """|z|^2 is four times DRR's Morlet (omega0 = 6) scalogram at the same frequency."""
    rng = np.random.default_rng(0)
    n = 1024
    t = np.arange(n) / FS
    signal = (
        np.sin(2 * np.pi * 7.0 * t) + 0.5 * np.sin(2 * np.pi * 19.0 * t) + rng.normal(0, 0.3, n)
    )
    wavelet = ResonanceDetector()._detect_with_wavelet(signal, FS, 0.1)
    index = int(np.argmin(np.abs(wavelet["frequency_axis"] - 7.0)))
    frequency = float(wavelet["frequency_axis"][index])
    band = band_limited_analytic_signal(signal, FS, frequency, DEFAULT_RELATIVE_BANDWIDTH)
    interior = slice(6 * band.edge_samples, n - 6 * band.edge_samples)
    assert band.bandwidth_hz == pytest.approx(frequency / 6.0)
    assert np.abs(band.signal[interior]) ** 2 == pytest.approx(
        4.0 * wavelet["scalogram"][index, interior], rel=1e-5
    )


def test_band_recovers_a_tone_amplitude_phase_and_frequency():
    n = 2048
    t = np.arange(n) / FS
    tone = 2.5 * np.cos(2 * np.pi * 9.3 * t + 0.4)
    band = band_limited_analytic_signal(tone, FS, 9.3)
    interior = slice(4 * band.edge_samples, n - 4 * band.edge_samples)
    assert band.amplitude[interior] == pytest.approx(2.5, rel=1e-3)
    expected = np.angle(np.exp(1j * (2 * np.pi * 9.3 * t + 0.4)))
    assert np.abs(np.angle(np.exp(1j * (band.phase - expected)))[interior]).max() < 1e-3
    frequency = instantaneous_frequency(band)
    assert frequency[interior] == pytest.approx(9.3, abs=1e-3)
    assert band.band_power_fraction > 0.95


def test_instantaneous_frequency_follows_a_sweep_and_masks_vanishing_amplitude():
    n = 4096
    t = np.arange(n) / FS
    sweep = np.cos(2 * np.pi * (8.0 * t + 0.5 * 0.1 * t**2))  # 8 Hz rising 0.1 Hz/s
    band = band_limited_analytic_signal(sweep, FS, 10.0, relative_bandwidth=0.4)
    frequency = instantaneous_frequency(band)
    middle = slice(n // 4, 3 * n // 4)
    assert frequency[middle] == pytest.approx(8.0 + 0.1 * t[middle], abs=0.05)
    silent = np.concatenate((np.zeros(512), np.sin(2 * np.pi * 10 * np.arange(1536) / FS)))
    masked = instantaneous_frequency(band_limited_analytic_signal(silent, FS, 10.0))
    assert np.isnan(masked[:300]).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"center_frequency": 0.0}, "Nyquist"),
        ({"center_frequency": 50.0}, "Nyquist"),
        ({"relative_bandwidth": 0.0}, "positive"),
    ],
)
def test_band_rejects_invalid_parameters(kwargs, match):
    arguments = {"data": np.random.default_rng(1).normal(size=128), "sampling_rate": FS}
    arguments.update({"center_frequency": 10.0, **kwargs})
    with pytest.raises(ValueError, match=match):
        band_limited_analytic_signal(**arguments)


def test_band_rejects_short_or_non_finite_input():
    with pytest.raises(ValueError, match="16 samples"):
        band_limited_analytic_signal(np.ones(8), FS, 10.0)
    data = np.random.default_rng(2).normal(size=128)
    data[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        band_limited_analytic_signal(data, FS, 10.0)


# ------------------------------------------------------------------ the statistic


def test_plv_is_bounded_amplitude_free_and_offset_invariant():
    rng = np.random.default_rng(3)
    phase = rng.uniform(-np.pi, np.pi, 4000)
    plv, offset = phase_locking_value(phase, phase - 0.7)
    assert plv == pytest.approx(1.0) and offset == pytest.approx(0.7)
    shifted, _ = phase_locking_value(phase + 1.3, phase + 1.3 - 0.7)
    assert shifted == pytest.approx(plv)
    unrelated, _ = phase_locking_value(phase, rng.uniform(-np.pi, np.pi, 4000))
    assert 0.0 <= unrelated < 0.05
    harmonic, _ = phase_locking_value(phase, np.angle(np.exp(1j * (2 * phase + 0.2))), n=2, m=1)
    assert harmonic == pytest.approx(1.0)


def test_farey_candidates_name_the_nearest_small_integer_ratio():
    assert farey_ratio_candidates(10.0, 20.3) == [(2, 1)]
    assert farey_ratio_candidates(10.0, 15.2) == [(3, 2)]
    assert farey_ratio_candidates(10.0, 6.6) == [(2, 3)]
    assert farey_ratio_candidates(10.0, 10.4) == [(1, 1)]
    assert farey_ratio_candidates(10.0, 13.7) == []
    assert farey_ratio_candidates(0.0, 10.0) == []
    assert farey_ratio_candidates(10.0, 12.5, max_order=4, relative_tolerance=0.01) == []
    assert farey_ratio_candidates(10.0, 12.5, max_order=5, relative_tolerance=0.01) == [(5, 4)]
    wide = farey_ratio_candidates(10.0, 14.5, max_order=4, relative_tolerance=0.1)
    assert wide[0] == (3, 2) and (4, 3) in wide
    with pytest.raises(ValueError):
        farey_ratio_candidates(10.0, 10.0, max_order=0)


def test_effective_samples_span_independent_phases_to_pure_tones():
    rng = np.random.default_rng(4)
    iid = np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 4000)))
    assert effective_phase_samples(iid[0], iid[1]) == pytest.approx(4000, rel=0.2)
    t = np.arange(4000)
    tone = np.exp(1j * 0.3 * t)
    assert effective_phase_samples(tone, tone * np.exp(1j * 0.5)) < 6
    # Two Wiener phases with diffusion D for T seconds: about T * D effective samples.
    dt, diffusion = 0.01, 2.0
    walks = np.cumsum(np.sqrt(2 * diffusion * dt) * rng.standard_normal((2, 4000)), axis=1)
    effective = effective_phase_samples(np.exp(1j * walks[0]), np.exp(1j * walks[1]))
    assert 0.4 * 40 * diffusion < effective < 2.5 * 40 * diffusion


def test_shift_test_is_exact_for_iid_phase_increments():
    """Proven statement: with i.i.d. increments the corrected test's p-value is uniform."""
    rng = np.random.default_rng(5)
    p_values = []
    for _ in range(400):
        walks = np.cumsum(rng.normal(0.25, 0.08, (2, 300)), axis=1)
        p_values.append(phase_shift_test(walks[0], walks[1], [(1, 1)])["p_value"])
    p_values = np.array(p_values)
    for level in (0.05, 0.1, 0.25, 0.5):
        assert np.mean(p_values <= level) == pytest.approx(level, abs=0.06)


def test_without_the_seam_correction_coherent_phases_reject_too_often():
    rng = np.random.default_rng(6)
    corrected, uncorrected = 0, 0
    for _ in range(200):
        walks = np.cumsum(rng.normal(0.3, 0.02, (2, 400)), axis=1)
        corrected += phase_shift_test(walks[0], walks[1], [(1, 1)])["p_value"] <= 0.05
        uncorrected += (
            phase_shift_test(walks[0], walks[1], [(1, 1)], seam_correction=False)["p_value"] <= 0.05
        )
    assert corrected / 200 < 0.1
    assert uncorrected / 200 > 0.2


def test_shift_test_detects_locking_and_cannot_tell_pure_tones():
    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(0.4, 0.15, 1000))
    locked = walk + 0.6 + rng.normal(0, 0.2, 1000)
    result = phase_shift_test(walk, locked, [(1, 1)])
    assert result["p_value"] == pytest.approx(result["minimum_attainable_p_value"])
    assert result["n_shifts"] == 999
    t = np.arange(1000)
    tones = phase_shift_test(0.3 * t, 0.3 * t + 2.0, [(1, 1)])
    assert tones["p_value"] == pytest.approx(1.0)
    assert phase_shift_test(walk, locked, [(1, 1)]) == result


def test_min_p_over_ratios_selects_the_locked_ratio_and_controls_the_family():
    rng = np.random.default_rng(8)
    walk = np.cumsum(rng.normal(0.3, 0.15, 1200))
    harmonic = 2 * walk + rng.normal(0, 0.3, 1200)
    result = phase_shift_test(walk, harmonic, [(1, 1), (2, 1)])
    assert result["selected_ratio"] == [2, 1]
    assert result["p_value"] <= 0.01
    assert len(result["marginal_p_values"]) == 2
    with pytest.raises(ValueError, match="ratio"):
        phase_shift_test(walk, harmonic, [])


def test_phase_dynamics_measure_mismatch_slips_and_diffusion():
    t = np.arange(4000) / FS
    phase_x = np.angle(np.exp(1j * 2 * np.pi * 10.0 * t))
    phase_y = np.angle(np.exp(1j * 2 * np.pi * 10.5 * t))
    dynamics = phase_difference_dynamics(phase_x, phase_y, 1, 1, FS)
    assert dynamics["frequency_mismatch_hz"] == pytest.approx(-0.5, abs=1e-6)
    assert dynamics["net_phase_slips"] == pytest.approx(-20.0)
    assert dynamics["phase_diffusion_rad2_per_s"] == pytest.approx(0.0, abs=1e-9)
    harmonic = phase_difference_dynamics(phase_x, np.angle(np.exp(2j * np.pi * 20.0 * t)), 2, 1, FS)
    assert harmonic["frequency_mismatch_hz"] == pytest.approx(0.0, abs=1e-6)


def test_windowed_locking_and_change_point_locate_a_coupling_onset():
    rng = np.random.default_rng(9)
    n = 4000
    walk = np.cumsum(rng.normal(0.3, 0.3, n))
    other = np.cumsum(rng.normal(0.3, 0.3, n))
    other[n // 2 :] = walk[n // 2 :] + rng.normal(0, 0.2, n - n // 2)
    centers, plv = windowed_phase_locking(walk, other, 1, 1, window=200, step=50)
    split = single_change_point(plv)
    onset = centers[int(split["index"])]
    assert abs(onset - n // 2) < 300
    assert split["mean_after"] > split["mean_before"] + 0.4
    with pytest.raises(ValueError):
        windowed_phase_locking(walk, other, 1, 1, window=1)


def test_holm_adjustment_matches_a_hand_computation():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.2])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert holm_adjust([]).size == 0


# ------------------------------------------------------------------ the analyzer


def test_analyzer_finds_harmonic_synchronization_that_lagged_correlation_cannot():
    data = _pair(10, fx=10.0, fy=20.3, coupling=12.0, ratio=[2, 1])
    result = PhaseSynchronyAnalyzer(FS).analyze(data, column_names=("drive", "response"))
    pair = result["pairs"][0]
    assert pair["status"] == "synchronized"
    assert pair["selected_ratio"] == [2, 1]
    assert pair["channels"] == ["drive", "response"]
    assert pair["effective_samples"] >= 20
    lagged = np.max([abs(np.corrcoef(data[:-lag, 0], data[lag:, 1])[0, 1]) for lag in range(1, 5)])
    assert lagged < 0.1
    json.dumps(result, allow_nan=False)


def test_analyzer_statuses_for_untestable_and_unidentifiable_pairs():
    unrelated_ratio = _pair(11, fx=10.0, fy=13.7)
    pair = PhaseSynchronyAnalyzer(FS).analyze(unrelated_ratio)["pairs"][0]
    assert pair["status"] == "not_tested" and "not within" in pair["reason"]

    t = np.arange(2048) / FS
    tones = np.column_stack((np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 10 * t + 1.0)))
    pair = PhaseSynchronyAnalyzer(FS).analyze(tones)["pairs"][0]
    assert pair["status"] == "inconclusive"
    assert "effective phase samples" in pair["reason"]

    slow = np.column_stack((np.sin(2 * np.pi * 0.3 * t), np.cos(2 * np.pi * 0.3 * t)))
    pair = PhaseSynchronyAnalyzer(FS).analyze(slow)["pairs"][0]
    assert pair["status"] == "inconclusive" and pair["p_value"] is None


def test_analyzer_applies_holm_across_pairs_and_validates_input():
    rng = np.random.default_rng(12)
    data = np.column_stack([_pair(13)[:, 0], _pair(14)[:, 0], rng.normal(size=2048)])
    result = PhaseSynchronyAnalyzer(FS).analyze(data)
    assert len(result["pairs"]) == 3
    tested = [p for p in result["pairs"] if p["p_value"] is not None]
    for pair in tested:
        assert pair["adjusted_p_value"] >= pair["p_value"]
    with pytest.raises(ValueError, match="channels"):
        PhaseSynchronyAnalyzer(FS).analyze(data[:, 0])
    with pytest.raises(ValueError, match="column_names"):
        PhaseSynchronyAnalyzer(FS).analyze(data, column_names=("a", "a", "b"))
    with pytest.raises(ValueError, match="alpha"):
        PhaseSynchronyAnalyzer(FS, alpha=1.5)
    bad = data.copy()
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        PhaseSynchronyAnalyzer(FS).analyze(bad)


def test_dominant_frequency_matches_the_depth_calculator_target():
    from drr_framework.modules import DepthCalculator

    data = _pair(15)[:, 0]
    assert dominant_frequency(data, FS) == DepthCalculator()._select_target_frequency(
        data, FS, None
    )
