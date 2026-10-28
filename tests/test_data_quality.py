"""The data-quality gate: scaling, drift, sampling, and when it abstains."""

from __future__ import annotations

import numpy as np
import pytest

from drr_framework.calibration import simulate_ar1
from drr_framework.data_quality import (
    DEFAULT_THRESHOLDS,
    assess_series_quality,
    detrended_fluctuation,
    dfa_scale_range,
    frequency_drift,
    regularize_series,
)
from drr_framework.dependent_noise import simulate_fractional_gaussian_noise

FS = 100.0


def _oscillator(seed: int, n: int = 2048, frequency: float = 10.0, diffusion: float = 2.0):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi) + np.cumsum(
        2 * np.pi * frequency / FS + np.sqrt(2 * diffusion / FS) * rng.standard_normal(n)
    )
    return np.cos(phase) + 0.3 * rng.standard_normal(n)


def _chirp(n: int = 2048, start: float = 8.0, stop: float = 12.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    rate = (stop - start) / t[-1]
    return np.cos(2 * np.pi * (start * t + 0.5 * rate * t**2)) + 0.2 * rng.standard_normal(n)


# ------------------------------------------------------------------ scaling


@pytest.mark.parametrize("hurst", [0.5, 0.8, 0.95])
def test_dfa_recovers_the_hurst_exponent_of_fractional_gaussian_noise(hurst):
    series = simulate_fractional_gaussian_noise(8192, 1, hurst, np.random.default_rng(1))[:, 0]
    result = detrended_fluctuation(series)
    assert result["alpha"] == pytest.approx(hurst, abs=0.1)
    assert result["r_squared"] > 0.95


def test_dfa_of_a_random_walk_is_one_and_a_half():
    walk = np.cumsum(np.random.default_rng(2).standard_normal(8192))
    assert detrended_fluctuation(walk)["alpha"] == pytest.approx(1.5, abs=0.12)


def test_dfa_is_unchanged_by_gain_and_offset_and_validates_scales():
    series = np.random.default_rng(3).standard_normal(1024)
    reference = detrended_fluctuation(series)
    rescaled = detrended_fluctuation(4.0 * series - 7.0)
    assert rescaled["alpha"] == pytest.approx(reference["alpha"], abs=1e-12)
    with pytest.raises(ValueError, match="too short"):
        detrended_fluctuation(np.random.default_rng(4).standard_normal(40))
    with pytest.raises(ValueError, match="three distinct"):
        detrended_fluctuation(series, scales=[16, 16, 32])
    with pytest.raises(ValueError, match="constant"):
        detrended_fluctuation(np.ones(1024))


def test_dfa_scales_start_above_two_periods():
    assert dfa_scale_range(2048, 100.0, 10.0) == (20, 512)
    assert dfa_scale_range(2048, 100.0, 0.3) == (16, 512)
    assert dfa_scale_range(2048, 100.0, 0.0) == (16, 512)


# ------------------------------------------------------------------ drift


def test_frequency_drift_measures_a_sweep_and_not_a_stationary_oscillation():
    sweep = frequency_drift(_chirp(), FS)
    assert sweep["assessed"]
    assert sweep["total_drift_hz"] == pytest.approx(4.0, abs=1.0)
    assert sweep["z"] > 10
    steady = frequency_drift(_oscillator(5), FS)
    assert steady["assessed"]
    assert abs(steady["z"]) < 3
    assert steady["phase_diffusion_rad2_per_s"] == pytest.approx(2.0, rel=0.5)
    slow = frequency_drift(np.sin(2 * np.pi * 0.5 * np.arange(2048) / FS), FS)
    assert not slow["assessed"] and "trimming" in slow["reason"]
    assert not frequency_drift(np.ones(64), FS)["assessed"]


def test_drift_standard_error_matches_brownian_phase_simulation():
    """The z statistic of stationary phase-diffusing oscillators has unit spread."""
    z = [frequency_drift(_oscillator(100 + seed, n=4096), FS)["z"] for seed in range(150)]
    assert np.std(z) == pytest.approx(1.0, abs=0.35)
    assert np.mean(np.abs(z) > 3) < 0.05


# ------------------------------------------------------------------ sampling


def test_regularize_fills_interior_gaps_exactly_for_a_line_and_drops_open_edges():
    values = np.arange(20, dtype=float)
    values[[0, 5, 6, 19]] = np.nan
    filled, report = regularize_series(values)
    assert filled == pytest.approx(np.arange(1, 19, dtype=float))
    assert report["missing_fraction"] == pytest.approx(4 / 20)
    assert report["longest_gap_samples"] == pytest.approx(3.0)
    assert report["dropped_edge_rows"] == 2


def test_regularize_resamples_irregular_timestamps_onto_the_grid():
    rng = np.random.default_rng(6)
    n = 500
    times = np.arange(n) / FS + rng.uniform(-0.3, 0.3, n) / FS
    signal = np.column_stack((np.sin(2 * np.pi * 2.0 * times), np.cos(2 * np.pi * 2.0 * times)))
    regular, report = regularize_series(signal, times, FS)
    grid = times[0] + np.arange(regular.shape[0]) / FS
    # Linear interpolation errs by at most h^2 max|x''| / 8 for spacing h <= 1.6 / FS.
    bound = (1.6 / FS) ** 2 * (2 * np.pi * 2.0) ** 2 / 8
    assert regular[:, 0] == pytest.approx(np.sin(2 * np.pi * 2.0 * grid), abs=bound)
    assert report["interval_cv"] > 0.1
    assert report["longest_gap_seconds"] < 2.0 / FS


@pytest.mark.parametrize(
    "values, timestamps, match",
    [
        (np.array([1.0, np.inf, 2.0]), None, "infinities"),
        (np.array([np.nan, np.nan, np.nan]), None, "observation"),
        (np.ones(3), np.array([0.0, 0.2, 0.1]), "increasing"),
        (np.ones(3), np.array([0.0, 0.1]), "one entry per row"),
    ],
)
def test_regularize_rejects_unusable_input(values, timestamps, match):
    with pytest.raises(ValueError, match=match):
        regularize_series(values, timestamps, FS if timestamps is not None else None)


# ------------------------------------------------------------------ the gate


def test_gate_passes_stationary_oscillators_and_red_noise():
    data = np.column_stack((_oscillator(7), _oscillator(8, frequency=10.3)))
    report = assess_series_quality(data, FS)
    assert not report["abstain"], report["reasons"]
    red = simulate_ar1(2048, 2, 0.9, np.random.default_rng(9))
    assert not assess_series_quality(red, FS)["abstain"]


def test_gate_does_not_call_white_noise_a_drifting_oscillation():
    noise = np.random.default_rng(10).standard_normal((2048, 2))
    report = assess_series_quality(noise, FS)
    assert not report["abstain"], report["reasons"]
    assert not report["channels"]["dim_0"]["frequency_drift"]["assessed"]


def test_gate_abstains_on_random_walks_and_shared_sweeps():
    walks = np.cumsum(np.random.default_rng(11).standard_normal((2048, 2)), axis=0)
    report = assess_series_quality(walks, FS)
    assert report["abstain"]
    assert any("DFA exponent" in reason for reason in report["reasons"])
    sweeps = np.column_stack((_chirp(seed=1), _chirp(seed=2)))
    report = assess_series_quality(sweeps, FS, column_names=("left", "right"))
    assert report["abstain"]
    assert any(reason.startswith("left: dominant frequency drifts") for reason in report["reasons"])


def test_gate_abstains_on_constant_short_or_over_interpolated_data():
    assert "constant series" in assess_series_quality(np.ones((512, 2)), FS)["reasons"][0]
    short = assess_series_quality(np.random.default_rng(12).standard_normal((64, 2)), FS)
    assert short["abstain"] and "samples" in short["reasons"][0]
    data = np.column_stack((_oscillator(13), _oscillator(14)))
    heavy = assess_series_quality(
        data, FS, regularization={"missing_fraction": 0.3, "longest_gap_samples": 2.0}
    )
    assert any("interpolated" in reason for reason in heavy["reasons"])
    gap = assess_series_quality(
        data, FS, regularization={"missing_fraction": 0.05, "longest_gap_samples": 25.0}
    )
    assert any("cycles" in reason for reason in gap["reasons"])
    light = assess_series_quality(
        data, FS, regularization={"missing_fraction": 0.05, "longest_gap_samples": 3.0}
    )
    assert not light["abstain"]
    assert any("interpolated onto a regular grid" in w for w in light["warnings"])


def test_gate_warns_about_aliasing_and_validates_its_inputs():
    t = np.arange(2048) / FS
    near_nyquist = np.column_stack((np.sin(2 * np.pi * 45 * t), np.cos(2 * np.pi * 45 * t)))
    near_nyquist += 0.1 * np.random.default_rng(15).standard_normal(near_nyquist.shape)
    report = assess_series_quality(near_nyquist, FS)
    assert any("aliasing" in warning for warning in report["warnings"])
    with pytest.raises(ValueError, match="unknown thresholds"):
        assess_series_quality(near_nyquist, FS, thresholds={"bogus": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        assess_series_quality(np.array([[1.0, np.nan], [2.0, 3.0]]), FS)
    assert report["thresholds"] == DEFAULT_THRESHOLDS
