"""Invariants the resonance-depth score must satisfy for any input.

These are properties, not regression values. A depth definition that breaks one
of them measures the analyst's choice of units or the FFT grid, not the signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.modules import DEPTH_METHOD_VERSION, DepthCalculator, _interpolate_peak
from drr_framework._spectral import welch

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 1024
WINDOW = 256


def _tone(cycles_per_sample: float, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    return np.sin(2 * np.pi * cycles_per_sample * t) + noise * rng.standard_normal(N_SAMPLES)


def _frequency_switch(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    first, second = np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.15 * t)
    return np.where(t < N_SAMPLES // 2, first, second) + 0.05 * rng.standard_normal(N_SAMPLES)


def _components(data: np.ndarray, sampling_rate: float) -> dict:
    result = DepthCalculator().calculate(data, WINDOW, sampling_rate=sampling_rate)
    return {"resonance_depth": result["resonance_depth"], **result["components"]}


@pytest.mark.parametrize("signal", ["tone", "frequency_switch", "noise"])
def test_depth_does_not_depend_on_the_time_unit(signal):
    """Relabeling samples from Hz to cycles per month must not move any score."""
    data = {
        "tone": _tone(0.0655),
        "frequency_switch": _frequency_switch(),
        "noise": np.random.default_rng(3).standard_normal(N_SAMPLES),
    }[signal]
    reference = _components(data, sampling_rate=200.0)
    for sampling_rate in (100.0, 12.0, 1.0, 1.0 / 30.0):
        scores = _components(data, sampling_rate=sampling_rate)
        for name, value in reference.items():
            assert scores[name] == pytest.approx(value, abs=1e-9), (name, sampling_rate)


def test_a_frequency_switch_is_not_persistent_at_low_sampling_rates():
    """The v1 score floored the tolerance at 0.5 Hz, so monthly data always looked persistent."""
    persistence = _components(_frequency_switch(), sampling_rate=1.0)["temporal_persistence"]
    assert persistence == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("bin_fraction", np.linspace(0.0, 1.0, 9))
def test_a_clean_tone_scores_the_same_wherever_it_falls_on_the_fft_grid(bin_fraction):
    cycles = 16 / WINDOW + bin_fraction / WINDOW
    components = _components(_tone(cycles), sampling_rate=1.0)
    assert components["phase_coherence"] > 0.95
    assert components["temporal_persistence"] > 0.95
    assert components["resonance_depth"] > 0.94


def test_depth_ignores_gain_and_offset():
    data = _tone(0.071)
    reference = _components(data, sampling_rate=1.0)
    rescaled = _components(37.5 * data - 4.0, sampling_rate=1.0)
    for name, value in reference.items():
        assert rescaled[name] == pytest.approx(value, abs=1e-9), name


def test_white_noise_sits_well_below_a_tone():
    rng = np.random.default_rng(11)
    noise_depths = [
        _components(rng.standard_normal(N_SAMPLES), sampling_rate=1.0)["resonance_depth"]
        for _ in range(40)
    ]
    tone_depth = _components(_tone(0.09, noise=0.3), sampling_rate=1.0)["resonance_depth"]
    assert max(noise_depths) < 0.35
    assert tone_depth - float(np.mean(noise_depths)) > 0.5


@pytest.mark.parametrize("bin_fraction", np.linspace(-0.45, 0.45, 7))
def test_peak_interpolation_recovers_sub_bin_frequencies(bin_fraction):
    nperseg = 256
    cycles = (20 + bin_fraction) / nperseg
    data = np.sin(2 * np.pi * cycles * np.arange(4 * nperseg))
    freqs, psd = welch(data, fs=1.0, nperseg=nperseg)
    peak = int(np.argmax(psd))
    error_in_bins = (_interpolate_peak(freqs, psd, peak) - cycles) * nperseg
    assert abs(error_in_bins) < 0.1


def test_method_version_names_the_depth_definition():
    result = DepthCalculator().calculate(_tone(0.05), WINDOW, sampling_rate=1.0)
    assert result["method"] == DEPTH_METHOD_VERSION == "drr_composite_v2"


def test_committed_quickstart_outputs_match_a_fresh_run():
    from examples.quickstart_resonance_export import analyze_sample, load_sample

    sampling_rate, data = load_sample()
    fresh = json.loads(json.dumps(analyze_sample(data, sampling_rate), sort_keys=True))
    for path in (
        ROOT / "results" / "expected" / "quickstart_expected_summary.json",
        ROOT / "data" / "processed" / "quickstart_expected_summary.json",
    ):
        committed = json.loads(path.read_text(encoding="utf-8"))
        assert committed.keys() == fresh.keys(), path
        assert committed["significant_edges"] == pytest.approx(fresh["significant_edges"]), path
        assert committed["resonance_depth_components"] == pytest.approx(
            fresh["resonance_depth_components"]
        ), path
        assert committed["resonance_depth"] == pytest.approx(fresh["resonance_depth"]), path


def test_committed_reproduction_summary_matches_a_fresh_run(tmp_path):
    from drr_framework.validation import run_reproduction_experiment

    fresh = run_reproduction_experiment(output_dir=tmp_path, random_state=42)
    committed = json.loads(
        (ROOT / "results" / "reproduction" / "drr_reproduction_summary.json").read_text(
            encoding="utf-8"
        )
    )
    for key in (
        "resonance_depth",
        "resonance_depth_components",
        "significant_edges_count",
        "detected_frequency_hz",
    ):
        assert committed[key] == pytest.approx(fresh[key]), key
