"""Tests for Morlet-wavelet resonance detection."""

import numpy as np
import pytest

from drr_framework.modules import ResonanceDetector


def _closest(detected: np.ndarray, target: float) -> float:
    assert len(detected) > 0
    return float(detected[np.argmin(np.abs(detected - target))])


def test_wavelet_detector_identifies_dominant_frequency():
    sampling_rate = 100.0
    t = np.arange(0, 4.0, 1.0 / sampling_rate)
    rng = np.random.default_rng(0)
    data = np.sin(2 * np.pi * 15.0 * t) + 0.1 * rng.normal(size=len(t))

    result = ResonanceDetector().detect(data, method="wavelet", sampling_rate=sampling_rate)

    assert result["method"] == "wavelet"
    assert np.isclose(result["dominant_freq"][0], 15.0, rtol=0.05)
    assert result["confidence"][0] > 0.5


def test_wavelet_detector_resolves_nonstationary_content():
    """A signal that switches frequency mid-record is the wavelet use case:
    both tones must be detected, and the scalogram must localize them in time.
    """
    sampling_rate = 200.0
    half = np.arange(0, 3.0, 1.0 / sampling_rate)
    data = np.concatenate([np.sin(2 * np.pi * 8.0 * half), np.sin(2 * np.pi * 30.0 * half)])

    result = ResonanceDetector().detect(data, method="wavelet", sampling_rate=sampling_rate)
    detected = result["dominant_freq"]

    assert np.isclose(_closest(detected, 8.0), 8.0, rtol=0.1)
    assert np.isclose(_closest(detected, 30.0), 30.0, rtol=0.1)

    # Time localization: the 8 Hz band should carry its power in the first
    # half of the record and the 30 Hz band in the second half.
    frequencies = result["frequency_axis"]
    scalogram = result["scalogram"]
    assert scalogram.shape == (len(frequencies), len(data))

    mid = len(data) // 2
    low_band = int(np.argmin(np.abs(frequencies - 8.0)))
    high_band = int(np.argmin(np.abs(frequencies - 30.0)))
    assert scalogram[low_band, :mid].mean() > 5 * scalogram[low_band, mid:].mean()
    assert scalogram[high_band, mid:].mean() > 5 * scalogram[high_band, :mid].mean()


def test_wavelet_detector_falls_back_to_fft_for_short_series():
    result = ResonanceDetector().detect(np.random.rand(8), method="wavelet", sampling_rate=100.0)
    assert result["method"] == "fft"


def test_wavelet_detector_validates_inputs():
    detector = ResonanceDetector()

    with pytest.raises(ValueError, match="sampling_rate"):
        detector.detect(np.random.rand(64), method="wavelet", sampling_rate=0)

    with pytest.raises(ValueError, match="peak_height_ratio"):
        detector.detect(
            np.random.rand(64), method="wavelet", sampling_rate=100, peak_height_ratio=2.0
        )


def test_wavelet_detector_handles_constant_signal():
    result = ResonanceDetector().detect(np.ones(128), method="wavelet", sampling_rate=100.0)
    assert len(result["dominant_freq"]) == 0
