
import pytest
import numpy as np
from drr_framework.modules import ResonanceDetector

def test_resonance_detector_accuracy():
    """
    Verify that ResonanceDetector correctly identifies the dominant frequency.
    This test should pass regardless of whether fft or rfft is used.
    """
    sampling_rate = 100.0
    duration = 2.0
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    target_freq = 15.0
    data = np.sin(2 * np.pi * target_freq * t)

    detector = ResonanceDetector()
    result = detector.detect(data, sampling_rate=sampling_rate)

    dominant_freqs = result['dominant_freq']
    assert len(dominant_freqs) > 0
    # The dominant frequency should be close to 15.0
    # FFT frequency resolution is fs/N = 100/200 = 0.5 Hz
    assert np.isclose(dominant_freqs[0], target_freq, atol=0.5)

if __name__ == "__main__":
    test_resonance_detector_accuracy()
