
import pytest
import numpy as np
from drr_framework.analysis import DynamicResonanceRooting
from scipy.fft import fft, rfft, fftfreq, rfftfreq

def test_rfft_equivalence():
    """
    Verify that the rfft implementation produces the same results as fft
    for positive frequencies on real-valued data.
    """
    sampling_rate = 100.0
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration))
    # Create a complex signal
    data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

    # 1. FFT
    fft_vals = fft(data)
    freqs = fftfreq(len(data), 1 / sampling_rate)
    power_spectrum = np.abs(fft_vals)**2

    pos_mask = freqs > 0
    freqs_fft = freqs[pos_mask]
    power_fft = power_spectrum[pos_mask]

    # 2. RFFT
    rfft_vals = rfft(data)
    rfreqs = rfftfreq(len(data), 1 / sampling_rate)
    rpower_spectrum = np.abs(rfft_vals)**2

    rpos_mask = rfreqs > 0
    freqs_rfft = rfreqs[rpos_mask]
    power_rfft = rpower_spectrum[rpos_mask]

    # Compare common frequencies (ignoring Nyquist if mismatched)
    # intersection of frequencies
    common_freqs = np.intersect1d(freqs_fft, freqs_rfft)

    # Filter both to common frequencies
    mask_fft = np.isin(freqs_fft, common_freqs)
    mask_rfft = np.isin(freqs_rfft, common_freqs)

    np.testing.assert_allclose(freqs_fft[mask_fft], freqs_rfft[mask_rfft], err_msg="Frequencies do not match")
    np.testing.assert_allclose(power_fft[mask_fft], power_rfft[mask_rfft], err_msg="Power spectrum does not match")

def test_drr_detect_resonances_output():
    """
    Verify that detect_resonances runs without error and returns expected keys.
    """
    sampling_rate = 100.0
    data = np.random.normal(0, 1, 1000)
    drr = DynamicResonanceRooting(sampling_rate=sampling_rate)

    resonances = drr.detect_resonances(data)

    assert isinstance(resonances, dict)
    assert 'dim_0' in resonances
    assert 'frequencies' in resonances['dim_0']
    assert 'power' in resonances['dim_0']
