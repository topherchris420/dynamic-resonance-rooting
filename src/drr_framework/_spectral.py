"""Spectral helpers with a NumPy fallback when SciPy is unavailable."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import find_peaks, hilbert, welch

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    rfft = np.fft.rfft
    rfftfreq = np.fft.rfftfreq

    def find_peaks(
        values: np.ndarray,
        *,
        height: Optional[float] = None,
        prominence: Optional[float] = None,
        **_: Any,
    ) -> Tuple[np.ndarray, dict]:
        """Return local maxima using the subset of SciPy's API used by DRR."""

        array = np.asarray(values, dtype=float)
        if array.size < 3:
            return np.array([], dtype=int), {}

        peak_mask = (array[1:-1] > array[:-2]) & (array[1:-1] >= array[2:])
        peaks = np.nonzero(peak_mask)[0] + 1
        properties: dict[str, np.ndarray] = {}

        if height is not None:
            peaks = peaks[array[peaks] >= float(height)]
            properties["peak_heights"] = array[peaks]

        if prominence is not None:
            prominences = np.array([_simple_prominence(array, peak) for peak in peaks])
            keep = prominences >= float(prominence)
            peaks = peaks[keep]
            properties["prominences"] = prominences[keep]

        return peaks.astype(int), properties

    def welch(
        values: np.ndarray,
        *,
        fs: float = 1.0,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        scaling: str = "density",
        **_: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate power spectral density with Welch's method.

        The fallback intentionally implements the compact one-dimensional API
        used by this package. SciPy remains the preferred backend when present.
        """

        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if fs <= 0:
            raise ValueError("fs must be positive")

        segment_length = int(nperseg or min(256, array.size))
        segment_length = max(1, min(segment_length, array.size))
        overlap = segment_length // 2 if noverlap is None else int(noverlap)
        overlap = min(max(overlap, 0), max(segment_length - 1, 0))
        step = max(1, segment_length - overlap)
        starts = range(0, array.size - segment_length + 1, step)

        window = np.hanning(segment_length) if segment_length > 1 else np.ones(1)
        window_power = max(float(np.sum(window**2)), np.finfo(float).eps)
        scale = window_power
        if scaling == "density":
            scale *= fs
        elif scaling != "spectrum":
            raise ValueError("scaling must be 'density' or 'spectrum'")

        periodograms = []
        for start in starts:
            segment = array[start : start + segment_length]
            centered = segment - np.mean(segment)
            spectrum = np.fft.rfft(centered * window)
            power = (np.abs(spectrum) ** 2) / scale
            if power.size > 1:
                if segment_length % 2 == 0:
                    power[1:-1] *= 2
                else:
                    power[1:] *= 2
            periodograms.append(power)

        psd = np.mean(periodograms, axis=0)
        freqs = np.fft.rfftfreq(segment_length, 1.0 / fs)
        return freqs, psd

    def hilbert(values: np.ndarray, **_: Any) -> np.ndarray:
        """Return the analytic signal using an FFT-domain Hilbert transform."""

        array = np.asarray(values, dtype=float)
        n = array.shape[0]
        if n == 0:
            return np.asarray(array, dtype=complex)

        spectrum = np.fft.fft(array, axis=0)
        multiplier = np.zeros(n)
        if n % 2 == 0:
            multiplier[0] = 1
            multiplier[n // 2] = 1
            multiplier[1 : n // 2] = 2
        else:
            multiplier[0] = 1
            multiplier[1 : (n + 1) // 2] = 2

        shape = [n] + [1] * (array.ndim - 1)
        return np.fft.ifft(spectrum * multiplier.reshape(shape), axis=0)

    def _simple_prominence(values: np.ndarray, peak: int) -> float:
        left_min = float(np.min(values[: peak + 1]))
        right_min = float(np.min(values[peak:]))
        return float(values[peak] - max(left_min, right_min))


__all__ = [
    "SCIPY_AVAILABLE",
    "find_peaks",
    "hilbert",
    "rfft",
    "rfftfreq",
    "welch",
]
