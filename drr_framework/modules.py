import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from pyinform import transfer_entropy

class ResonanceDetector:
    """
    Detects resonances in a time series.
    """

    def detect(self, data: np.ndarray, method: str = 'fft') -> dict:
        """
        Detects resonances using the specified method.
        Args:
            data (np.ndarray): The input time series.
            method (str, optional): The method to use for resonance detection.
                                     Defaults to 'fft'.
        Returns:
            dict: A dictionary of detected resonances.
        """
        if method == 'fft':
            return self._detect_with_fft(data)
        elif method == 'wavelet':
            # Placeholder for wavelet-based detection
            return self._detect_with_wavelet(data)
        else:
            raise ValueError(f"Unknown resonance detection method: {method}")

    def _detect_with_fft(self, data: np.ndarray) -> dict:
        """
        Detects resonances using the Fast Fourier Transform (FFT).
        """
        n = len(data)
        yf = fft(data)
        xf = np.fft.fftfreq(n, 1 / 100)  # Assuming sampling rate of 100 Hz

        # Find peaks in the positive frequency domain
        peaks, _ = find_peaks(np.abs(yf[:n//2]), height=0.1)
        dominant_freq = xf[peaks]

        return {'dominant_freq': dominant_freq}

    def _detect_with_wavelet(self, data: np.ndarray) -> dict:
        """
        Placeholder for wavelet-based resonance detection.
        """
        # In a real implementation, you would use a library like PyWavelets
        print("Wavelet detection is not yet implemented.")
        return {}


class RootingAnalyzer:
    """
    Analyzes the hierarchical and causal dependencies among system components.
    """

    def analyze(self, data: np.ndarray) -> dict:
        """
        Performs rooting analysis using transfer entropy.
        Args:
            data (np.ndarray): The input multivariate time series.
        Returns:
            dict: A dictionary of rooting analysis results.
        """
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Data must be a multivariate time series with at least two variables.")

        n_variables = data.shape[1]
        te_matrix = np.zeros((n_variables, n_variables))

        for i in range(n_variables):
            for j in range(n_variables):
                if i != j:
                    te_matrix[i, j] = transfer_entropy(data[:, i], data[:, j], k=1)

        return {'transfer_entropy': te_matrix}


class DepthCalculator:
    """
    Calculates the resonance depth.
    """

    def calculate(self, data: np.ndarray, resonances: dict) -> dict:
        """
        Calculates the resonance depth for each detected resonance.
        Args:
            data (np.ndarray): The input time series.
            resonances (dict): The detected resonances.
        Returns:
            dict: A dictionary of resonance depths.
        """
        # For now, we'll use the standard deviation of the signal as a simple measure of resonance depth.
        return {'resonance_depth': np.std(data)}
