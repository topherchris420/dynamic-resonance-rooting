import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import pandas as pd

# Handle pyinform import gracefully
try:
    from pyinform import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    print("Warning: pyinform not available. Transfer entropy analysis will be limited.")


class ResonanceDetector:
    """
    Detects resonances in a time series.
    """

    def detect(self, data: np.ndarray, method: str = 'fft', sampling_rate: int = 100) -> dict:
        """
        Detects resonances using the specified method.
        Args:
            data (np.ndarray): The input time series.
            method (str, optional): The method to use for resonance detection.
                                     Defaults to 'fft'.
            sampling_rate (int, optional): The sampling rate of the time series.
                                           Defaults to 100.
        Returns:
            dict: A dictionary of detected resonances.
        """
        if method == 'fft':
            return self._detect_with_fft(data, sampling_rate)
        elif method == 'wavelet':
            # Placeholder for wavelet-based detection
            return self._detect_with_wavelet(data)
        else:
            raise ValueError(f"Unknown resonance detection method: {method}")

    def _detect_with_fft(self, data: np.ndarray, sampling_rate: int) -> dict:
        """
        Detects resonances using the Fast Fourier Transform (FFT).
        """
        n = len(data)
        yf = fft(data)
        xf = np.fft.fftfreq(n, 1 / sampling_rate)

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
        return {'dominant_freq': np.array([])}


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

        if PYINFORM_AVAILABLE:
            # Discretize data for transfer entropy
            data_discrete = self._discretize_data(data)
            
            for i in range(n_variables):
                for j in range(n_variables):
                    if i != j:
                        try:
                            te_matrix[i, j] = transfer_entropy(data_discrete[:, i], data_discrete[:, j], k=1)
                        except Exception as e:
                            print(f"Warning: Transfer entropy calculation failed for pair ({i},{j}): {e}")
                            te_matrix[i, j] = 0
        else:
            # Fallback to correlation-based analysis
            print("Using correlation-based analysis instead of transfer entropy")
            corr_matrix = np.corrcoef(data.T)
            te_matrix = np.abs(corr_matrix)
            np.fill_diagonal(te_matrix, 0)

        return {'transfer_entropy': te_matrix}

    def _discretize_data(self, data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """
        Discretize continuous data for information-theoretic measures.
        """
        data_discrete = np.zeros_like(data, dtype=int)
        
        for i in range(data.shape[1]):
            # Use quantile-based binning for robustness
            _, bin_edges = np.histogram(data[:, i], bins=n_bins)
            data_discrete[:, i] = np.digitize(data[:, i], bin_edges[1:-1])
        
        return data_discrete


class DepthCalculator:
    """
    Calculates the resonance depth.
    """

    def calculate(self, data: np.ndarray, window_size: int) -> dict:
        """
        Calculates the resonance depth for each detected resonance.
        Args:
            data (np.ndarray): The input time series.
            window_size (int): The size of the rolling window.
        Returns:
            dict: A dictionary of resonance depths.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) < window_size:
            return {'resonance_depth': 0.0}
        
        # We'll use a rolling standard deviation to capture changes in volatility.
        try:
            rolling_std = pd.Series(data).rolling(window=window_size, min_periods=1).std().iloc[-1]
            if pd.isna(rolling_std):
                rolling_std = 0.0
        except Exception as e:
            print(f"Warning: Error calculating rolling std: {e}")
            rolling_std = np.std(data) if len(data) > 0 else 0.0
        
        return {'resonance_depth': float(rolling_std)}


class AnomalyDetector:
    """
    Detects anomalies based on resonance depth.
    """

    def __init__(self, threshold: float):
        """
        Initializes the anomaly detector.
        Args:
            threshold (float): The anomaly detection threshold.
        """
        self.threshold = threshold

    def detect(self, resonance_depth: float) -> bool:
        """
        Detects an anomaly based on the resonance depth.
        Args:
            resonance_depth (float): The calculated resonance depth.
        Returns:
            bool: True if an anomaly is detected, False otherwise.
        """
        return resonance_depth > self.threshold

    def reset_history(self):
        """Reset any internal state (placeholder for compatibility)"""
        pass
