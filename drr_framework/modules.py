import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import pandas as pd
from sklearn.cluster import KMeans

# Handle pyinform import gracefully
try:
    from pyinform import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    print("Warning: pyinform not available. Transfer entropy analysis will be limited.")


class MarkovChain:
    """
    A class to represent a first-order Markov chain.
    """

    def __init__(self, states: np.ndarray):
        """
        Initializes the MarkovChain object.
        Args:
            states (np.ndarray): A 1D array of discrete states.
        """
        self.states = states
        self.unique_states = np.unique(states)
        self.n_states = len(self.unique_states)
        self.state_map = {state: i for i, state in enumerate(self.unique_states)}

        self.transition_matrix = self._calculate_transition_matrix()
        self.stationary_distribution = self._calculate_stationary_distribution()

    def _calculate_transition_matrix(self) -> np.ndarray:
        """
        Calculates the transition matrix from the sequence of states.
        Returns:
            np.ndarray: The transition matrix.
        """
        matrix = np.zeros((self.n_states, self.n_states))
        for i in range(len(self.states) - 1):
            current_state_idx = self.state_map[self.states[i]]
            next_state_idx = self.state_map[self.states[i+1]]
            matrix[current_state_idx, next_state_idx] += 1

        # Normalize rows to get probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for states that are not visited
        row_sums[row_sums == 0] = 1
        return matrix / row_sums

    def _calculate_stationary_distribution(self) -> np.ndarray:
        """
        Calculates the stationary distribution of the Markov chain.
        This is the left eigenvector of the transition matrix with eigenvalue 1.
        Returns:
            np.ndarray: The stationary distribution.
        """
        try:
            # We need to find the eigenvector of the transposed matrix
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

            # Find the eigenvector corresponding to the eigenvalue 1
            one_eigenvalue_idx = np.isclose(eigenvalues, 1)

            if np.any(one_eigenvalue_idx):
                stationary_vector = eigenvectors[:, one_eigenvalue_idx].flatten().real
                # Normalize to get a probability distribution
                return stationary_vector / stationary_vector.sum()
            else:
                # Fallback for chains that may not have a clear stationary distribution
                # (e.g., periodic chains)
                return np.ones(self.n_states) / self.n_states
        except np.linalg.LinAlgError:
            # Fallback if eigenvalue decomposition fails
            return np.ones(self.n_states) / self.n_states


class ResonanceDetector:
    """
    Detects resonances in a time series.
    """

    def detect(
        self,
        data: np.ndarray,
        method: str = 'fft',
        sampling_rate: int = 100,
        n_clusters: int = 4,
        peak_height_ratio: float = 0.1,
    ) -> dict:
        """
        Detects resonances using the specified method.
        Args:
            data (np.ndarray): The input time series.
            method (str, optional): The method to use for resonance detection.
                                     Defaults to 'fft'.
            sampling_rate (int, optional): The sampling rate of the time series.
                                           Defaults to 100.
            n_clusters (int, optional): The number of clusters to use for the Markov method.
                                        Defaults to 4.
        Returns:
            dict: A dictionary of detected resonances.
        """
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError("data must not be empty")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains non-finite values")

        if method == 'fft':
            return self._detect_with_fft(data, sampling_rate, peak_height_ratio)
        elif method == 'wavelet':
            # Placeholder for wavelet-based detection
            return self._detect_with_wavelet(data)
        elif method == 'markov':
            return self._detect_with_markov(data, n_clusters)
        else:
            raise ValueError(f"Unknown resonance detection method: {method}")

    def _detect_with_fft(self, data: np.ndarray, sampling_rate: int, peak_height_ratio: float) -> dict:
        """
        Detects resonances using the Fast Fourier Transform (FFT).
        """
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if not 0 < peak_height_ratio <= 1:
            raise ValueError("peak_height_ratio must be in the interval (0, 1]")

        n = len(data)
        yf = fft(data)
        xf = np.fft.fftfreq(n, 1 / sampling_rate)
        magnitudes = np.abs(yf)

        # Find peaks in the positive frequency domain
        peaks, _ = find_peaks(magnitudes[:n//2], height=np.max(magnitudes[:n//2]) * peak_height_ratio)
        dominant_freq = xf[peaks]
        peak_magnitudes = magnitudes[peaks]

        # Sort peaks by magnitude (largest first) for easier downstream usage.
        if len(peak_magnitudes) > 0:
            order = np.argsort(peak_magnitudes)[::-1]
            dominant_freq = dominant_freq[order]
            peak_magnitudes = peak_magnitudes[order]

        return {
            'dominant_freq': dominant_freq,
            'peak_magnitude': peak_magnitudes,
        }

    def _detect_with_wavelet(self, data: np.ndarray) -> dict:
        """
        Placeholder for wavelet-based resonance detection.
        """
        # In a real implementation, you would use a library like PyWavelets
        print("Wavelet detection is not yet implemented.")
        return {'dominant_freq': np.array([])}

    def _detect_with_markov(self, data: np.ndarray, n_clusters: int) -> dict:
        """
        Detects resonances using a Markov chain analysis of clustered states.
        """
        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) < n_clusters:
            raise ValueError("n_clusters cannot exceed number of observations")

        # 1. Cluster the data to get discrete states
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        states = kmeans.fit_predict(data)

        # 2. Create and analyze the Markov chain
        mc = MarkovChain(states)

        # 3. Identify "resonance"
        # Here, we define resonance as states with high self-transition probability
        resonant_states = np.where(np.diag(mc.transition_matrix) > 0.9)[0]

        return {
            'transition_matrix': mc.transition_matrix,
            'stationary_distribution': mc.stationary_distribution,
            'resonant_states': resonant_states,
            'clusters': states
        }


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
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")

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
