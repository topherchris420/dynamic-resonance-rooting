import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks, welch
import pandas as pd
from typing import Dict, Union, Optional
import warnings

# Handle pyinform import gracefully
try:
    from pyinform import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    warnings.warn("pyinform not available. Transfer entropy analysis will be limited.")


class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass


def validate_timeseries(data: np.ndarray, min_length: int = 10, name: str = "data") -> np.ndarray:
    """
    Validate input time series data.
    
    Args:
        data: Input array to validate
        min_length: Minimum required length
        name: Name of the data for error messages
        
    Returns:
        np.ndarray: Validated and potentially converted data
        
    Raises:
        ValidationError: If data is invalid
    """
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except Exception as e:
            raise ValidationError(f"{name} must be convertible to numpy array: {e}")
    
    if data.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if len(data) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} samples, got {len(data)}")
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    
    return data


class ResonanceDetector:
    """
    Enhanced resonance detector with improved error handling and multiple methods.
    """

    def __init__(self, min_frequency: float = 0.1, max_frequency: Optional[float] = None):
        """
        Initialize resonance detector.
        
        Args:
            min_frequency: Minimum frequency to consider (Hz)
            max_frequency: Maximum frequency to consider (Hz). If None, uses Nyquist frequency.
        """
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def detect(self, data: np.ndarray, method: str = 'fft', sampling_rate: float = 100) -> Dict:
        """
        Detect resonances using the specified method.
        
        Args:
            data: Input time series
            method: Detection method ('fft', 'welch', or 'peaks')
            sampling_rate: Sampling rate of the data
            
        Returns:
            Dict: Detected resonances information
            
        Raises:
            ValidationError: If input parameters are invalid
        """
        # Validate inputs
        data = validate_timeseries(data, min_length=50, name="input data")
        
        if sampling_rate <= 0:
            raise ValidationError("sampling_rate must be positive")
        
        if method not in ['fft', 'welch', 'peaks']:
            raise ValidationError(f"Unknown method '{method}'. Use 'fft', 'welch', or 'peaks'")
        
        # Set max frequency if not specified
        nyquist_freq = sampling_rate / 2
        max_freq = min(self.max_frequency or nyquist_freq, nyquist_freq)
        
        try:
            if method == 'fft':
                return self._detect_with_fft(data, sampling_rate, max_freq)
            elif method == 'welch':
                return self._detect_with_welch(data, sampling_rate, max_freq)
            elif method == 'peaks':
                return self._detect_with_peaks(data, sampling_rate)
        except Exception as e:
            warnings.warn(f"Resonance detection failed: {e}")
            return {'dominant_freq': np.array([]), 'power': np.array([]), 'method': method}

    def _detect_with_fft(self, data: np.ndarray, sampling_rate: float, max_freq: float) -> Dict:
        """FFT-based resonance detection"""
        n = len(data)
        yf = fft(data)
        xf = np.fft.fftfreq(n, 1 / sampling_rate)
        
        # Only positive frequencies
        pos_mask = (xf > self.min_frequency) & (xf <= max_freq)
        freqs_pos = xf[pos_mask]
        power_pos = np.abs(yf[pos_mask])**2
        
        if len(power_pos) == 0:
            return {'dominant_freq': np.array([]), 'power': np.array([]), 'method': 'fft'}
        
        # Find peaks
        mean_power = np.mean(power_pos)
        std_power = np.std(power_pos)
        threshold = mean_power + 2 * std_power  # 2-sigma threshold
        
        peaks, properties = find_peaks(power_pos, height=threshold, distance=5)
        
        return {
            'dominant_freq': freqs_pos[peaks],
            'power': power_pos[peaks],
            'method': 'fft',
            'threshold': threshold
        }
    
    def _detect_with_welch(self, data: np.ndarray, sampling_rate: float, max_freq: float) -> Dict:
        """Welch method for power spectral density"""
        nperseg = min(256, len(data) // 4)
        freqs, psd = welch(data, fs=sampling_rate, nperseg=nperseg)
        
        # Filter frequency range
        freq_mask = (freqs >= self.min_frequency) & (freqs <= max_freq)
        freqs_filt = freqs[freq_mask]
        psd_filt = psd[freq_mask]
        
        if len(psd_filt) == 0:
            return {'dominant_freq': np.array([]), 'power': np.array([]), 'method': 'welch'}
        
        # Find peaks
        threshold = np.mean(psd_filt) + 2 * np.std(psd_filt)
        peaks, _ = find_peaks(psd_filt, height=threshold)
        
        return {
            'dominant_freq': freqs_filt[peaks],
            'power': psd_filt[peaks],
            'method': 'welch',
            'psd_full': psd,
            'freqs_full': freqs
        }
    
    def _detect_with_peaks(self, data: np.ndarray, sampling_rate: float) -> Dict:
        """Simple peak detection in time domain"""
        # Find peaks in the signal itself
        mean_val = np.mean(data)
        std_val = np.std(data)
        threshold = mean_val + 1.5 * std_val
        
        peaks, properties = find_peaks(data, height=threshold, distance=10)
        
        if len(peaks) < 2:
            return {'dominant_freq': np.array([]), 'power': np.array([]), 'method': 'peaks'}
        
        # Estimate frequency from peak intervals
        intervals = np.diff(peaks) / sampling_rate
        freq_estimate = 1 / np.mean(intervals) if len(intervals) > 0 else 0
        
        return {
            'dominant_freq': np.array([freq_estimate]),
            'power': np.array([np.mean(data[peaks])]),
            'method': 'peaks',
            'peak_indices': peaks
        }


class RootingAnalyzer:
    """
    Enhanced rooting analyzer for causal relationship mapping.
    """

    def __init__(self, max_lag: int = 5):
        """
        Initialize rooting analyzer.
        
        Args:
            max_lag: Maximum lag to consider for transfer entropy
        """
        self.max_lag = max_lag

    def analyze(self, data: np.ndarray, method: str = 'transfer_entropy') -> Dict:
        """
        Analyze causal relationships between variables.
        
        Args:
            data: Multivariate time series (samples x variables)
            method: Analysis method ('transfer_entropy' or 'correlation')
            
        Returns:
            Dict: Analysis results
            
        Raises:
            ValidationError: If input is invalid
        """
        # Validate input
        data = validate_timeseries(data, min_length=100, name="multivariate data")
        
        if data.ndim != 2:
            raise ValidationError("Data must be 2D (samples x variables)")
        
        if data.shape[1] < 2:
            raise ValidationError("Need at least 2 variables for rooting analysis")
        
        if method == 'transfer_entropy':
            return self._analyze_transfer_entropy(data)
        elif method == 'correlation':
            return self._analyze_correlation(data)
        else:
            raise ValidationError(f"Unknown method '{method}'")

    def _analyze_transfer_entropy(self, data: np.ndarray) -> Dict:
        """Compute transfer entropy matrix"""
        if not PYINFORM_AVAILABLE:
            warnings.warn("pyinform not available, using correlation instead")
            return self._analyze_correlation(data)
        
        n_vars = data.shape[1]
        te_matrix = np.zeros((n_vars, n_vars))
        
        # Discretize data for transfer entropy calculation
        data_discrete = self._discretize_data(data)
        
        try:
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        te_matrix[i, j] = transfer_entropy(
                            data_discrete[:, i], 
                            data_discrete[:, j], 
                            k=1
                        )
        except Exception as e:
            warnings.warn(f"Transfer entropy calculation failed: {e}")
            return self._analyze_correlation(data)
        
        return {
            'transfer_entropy': te_matrix,
            'method': 'transfer_entropy',
            'interpretation': 'Higher values indicate stronger causal influence'
        }
    
    def _analyze_correlation(self, data: np.ndarray) -> Dict:
        """Fallback correlation analysis"""
        corr_matrix = np.corrcoef(data.T)
        
        # Convert to "influence" by taking absolute values and zeroing diagonal
        influence_matrix = np.abs(corr_matrix)
        np.fill_diagonal(influence_matrix, 0)
        
        return {
            'transfer_entropy': influence_matrix,
            'correlation_matrix': corr_matrix,
            'method': 'correlation',
            'interpretation': 'Higher values indicate stronger linear relationships'
        }
    
    def _discretize_data(self, data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous data for information-theoretic measures"""
        data_discrete = np.zeros_like(data, dtype=int)
        
        for i in range(data.shape[1]):
            # Use quantile-based binning for robustness
            _, bin_edges = np.histogram(data[:, i], bins=n_bins)
            data_discrete[:, i] = np.digitize(data[:, i], bin_edges[1:-1])
        
        return data_discrete


class DepthCalculator:
    """
    Enhanced depth calculator with multiple stability metrics.
    """

    def __init__(self, method: str = 'rolling_std'):
        """
        Initialize depth calculator.
        
        Args:
            method: Calculation method ('rolling_std', 'variance', 'persistence')
        """
        if method not in ['rolling_std', 'variance', 'persistence']:
            raise ValueError("method must be 'rolling_std', 'variance', or 'persistence'")
        self.method = method

    def calculate(self, data: np.ndarray, window_size: int) -> Dict:
        """
        Calculate resonance depth for time series.
        
        Args:
            data: Input time series
            window_size: Window size for calculations
            
        Returns:
            Dict: Depth calculation results
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        data = validate_timeseries(data, min_length=window_size*2, name="time series")
        
        if window_size <= 0:
            raise ValidationError("window_size must be positive")
        
        if window_size >= len(data):
            raise ValidationError("window_size must be smaller than data length")
        
        try:
            if self.method == 'rolling_std':
                return self._calculate_rolling_std(data, window_size)
            elif self.method == 'variance':
                return self._calculate_variance(data, window_size)
            elif self.method == 'persistence':
                return self._calculate_persistence(data, window_size)
        except Exception as e:
            warnings.warn(f"Depth calculation failed: {e}")
            return {'resonance_depth': 0.0, 'method': self.method, 'error': str(e)}

    def _calculate_rolling_std(self, data: np.ndarray, window_size: int) -> Dict:
        """Calculate depth using rolling standard deviation"""
        df = pd.Series(data)
        rolling_std = df.rolling(window=window_size, min_periods=1).std()
        
        # Use the final rolling std as the depth measure
        final_std = rolling_std.iloc[-1]
        
        # Normalize by global std for comparability
        global_std = np.std(data)
        normalized_depth = final_std / global_std if global_std > 0 else 0
        
        return {
            'resonance_depth': normalized_depth,
            'rolling_std_series': rolling_std.values,
            'method': 'rolling_std'
        }
    
    def _calculate_variance(self, data: np.ndarray, window_size: int) -> Dict:
        """Calculate depth using windowed variance"""
        variances = []
        
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            variances.append(np.var(window_data))
        
        # Use coefficient of variation of variances as depth measure
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        cv_variance = std_var / mean_var if mean_var > 0 else 0
        
        return {
            'resonance_depth': cv_variance,
            'variance_series': np.array(variances),
            'method': 'variance'
        }
    
    def _calculate_persistence(self, data: np.ndarray, window_size: int) -> Dict:
        """Calculate depth using pattern persistence"""
        # Look for repeating patterns
        n_windows = len(data) - window_size + 1
        windows = np.array([data[i:i + window_size] for i in range(n_windows)])
        
        # Calculate pairwise correlations between windows
        correlations = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                if np.std(windows[i]) > 0 and np.std(windows[j]) > 0:
                    corr = np.corrcoef(windows[i], windows[j])[0, 1]
                    correlations.append(abs(corr))
        
        # High persistence = high average correlation between windows
        persistence = np.mean(correlations) if correlations else 0
        
        return {
            'resonance_depth': persistence,
            'window_correlations': np.array(correlations),
            'method': 'persistence'
        }


class AnomalyDetector:
    """
    Enhanced anomaly detector with multiple detection strategies.
    """

    def __init__(self, method: str = 'threshold', threshold: float = 0.5):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('threshold', 'zscore', 'iqr')
            threshold: Threshold value (interpretation depends on method)
        """
        if method not in ['threshold', 'zscore', 'iqr']:
            raise ValueError("method must be 'threshold', 'zscore', or 'iqr'")
        
        self.method = method
        self.threshold = threshold
        self.history = []

    def detect(self, value: Union[float, np.ndarray], update_history: bool = True) -> Union[bool, np.ndarray]:
        """
        Detect anomalies in input value(s).
        
        Args:
            value: Value(s) to check for anomalies
            update_history: Whether to update internal history
            
        Returns:
            bool or np.ndarray: Anomaly detection results
        """
        if update_history:
            if isinstance(value, (int, float)):
                self.history.append(value)
            else:
                self.history.extend(value.flatten())
        
        if self.method == 'threshold':
            return self._detect_threshold(value)
        elif self.method == 'zscore':
            return self._detect_zscore(value)
        elif self.method == 'iqr':
            return self._detect_iqr(value)
    
    def _detect_threshold(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Simple threshold-based detection"""
        return value > self.threshold
    
    def _detect_zscore(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Z-score based detection using historical data"""
        if len(self.history) < 10:
            # Not enough history, fall back to threshold
            return self._detect_threshold(value)
        
        mean_hist = np.mean(self.history)
        std_hist = np.std(self.history)
        
        if std_hist == 0:
            return False if isinstance(value, (int, float)) else np.zeros_like(value, dtype=bool)
        
        zscore = np.abs(value - mean_hist) / std_hist
        return zscore > self.threshold
    
    def _detect_iqr(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Interquartile range based detection"""
        if len(self.history) < 10:
            return self._detect_threshold(value)
        
        q1 = np.percentile(self.history, 25)
        q3 = np.percentile(self.history, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        return (value < lower_bound) | (value > upper_bound)
    
    def reset_history(self):
        """Reset the internal history"""
        self.history = []
    
    def get_statistics(self) -> Dict:
        """Get statistics about detected values"""
        if not self.history:
            return {'count': 0}
        
        return {
            'count': len(self.history),
            'mean': np.mean(self.history),
            'std': np.std(self.history),
            'min': np.min(self.history),
            'max': np.max(self.history),
            'median': np.median(self.history)
        }        n = len(data)
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

    def calculate(self, data: np.ndarray, window_size: int) -> dict:
        """
        Calculates the resonance depth for each detected resonance.
        Args:
            data (np.ndarray): The input time series.
            window_size (int): The size of the rolling window.
        Returns:
            dict: A dictionary of resonance depths.
        """
        # We'll use a rolling standard deviation to capture changes in volatility.
        rolling_std = pd.Series(data).rolling(window=window_size).std().iloc[-1]
        return {'resonance_depth': rolling_std}


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
