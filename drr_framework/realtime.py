from __future__ import annotations

import numpy as np
from collections import deque
from .modules import ResonanceDetector, DepthCalculator, AnomalyDetector

class RealTimeDRR:
    """
    Performs real-time DRR analysis on streaming data.
    """

    def __init__(
        self,
        window_size: int,
        n_variables: int = 1,
        threshold: float = 0.5,
        resonance_options: dict | None = None,
    ):
        """
        Initializes the real-time DRR analyzer.

        Args:
            window_size (int): The size of the sliding window.
            n_variables (int): The number of variables in the time series.
            threshold (float): The anomaly detection threshold.
            resonance_options (dict | None): Optional keyword arguments forwarded
                to :meth:`ResonanceDetector.detect`, e.g. ``{"method": "markov"}``.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if n_variables <= 0:
            raise ValueError("n_variables must be a positive integer")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be a finite float")

        self.window_size = window_size
        self.n_variables = n_variables
        self.data_window = deque(maxlen=window_size)
        self.resonance_options = resonance_options or {}
        self.resonance_detector = ResonanceDetector()
        self.depth_calculator = DepthCalculator()
        self.anomaly_detector = AnomalyDetector(threshold)
        self.latest_results = None

    def process_data_point(self, data_point: np.ndarray) -> dict:
        """
        Processes a single data point.

        Args:
            data_point: The new data point, shape (n_variables,) or scalar.

        Returns:
            dict: A dictionary of analysis results, including any detected anomalies.
        """
        # Handle scalar input for single variable case
        if np.isscalar(data_point):
            if self.n_variables != 1:
                raise ValueError(f"Expected {self.n_variables} variables, got scalar value")
            data_point = np.array([data_point])
        else:
            data_point = np.asarray(data_point)
            if data_point.shape != (self.n_variables,):
                raise ValueError(f"data_point must have shape ({self.n_variables},), got {data_point.shape}")

        self.data_window.append(data_point)

        if len(self.data_window) == self.window_size:
            data_array = np.array(self.data_window)
            results = []
            
            for i in range(self.n_variables):
                series = data_array[:, i]
                
                try:
                    resonances = self.resonance_detector.detect(series, **self.resonance_options)
                    depth = self.depth_calculator.calculate(series, self.window_size)
                    anomaly = self.anomaly_detector.detect(depth['resonance_depth'])
                    
                    results.append({
                        'resonances': resonances, 
                        'depth': depth, 
                        'anomaly': anomaly
                    })
                except Exception as e:
                    # Handle errors gracefully
                    print(f"Warning: Error processing variable {i}: {e}")
                    results.append({
                        'resonances': {'dominant_freq': np.array([])}, 
                        'depth': {'resonance_depth': 0.0}, 
                        'anomaly': False,
                        'error': str(e)
                    })
            
            self.latest_results = results
            return results
        else:
            return None

    def reset(self):
        """Reset the data window"""
        self.data_window.clear()
        self.latest_results = None

    def get_window_data(self):
        """Get current window data as numpy array"""
        if len(self.data_window) == 0:
            return None
        return np.array(self.data_window)

    def summarize_latest_results(self):
        """Return a compact summary for the latest fully processed window.

        Returns:
            dict | None: None if no completed window exists yet, otherwise a summary
                with per-variable depths, anomaly count, and average depth.
        """
        if self.latest_results is None:
            return None

        depths = [item['depth']['resonance_depth'] for item in self.latest_results]
        anomaly_count = sum(1 for item in self.latest_results if item['anomaly'])
        return {
            'n_variables': self.n_variables,
            'depths': depths,
            'average_depth': float(np.mean(depths)) if depths else 0.0,
            'anomaly_count': anomaly_count,
            'has_anomaly': anomaly_count > 0,
        }
