import numpy as np
from collections import deque
from .modules import ResonanceDetector, DepthCalculator, AnomalyDetector

class RealTimeDRR:
    """
    Performs real-time DRR analysis on streaming data.
    """

    def __init__(self, window_size: int, n_variables: int = 1, threshold: float = 0.5):
        """
        Initializes the real-time DRR analyzer.

        Args:
            window_size (int): The size of the sliding window.
            n_variables (int): The number of variables in the time series.
            threshold (float): The anomaly detection threshold.
        """
        self.window_size = window_size
        self.n_variables = n_variables
        self.data_window = deque(maxlen=window_size)
        self.resonance_detector = ResonanceDetector()
        self.depth_calculator = DepthCalculator()
        self.anomaly_detector = AnomalyDetector(threshold)

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
                    resonances = self.resonance_detector.detect(series)
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
            
            return results
        else:
            return None

    def reset(self):
        """Reset the data window"""
        self.data_window.clear()

    def get_window_data(self):
        """Get current window data as numpy array"""
        if len(self.data_window) == 0:
            return None
        return np.array(self.data_window)
