import numpy as np
from collections import deque
from drr_framework.modules import ResonanceDetector, DepthCalculator

class RealTimeDRR:
    """
    Performs real-time DRR analysis on streaming data.
    """

    def __init__(self, window_size: int, threshold: float):
        """
        Initializes the real-time DRR analyzer.

        Args:
            window_size (int): The size of the sliding window.
            threshold (float): The anomaly detection threshold.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.resonance_detector = ResonanceDetector()
        self.depth_calculator = DepthCalculator()

    def process_data_point(self, data_point: float) -> dict:
        """
        Processes a single data point.

        Args:
            data_point (float): The new data point.

        Returns:
            dict: A dictionary of analysis results, including any detected anomalies.
        """
        self.data_window.append(data_point)

        if len(self.data_window) == self.window_size:
            data_array = np.array(self.data_window)
            resonances = self.resonance_detector.detect(data_array)
            depth = self.depth_calculator.calculate(data_array, resonances)

            anomaly = depth['resonance_depth'] > self.threshold
            return {'resonances': resonances, 'depth': depth, 'anomaly': anomaly}
        else:
            return None
