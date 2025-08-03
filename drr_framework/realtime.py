import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union, Tuple
import time
import threading
import warnings
from datetime import datetime

from .modules import ResonanceDetector, DepthCalculator, AnomalyDetector, ValidationError


class RealTimeDRR:
    """
    Enhanced real-time DRR analyzer with buffering, threading, and performance monitoring.
    """

    def __init__(
        self, 
        window_size: int, 
        n_variables: int = 1, 
        threshold: float = 0.5,
        overlap: float = 0.5,
        max_history: int = 10000,
        sampling_rate: float = 100.0
    ):
        """
        Initialize real-time DRR analyzer.

        Args:
            window_size: Size of the sliding window
            n_variables: Number of variables in the time series
            threshold: Anomaly detection threshold
            overlap: Overlap between consecutive windows (0-1)
            max_history: Maximum number of samples to keep in history
            sampling_rate: Expected sampling rate in Hz
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if n_variables <= 0:
            raise ValueError("n_variables must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 and 1")
        
        self.window_size = window_size
        self.n_variables = n_variables
        self.overlap = overlap
        self.step_size = max(1, int(window_size * (1 - overlap)))
        self.sampling_rate = sampling_rate
        
        # Data buffers
        self.data_window = deque(maxlen=window_size)
        self.history = deque(maxlen=max_history)
        
        # Analysis components
        self.resonance_detector = ResonanceDetector()
        self.depth_calculator = DepthCalculator()
        self.anomaly_detector = AnomalyDetector(threshold=threshold, method='zscore')
        
        # State tracking
        self.samples_processed = 0
        self.last_analysis_time = None
        self.analysis_intervals = deque(maxlen=100)  # Track timing
        self.results_history = deque(maxlen=1000)
        
        # Threading support
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Performance metrics
        self.performance_stats = {
            'total_samples': 0,
            'total_analyses': 0,
            'total_anomalies': 0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0
        }

    def process_data_point(self, data_point: np.ndarray) -> Optional[Dict]:
        """
        Process a single data point with enhanced error handling and performance tracking.

        Args:
            data_point: New data point, shape (n_variables,) or scalar

        Returns:
            Optional[Dict]: Analysis results if window is complete, None otherwise
        """
        start_time = time.time()
        
        try:
            # Validate and normalize input
            if np.isscalar(data_point):
                if self.n_variables != 1:
                    raise ValidationError(f"Expected {self.n_variables} variables, got scalar")
                data_point = np.array([data_point])
            else:
                data_point = np.asarray(data_point)
                if data_point.shape != (self.n_variables,):
                    raise ValidationError(f"Expected shape ({self.n_variables},), got {data_point.shape}")
            
            # Check for invalid values
            if np.any(np.isnan(data_point)) or np.any(np.isinf(data_point)):
                warnings.warn("Received invalid data point (NaN or inf), skipping")
                return None
            
            with self._lock:
                # Add to buffers
                self.data_window.append(data_point)
                self.history.append(data_point)
                self.samples_processed += 1
                self.performance_stats['total_samples'] += 1
                
                # Check if we have enough data and should analyze
                if len(self.data_window) == self.window_size:
                    if self.samples_processed % self.step_size == 0:
                        result = self._analyze_current_window()
                        
                        # Update performance stats
                        processing_time = time.time() - start_time
                        self._update_performance_stats(processing_time)
                        
                        return result
                
                return None
                
        except Exception as e:
            warnings.warn(f"Error processing data point: {e}")
            return None

    def _analyze_current_window(self) -> Dict:
        """Analyze the current window of data"""
        data_array = np.array(self.data_window)
        timestamp = datetime.now()
        
        results = {
            'timestamp': timestamp,
            'window_index': self.performance_stats['total_analyses'],
            'variables': []
        }
        
        # Analyze each variable
        for i in range(self.n_variables):
            var_results = self._analyze_variable(data_array[:, i], i)
            results['variables'].append(var_results)
        
        # Overall anomaly detection
        results['overall_anomaly'] = any(var['anomaly'] for var in results['variables'])
        
        # Store results
        self.results_history.append(results)
        self.performance_stats['total_analyses'] += 1
        
        if results['overall_anomaly']:
            self.performance_stats['total_anomalies'] += 1
        
        return results

    def _analyze_variable(self, series: np.ndarray, var_index: int) -> Dict:
        """Analyze a single variable time series"""
        try:
            # Resonance detection
            resonances = self.resonance_detector.detect(
                series, 
                method='welch', 
                sampling_rate=self.sampling_rate
            )
            
            # Depth calculation
            depth_result = self.depth_calculator.calculate(series, self.window_size // 4)
            depth_value = depth_result['resonance_depth']
            
            # Anomaly detection
            anomaly = self.anomaly_detector.detect(depth_value)
            
            return {
                'variable_index': var_index,
                'resonances': resonances,
                'depth': depth_result,
                'anomaly': bool(anomaly),
                'depth_value': float(depth_value)
            }
            
        except Exception as e:
            warnings.warn(f"Error analyzing variable {var_index}: {e}")
            return {
                'variable_index': var_index,
                'resonances': {'dominant_freq': np.array([]), 'power': np.array([])},
                'depth': {'resonance_depth': 0.0},
                'anomaly': False,
                'depth_value': 0.0,
                'error': str(e)
            }

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.analysis_intervals.append(processing_time)
        
        self.performance_stats['avg_processing_time'] = np.mean(self.analysis_intervals)
        self.performance_stats['max_processing_time'] = max(
            self.performance_stats['max_processing_time'], 
            processing_time
        )

    def get_recent_results(self, n: int = 10) -> List[Dict]:
        """Get the n most recent analysis results"""
        with self._lock:
            return list(self.results_history)[-n:]

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        with self._lock:
            stats = self.performance_stats.copy()
            
            # Add derived metrics
            if stats['total_analyses'] > 0:
                stats['anomaly_rate'] = stats['total_anomalies'] / stats['total_analyses']
                stats['samples_per_analysis'] = stats['total_samples'] / stats['total_analyses']
            else:
                stats['anomaly_rate'] = 0.0
                stats['samples_per_analysis'] = 0.0
            
            stats['buffer_usage'] = len(self.data_window) / self.window_size
            stats['history_usage'] = len(self.history) / self.history.maxlen
            
            return stats

    def reset(self):
        """Reset all buffers and statistics"""
        with self._lock:
            self.data_window.clear()
            self.history.clear()
            self.results_history.clear()
            self.analysis_intervals.clear()
            
            self.samples_processed = 0
            self.last_analysis_time = None
            
            # Reset performance stats
            self.performance_stats = {
                'total_samples': 0,
                'total_analyses': 0,
                'total_anomalies': 0,
                'avg_processing_time': 0.0,
                'max_processing_time': 0.0
            }
            
            # Reset detector histories
            self.anomaly_detector.reset_history()

    def process_batch(self, data_batch: np.ndarray) -> List[Dict]:
        """
        Process a batch of data points.
        
        Args:
            data_batch: Batch of data points, shape (n_samples, n_variables)
            
        Returns:
            List[Dict]: List of analysis results
        """
        if data_batch.ndim == 1 and self.n_variables == 1:
            data_batch = data_batch.reshape(-1, 1)
        elif data_batch.ndim == 1:
            raise ValidationError(f"Expected multivariate data with {self.n_variables} variables")
        
        results = []
        for i in range(len(data_batch)):
            result = self.process_data_point(data_batch[i])
            if result is not None:
                results.append(result)
        
        return results

    def start_continuous_monitoring(self, data_generator, callback=None):
        """
        Start continuous monitoring with a data generator.
        
        Args:
            data_generator: Generator that yields data points
            callback: Optional callback function for results
        """
        def monitor():
            try:
                for data_point in data_generator:
                    if self._stop_event.is_set():
                        break
                    
                    result = self.process_data_point(data_point)
                    if result is not None and callback is not None:
                        callback(result)
                        
            except Exception as e:
                warnings.warn(f"Monitoring stopped due to error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._stop_event.set()
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


class StreamingDRRAnalyzer:
    """
    High-level streaming analyzer that manages multiple real-time DRR instances.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize streaming analyzer from configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analyzers = {}
        self.global_stats = {
            'start_time': datetime.now(),
            'total_streams': 0,
            'active_streams': 0
        }

    def add_stream(self, stream_id: str, **kwargs) -> RealTimeDRR:
        """Add a new data stream for analysis"""
        if stream_id in self.analyzers:
            raise ValueError(f"Stream {stream_id} already exists")
        
        # Merge config with kwargs
        stream_config = self.config.copy()
        stream_config.update(kwargs)
        
        analyzer = RealTimeDRR(**stream_config)
        self.analyzers[stream_id] = analyzer
        
        self.global_stats['total_streams'] += 1
        self.global_stats['active_streams'] += 1
        
        return analyzer

    def remove_stream(self, stream_id: str):
        """Remove a data stream"""
        if stream_id in self.analyzers:
            self.analyzers[stream_id].stop_monitoring()
            del self.analyzers[stream_id]
            self.global_stats['active_streams'] -= 1

    def get_all_stats(self) -> Dict:
        """Get statistics from all streams"""
        stats = {
            'global': self.global_stats,
            'streams': {}
        }
        
        for stream_id, analyzer in self.analyzers.items():
            stats['streams'][stream_id] = analyzer.get_performance_stats()
        
        return stats

    def reset_all(self):
        """Reset all analyzers"""
        for analyzer in self.analyzers.values():
            analyzer.reset()

    def __del__(self):
        """Cleanup on deletion"""
        for analyzer in self.analyzers.values():
            analyzer.stop_monitoring()            raise ValueError(f"data_point must have shape ({self.n_variables},)")

        self.data_window.append(data_point)

        if len(self.data_window) == self.window_size:
            data_array = np.array(self.data_window)
            results = []
            for i in range(self.n_variables):
                series = data_array[:, i]
                resonances = self.resonance_detector.detect(series)
                depth = self.depth_calculator.calculate(series, self.window_size)
                anomaly = self.anomaly_detector.detect(depth['resonance_depth'])
                results.append({'resonances': resonances, 'depth': depth, 'anomaly': anomaly})
            return results
        else:
            return None
