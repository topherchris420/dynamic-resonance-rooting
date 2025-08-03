import numpy as np
import pytest
from drr_framework.modules import ResonanceDetector, RootingAnalyzer, DepthCalculator, AnomalyDetector
from drr_framework.benchmarks import BenchmarkSystems
from drr_framework.realtime import RealTimeDRR

def test_resonance_detector():
    """
    Tests the ResonanceDetector module.
    """
    data = np.sin(np.linspace(0, 100, 1000))
    detector = ResonanceDetector()
    result = detector.detect(data, sampling_rate=100)
    assert 'dominant_freq' in result
    assert isinstance(result['dominant_freq'], np.ndarray)

def test_rooting_analyzer():
    """
    Tests the RootingAnalyzer module.
    """
    data = np.random.rand(100, 2)
    analyzer = RootingAnalyzer()
    result = analyzer.analyze(data)
    assert 'transfer_entropy' in result
    assert result['transfer_entropy'].shape == (2, 2)

def test_depth_calculator():
    """
    Tests the DepthCalculator module.
    """
    data = np.random.rand(100)
    calculator = DepthCalculator()
    result = calculator.calculate(data, window_size=10)
    assert 'resonance_depth' in result
    assert isinstance(result['resonance_depth'], float)

def test_anomaly_detector():
    """
    Tests the AnomalyDetector module.
    """
    detector = AnomalyDetector(threshold=0.5)
    assert detector.detect(0.6) is True
    assert detector.detect(0.4) is False

def test_benchmark_systems():
    """
    Tests the BenchmarkSystems module.
    """
    # Test Lorenz system
    t, data = BenchmarkSystems.generate_lorenz_data(duration=1, dt=0.01)
    assert len(t) == 100
    assert data.shape == (100, 3)

    # Test Rossler system
    t, data = BenchmarkSystems.generate_rossler_data(duration=1, dt=0.01)
    assert len(t) == 100
    assert data.shape == (100, 3)

def test_real_time_drr():
    """
    Tests the RealTimeDRR module.
    """
    rt_drr = RealTimeDRR(window_size=10, n_variables=2, threshold=0.5)
    
    # Feed it 10 data points to fill the window
    result = None
    for i in range(10):
        result = rt_drr.process_data_point(np.random.rand(2))
    
    # Should have a result after 10 points
    assert result is not None
    assert len(result) == 2  # 2 variables
    assert 'resonances' in result[0]
    assert 'depth' in result[0]
    assert 'anomaly' in result[0]

def test_real_time_drr_single_variable():
    """
    Test RealTimeDRR with single variable (scalar input).
    """
    rt_drr = RealTimeDRR(window_size=5, n_variables=1, threshold=0.5)
    
    # Feed scalar values
    result = None
    for i in range(5):
        result = rt_drr.process_data_point(np.random.rand())
    
    assert result is not None
    assert len(result) == 1  # 1 variable
