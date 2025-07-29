import numpy as np
import pytest
from drr_framework.modules import ResonanceDetector, RootingAnalyzer, DepthCalculator
from drr_framework.benchmarks import BenchmarkSystems
from drr_framework.realtime import RealTimeDRR

def test_resonance_detector():
    """
    Tests the ResonanceDetector module.
    """
    data = np.sin(np.linspace(0, 100, 1000))
    detector = ResonanceDetector()
    result = detector.detect(data)
    assert 'dominant_freq' in result
    assert len(result['dominant_freq']) > 0

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
    result = calculator.calculate(data, {})
    assert 'resonance_depth' in result
    assert isinstance(result['resonance_depth'], float)

def test_benchmark_systems():
    """
    Tests the BenchmarkSystems module.
    """
    t, data = BenchmarkSystems.generate_lorenz_data(duration=1, dt=0.01)
    assert len(t) == 100
    assert data.shape == (100, 3)

    t, data = BenchmarkSystems.generate_rossler_data(duration=1, dt=0.01)
    assert len(t) == 100
    assert data.shape == (100, 3)

    t, data = BenchmarkSystems.generate_fitzhugh_nagumo_data(duration=10, dt=0.1)
    assert len(t) == 100
    assert data.shape == (100, 2)

def test_real_time_drr():
    """
    Tests the RealTimeDRR module.
    """
    rt_drr = RealTimeDRR(window_size=10, threshold=0.5)
    for i in range(10):
        result = rt_drr.process_data_point(np.random.rand())
    assert result is not None
    assert 'resonances' in result
    assert 'depth' in result
    assert 'anomaly' in result
