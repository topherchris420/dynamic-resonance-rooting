import numpy as np
import pytest
from drr_framework.modules import (
    MarkovChain,
    ResonanceDetector,
    RootingAnalyzer,
    DepthCalculator,
    AnomalyDetector,
)
from drr_framework.benchmarks import BenchmarkSystems
from drr_framework.realtime import RealTimeDRR


def test_resonance_detector():
    """
    Tests the ResonanceDetector module.
    """
    data = np.sin(np.linspace(0, 100, 1000))
    detector = ResonanceDetector()
    result = detector.detect(data, sampling_rate=100)
    assert "dominant_freq" in result
    assert "peak_magnitude" in result
    assert isinstance(result["dominant_freq"], np.ndarray)
    assert isinstance(result["peak_magnitude"], np.ndarray)


def test_resonance_detector_validates_inputs():
    detector = ResonanceDetector()

    with pytest.raises(ValueError, match="empty"):
        detector.detect(np.array([]))

    with pytest.raises(ValueError, match="non-finite"):
        detector.detect(np.array([0.0, np.nan, 1.0]))

    with pytest.raises(ValueError, match="sampling_rate"):
        detector.detect(np.array([0.0, 1.0, 0.0]), sampling_rate=0)

    with pytest.raises(ValueError, match="peak_height_ratio"):
        detector.detect(np.array([0.0, 1.0, 0.0]), peak_height_ratio=1.5)


def test_markov_detector_validates_cluster_configuration():
    detector = ResonanceDetector()

    with pytest.raises(ValueError, match="greater than 1"):
        detector.detect(np.random.rand(10), method="markov", n_clusters=1)

    with pytest.raises(ValueError, match="cannot exceed"):
        detector.detect(np.random.rand(3), method="markov", n_clusters=4)


def test_wavelet_detector_returns_standard_schema():
    detector = ResonanceDetector()
    result = detector.detect(np.random.rand(64), method="wavelet", sampling_rate=100)

    assert result["method"] == "wavelet"
    for key in ("dominant_freq", "peak_magnitude", "confidence", "spectrum", "scalogram"):
        assert key in result


def test_rooting_analyzer():
    """
    Tests the RootingAnalyzer module.
    """
    data = np.random.rand(100, 2)
    analyzer = RootingAnalyzer()
    result = analyzer.analyze(data)
    assert "transfer_entropy" in result
    assert result["transfer_entropy"].shape == (2, 2)


def test_lagged_correlation_matches_naive_reference():
    """
    The vectorized lagged-correlation scores must match a per-pair
    corrcoef reference, including constant (zero-variance) columns.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(size=(120, 4))
    data[:, 2] = data[:, 2] * 0 + 3.0  # constant column
    data[5:, 3] = 0.8 * data[:-5, 0] + 0.1 * rng.normal(size=115)  # lagged driver
    max_lag = 6

    analyzer = RootingAnalyzer()
    scores, lags = analyzer._lagged_correlation_scores(data, max_lag)

    eps = np.finfo(float).eps
    n_variables = data.shape[1]
    for source in range(n_variables):
        for target in range(n_variables):
            if source == target:
                continue
            best_score, best_lag = 0.0, 1
            for lag in range(1, max_lag + 1):
                x = data[:-lag, source]
                y = data[lag:, target]
                if np.std(x) <= eps or np.std(y) <= eps:
                    score = 0.0
                else:
                    score = abs(float(np.corrcoef(x, y)[0, 1]))
                if score > best_score:
                    best_score, best_lag = score, lag
            assert np.isclose(scores[source, target], best_score, atol=1e-10)
            assert lags[source, target] == best_lag

    # The injected lag-5 dependency should be recovered.
    assert lags[0, 3] == 5
    assert scores[0, 3] > 0.9


def test_markov_chain_stationary_distribution_reducible_chain():
    """
    A transition matrix with eigenvalue 1 of multiplicity > 1 (two absorbing
    states) must still yield a valid distribution of shape (n_states,).
    """
    mc = MarkovChain(np.array([0, 1, 0, 1]))
    mc.transition_matrix = np.eye(2)  # reducible: both states absorbing
    distribution = mc._calculate_stationary_distribution()

    assert distribution.shape == (2,)
    assert np.isclose(distribution.sum(), 1.0)
    assert np.all(distribution >= 0)


def test_depth_calculator():
    """
    Tests the DepthCalculator module.
    """
    data = np.random.rand(100)
    calculator = DepthCalculator()
    result = calculator.calculate(data, window_size=10)
    assert "resonance_depth" in result
    assert isinstance(result["resonance_depth"], float)


def test_depth_calculator_rejects_non_positive_window():
    calculator = DepthCalculator()
    with pytest.raises(ValueError, match="positive"):
        calculator.calculate(np.random.rand(100), window_size=0)


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
    assert "resonances" in result[0]
    assert "depth" in result[0]
    assert "anomaly" in result[0]


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


def test_real_time_drr_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="window_size"):
        RealTimeDRR(window_size=0)

    with pytest.raises(ValueError, match="n_variables"):
        RealTimeDRR(window_size=5, n_variables=0)

    with pytest.raises(ValueError, match="threshold"):
        RealTimeDRR(window_size=5, n_variables=1, threshold=1.5)

    with pytest.raises(ValueError, match="analysis_interval"):
        RealTimeDRR(window_size=5, n_variables=1, analysis_interval=0)


def test_real_time_drr_analysis_interval_throttles_updates():
    """
    With analysis_interval=3 and window_size=5, results should come back on
    points 5, 8, and 11 only.
    """
    rt_drr = RealTimeDRR(window_size=5, n_variables=1, analysis_interval=3)

    analyzed_points = [
        i for i in range(1, 12) if rt_drr.process_data_point(np.random.rand()) is not None
    ]
    assert analyzed_points == [5, 8, 11]


def test_heston_benchmark_is_reproducible_with_seed():
    t1, data1 = BenchmarkSystems.generate_heston_data(duration=1, dt=1 / 50, random_state=42)
    t2, data2 = BenchmarkSystems.generate_heston_data(duration=1, dt=1 / 50, random_state=42)

    assert data1.shape == (50, 2)
    np.testing.assert_allclose(data1, data2)
    assert np.all(data1[:, 0] > 0)  # prices stay positive
    assert np.all(data1[:, 1] >= 0)  # variance stays non-negative


def test_drr_framework_rejects_invalid_configuration():
    from drr_framework.analysis import DynamicResonanceRooting

    with pytest.raises(ValueError, match="embedding_dim"):
        DynamicResonanceRooting(embedding_dim=0)

    with pytest.raises(ValueError, match="tau"):
        DynamicResonanceRooting(tau=0)

    with pytest.raises(ValueError, match="sampling_rate"):
        DynamicResonanceRooting(sampling_rate=0)


def test_analyze_system_is_deterministic_across_repeated_calls():
    """
    Re-analyzing the same data on the same instance must not carry belief
    state over from the previous run.
    """
    from drr_framework.analysis import DynamicResonanceRooting

    t = np.linspace(0, 10, 1000, endpoint=False)
    data = np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)])

    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=100.0)
    first = drr.analyze_system(data, multivariate=True, window_size=128)
    second = drr.analyze_system(data, multivariate=True, window_size=128)

    assert first["agent_belief"] == second["agent_belief"]
    assert first["is_rooted"] == second["is_rooted"]


def test_analyze_system_passes_spectral_method_through():
    """
    The spectral method selected in analyze_system must reach the detector
    instead of silently falling back to FFT.
    """
    from drr_framework.analysis import DynamicResonanceRooting

    sampling_rate = 100.0
    t = np.linspace(0, 10, 1000, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t)

    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)

    seen_methods = []
    original_detect = drr.resonance_detector.detect

    def spying_detect(*args, **kwargs):
        seen_methods.append(kwargs.get("method"))
        return original_detect(*args, **kwargs)

    drr.resonance_detector.detect = spying_detect
    results = drr.analyze_system(data, window_size=128, method="welch")

    assert seen_methods and set(seen_methods) == {"welch"}
    assert np.isclose(results["resonances"]["dim_0"]["dominant_freq"], 5.0, atol=0.5)


def test_analyze_system_propagates_errors():
    """
    Invalid input must raise instead of silently returning an empty dict.
    """
    from drr_framework.analysis import DynamicResonanceRooting

    drr = DynamicResonanceRooting(embedding_dim=3, tau=2)
    with pytest.raises(ValueError, match="too short"):
        drr.analyze_system(np.array([1.0, 2.0]))
