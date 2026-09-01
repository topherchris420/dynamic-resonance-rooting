"""Unit tests for BenchmarkSuite and evaluation metrics."""

import pytest
import numpy as np
from drr_framework.evaluation import compute_binary_classification_metrics
from drr_framework.benchmarks_suite import BenchmarkSuite, run_temporal_out_of_sample_benchmark


def test_compute_binary_classification_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    metrics = compute_binary_classification_metrics(y_true, y_score)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_benchmark_suite_methods():
    data = np.random.normal(size=(100, 2))
    vol = BenchmarkSuite.rolling_volatility_indicator(data[:, 0], window_size=20)
    assert len(vol) == 100
    corr = BenchmarkSuite.rolling_correlation_indicator(data, window_size=20)
    assert len(corr) == 100
    var_res = BenchmarkSuite.var_granger_proxy(data)
    assert len(var_res) == 100


def test_run_temporal_out_of_sample_benchmark():
    data = np.random.normal(size=(200, 2))
    events = (np.arange(200) > 150).astype(int)
    results = run_temporal_out_of_sample_benchmark(data, events, window_size=50)
    assert "DRR_Resonance_Depth" in results
    assert "Rolling_Volatility" in results
    assert "Rolling_Correlation" in results
    assert "VAR_Residual_Variance" in results
