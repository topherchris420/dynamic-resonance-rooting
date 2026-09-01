"""Comprehensive Benchmark Suite & Temporal Leakage Prevention Pipeline for DRR."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator
from .evaluation import compute_binary_classification_metrics

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Benchmark suite evaluating DRR against conventional econometric and baseline early warning models."""

    @staticmethod
    def rolling_volatility_indicator(data: np.ndarray, window_size: int = 64) -> np.ndarray:
        """Rolling volatility standard deviation early warning proxy."""
        n = len(data)
        scores = np.zeros(n)
        for t in range(window_size, n):
            scores[t] = np.std(data[t - window_size : t])
        if n > window_size:
            scores[:window_size] = scores[window_size]
        return scores

    @staticmethod
    def rolling_correlation_indicator(data: np.ndarray, window_size: int = 64) -> np.ndarray:
        """Rolling average cross-correlation early warning proxy for multivariate data."""
        if data.ndim == 1:
            return BenchmarkSuite.rolling_volatility_indicator(data, window_size)
        n, p = data.shape
        scores = np.zeros(n)
        for t in range(window_size, n):
            window = data[t - window_size : t]
            corr = np.corrcoef(window.T)
            np.fill_diagonal(corr, 0)
            scores[t] = np.mean(np.abs(corr))
        if n > window_size:
            scores[:window_size] = scores[window_size]
        return scores

    @staticmethod
    def persistence_baseline(data: np.ndarray) -> np.ndarray:
        """Naive persistence baseline predicting metric value at t as observation at t-1."""
        if data.ndim == 1:
            res = np.roll(data, 1)
            res[0] = data[0]
            return res
        res = np.roll(data, 1, axis=0)
        res[0] = data[0]
        return res

    @staticmethod
    def var_granger_proxy(data: np.ndarray, lag: int = 2) -> np.ndarray:
        """VAR linear residual variance proxy for dynamic coupling change."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, p = data.shape
        scores = np.zeros(n)
        if n <= lag + p:
            return scores

        for t in range(lag + p + 10, n):
            # Strict rolling window to prevent leakage
            win = data[:t]
            y = win[lag:]
            x_cols = []
            for lag_idx in range(1, lag + 1):
                x_cols.append(win[lag - lag_idx : len(win) - lag_idx])
            X = np.hstack(x_cols)
            # Ordinary least squares
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                pred = X[-1] @ beta
                scores[t] = float(np.mean((y[-1] - pred) ** 2))
            except Exception:
                scores[t] = 0.0
        return scores

    @staticmethod
    def logistic_ews(data: np.ndarray, target: np.ndarray, train_ratio: float = 0.5) -> np.ndarray:
        """Regularized logistic regression early-warning baseline using expanding window to prevent look-ahead bias."""
        n = len(data)
        if data.ndim == 1:
            X = data.reshape(-1, 1)
        else:
            X = data

        split_idx = int(n * train_ratio)
        scores = np.zeros(n)

        for t in range(split_idx, n):
            X_train = X[:t]
            y_train = target[:t]

            # Simple Ridge/Sigmoid approximation
            mean_x = np.mean(X_train, axis=0)
            std_x = np.std(X_train, axis=0)
            std_x[std_x == 0] = 1.0
            X_norm = (X_train - mean_x) / std_x

            w = np.linalg.pinv(X_norm.T @ X_norm + 1.0 * np.eye(X.shape[1])) @ (X_norm.T @ y_train)

            x_curr = (X[t] - mean_x) / std_x
            logit = x_curr @ w
            scores[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

        return scores


def run_temporal_out_of_sample_benchmark(
    data: np.ndarray,
    ground_truth_events: np.ndarray,
    window_size: int = 128,
    lead_steps: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Strict expanding window out-of-sample evaluation of DRR vs baselines."""
    n = len(data)
    drr_scores = np.zeros(n)

    detector = ResonanceDetector()
    depth_calc = DepthCalculator()

    # Calculate DRR Resonance Depth over expanding/rolling windows without future leakage
    for t in range(window_size, n):
        win_data = data[t - window_size : t]
        if win_data.ndim > 1:
            series = win_data[:, 0]
        else:
            series = win_data

        try:
            det = detector.detect(series, method="welch", sampling_rate=100.0)
            res_freqs = det.get("dominant_freq", np.array([]))
            depth_res = depth_calc.calculate(
                series,
                window_size=min(window_size, len(series)),
                sampling_rate=100.0,
                resonance_frequencies=res_freqs,
            )
            drr_scores[t] = float(depth_res["resonance_depth"])
        except Exception:
            drr_scores[t] = 0.0

    vol_scores = BenchmarkSuite.rolling_volatility_indicator(
        data[:, 0] if data.ndim > 1 else data, window_size
    )
    corr_scores = BenchmarkSuite.rolling_correlation_indicator(data, window_size)
    var_scores = BenchmarkSuite.var_granger_proxy(data)

    eval_start = window_size
    results = {
        "DRR_Resonance_Depth": compute_binary_classification_metrics(
            ground_truth_events[eval_start:], drr_scores[eval_start:]
        ),
        "Rolling_Volatility": compute_binary_classification_metrics(
            ground_truth_events[eval_start:], vol_scores[eval_start:]
        ),
        "Rolling_Correlation": compute_binary_classification_metrics(
            ground_truth_events[eval_start:], corr_scores[eval_start:]
        ),
        "VAR_Residual_Variance": compute_binary_classification_metrics(
            ground_truth_events[eval_start:], var_scores[eval_start:]
        ),
    }

    return results
