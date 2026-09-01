"""
Falsifiable Hypothesis & Quantitative Benchmark Framework for DRR Evaluation.

This module defines the quantitative evaluation harness used to compare DRR against
conventional quantitative finance, supervisory, and econometric baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QuantitativeMetrics:
    """Standard evaluation metrics for early warning and stability models."""

    lead_time: float  # Advance warning horizon (steps/time before stress event)
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    auroc: float
    auprc: float
    brier_score: float
    regime_stability_score: float  # Variance of metric across historical regimes
    parameter_fragility_score: float  # Normalized drop in score under perturbation

    def to_dict(self) -> Dict[str, float]:
        return {
            "lead_time": float(self.lead_time),
            "false_positive_rate": float(self.false_positive_rate),
            "false_negative_rate": float(self.false_negative_rate),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "auroc": float(self.auroc),
            "auprc": float(self.auprc),
            "brier_score": float(self.brier_score),
            "regime_stability_score": float(self.regime_stability_score),
            "parameter_fragility_score": float(self.parameter_fragility_score),
        }


def compute_binary_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, FPR, FNR, Precision, Recall, and Brier Score without external scikit-learn dependency."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("y_true and y_score must be non-empty and of matching length")

    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    brier = float(np.mean((y_score - y_true) ** 2))

    # Calculate AUROC rank-based approximation
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
        auprc = 0.0
    else:
        # AUROC via Wilcoxon-Mann-Whitney statistic
        ranks = np.argsort(np.argsort(y_score))
        sum_pos_ranks = np.sum(ranks[pos_mask])
        auroc = float((sum_pos_ranks - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))

        # AUPRC via step trapezoidal integration
        order = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[order]
        tp_cum = np.cumsum(y_true_sorted)
        fp_cum = np.cumsum(1 - y_true_sorted)
        recalls = tp_cum / n_pos
        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = np.concatenate(([0.0], recalls))
        precisions = np.concatenate(([1.0], precisions))
        auprc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

    return {
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": float(np.clip(auroc, 0.0, 1.0)),
        "auprc": float(np.clip(auprc, 0.0, 1.0)),
        "brier_score": brier,
    }
