"""Causal ridge-VAR structural innovation scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StructuralSurpriseResult:
    score: np.ndarray
    innovations: np.ndarray
    contributions: np.ndarray
    feature_names: Tuple[str, ...]
    group_contributions: Mapping[str, np.ndarray]


def structural_surprise(
    fingerprints: np.ndarray,
    min_history: int = 20,
    ridge: float = 1e-3,
    covariance_ridge: float = 1e-4,
    feature_names: Optional[Tuple[str, ...]] = None,
    groups: Optional[Mapping[str, Tuple[int, ...]]] = None,
) -> StructuralSurpriseResult:
    """Compute expanding-window, one-step surprise without using future rows."""
    z = np.asarray(fingerprints, dtype=float)
    if z.ndim != 2 or len(z) <= min_history or not np.all(np.isfinite(z)):
        raise ValueError("fingerprints must be a finite 2D array longer than min_history")
    if min_history < 3 or ridge < 0 or covariance_ridge <= 0:
        raise ValueError("invalid regularization or history")
    names = feature_names or tuple(f"feature_{i}" for i in range(z.shape[1]))
    scores = np.full(len(z), np.nan)
    innovations = np.full_like(z, np.nan)
    contributions = np.full_like(z, np.nan)
    for t in range(min_history, len(z)):
        x, y = z[: t - 1], z[1:t]
        design = np.column_stack((x, np.ones(len(x))))
        penalty = np.eye(design.shape[1]) * ridge
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        residuals = y - design @ coefficients
        covariance = residuals.T @ residuals / max(len(residuals) - 1, 1)
        scale = max(float(np.trace(covariance)) / z.shape[1], np.finfo(float).eps)
        covariance += np.eye(z.shape[1]) * covariance_ridge * scale
        innovation = z[t] - np.r_[z[t - 1], 1.0] @ coefficients
        precision_innovation = np.linalg.solve(covariance, innovation)
        component = innovation * precision_innovation
        scores[t] = np.sqrt(max(float(component.sum()), 0.0))
        innovations[t] = innovation
        contributions[t] = component
    grouped = {
        name: np.nansum(contributions[:, indices], axis=1)
        for name, indices in (groups or {}).items()
    }
    return StructuralSurpriseResult(scores, innovations, contributions, tuple(names), grouped)
