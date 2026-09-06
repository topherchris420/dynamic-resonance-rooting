"""Positive-semidefinite kernels from structural resonance loadings."""

from __future__ import annotations

import numpy as np


def resonance_risk_kernel(loadings: np.ndarray, center: bool = False) -> np.ndarray:
    """Return a unit-diagonal Gram kernel describing shared DRR exposures."""
    features = np.asarray(loadings, dtype=float)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("loadings must be a finite asset-by-feature matrix")
    if center:
        features = features - features.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    return normalized @ normalized.T


def resonance_regularized_covariance(
    covariance: np.ndarray, kernel: np.ndarray, strength: float
) -> np.ndarray:
    """Add ``strength * D K D`` while preserving covariance units and PSD."""
    sigma, kernel = np.asarray(covariance, float), np.asarray(kernel, float)
    if sigma.shape != kernel.shape or sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance and kernel must be equal-size square matrices")
    if strength < 0 or not np.allclose(kernel, kernel.T):
        raise ValueError("strength must be non-negative and kernel symmetric")
    if np.linalg.eigvalsh(kernel).min() < -1e-10:
        raise ValueError("kernel must be positive semidefinite")
    sigma = 0.5 * (sigma + sigma.T)
    volatility = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    adjusted = sigma + strength * (volatility[:, None] * kernel * volatility[None, :])
    return 0.5 * (adjusted + adjusted.T)
