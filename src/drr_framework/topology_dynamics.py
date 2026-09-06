"""Temporal statistics for sequences of directed rooting-score matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _adjacency(matrix: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(matrix, dtype=float), 0.0).copy()
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError("adjacency must be a finite square matrix")
    np.fill_diagonal(value, 0.0)
    return value


def root_distribution(adjacency: np.ndarray) -> np.ndarray:
    """Normalize non-negative outgoing influence; use uniform mass if empty."""
    outflow = _adjacency(adjacency).sum(axis=1)
    total = float(outflow.sum())
    return outflow / total if total > 0 else np.full(len(outflow), 1.0 / len(outflow))


def root_migration(previous: np.ndarray, current: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence of consecutive root distributions [0, 1]."""
    p, q = root_distribution(previous), root_distribution(current)
    if p.shape != q.shape:
        raise ValueError("graphs must have equal size")
    midpoint = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / midpoint[mask])))

    return float(np.clip(0.5 * (kl(p) + kl(q)), 0.0, 1.0))


def topology_drift(previous: np.ndarray, current: np.ndarray) -> float:
    """Scale-invariant angular Frobenius distance in [0, 1]."""
    p, q = _adjacency(previous), _adjacency(current)
    if p.shape != q.shape:
        raise ValueError("graphs must have equal size")
    if np.array_equal(p, q):
        return 0.0
    pn, qn = np.linalg.norm(p), np.linalg.norm(q)
    if pn == 0 and qn == 0:
        return 0.0
    if pn == 0 or qn == 0:
        return 1.0
    similarity = float(np.sum(p * q) / (pn * qn))
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


@dataclass(frozen=True)
class RootingTopologySummary:
    inflow: np.ndarray
    outflow: np.ndarray
    root_share: np.ndarray
    density: float
    root_concentration: float
    rooting_entropy: float


def summarize_topology(adjacency: np.ndarray) -> RootingTopologySummary:
    matrix = _adjacency(adjacency)
    share = root_distribution(matrix)
    positive = share > 0
    entropy = -np.sum(share[positive] * np.log(share[positive]))
    normalized = float(entropy / np.log(len(share))) if len(share) > 1 else 0.0
    possible = matrix.shape[0] * (matrix.shape[0] - 1)
    return RootingTopologySummary(
        inflow=matrix.sum(axis=0),
        outflow=matrix.sum(axis=1),
        root_share=share,
        density=float(np.count_nonzero(matrix) / possible) if possible else 0.0,
        root_concentration=float(np.sum(share**2)),
        rooting_entropy=normalized,
    )
