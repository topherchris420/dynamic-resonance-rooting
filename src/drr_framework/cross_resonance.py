"""Frequency-resolved multivariate resonance geometry.

This module is an independent DRR implementation built from the mathematical
definition of Welch cross spectra.  It does not contain code from an external
quantitative platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import csd


@dataclass(frozen=True)
class CrossResonanceTensor:
    """Interpretable components of a frequency-resolved resonance field.

    Arrays use ``(frequency, source, target)`` ordering.  ``phase[i,j]`` is the
    phase of the cross spectrum ``conj(X_i) X_j`` and is therefore
    antisymmetric.  Coherence is unitless; power weights sum to one per pair.
    """

    frequencies: np.ndarray
    coherence: np.ndarray
    phase: np.ndarray
    power_weight: np.ndarray
    sampling_rate: float
    nperseg: int
    column_names: Tuple[str, ...]

    def dominant_frequency(self, source: int, target: int) -> float:
        score = self.coherence[:, source, target] * self.power_weight[:, source, target]
        return float(self.frequencies[int(np.argmax(score))])

    def band_coupling(self, low: float = 0.0, high: Optional[float] = None) -> np.ndarray:
        """Return a PSD, trace-normalized coupling matrix for a frequency band."""
        upper = np.inf if high is None else high
        mask = (self.frequencies >= low) & (self.frequencies < upper)
        if not np.any(mask):
            raise ValueError("frequency band contains no Fourier bins")
        weights = np.mean(self.power_weight[mask], axis=(1, 2))
        weights = weights / max(float(weights.sum()), np.finfo(float).eps)
        coupling = np.einsum("f,fij->ij", weights, self.coherence[mask])
        coupling = 0.5 * (coupling + coupling.T)
        values, vectors = np.linalg.eigh(coupling)
        coupling = (vectors * np.maximum(values, 0.0)) @ vectors.T
        trace = float(np.trace(coupling))
        return coupling / trace if trace > 0 else coupling


@dataclass(frozen=True)
class CollectiveResonanceModes:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    dominant_mode_strength: float
    effective_rank: float
    participation_ratio: float
    synchronization_entropy: float


def estimate_cross_resonance(
    data: np.ndarray,
    sampling_rate: float = 1.0,
    nperseg: Optional[int] = None,
    column_names: Optional[Tuple[str, ...]] = None,
) -> CrossResonanceTensor:
    """Estimate pairwise magnitude-squared coherence, phase, and power weight.

    The same Welch partition is used for every pair.  Pairwise cross spectra
    retain frequency and phase information that a time-domain correlation
    necessarily aggregates away.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 2:
        raise ValueError("data must have shape (time >= 8, dimensions >= 2)")
    if not np.all(np.isfinite(x)) or sampling_rate <= 0:
        raise ValueError("data must be finite and sampling_rate positive")
    segment = min(nperseg or min(256, x.shape[0] // 2), x.shape[0])
    if segment < 4:
        raise ValueError("nperseg must be at least four")
    names = column_names or tuple(f"dim_{i}" for i in range(x.shape[1]))
    if len(names) != x.shape[1]:
        raise ValueError("column_names must match the number of dimensions")

    spectra = []
    frequencies = None
    for i in range(x.shape[1]):
        row = []
        for j in range(x.shape[1]):
            frequencies, spectrum = csd(
                x[:, i], x[:, j], fs=sampling_rate, nperseg=segment, detrend="constant"
            )
            row.append(spectrum)
        spectra.append(row)
    cross = np.moveaxis(np.asarray(spectra), -1, 0)
    auto = np.maximum(np.real(np.diagonal(cross, axis1=1, axis2=2)), 0.0)
    denominator = auto[:, :, None] * auto[:, None, :]
    coherence = np.divide(
        np.abs(cross) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    phase = np.angle(cross)
    pair_power = np.sqrt(denominator)
    totals = pair_power.sum(axis=0, keepdims=True)
    power_weight = np.divide(pair_power, totals, out=np.zeros_like(pair_power), where=totals > 0)
    return CrossResonanceTensor(
        frequencies=np.asarray(frequencies),
        coherence=coherence,
        phase=phase,
        power_weight=power_weight,
        sampling_rate=float(sampling_rate),
        nperseg=segment,
        column_names=tuple(names),
    )


def collective_modes(coupling: np.ndarray) -> CollectiveResonanceModes:
    """Summarize eigenstructure of a symmetric PSD resonance coupling matrix."""
    matrix = np.asarray(coupling, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coupling must be square")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values[::-1], 0.0)
    vectors = vectors[:, ::-1]
    total = float(values.sum())
    probabilities = values / total if total > 0 else np.full(len(values), 1 / len(values))
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    normalized_entropy = entropy / np.log(len(values)) if len(values) > 1 else 0.0
    participation = float(1.0 / np.sum(probabilities**2))
    return CollectiveResonanceModes(
        eigenvalues=values,
        eigenvectors=vectors,
        dominant_mode_strength=float(probabilities[0]),
        effective_rank=float(np.exp(entropy)),
        participation_ratio=participation,
        synchronization_entropy=float(normalized_entropy),
    )
