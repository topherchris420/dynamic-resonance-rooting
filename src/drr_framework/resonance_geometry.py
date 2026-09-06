"""Typed high-level API for multivariate Dynamic Resonance Geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .cross_resonance import (
    CollectiveResonanceModes,
    CrossResonanceTensor,
    collective_modes,
    estimate_cross_resonance,
)


@dataclass(frozen=True)
class StructuralResonanceFingerprint:
    """Canonical, serialization-friendly system state for one trailing window."""

    timestamp: object
    collective_mode_strength: float
    effective_resonance_rank: float
    synchronization_entropy: float
    algorithm_version: str = "cross-resonance-v1"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResonanceGeometryResult:
    cross_resonance: CrossResonanceTensor
    collective_modes: CollectiveResonanceModes
    fingerprint: StructuralResonanceFingerprint
    parameters: Dict[str, object]


class DynamicResonanceGeometry:
    """Additive facade that leaves the established DRR API unchanged."""

    def __init__(self, sampling_rate: float = 1.0, nperseg: Optional[int] = None):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg

    def analyze(
        self,
        data: np.ndarray,
        timestamp: object = None,
        column_names: Optional[Tuple[str, ...]] = None,
        frequency_band: Tuple[float, Optional[float]] = (0.0, None),
    ) -> ResonanceGeometryResult:
        tensor = estimate_cross_resonance(
            data,
            sampling_rate=self.sampling_rate,
            nperseg=self.nperseg,
            column_names=column_names,
        )
        coupling = tensor.band_coupling(*frequency_band)
        modes = collective_modes(coupling)
        fingerprint = StructuralResonanceFingerprint(
            timestamp=timestamp,
            collective_mode_strength=modes.dominant_mode_strength,
            effective_resonance_rank=modes.effective_rank,
            synchronization_entropy=modes.synchronization_entropy,
        )
        return ResonanceGeometryResult(
            tensor,
            modes,
            fingerprint,
            {
                "sampling_rate": self.sampling_rate,
                "nperseg": tensor.nperseg,
                "frequency_band": frequency_band,
                "column_names": tensor.column_names,
            },
        )

    def rolling(
        self, data: np.ndarray, window: int, step: int = 1
    ) -> Tuple[ResonanceGeometryResult, ...]:
        """Analyze trailing windows; every result at endpoint ``t`` uses ``[:t]`` only."""
        values = np.asarray(data, dtype=float)
        if window < 8 or step < 1 or len(values) < window:
            raise ValueError("invalid window, step, or data length")
        return tuple(
            self.analyze(values[end - window : end], timestamp=end - 1)
            for end in range(window, len(values) + 1, step)
        )
