"""DRR Evidence Card Builder and Schema Definition."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DRREvidenceCard:
    """Standardized machine-readable and human-readable evidence card for DRR signals."""

    signal_id: str
    timestamp: float
    variables: List[str]
    methodology: str
    parameter_configuration: Dict[str, Any]
    statistical_significance: Dict[str, float]  # p-values, z-scores
    effect_size: Dict[str, float]  # resonance depth, transfer entropy weights
    robustness_score: float  # score across parameter variations
    benchmark_comparison: Dict[str, Any]  # comparison vs baselines
    historical_precedent: Optional[str]
    uncertainty: Dict[str, float]  # confidence intervals
    data_provenance: Dict[str, str]
    model_version: str
    reproducibility_hash: str
    detection_vs_interpretation: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_drr_evidence_card(
    signal_id: str,
    variables: List[str],
    methodology: str,
    parameter_configuration: Dict[str, Any],
    p_value: float,
    effect_size_dict: Dict[str, float],
    robustness_score: float,
    benchmark_comparison: Dict[str, Any],
    confidence_interval: tuple[float, float],
    data_provenance: Dict[str, str],
    detection_statement: str,
    interpretation_statement: str,
    model_version: str = "4.3.0",
    historical_precedent: Optional[str] = None,
) -> DRREvidenceCard:
    """Construct an immutable evidence card with an automated reproducibility hash."""
    timestamp = time.time()

    # Compute SHA-256 reproducibility hash
    raw_payload = f"{signal_id}:{variables}:{methodology}:{json.dumps(parameter_configuration, sort_keys=True)}:{p_value}:{effect_size_dict}:{robustness_score}"
    reproducibility_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    return DRREvidenceCard(
        signal_id=signal_id,
        timestamp=timestamp,
        variables=variables,
        methodology=methodology,
        parameter_configuration=parameter_configuration,
        statistical_significance={"p_value": p_value},
        effect_size=effect_size_dict,
        robustness_score=robustness_score,
        benchmark_comparison=benchmark_comparison,
        historical_precedent=historical_precedent,
        uncertainty={
            "ci_lower": confidence_interval[0],
            "ci_upper": confidence_interval[1],
        },
        data_provenance=data_provenance,
        model_version=model_version,
        reproducibility_hash=reproducibility_hash,
        detection_vs_interpretation={
            "detection": detection_statement,
            "interpretation": interpretation_statement,
            "caveat": "Detection of statistical resonance/coupling does not automatically imply economic cause or supervisory failure.",
        },
    )
