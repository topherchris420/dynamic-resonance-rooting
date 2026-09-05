"""
Data Contracts and Provenance Types for DRR Quant Lab.

Defines structural market resonance representations, data containers, and experiment result types.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketResonanceState:
    """
    Structured representation of financial market dynamics inferred by Dynamic Resonance Rooting (DRR).

    Does NOT equate resonance with systemic risk or crash probability.
    High resonance indicates strong structured dynamical behavior across market dimensions.
    """

    timestamp: Optional[pd.Timestamp]

    # Global Depth Aggregations
    mean_depth: float
    max_depth: float
    min_depth: float
    depth_dispersion: float

    # Dynamical & Spectral Properties
    spectral_concentration: float
    temporal_persistence: float
    phase_coherence: float
    amplitude_stability: float

    # Causal Topology & Network Structure
    network_density: float
    significant_edge_count: int

    # Epistemic Agent Belief & Rooting State
    agent_belief: float
    is_rooted: bool

    # Backend Execution Provenance
    effective_rooting_method: str  # 'transfer_entropy' or 'lagged_correlation'
    requested_rooting_method: str = "transfer_entropy"

    # State Space Diagnostics
    state_space_stability: Optional[float] = None

    # Per-Asset / Per-Dimension Detail
    resonance_depths: Dict[str, float] = field(default_factory=dict)
    state_space_diagnostics: Optional[Dict[str, Any]] = None

    # Full Lineage Provenance
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """Container for validated market price and return series across observation and portfolio universes."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    observation_symbols: List[str]
    portfolio_symbols: List[str]


@dataclass
class QuantExperimentResult:
    """Structured output container for quantitative research experiments."""

    config: Any  # QuantResearchConfig instance
    drr_states: pd.DataFrame
    predictions: Optional[pd.DataFrame]
    weights: Optional[pd.DataFrame]
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def export(self, output_dir: Union[str, Path]) -> None:
        """Export experiment result artifacts to CSV and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Export returns & turnover
        ret_df = pd.DataFrame(
            {
                "net_return": self.net_returns,
                "gross_return": self.gross_returns,
                "turnover": self.turnover,
            }
        )
        ret_df.to_csv(out_path / "returns.csv")

        # 2. Export DRR states
        if self.drr_states is not None and not self.drr_states.empty:
            self.drr_states.to_csv(out_path / "drr_states.csv")

        # 3. Export weights if present
        if self.weights is not None and not self.weights.empty:
            self.weights.to_csv(out_path / "weights.csv")

        # 4. Export predictions if present
        if self.predictions is not None and not self.predictions.empty:
            self.predictions.to_csv(out_path / "predictions.csv")

        # 5. Export summary JSON
        config_dict = (
            self.config.__dict__
            if hasattr(self.config, "__dict__")
            else getattr(self.config, "_asdict", lambda: {})()
        )
        summary_payload = {
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "config": config_dict,
        }
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)

        logger.info("Exported experiment results to %s", out_path)
