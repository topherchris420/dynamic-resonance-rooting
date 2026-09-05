"""
DRR Market-State Adapter & Portfolio Regime Policy Module.

Interfaces rolling market-return windows with Dynamic Resonance Rooting (DRR),
aggregates structural metrics into MarketResonanceState, and defines no-lookahead regime policies.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..analysis import DynamicResonanceRooting

logger = logging.getLogger(__name__)


@dataclass
class MarketResonanceState:
    """Summary dataclass representing the structural state inferred by DRR for a market window."""

    timestamp: Optional[pd.Timestamp]
    mean_depth: float
    max_depth: float
    depth_dispersion: float
    network_density: float
    significant_edge_count: int
    effective_rooting_method: str
    agent_belief: float
    is_rooted: bool
    resonance_depths: Dict[str, float] = field(default_factory=dict)
    state_space_diagnostics: Optional[Dict[str, Any]] = None


def analyze_market_regime(
    returns_window: Union[pd.DataFrame, np.ndarray],
    sampling_rate: float = 1.0,
    embedding_dim: int = 3,
    tau: int = 1,
    spectral_method: str = "welch",
    rooting_method: str = "transfer_entropy",
    rooting_n_surrogates: int = 199,
    rooting_random_state: int = 42,
    rooting_alpha: float = 0.05,
    state_space: bool = False,
    timestamp: Optional[pd.Timestamp] = None,
) -> MarketResonanceState:
    """
    Adapter that takes a rolling window of market returns and executes DRR system analysis.

    Args:
        returns_window: Returns data (DataFrame or 2D NumPy array)
        sampling_rate: Sampling rate in Hz (1.0 for daily market data)
        embedding_dim: Embedding dimension
        tau: Time delay
        spectral_method: Spectral method ('welch', 'fft', 'wavelet', 'markov')
        rooting_method: 'transfer_entropy' or 'lagged_correlation'
        rooting_n_surrogates: Number of surrogates for rooting significance
        rooting_random_state: Random seed for surrogates
        rooting_alpha: Significance threshold
        state_space: Whether to run DSGE state-space diagnostics
        timestamp: Optional timestamp corresponding to the end of the window

    Returns:
        MarketResonanceState summarizing system-level resonance metrics.
    """
    if isinstance(returns_window, pd.DataFrame):
        data_matrix = returns_window.to_numpy()
        if timestamp is None and isinstance(returns_window.index, pd.DatetimeIndex):
            timestamp = returns_window.index[-1]
    else:
        data_matrix = np.asarray(returns_window)

    if data_matrix.ndim != 2:
        raise ValueError(f"returns_window must be 2D, got shape {data_matrix.shape}")

    n_samples, n_vars = data_matrix.shape
    if n_samples < 20:
        raise ValueError(f"Insufficient window length for DRR analysis ({n_samples} samples).")

    drr = DynamicResonanceRooting(
        embedding_dim=embedding_dim,
        tau=tau,
        sampling_rate=sampling_rate,
    )

    is_multivariate = n_vars > 1
    analysis_results = drr.analyze_system(
        data=data_matrix,
        multivariate=is_multivariate,
        window_size=min(100, n_samples // 2),
        state_space=state_space,
        method=spectral_method,
        rooting_method=rooting_method,
        rooting_n_surrogates=rooting_n_surrogates,
        rooting_random_state=rooting_random_state,
        rooting_alpha=rooting_alpha,
    )

    resonance_depths = analysis_results.get("resonance_depths", {})
    depth_vals = list(resonance_depths.values())

    if depth_vals:
        mean_depth = float(np.mean(depth_vals))
        max_depth = float(np.max(depth_vals))
        depth_dispersion = float(np.std(depth_vals))
    else:
        mean_depth, max_depth, depth_dispersion = 0.0, 0.0, 0.0

    # Network density calculation E / [N * (N - 1)] for directed graphs
    rooting_analysis = analysis_results.get("rooting_analysis", {})
    sig_edges = rooting_analysis.get("significant_edges", [])
    sig_edge_count = len(sig_edges)
    effective_method = rooting_analysis.get("method", rooting_method)

    if is_multivariate and n_vars > 1:
        max_possible_edges = n_vars * (n_vars - 1)
        network_density = (
            float(sig_edge_count / max_possible_edges) if max_possible_edges > 0 else 0.0
        )
    else:
        network_density = 0.0

    agent_beliefs = analysis_results.get("agent_belief", {})
    agent_belief = float(np.mean(list(agent_beliefs.values()))) if agent_beliefs else 0.5
    is_rooted = bool(analysis_results.get("is_rooted", False))

    state_space_diag = analysis_results.get("state_space_analysis", None)

    return MarketResonanceState(
        timestamp=timestamp,
        mean_depth=mean_depth,
        max_depth=max_depth,
        depth_dispersion=depth_dispersion,
        network_density=network_density,
        significant_edge_count=sig_edge_count,
        effective_rooting_method=effective_method,
        agent_belief=agent_belief,
        is_rooted=is_rooted,
        resonance_depths=resonance_depths,
        state_space_diagnostics=state_space_diag,
    )


class PortfolioRegimePolicy:
    """
    Policy abstraction mapping MarketResonanceState to a portfolio risk policy mode.

    Modes supported: 'standard' (e.g. Mean-Variance) vs 'high_resonance' (e.g. CVaR).

    Threshold types:
    - 'fixed': Fixed threshold on mean_depth
    - 'expanding_percentile': Percentile threshold computed over expanding historical history
    - 'rolling_percentile': Percentile threshold computed over rolling historical window
    - 'z_score': Z-score threshold computed over historical history
    """

    def __init__(
        self,
        threshold_type: str = "expanding_percentile",
        fixed_threshold: float = 0.70,
        percentile: float = 80.0,
        rolling_window: int = 252,
        z_threshold: float = 1.0,
        metric_name: str = "mean_depth",
    ):
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        self.percentile = percentile
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.metric_name = metric_name

        self._history: list[float] = []

    def choose_policy(self, state: MarketResonanceState) -> str:
        """
        Determine policy mode ('standard' vs 'high_resonance') using only history available up to t.

        Guarantees NO lookahead bias by appending current state metric AFTER computing regime threshold.
        """
        current_val = getattr(state, self.metric_name, state.mean_depth)

        if self.threshold_type == "fixed":
            is_high = current_val > self.fixed_threshold

        elif self.threshold_type == "expanding_percentile":
            if len(self._history) < 10:  # Warm-up fallback to fixed
                is_high = current_val > self.fixed_threshold
            else:
                cutoff = float(np.percentile(self._history, self.percentile))
                is_high = current_val > cutoff

        elif self.threshold_type == "rolling_percentile":
            if len(self._history) < 10:
                is_high = current_val > self.fixed_threshold
            else:
                window = self._history[-self.rolling_window :]
                cutoff = float(np.percentile(window, self.percentile))
                is_high = current_val > cutoff

        elif self.threshold_type == "z_score":
            if len(self._history) < 10:
                is_high = current_val > self.fixed_threshold
            else:
                hist = np.array(self._history)
                mean = np.mean(hist)
                std = np.std(hist)
                z = (current_val - mean) / std if std > 1e-8 else 0.0
                is_high = z > self.z_threshold

        else:
            raise ValueError(f"Unknown threshold_type: {self.threshold_type}")

        # Update historical memory after making decision for t (Strict no-lookahead)
        self._history.append(current_val)

        return "high_resonance" if is_high else "standard"
