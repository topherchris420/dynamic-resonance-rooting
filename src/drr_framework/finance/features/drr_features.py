"""
DRR Feature Generator & Rolling Matrix Builder.

Transforms historical market return windows into DRR structural state representations
and constructs rolling, strictly causal DRR feature matrices without lookahead leakage.
"""

import logging
from typing import Dict, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...analysis import DynamicResonanceRooting
from ..config import QuantResearchConfig
from ..types import MarketResonanceState
from .market_state import aggregate_market_resonance_state

logger = logging.getLogger(__name__)


class DRRMarketFeatureGenerator:
    """
    Feature generator that executes DRR analysis on rolling market return windows.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        tau: int = 1,
        sampling_rate: float = 1.0,
        spectral_method: str = "welch",
        rooting_method: str = "transfer_entropy",
        rooting_n_surrogates: int = 199,
        rooting_random_state: int = 42,
        rooting_alpha: float = 0.05,
        state_space: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.sampling_rate = sampling_rate
        self.spectral_method = spectral_method
        self.rooting_method = rooting_method
        self.rooting_n_surrogates = rooting_n_surrogates
        self.rooting_random_state = rooting_random_state
        self.rooting_alpha = rooting_alpha
        self.state_space = state_space

    def transform(
        self,
        market_window: Union[pd.DataFrame, np.ndarray],
        as_of: Optional[pd.Timestamp] = None,
    ) -> MarketResonanceState:
        """
        Transform trailing market return window <= as_of into a MarketResonanceState object.
        """
        if isinstance(market_window, pd.DataFrame):
            data_matrix = market_window.to_numpy()
            if as_of is None and isinstance(market_window.index, pd.DatetimeIndex):
                as_of = market_window.index[-1]
        else:
            data_matrix = np.asarray(market_window)

        if data_matrix.ndim != 2:
            raise ValueError(f"market_window must be 2D array, got shape {data_matrix.shape}")

        n_samples, n_vars = data_matrix.shape
        if n_samples < 20:
            raise ValueError(f"Insufficient market history for DRR analysis ({n_samples} samples).")

        drr = DynamicResonanceRooting(
            embedding_dim=self.embedding_dim,
            tau=self.tau,
            sampling_rate=self.sampling_rate,
        )

        is_multivariate = n_vars > 1
        analysis_results = drr.analyze_system(
            data=data_matrix,
            multivariate=is_multivariate,
            window_size=min(100, n_samples // 2),
            state_space=self.state_space,
            method=self.spectral_method,
            rooting_method=self.rooting_method,
            rooting_n_surrogates=self.rooting_n_surrogates,
            rooting_random_state=self.rooting_random_state,
            rooting_alpha=self.rooting_alpha,
        )

        return aggregate_market_resonance_state(
            analysis_results=analysis_results,
            timestamp=as_of,
            requested_rooting_method=self.rooting_method,
            n_vars=n_vars,
        )


def build_drr_feature_matrix(
    returns: pd.DataFrame,
    depth_window: int = 126,
    step_size: int = 1,
    config: Optional[QuantResearchConfig] = None,
) -> pd.DataFrame:
    """
    Construct a rolling time series of DRR features across market returns.

    Enforces strict causal alignment: row at index t uses returns in [t - depth_window, t].

    Args:
        returns: Market returns DataFrame indexed by DatetimeIndex
        depth_window: Lookback window length in trading days
        step_size: Frequency of feature evaluation (e.g. 1 for daily, 5 for weekly)
        config: Optional QuantResearchConfig instance

    Returns:
        pd.DataFrame containing rolling DRR features indexed by timestamp.
    """
    if returns.empty or len(returns) < depth_window:
        raise ValueError(
            f"Insufficient return history ({len(returns)}) for depth window ({depth_window})."
        )

    cfg = config or QuantResearchConfig(depth_window=depth_window)
    generator = DRRMarketFeatureGenerator(
        embedding_dim=cfg.embedding_dim,
        tau=cfg.tau,
        sampling_rate=cfg.sampling_rate,
        spectral_method=cfg.spectral_method,
        rooting_method=cfg.rooting_method,
        rooting_n_surrogates=cfg.rooting_surrogates,
        rooting_random_state=cfg.random_state,
        state_space=cfg.state_space,
    )

    records = []
    dates = returns.index

    for i in range(depth_window, len(returns), step_size):
        dt = dates[i]
        window = returns.iloc[i - depth_window : i]
        state = generator.transform(window, as_of=dt)

        row = {
            "timestamp": dt,
            "drr_mean_depth": state.mean_depth,
            "drr_max_depth": state.max_depth,
            "drr_min_depth": state.min_depth,
            "drr_depth_dispersion": state.depth_dispersion,
            "drr_spectral_concentration": state.spectral_concentration,
            "drr_temporal_persistence": state.temporal_persistence,
            "drr_phase_coherence": state.phase_coherence,
            "drr_amplitude_stability": state.amplitude_stability,
            "drr_network_density": state.network_density,
            "drr_significant_edges": state.significant_edge_count,
            "drr_agent_belief": state.agent_belief,
            "drr_is_rooted": float(state.is_rooted),
            "drr_effective_method": state.effective_rooting_method,
        }

        # Add per-asset dimension depths
        for asset, depth in state.resonance_depths.items():
            row[f"drr_depth_{asset}"] = depth

        records.append(row)

    df_drr = pd.DataFrame(records).set_index("timestamp")
    return df_drr
