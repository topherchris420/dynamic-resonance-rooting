"""
DRR Market-State Adapter & Portfolio Regime Policy Module.

Interfaces rolling market-return windows with Dynamic Resonance Rooting (DRR),
aggregates structural metrics into MarketResonanceState, and defines no-lookahead regime policies.
"""

import logging
from typing import Dict, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .types import MarketResonanceState
from .features.drr_features import DRRMarketFeatureGenerator

logger = logging.getLogger(__name__)


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
    """
    generator = DRRMarketFeatureGenerator(
        embedding_dim=embedding_dim,
        tau=tau,
        sampling_rate=sampling_rate,
        spectral_method=spectral_method,
        rooting_method=rooting_method,
        rooting_n_surrogates=rooting_n_surrogates,
        rooting_random_state=rooting_random_state,
        rooting_alpha=rooting_alpha,
        state_space=state_space,
    )
    return generator.transform(returns_window, as_of=timestamp)


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
