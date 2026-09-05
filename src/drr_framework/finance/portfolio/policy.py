"""
Portfolio Policy, Constraints, and Conversions.
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np
import pandas as pd

from ..types import MarketResonanceState


def annual_to_daily_rf(annual_rf: float, trading_days: int = 252) -> float:
    """Convert annualized risk-free rate to daily compounding rate."""
    if annual_rf < -1.0:
        raise ValueError("Annual risk-free rate cannot be less than -100%.")
    return float((1.0 + annual_rf) ** (1.0 / trading_days) - 1.0)


class DRRRegimePortfolioPolicy:
    """
    Policy abstraction mapping MarketResonanceState to a portfolio risk policy mode.
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
        Choose portfolio policy ('standard' vs 'high_resonance') using only history available up to t.
        """
        current_val = getattr(state, self.metric_name, state.mean_depth)

        if self.threshold_type == "fixed":
            is_high = current_val > self.fixed_threshold

        elif self.threshold_type == "expanding_percentile":
            if len(self._history) < 10:
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

        # Update historical memory strictly AFTER making decision for time t
        self._history.append(current_val)

        return "high_resonance" if is_high else "standard"
