"""
Experiment Configuration Registry for DRR Quant Lab.

Provides deterministic, reproducible configuration dataclasses for walk-forward experiments,
data providers, feature generators, and model training.
"""

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Dict, Any, Optional


@dataclass(frozen=True)
class QuantResearchConfig:
    """
    Deterministic configuration container for DRR Quantitative Research Lab experiments.

    All stochastic behavior, universe specifications, windows, and parameters are pinned here.
    """

    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"

    observation_symbols: Tuple[str, ...] = ("SPY", "TLT", "GLD", "HYG", "VIXY")
    portfolio_symbols: Tuple[str, ...] = ("SPY", "TLT", "GLD", "HYG")
    prediction_symbols: Tuple[str, ...] = ("SPY",)

    lookback: int = 252  # Trailing window for return/covariance estimation
    depth_window: int = 126  # Trailing window for DRR analysis

    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly', 'quarterly'

    # DRR Hyperparameters
    embedding_dim: int = 3
    tau: int = 1
    sampling_rate: float = 1.0
    spectral_method: str = "welch"  # 'welch', 'fft', 'wavelet', 'markov'
    rooting_method: str = "transfer_entropy"  # 'transfer_entropy' or 'lagged_correlation'
    rooting_surrogates: int = 199
    rooting_alpha: float = 0.05
    state_space: bool = False

    # Portfolio & Risk Hyperparameters
    transaction_cost_bps: float = 5.0
    annual_rf: float = 0.04
    max_weight: float = 0.50
    min_weight: float = 0.0
    threshold_type: str = "expanding_percentile"  # 'expanding_percentile', 'rolling_percentile', 'fixed', 'z_score'
    percentile: float = 80.0
    fixed_threshold: float = 0.70

    # ML & Qlib Parameters
    model_family: str = "linear"  # 'linear', 'lightgbm', 'mlp'
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "observation_symbols": list(self.observation_symbols),
            "portfolio_symbols": list(self.portfolio_symbols),
            "prediction_symbols": list(self.prediction_symbols),
            "lookback": self.lookback,
            "depth_window": self.depth_window,
            "rebalance_frequency": self.rebalance_frequency,
            "embedding_dim": self.embedding_dim,
            "tau": self.tau,
            "sampling_rate": self.sampling_rate,
            "spectral_method": self.spectral_method,
            "rooting_method": self.rooting_method,
            "rooting_surrogates": self.rooting_surrogates,
            "transaction_cost_bps": self.transaction_cost_bps,
            "annual_rf": self.annual_rf,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "threshold_type": self.threshold_type,
            "percentile": self.percentile,
            "model_family": self.model_family,
            "random_state": self.random_state,
        }
