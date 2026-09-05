"""
Walk-Forward Backtesting Framework & Exporter for DRR Quant Lab.

Executes strict out-of-sample walk-forward portfolio rebalancing conditioned on DRR structural state.
Ensures zero lookahead leakage and exports results to CSV and JSON.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .types import MarketData
from .data.base import DEFAULT_OBSERVATION_UNIVERSE, DEFAULT_PORTFOLIO_UNIVERSE
from .regimes import analyze_market_regime, PortfolioRegimePolicy, MarketResonanceState
from .portfolio import optimize_portfolio
from .metrics import calculate_performance_metrics, validate_drr_predictive_signal

logger = logging.getLogger(__name__)


@dataclass
class QuantMacroConfig:
    """Configuration container for DRR Quant Lab backtest experiments."""

    lookback: int = 252  # Trailing returns window for portfolio estimation
    depth_window: int = 126  # Trailing window for DRR analysis
    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_cost_bps: float = 5.0  # Basis points per turnover unit (5 bps = 0.0005)
    annual_rf: float = 0.04  # Annualized risk-free rate (4%)
    max_weight: float = 0.50  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    threshold_type: str = "expanding_percentile"
    percentile: float = 80.0
    fixed_threshold: float = 0.70
    rooting_method: str = "transfer_entropy"
    rooting_surrogates: int = 199
    random_state: int = 42
    spectral_method: str = "welch"


@dataclass
class BacktestResult:
    """Container holding out-of-sample backtest results and diagnostic histories."""

    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    resonance_states: pd.DataFrame
    turnover: pd.Series
    metrics: Dict[str, float]
    predictive_signal_analysis: Dict[str, Any]
    config: QuantMacroConfig

    def export(self, output_dir: Union[str, Path]) -> None:
        """Export backtest results to CSV and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Export returns and turnover
        ret_df = pd.DataFrame(
            {
                "net_return": self.returns,
                "gross_return": self.gross_returns,
                "turnover": self.turnover,
                "regime": self.regimes,
            }
        )
        ret_df.to_csv(out_path / "results.csv")

        # 2. Export portfolio weights
        self.weights.to_csv(out_path / "weights.csv")

        # 3. Export DRR state variables
        self.resonance_states.to_csv(out_path / "drr_states.csv")

        # 4. Export summary JSON
        summary_payload = {
            "metrics": self.metrics,
            "predictive_signal_analysis": self.predictive_signal_analysis,
            "config": {
                "lookback": self.config.lookback,
                "depth_window": self.config.depth_window,
                "rebalance_frequency": self.config.rebalance_frequency,
                "transaction_cost_bps": self.config.transaction_cost_bps,
                "annual_rf": self.config.annual_rf,
                "max_weight": self.config.max_weight,
                "threshold_type": self.config.threshold_type,
                "percentile": self.config.percentile,
                "rooting_method": self.config.rooting_method,
            },
        }
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)

        logger.info("Exported backtest results to %s", out_path)


class WalkForwardBacktester:
    """
    Executes strict out-of-sample walk-forward portfolio rebalancing.

    At each rebalance date t:
      1. Slice trailing market returns <= t.
      2. Compute DRR state on trailing window <= t.
      3. Select policy mode using threshold derived ONLY from state history <= t.
      4. Optimize portfolio weights using returns <= t.
      5. Apply weights to subsequent out-of-sample period (t, t+1].
      6. Calculate turnover and transaction costs from previous weights.
    """

    def __init__(self, config: Optional[QuantMacroConfig] = None):
        self.config = config or QuantMacroConfig()
        self.policy = PortfolioRegimePolicy(
            threshold_type=self.config.threshold_type,
            fixed_threshold=self.config.fixed_threshold,
            percentile=self.config.percentile,
            metric_name="mean_depth",
        )

    def run(
        self,
        market_data: MarketData,
        policy_override: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest across market_data.

        Args:
            market_data: MarketData container
            policy_override: Fixed policy name ('standard', 'high_resonance', 'equal_weight', 'spy')
                             to bypass DRR conditioning for baseline comparisons.
        """
        returns_obs = market_data.returns[market_data.observation_symbols]
        returns_port = market_data.returns[market_data.portfolio_symbols]

        dates = returns_obs.index
        n_days = len(dates)

        if n_days <= self.config.lookback:
            raise ValueError(
                f"Data length ({n_days}) is insufficient for lookback window ({self.config.lookback})."
            )

        # Identify rebalance dates based on frequency
        rebalance_dates = self._get_rebalance_dates(dates, freq=self.config.rebalance_frequency)

        # Filter rebalance dates that have sufficient lookback history
        rebalance_dates = [d for d in rebalance_dates if dates.get_loc(d) >= self.config.lookback]

        if not rebalance_dates:
            raise ValueError("No valid rebalance dates found after applying lookback window.")

        tc_factor = self.config.transaction_cost_bps / 10000.0

        daily_net_returns = pd.Series(index=dates[dates.get_loc(rebalance_dates[0]) :], dtype=float)
        daily_gross_returns = pd.Series(index=daily_net_returns.index, dtype=float)
        daily_turnover = pd.Series(0.0, index=daily_net_returns.index)
        daily_regimes = pd.Series(index=daily_net_returns.index, dtype=object)

        weights_records = []
        state_records = []

        current_weights = pd.Series(
            1.0 / len(market_data.portfolio_symbols), index=market_data.portfolio_symbols
        )

        for i, reb_date in enumerate(rebalance_dates):
            reb_loc = dates.get_loc(reb_date)

            # Determine start and end of out-of-sample period
            next_reb_loc = (
                dates.get_loc(rebalance_dates[i + 1]) if i + 1 < len(rebalance_dates) else n_days
            )
            oos_dates = dates[reb_loc:next_reb_loc]

            # 1. Trailing historical window strictly <= reb_date
            hist_obs = returns_obs.iloc[reb_loc - self.config.lookback : reb_loc]
            hist_port = returns_port.iloc[reb_loc - self.config.lookback : reb_loc]
            hist_drr = returns_obs.iloc[reb_loc - self.config.depth_window : reb_loc]

            # 2. Compute DRR state at t
            drr_state = analyze_market_regime(
                returns_window=hist_drr,
                sampling_rate=1.0,
                embedding_dim=3,
                tau=1,
                spectral_method=self.config.spectral_method,
                rooting_method=self.config.rooting_method,
                rooting_n_surrogates=self.config.rooting_surrogates,
                rooting_random_state=self.config.random_state,
                state_space=False,
                timestamp=reb_date,
            )

            # 3. Determine regime policy
            if policy_override is not None:
                selected_policy = policy_override
            else:
                selected_policy = self.policy.choose_policy(drr_state)

            # Record DRR state
            state_records.append(
                {
                    "timestamp": reb_date,
                    "mean_depth": drr_state.mean_depth,
                    "max_depth": drr_state.max_depth,
                    "depth_dispersion": drr_state.depth_dispersion,
                    "network_density": drr_state.network_density,
                    "significant_edge_count": drr_state.significant_edge_count,
                    "effective_rooting_method": drr_state.effective_rooting_method,
                    "agent_belief": drr_state.agent_belief,
                    "policy": selected_policy,
                }
            )

            # 4. Portfolio Optimization
            if selected_policy == "equal_weight":
                new_weights = pd.Series(
                    1.0 / len(market_data.portfolio_symbols), index=market_data.portfolio_symbols
                )
            elif selected_policy == "spy":
                new_weights = pd.Series(0.0, index=market_data.portfolio_symbols)
                if "SPY" in new_weights.index:
                    new_weights["SPY"] = 1.0
                else:
                    new_weights.iloc[0] = 1.0
            else:
                new_weights = optimize_portfolio(
                    returns=hist_port,
                    policy_mode=selected_policy,
                    annual_rf=self.config.annual_rf,
                    min_weight=self.config.min_weight,
                    max_weight=self.config.max_weight,
                )

            # 5. Compute turnover on rebalance date
            turnover_value = float((new_weights - current_weights).abs().sum())
            current_weights = new_weights.copy()

            for w_date in oos_dates:
                weights_records.append(new_weights.rename(w_date))

            # 6. Out-of-sample return generation
            oos_returns = returns_port.loc[oos_dates]
            gross_series = (oos_returns * new_weights).sum(axis=1)

            # Apply transaction cost on the rebalance day
            net_series = gross_series.copy()
            if len(net_series) > 0:
                net_series.iloc[0] -= turnover_value * tc_factor

            daily_gross_returns.loc[oos_dates] = gross_series
            daily_net_returns.loc[oos_dates] = net_series
            daily_turnover.loc[reb_date] = turnover_value
            daily_regimes.loc[oos_dates] = selected_policy

        weights_df = pd.DataFrame(weights_records)
        weights_df = weights_df[~weights_df.index.duplicated(keep="first")]

        states_df = pd.DataFrame(state_records).set_index("timestamp")

        # Compute summary performance metrics
        metrics = calculate_performance_metrics(
            returns=daily_net_returns,
            gross_returns=daily_gross_returns,
            annual_rf=self.config.annual_rf,
            trading_days=252,
            turnover=daily_turnover,
            regimes=daily_regimes,
        )

        # Statistical signal validation against market
        predictive_analysis = validate_drr_predictive_signal(
            drr_states=states_df,
            market_returns=returns_obs,
            horizons=(5, 20, 60),
        )

        return BacktestResult(
            returns=daily_net_returns,
            gross_returns=daily_gross_returns,
            weights=weights_df,
            regimes=daily_regimes,
            resonance_states=states_df,
            turnover=daily_turnover,
            metrics=metrics,
            predictive_signal_analysis=predictive_analysis,
            config=self.config,
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """Helper to compute rebalance dates from DatetimeIndex based on frequency."""
        df_dates = pd.DataFrame(index=dates)

        if freq == "daily":
            return list(dates)

        elif freq == "weekly":
            # Rebalance on first business day of each week
            resamp = df_dates.resample("W-MON").first()
            return [d for d in resamp.index if d in dates]

        elif freq == "monthly":
            # Rebalance on first business day of each month
            resamp = df_dates.resample("MS").first()
            # Map to nearest actual trading date in index
            reb_dates = []
            for dt in resamp.index:
                loc = dates.searchsorted(dt)
                if loc < len(dates):
                    reb_dates.append(dates[loc])
            return list(dict.fromkeys(reb_dates))  # Unique ordered

        elif freq == "quarterly":
            resamp = df_dates.resample("QS").first()
            reb_dates = []
            for dt in resamp.index:
                loc = dates.searchsorted(dt)
                if loc < len(dates):
                    reb_dates.append(dates[loc])
            return list(dict.fromkeys(reb_dates))

        else:
            raise ValueError(f"Unknown rebalance frequency: {freq}")
