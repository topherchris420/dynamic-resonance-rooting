"""
VectorBT Validation Adapter & Parameter Robustness Sweeps.
"""

import logging
from typing import Dict, Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VectorBTAdapter:
    """
    Independent validation adapter using VectorBT (if available) for high-speed backtesting and parameter sweeps.
    """

    def __init__(self, transaction_cost_bps: float = 5.0):
        self.transaction_cost_bps = transaction_cost_bps
        self.tc_factor = transaction_cost_bps / 10000.0

    def run_backtest_validation(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Reconstruct backtest independently using VectorBT or vectorized Pandas fallback.

        Args:
            prices: Asset prices DataFrame indexed by timestamp
            weights: Target asset weights DataFrame indexed by timestamp

        Returns:
            Dict containing total return, Sharpe ratio, max drawdown, and strategy returns series.
        """
        try:
            import vectorbt as vbt

            # Align prices and weights
            common_idx = prices.index.intersection(weights.index)
            p_sub = prices.loc[common_idx]
            w_sub = weights.loc[common_idx]

            pf = vbt.Portfolio.from_orders(
                close=p_sub,
                size=w_sub,
                size_type="targetpercent",
                fees=self.tc_factor,
                freq="1D",
            )
            vbt_returns = pf.returns()
            return {
                "total_return": float(pf.total_return().mean()),
                "sharpe_ratio": float(pf.sharpe_ratio().mean()),
                "max_drawdown": float(pf.max_drawdown().mean()),
                "vbt_returns": vbt_returns,
                "engine": "vectorbt",
            }

        except ImportError:
            logger.info(
                "vectorbt package not installed; running vectorized Pandas backtest validation fallback."
            )
            return self._pandas_vectorized_validation(prices, weights)

    def run_parameter_robustness_sweep(
        self,
        market_data: Any,
        lookbacks: Sequence[int] = (63, 126, 252, 504),
        percentiles: Sequence[float] = (70.0, 80.0, 90.0),
        depth_windows: Sequence[int] = (63, 126),
    ) -> pd.DataFrame:
        """
        Execute parameter robustness sweep over lookback windows, depth windows, and regime quantiles.
        """
        from ..backtest import WalkForwardBacktester, QuantMacroConfig

        records = []
        for lb in lookbacks:
            for dw in depth_windows:
                if dw > lb:
                    continue
                for pct in percentiles:
                    cfg = QuantMacroConfig(
                        lookback=lb,
                        depth_window=dw,
                        percentile=pct,
                        transaction_cost_bps=self.transaction_cost_bps,
                        rooting_surrogates=10,
                    )
                    tester = WalkForwardBacktester(config=cfg)
                    try:
                        res = tester.run(market_data)
                        m = res.metrics
                        records.append(
                            {
                                "lookback": lb,
                                "depth_window": dw,
                                "percentile": pct,
                                "cagr": m["cagr"],
                                "annual_volatility": m["annualized_volatility"],
                                "sharpe": m["sharpe_ratio"],
                                "max_drawdown": m["max_drawdown"],
                                "cvar_95": m["historical_cvar_95"],
                                "turnover": m["average_turnover"],
                            }
                        )
                    except Exception as exc:
                        logger.debug("Parameter sweep combination failed: %s", exc)

        return pd.DataFrame(records)

    def _pandas_vectorized_validation(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> Dict[str, Any]:
        common_idx = prices.index.intersection(weights.index)
        p_sub = prices.loc[common_idx]
        w_sub = weights.loc[common_idx]

        returns = p_sub.pct_change().dropna()
        w_sub = w_sub.reindex(returns.index).ffill().fillna(0.0)

        # Gross strategy returns
        gross_rets = (returns * w_sub).sum(axis=1)

        # Turnover
        weight_diffs = w_sub.diff().abs().sum(axis=1).fillna(0.0)
        net_rets = gross_rets - weight_diffs * self.tc_factor

        cum_ret = float((1.0 + net_rets).prod() - 1.0)
        ann_vol = float(net_rets.std() * np.sqrt(252))
        sharpe = (
            float((net_rets.mean() / net_rets.std()) * np.sqrt(252))
            if net_rets.std() > 1e-8
            else 0.0
        )

        wealth = (1.0 + net_rets).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        max_dd = float(dd.min())

        return {
            "total_return": cum_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "vbt_returns": net_rets,
            "engine": "pandas_fallback",
        }
