"""
Riskfolio-Lib Adapter & SciPy Optimization Fallback.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskfolioAllocator:
    """
    Portfolio optimization adapter interfacing with Riskfolio-Lib with a robust SciPy fallback.
    """

    def __init__(
        self,
        annual_rf: float = 0.04,
        trading_days: int = 252,
        min_weight: float = 0.0,
        max_weight: float = 0.50,
        allow_short: bool = False,
    ):
        self.annual_rf = annual_rf
        self.trading_days = trading_days
        self.daily_rf = (1.0 + annual_rf) ** (1.0 / trading_days) - 1.0
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.allow_short = allow_short

    def optimize(
        self,
        returns: pd.DataFrame,
        policy_mode: str = "standard",
    ) -> pd.Series:
        """
        Optimize portfolio weights given historical return matrix.

        Args:
            returns: Asset returns DataFrame
            policy_mode: 'standard' (Mean-Variance Sharpe) or 'high_resonance' (CVaR)

        Returns:
            pd.Series of normalized asset weights.
        """
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        symbols = list(returns.columns)
        n_assets = len(symbols)

        # Attempt Riskfolio optimization
        weights = self._try_riskfolio(returns, policy_mode)

        # Fallback to SciPy
        if weights is None:
            weights = self._scipy_optimize(returns, policy_mode)

        if weights is None or weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
            logger.warning("Optimization failed; returning Equal Weight fallback.")
            weights = pd.Series(1.0 / n_assets, index=symbols)

        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
        else:
            weights = pd.Series(1.0 / n_assets, index=symbols)

        return weights

    def _try_riskfolio(self, returns: pd.DataFrame, policy_mode: str) -> Optional[pd.Series]:
        try:
            import riskfolio as rp

            symbols = list(returns.columns)
            port = rp.Portfolio(returns=returns)
            port.assets_stats(
                method_mu="hist",
                method_cov="ledoit" if policy_mode == "high_resonance" else "hist",
            )
            port.upperbound = self.max_weight
            port.lowerbound = self.min_weight

            rm = "CVaR" if policy_mode == "high_resonance" else "MV"
            w = port.optimization(
                model="Classic",
                rm=rm,
                obj="Sharpe",
                rf=self.daily_rf,
                l=0,
                hist=True,
            )
            if w is not None and not w.empty:
                s = w.iloc[:, 0]
                s.index = symbols
                return s
        except Exception as exc:
            logger.debug("Riskfolio optimization fallback triggered: %s", exc)

        return None

    def _scipy_optimize(self, returns: pd.DataFrame, policy_mode: str) -> Optional[pd.Series]:
        symbols = list(returns.columns)
        n_assets = len(symbols)

        mean_returns = returns.mean().to_numpy()
        cov_matrix = returns.cov().to_numpy()

        min_w = 0.0 if not self.allow_short and self.min_weight < 0 else self.min_weight
        bounds = tuple((min_w, self.max_weight) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        init_weights = np.ones(n_assets) / n_assets

        if policy_mode == "high_resonance":
            r_matrix = returns.to_numpy()

            def cvar_obj(w):
                port_rets = r_matrix @ w
                var_thresh = np.percentile(port_rets, 5.0)
                tail_losses = -port_rets[port_rets <= var_thresh]
                return float(np.mean(tail_losses)) if len(tail_losses) > 0 else -var_thresh

            res = minimize(
                cvar_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
            )
        else:

            def neg_sharpe_obj(w):
                p_ret = np.sum(mean_returns * w) - self.daily_rf
                p_vol = np.sqrt(w.T @ cov_matrix @ w)
                return -p_ret / p_vol if p_vol > 1e-8 else 0.0

            res = minimize(
                neg_sharpe_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
            )

        if res.success and res.x is not None:
            return pd.Series(res.x, index=symbols)

        return None
