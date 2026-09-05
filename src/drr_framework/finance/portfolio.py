"""
Portfolio Optimization Module for DRR Quant Lab.

Implements regime-conditioned portfolio optimization using Riskfolio-Lib with a robust
SciPy fallback optimizer. Enforces long-only constraints, asset weights bounds,
and explicit daily risk-free rate conversions.
"""

import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def annual_to_daily_rf(annual_rf: float, trading_days: int = 252) -> float:
    """Convert an annualized risk-free rate to a compounding daily risk-free rate."""
    if annual_rf < -1.0:
        raise ValueError("Annual risk-free rate cannot be less than -100%.")
    return float((1.0 + annual_rf) ** (1.0 / trading_days) - 1.0)


def optimize_portfolio(
    returns: pd.DataFrame,
    policy_mode: str = "standard",
    annual_rf: float = 0.04,
    min_weight: float = 0.0,
    max_weight: float = 0.50,
    allow_short: bool = False,
    trading_days: int = 252,
) -> pd.Series:
    """
    Optimize portfolio weights conditioned on policy_mode.

    Args:
        returns: DataFrame of asset returns (columns are asset symbols)
        policy_mode: 'standard' (Mean-Variance Sharpe) or 'high_resonance' (CVaR / Risk-Adjusted)
        annual_rf: Annualized risk-free rate (e.g. 0.04 = 4%)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        allow_short: Whether short positions are permitted
        trading_days: Number of trading days per year

    Returns:
        pd.Series of asset weights summing to 1.0
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    symbols = list(returns.columns)
    n_assets = len(symbols)

    if n_assets == 0:
        raise ValueError("No assets provided in returns DataFrame.")

    # Convert annual risk-free rate to daily compounding rate
    daily_rf = annual_to_daily_rf(annual_rf, trading_days=trading_days)

    # Attempt optimization via Riskfolio-Lib if available
    weights = _try_riskfolio_optimize(
        returns=returns,
        policy_mode=policy_mode,
        daily_rf=daily_rf,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # Fall back to SciPy optimization if Riskfolio is unavailable or fails
    if weights is None:
        weights = _scipy_optimize(
            returns=returns,
            policy_mode=policy_mode,
            daily_rf=daily_rf,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
        )

    # Validate resulting weights
    if weights is None or weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        logger.warning("Portfolio optimization failed; returning Equal Weight fallback.")
        weights = pd.Series(1.0 / n_assets, index=symbols)

    # Normalize weights to sum exactly to 1.0
    total_w = weights.sum()
    if total_w > 0:
        weights = weights / total_w
    else:
        weights = pd.Series(1.0 / n_assets, index=symbols)

    return weights


def _try_riskfolio_optimize(
    returns: pd.DataFrame,
    policy_mode: str,
    daily_rf: float,
    min_weight: float,
    max_weight: float,
) -> Optional[pd.Series]:
    """Internal helper to attempt Riskfolio-Lib portfolio optimization."""
    try:
        import riskfolio as rp
    except ImportError:
        return None

    try:
        symbols = list(returns.columns)
        port = rp.Portfolio(returns=returns)

        # Estimate statistics
        port.assets_stats(
            method_mu="hist", method_cov="ledoit" if policy_mode == "high_resonance" else "hist"
        )

        # Set bounds
        port.upperbound = max_weight
        port.lowerbound = min_weight

        if policy_mode == "high_resonance":
            # High resonance regime -> CVaR Optimization
            w = port.optimization(
                model="Classic",
                rm="CVaR",
                obj="Sharpe",
                rf=daily_rf,
                l=0,
                hist=True,
            )
        else:
            # Standard regime -> Mean-Variance Optimization
            w = port.optimization(
                model="Classic",
                rm="MV",
                obj="Sharpe",
                rf=daily_rf,
                l=0,
                hist=True,
            )

        if w is not None and not w.empty:
            weights_series = w.iloc[:, 0]
            weights_series.index = symbols
            return weights_series

    except Exception as exc:
        logger.debug("Riskfolio-Lib optimization error: %s", exc)

    return None


def _scipy_optimize(
    returns: pd.DataFrame,
    policy_mode: str,
    daily_rf: float,
    min_weight: float,
    max_weight: float,
    allow_short: bool,
) -> Optional[pd.Series]:
    """Fallback portfolio optimization using SciPy minimize."""
    symbols = list(returns.columns)
    n_assets = len(symbols)

    mean_returns = returns.mean().to_numpy()
    cov_matrix = returns.cov().to_numpy()

    if min_weight < 0 and not allow_short:
        min_weight = 0.0

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    init_weights = np.ones(n_assets) / n_assets

    if policy_mode == "high_resonance":
        # Objective: Minimize Conditional Value at Risk (CVaR 95%) or Maximize Risk-Adjusted CVaR
        # Estimate empirical 95% CVaR for return series w @ R.T
        r_matrix = returns.to_numpy()

        def cvar_objective(w):
            port_returns = r_matrix @ w
            alpha = 0.05
            var_thresh = np.percentile(port_returns, alpha * 100)
            tail_losses = -port_returns[port_returns <= var_thresh]
            cvar = np.mean(tail_losses) if len(tail_losses) > 0 else -var_thresh
            return cvar

        res = minimize(
            cvar_objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

    else:
        # Objective: Maximize Sharpe Ratio (Minimize Negative Sharpe Ratio)
        def neg_sharpe_objective(w):
            port_return = np.sum(mean_returns * w) - daily_rf
            port_vol = np.sqrt(w.T @ cov_matrix @ w)
            if port_vol <= 1e-8:
                return 0.0
            return -port_return / port_vol

        res = minimize(
            neg_sharpe_objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

    if res.success and res.x is not None:
        return pd.Series(res.x, index=symbols)

    return None
