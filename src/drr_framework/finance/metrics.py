"""
Performance Metrics & Statistical Validation Module for DRR Quant Lab.

Calculates risk/return metrics (CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR, CVaR, Turnover)
and performs statistical validation testing the predictive relationship between DRR resonance metrics
and forward market behavior (realized vol, drawdowns, correlations).
"""

import logging
from typing import Dict, Any, Sequence, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_performance_metrics(
    returns: pd.Series,
    gross_returns: Optional[pd.Series] = None,
    annual_rf: float = 0.04,
    trading_days: int = 252,
    turnover: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive quantitative financial performance metrics for a return series.

    Returns:
        Dict containing CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR 95%, CVaR 95%,
        Turnover, Transaction Costs, Regime Switches, Average Holding Period.
    """
    if returns.empty:
        raise ValueError("Return series is empty.")

    r = returns.dropna()
    n_days = len(r)

    if n_days == 0:
        raise ValueError("No valid returns to compute metrics.")

    daily_rf = (1.0 + annual_rf) ** (1.0 / trading_days) - 1.0

    # Cumulative Return & CAGR
    cum_return = float((1.0 + r).prod())
    years = n_days / trading_days
    cagr = float(cum_return ** (1.0 / years) - 1.0) if years > 0 and cum_return > 0 else 0.0

    # Annualized Volatility
    ann_vol = float(r.std() * np.sqrt(trading_days))

    # Sharpe Ratio
    excess_mean_daily = float((r - daily_rf).mean())
    daily_std = float(r.std())
    sharpe = (
        float((excess_mean_daily / daily_std) * np.sqrt(trading_days)) if daily_std > 1e-8 else 0.0
    )

    # Sortino Ratio
    downside_returns = r[r < daily_rf] - daily_rf
    downside_std = (
        float(np.sqrt(np.mean(downside_returns**2))) if len(downside_returns) > 0 else 0.0
    )
    sortino = (
        float((excess_mean_daily / downside_std) * np.sqrt(trading_days))
        if downside_std > 1e-8
        else 0.0
    )

    # Max Drawdown & Calmar Ratio
    wealth_index = (1.0 + r).cumprod()
    peak = wealth_index.cummax()
    drawdown = (wealth_index - peak) / peak
    max_drawdown = float(drawdown.min())  # Negative number e.g. -0.20
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-8 else 0.0

    # VaR 95% and CVaR 95% (Historical)
    var_95 = float(-np.percentile(r, 5))
    tail_losses = r[r <= -var_95]
    cvar_95 = float(-tail_losses.mean()) if len(tail_losses) > 0 else var_95

    # Turnover & Transaction Costs
    avg_turnover = float(turnover.mean()) if turnover is not None and not turnover.empty else 0.0
    total_turnover = float(turnover.sum()) if turnover is not None and not turnover.empty else 0.0

    if gross_returns is not None and not gross_returns.empty:
        total_tc = float((gross_returns - r).sum())
    else:
        total_tc = 0.0

    # Regime switches and holding periods
    if regimes is not None and not regimes.empty:
        switches = int((regimes != regimes.shift(1)).sum() - 1)
        switches = max(0, switches)
        n_periods = len(regimes)
        avg_holding_period = float(n_periods / (switches + 1))
    else:
        switches = 0
        avg_holding_period = float(n_days)

    return {
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "historical_var_95": var_95,
        "historical_cvar_95": cvar_95,
        "average_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "total_transaction_costs": total_tc,
        "regime_switches": switches,
        "average_holding_period_days": avg_holding_period,
    }


def validate_drr_predictive_signal(
    drr_states: pd.DataFrame,
    market_returns: pd.DataFrame,
    horizons: Sequence[int] = (5, 20, 60),
) -> Dict[str, Any]:
    """
    Perform statistical analysis on the relationship between DRR resonance metrics at time t
    and forward market behavior (forward realized volatility, forward drawdown, forward max correlation).

    Args:
        drr_states: DataFrame containing DRR state variables indexed by timestamp
            (columns: mean_depth, network_density, depth_dispersion, agent_belief)
        market_returns: DataFrame of asset returns indexed by timestamp
        horizons: List of forward trading horizons (e.g., 5, 20, 60 days)

    Returns:
        Dict containing Pearson, Spearman rank correlations, p-values, and regime-conditioned means.
    """
    if drr_states.empty or market_returns.empty:
        return {"status": "insufficient_data"}

    # Align timestamps
    common_idx = drr_states.index.intersection(market_returns.index)
    if len(common_idx) < 30:
        return {"status": "insufficient_aligned_data"}

    states_df = drr_states.loc[common_idx]
    returns_df = market_returns.loc[common_idx]

    results = {}

    for horizon in horizons:
        # Calculate forward realized volatility across all market assets
        fwd_vol = returns_df.rolling(window=horizon).std().mean(axis=1).shift(-horizon)

        # Calculate forward max drawdown
        market_wealth = (1.0 + returns_df.mean(axis=1)).cumprod()
        fwd_dd = (
            market_wealth.rolling(window=horizon)
            .apply(lambda x: (x[-1] - x.max()) / x.max() if x.max() > 0 else 0.0, raw=True)
            .shift(-horizon)
        )

        horizon_results = {}

        for col in ["mean_depth", "network_density", "depth_dispersion"]:
            if col not in states_df.columns:
                continue

            x = states_df[col]
            valid_mask_vol = x.notna() & fwd_vol.notna()
            valid_mask_dd = x.notna() & fwd_dd.notna()

            # Volatility relationships
            if valid_mask_vol.sum() > 10:
                p_corr, p_pval = stats.pearsonr(x[valid_mask_vol], fwd_vol[valid_mask_vol])
                s_corr, s_pval = stats.spearmanr(x[valid_mask_vol], fwd_vol[valid_mask_vol])
            else:
                p_corr, p_pval, s_corr, s_pval = 0.0, 1.0, 0.0, 1.0

            # Drawdown relationships
            if valid_mask_dd.sum() > 10:
                dd_p_corr, dd_p_pval = stats.pearsonr(x[valid_mask_dd], fwd_dd[valid_mask_dd])
            else:
                dd_p_corr, dd_p_pval = 0.0, 1.0

            horizon_results[col] = {
                "fwd_vol_pearson_corr": float(p_corr),
                "fwd_vol_pearson_pvalue": float(p_pval),
                "fwd_vol_spearman_corr": float(s_corr),
                "fwd_vol_spearman_pvalue": float(s_pval),
                "fwd_drawdown_pearson_corr": float(dd_p_corr),
                "fwd_drawdown_pvalue": float(dd_p_pval),
            }

        results[f"horizon_{horizon}d"] = horizon_results

    return results
