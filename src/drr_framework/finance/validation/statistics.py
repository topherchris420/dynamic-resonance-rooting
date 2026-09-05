"""
Statistical Validation & Negative Controls Module.

Implements rigorous statistical testing (Pearson/Spearman correlations, Newey-West HAC standard errors,
FDR p-value corrections) and negative controls (date-shuffled features, noise baseline) for DRR market signals.
"""

import logging
from typing import Dict, Any, Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_hac_standard_errors(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 5,
) -> Tuple[float, float, float]:
    """
    Compute OLS regression coefficient and Newey-West / HAC heteroskedasticity and autocorrelation robust standard error.

    Returns:
        Tuple of (beta_coefficient, hac_std_error, t_statistic).
    """
    valid = x.notna() & y.notna()
    x_c, y_c = x[valid].to_numpy(), y[valid].to_numpy()
    n = len(x_c)

    if n < 10:
        return 0.0, 1.0, 0.0

    X = np.column_stack([np.ones(n), x_c])
    beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
    residuals = y_c - X @ beta

    # Compute HAC covariance matrix
    xtx_inv = np.linalg.inv(X.T @ X)
    S = (X.T * residuals) @ (X * residuals[:, None]) / n

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = (X[lag:].T * residuals[lag:]) @ (X[:-lag] * residuals[:-lag, None]) / n
        S += weight * (gamma + gamma.T)

    hac_cov = n * (xtx_inv @ S @ xtx_inv)
    hac_se = np.sqrt(max(1e-12, hac_cov[1, 1]))
    t_stat = beta[1] / hac_se if hac_se > 1e-8 else 0.0

    return float(beta[1]), float(hac_se), float(t_stat)


def benjamini_hochberg_fdr(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) procedure to control for multiple hypothesis testing.

    Returns:
        Tuple of (boolean array indicating significance, adjusted p-values).
    """
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([])

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # Calculate adjusted p-values
    adj_pvals = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_p = (sorted_pvals[i] * n) / rank
        cum_min = min(cum_min, adj_p)
        adj_pvals[i] = cum_min

    adj_pvals = np.clip(adj_pvals, 0.0, 1.0)

    # Re-order to original sequence
    reordered_adj_pvals = np.zeros(n)
    reordered_adj_pvals[sorted_idx] = adj_pvals
    is_significant = reordered_adj_pvals <= alpha

    return is_significant, reordered_adj_pvals


def run_negative_controls(
    drr_states: pd.DataFrame,
    market_returns: pd.DataFrame,
    horizon: int = 20,
    n_shuffles: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute negative control analysis by comparing original DRR signal correlation against date-shuffled DRR features.
    """
    if drr_states.empty or market_returns.empty:
        return {"status": "insufficient_data"}

    common_idx = drr_states.index.intersection(market_returns.index)
    if len(common_idx) < 30:
        return {"status": "insufficient_data"}

    states_df = drr_states.loc[common_idx]
    returns_df = market_returns.loc[common_idx]

    fwd_vol = returns_df.rolling(window=horizon).std().mean(axis=1).shift(-horizon)

    x_orig = states_df["mean_depth"] if "mean_depth" in states_df.columns else states_df.iloc[:, 0]
    valid = x_orig.notna() & fwd_vol.notna()

    if valid.sum() < 10:
        return {"status": "insufficient_valid_samples"}

    orig_ic, _ = stats.spearmanr(x_orig[valid], fwd_vol[valid])

    # Date-shuffled controls
    rng = np.random.default_rng(seed)
    shuffled_ics = []

    x_vals = x_orig[valid].to_numpy()
    y_vals = fwd_vol[valid].to_numpy()

    for _ in range(n_shuffles):
        shuffled_x = rng.permutation(x_vals)
        s_ic, _ = stats.spearmanr(shuffled_x, y_vals)
        shuffled_ics.append(s_ic)

    shuffled_ics = np.array(shuffled_ics)
    p_value_empirical = float(np.mean(np.abs(shuffled_ics) >= np.abs(orig_ic)))

    return {
        "original_spearman_ic": float(orig_ic),
        "shuffled_ic_mean": float(np.mean(shuffled_ics)),
        "shuffled_ic_std": float(np.std(shuffled_ics)),
        "empirical_p_value": p_value_empirical,
        "is_statistically_distinct": bool(p_value_empirical < 0.05),
    }
