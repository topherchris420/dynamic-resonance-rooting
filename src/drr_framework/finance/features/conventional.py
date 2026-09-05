"""
Conventional Financial Market Features Extractor.

Extracts standard statistical/financial market features (simple/log returns, realized volatility,
rolling asset correlations, momentum, volume) cleanly separated from DRR features for ablation testing.
"""

from typing import Optional, Sequence
import numpy as np
import pandas as pd


def extract_conventional_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    vol_windows: Sequence[int] = (5, 20, 60),
    mom_windows: Sequence[int] = (5, 20, 60),
) -> pd.DataFrame:
    """
    Extract conventional financial market features.

    Calculates:
      - Asset returns
      - Realized volatility across multiple lookback windows
      - Rolling momentum
      - Rolling average cross-asset correlation

    Returns:
      pd.DataFrame indexed by date.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    features = pd.DataFrame(index=returns.index)

    # 1. Asset returns
    for col in returns.columns:
        features[f"{col}_return"] = returns[col]

    # 2. Realized volatilities
    for w in vol_windows:
        for col in returns.columns:
            features[f"{col}_vol_{w}d"] = returns[col].rolling(window=w).std() * np.sqrt(252)

    # 3. Rolling momentum (cumulative return over window)
    for w in mom_windows:
        for col in prices.columns:
            features[f"{col}_mom_{w}d"] = prices[col].pct_change(w)

    # 4. Average cross-asset correlation
    for w in (20, 60):
        corr_series = returns.rolling(window=w).corr()
        # Compute mean off-diagonal correlation per date
        mean_corrs = []
        for dt in returns.index:
            try:
                c_mat = corr_series.loc[dt]
                if isinstance(c_mat, pd.DataFrame) and c_mat.shape[0] > 1:
                    vals = c_mat.to_numpy()
                    mask = ~np.eye(vals.shape[0], dtype=bool)
                    off_diag = vals[mask]
                    valid_off = off_diag[~np.isnan(off_diag)]
                    if len(valid_off) > 0:
                        mean_corrs.append(float(np.mean(valid_off)))
                    else:
                        mean_corrs.append(np.nan)
                else:
                    mean_corrs.append(np.nan)
            except Exception:
                mean_corrs.append(np.nan)
        features[f"mean_cross_asset_corr_{w}d"] = mean_corrs

    return features.dropna()
