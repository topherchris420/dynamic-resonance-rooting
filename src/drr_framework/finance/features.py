"""
Market Data & Feature Engineering Layer for DRR Quant Lab.

Provides market data loading, synthetic data generation, validation, and return calculation
for cross-asset market observation and portfolio universes.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OBSERVATION_UNIVERSE = ["SPY", "TLT", "GLD", "HYG", "VIXY"]
DEFAULT_PORTFOLIO_UNIVERSE = ["SPY", "TLT", "GLD", "HYG"]


@dataclass
class MarketData:
    """Container for validated market price and return series."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    observation_symbols: List[str]
    portfolio_symbols: List[str]


def validate_market_data(
    df: pd.DataFrame,
    required_symbols: Sequence[str],
    min_history_length: int = 20,
) -> None:
    """
    Validate historical market price or return data.

    Raises ValueError or TypeError if data quality checks fail.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Market data DataFrame is empty.")

    # Check required symbols
    missing_cols = [sym for sym in required_symbols if sym not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required symbols in market data: {missing_cols}")

    # Check datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Market data DataFrame index must be a pandas DatetimeIndex.")

    if df.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates in the index.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Market data index timestamps are not strictly monotonic increasing.")

    if len(df) < min_history_length:
        raise ValueError(
            f"Insufficient history: received {len(df)} rows, minimum required is {min_history_length}."
        )

    # Check for NaNs and Inf values
    sub_df = df[list(required_symbols)]
    if sub_df.isna().any().any():
        raise ValueError("Market data contains NaN values.")

    if np.isinf(sub_df.to_numpy()).any():
        raise ValueError("Market data contains infinite values.")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price DataFrame into simple return DataFrame using explicit pct_change without silent filling.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna()  # Ensure clean return matrix without NaNs
    return returns


def generate_synthetic_market_data(
    symbols: Optional[Sequence[str]] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
    portfolio_symbols: Optional[Sequence[str]] = None,
) -> MarketData:
    """
    Generate deterministic synthetic cross-asset price and return series for offline research and testing.

    Includes realistic cross-asset correlations, regime-dependent volatility, and multi-factor structure.
    """
    if symbols is None:
        symbols = DEFAULT_OBSERVATION_UNIVERSE
    if portfolio_symbols is None:
        portfolio_symbols = [s for s in DEFAULT_PORTFOLIO_UNIVERSE if s in symbols]

    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(date_range)

    if n_days < 30:
        raise ValueError(f"Date range generates insufficient business days ({n_days}).")

    n_assets = len(symbols)
    # Asset parameters (annualized drift and vol)
    base_vols = {
        "SPY": 0.16,
        "TLT": 0.12,
        "GLD": 0.14,
        "HYG": 0.10,
        "VIXY": 0.45,
    }
    base_drifts = {
        "SPY": 0.08,
        "TLT": 0.02,
        "GLD": 0.04,
        "HYG": 0.05,
        "VIXY": -0.15,
    }

    # Generate correlated random returns
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            s1, s2 = symbols[i], symbols[j]
            if (s1 == "SPY" and s2 == "TLT") or (s1 == "TLT" and s2 == "SPY"):
                c = -0.3
            elif (s1 == "SPY" and s2 == "HYG") or (s1 == "HYG" and s2 == "SPY"):
                c = 0.7
            elif (s1 == "SPY" and s2 == "VIXY") or (s1 == "VIXY" and s2 == "SPY"):
                c = -0.75
            elif (s1 == "GLD" and s2 == "SPY") or (s1 == "SPY" and s2 == "GLD"):
                c = 0.1
            else:
                c = 0.2
            corr[i, j] = c
            corr[j, i] = c

    # Make correlation matrix positive definite by eigenvalue clipping
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 1e-4)
    corr = vecs @ np.diag(vals) @ vecs.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = d_inv @ corr @ d_inv
    L = np.linalg.cholesky(corr)

    dt = 1.0 / 252.0
    daily_returns = np.zeros((n_days, n_assets))

    for t in range(n_days):
        # Inject regime shift / stress periods
        regime_factor = 1.8 if (100 <= t <= 160 or 400 <= t <= 480) else 1.0
        z = rng.normal(size=n_assets)
        corr_z = L @ z

        for i, sym in enumerate(symbols):
            vol = base_vols.get(sym, 0.18) * regime_factor
            drift = base_drifts.get(sym, 0.03)
            daily_returns[t, i] = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * corr_z[i]

    # Convert returns to prices
    initial_price = 100.0
    prices_data = initial_price * np.exp(np.cumsum(daily_returns, axis=0))
    prices_df = pd.DataFrame(prices_data, index=date_range, columns=list(symbols))

    # Calculate returns explicitly
    returns_df = calculate_returns(prices_df)

    validate_market_data(prices_df, symbols)
    validate_market_data(returns_df, symbols)

    return MarketData(
        prices=prices_df,
        returns=returns_df,
        observation_symbols=list(symbols),
        portfolio_symbols=list(portfolio_symbols),
    )


def load_market_data(
    symbols: Optional[Sequence[str]] = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    provider: str = "synthetic",
    portfolio_symbols: Optional[Sequence[str]] = None,
) -> MarketData:
    """
    Load market price data from provider or synthetic generator and return validated MarketData.

    Args:
        symbols: Observation universe symbols.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD) or None.
        provider: Provider name ("synthetic", "yfinance", or "openbb").
        portfolio_symbols: Portfolio universe symbols.
    """
    if symbols is None:
        symbols = DEFAULT_OBSERVATION_UNIVERSE
    if portfolio_symbols is None:
        portfolio_symbols = [s for s in DEFAULT_PORTFOLIO_UNIVERSE if s in symbols]

    if provider == "synthetic":
        end_d = end_date or "2023-12-31"
        return generate_synthetic_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_d,
            portfolio_symbols=portfolio_symbols,
        )

    elif provider == "yfinance":
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for provider='yfinance'. Install via `pip install yfinance`."
            ) from e

        data = yf.download(
            tickers=list(symbols),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                prices = data["Close"]
            else:
                prices = data.xs("Close", level=0, axis=1)
        else:
            prices = data

        prices = prices.dropna(how="all").dropna()
        validate_market_data(prices, symbols)
        returns = calculate_returns(prices)

        return MarketData(
            prices=prices,
            returns=returns,
            observation_symbols=list(symbols),
            portfolio_symbols=list(portfolio_symbols),
        )

    elif provider == "openbb":
        try:
            from openbb import obb
        except ImportError as e:
            raise ImportError(
                "OpenBB is required for provider='openbb'. Install via `pip install openbb` or `pip install drr-framework[quant]`."
            ) from e

        # Fetch using OpenBB platform API
        price_dict = {}
        for sym in symbols:
            res = obb.equity.price.historical(symbol=sym, start_date=start_date, end_date=end_date)
            df_sym = res.to_df()
            if "close" in df_sym.columns:
                price_dict[sym] = df_sym["close"]
            elif "adj_close" in df_sym.columns:
                price_dict[sym] = df_sym["adj_close"]
            else:
                price_dict[sym] = df_sym.iloc[:, 0]

        prices = pd.DataFrame(price_dict).dropna()
        validate_market_data(prices, symbols)
        returns = calculate_returns(prices)

        return MarketData(
            prices=prices,
            returns=returns,
            observation_symbols=list(symbols),
            portfolio_symbols=list(portfolio_symbols),
        )

    else:
        raise ValueError(f"Unknown market data provider: {provider}")
