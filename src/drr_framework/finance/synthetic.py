"""
Synthetic Market Data Generation Utilities.
"""

from typing import Optional, Sequence
import numpy as np
import pandas as pd

from .data.base import (
    DEFAULT_OBSERVATION_UNIVERSE,
    DEFAULT_PORTFOLIO_UNIVERSE,
    calculate_returns,
    validate_market_data,
)
from .types import MarketData


def generate_synthetic_market_data(
    symbols: Optional[Sequence[str]] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
    portfolio_symbols: Optional[Sequence[str]] = None,
) -> MarketData:
    """
    Generate deterministic synthetic cross-asset price and return series for offline research and testing.
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

    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 1e-4)
    corr = vecs @ np.diag(vals) @ vecs.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = d_inv @ corr @ d_inv
    L = np.linalg.cholesky(corr)

    dt = 1.0 / 252.0
    daily_returns = np.zeros((n_days, n_assets))

    for t in range(n_days):
        regime_factor = 1.8 if (100 <= t <= 160 or 400 <= t <= 480) else 1.0
        z = rng.normal(size=n_assets)
        corr_z = L @ z

        for i, sym in enumerate(symbols):
            vol = base_vols.get(sym, 0.18) * regime_factor
            drift = base_drifts.get(sym, 0.03)
            daily_returns[t, i] = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * corr_z[i]

    initial_price = 100.0
    prices_data = initial_price * np.exp(np.cumsum(daily_returns, axis=0))
    prices_df = pd.DataFrame(prices_data, index=date_range, columns=list(symbols))

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
    Load market price data from provider or synthetic generator.
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
    elif provider == "openbb":
        from .data.openbb_adapter import OpenBBMarketDataProvider

        p = OpenBBMarketDataProvider()
        prices = p.load_prices(symbols=symbols, start=start_date, end=end_date)
        returns = calculate_returns(prices)
        return MarketData(
            prices=prices,
            returns=returns,
            observation_symbols=list(symbols),
            portfolio_symbols=list(portfolio_symbols),
        )
    elif provider == "qlib":
        from .data.qlib_adapter import QlibMarketDataProvider

        qp = QlibMarketDataProvider()
        prices = qp.load_prices(symbols=symbols, start=start_date, end=end_date)
        returns = calculate_returns(prices)
        return MarketData(
            prices=prices,
            returns=returns,
            observation_symbols=list(symbols),
            portfolio_symbols=list(portfolio_symbols),
        )
    else:
        raise ValueError(f"Unknown market data provider: {provider}")
