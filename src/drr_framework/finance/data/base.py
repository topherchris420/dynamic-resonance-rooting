"""
Base Market Data Provider Protocol & Common Data Quality Validation.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd

from ..types import MarketData

logger = logging.getLogger(__name__)

DEFAULT_OBSERVATION_UNIVERSE = ("SPY", "TLT", "GLD", "HYG", "VIXY")
DEFAULT_PORTFOLIO_UNIVERSE = ("SPY", "TLT", "GLD", "HYG")


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for financial market data providers."""

    def load_prices(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load asset price DataFrame indexed by DatetimeIndex."""
        ...


def validate_market_data(
    df: pd.DataFrame,
    required_symbols: Sequence[str],
    min_history_length: int = 20,
) -> None:
    """
    Validate historical market price or return data.

    Raises ValueError or TypeError if quality checks fail. Never silently forward fills returns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Market data DataFrame is empty.")

    missing_cols = [sym for sym in required_symbols if sym not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required symbols in market data: {missing_cols}")

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

    sub_df = df[list(required_symbols)]
    if sub_df.isna().any().any():
        raise ValueError("Market data contains NaN values.")

    if np.isinf(sub_df.to_numpy()).any():
        raise ValueError("Market data contains infinite values.")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price DataFrame into simple return DataFrame without silent forward-filling.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna()
    return returns


class DataFrameMarketDataProvider:
    """Offline MarketDataProvider serving pre-loaded or synthetic DataFrames."""

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices

    def load_prices(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        sub = self._prices[list(symbols)].copy()
        if start:
            sub = sub[sub.index >= pd.Timestamp(start)]
        if end:
            sub = sub[sub.index <= pd.Timestamp(end)]
        validate_market_data(sub, symbols)
        return sub
