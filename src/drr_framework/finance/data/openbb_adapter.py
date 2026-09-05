"""
OpenBB Financial Data Infrastructure Adapter.
"""

import logging
from typing import Optional, Sequence

import pandas as pd

from .base import validate_market_data, calculate_returns

logger = logging.getLogger(__name__)


class OpenBBMarketDataProvider:
    """MarketDataProvider adapter interfacing with OpenBB Platform API."""

    def __init__(self, provider_backend: str = "yfinance"):
        self.provider_backend = provider_backend

    def load_prices(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load prices via OpenBB platform."""
        try:
            from openbb import obb
        except ImportError as exc:
            raise ImportError(
                "OpenBB package is not installed. "
                'Please install via `pip install "drr-framework[quant-data]"` or `pip install openbb`.'
            ) from exc

        price_dict = {}
        for sym in symbols:
            res = obb.equity.price.historical(
                symbol=sym,
                start_date=start,
                end_date=end,
                provider=self.provider_backend,
            )
            df_sym = res.to_df()
            if "close" in df_sym.columns:
                price_dict[sym] = df_sym["close"]
            elif "adj_close" in df_sym.columns:
                price_dict[sym] = df_sym["adj_close"]
            else:
                price_dict[sym] = df_sym.iloc[:, 0]

        prices = pd.DataFrame(price_dict).dropna()
        validate_market_data(prices, symbols)
        return prices
