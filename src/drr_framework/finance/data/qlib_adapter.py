"""
Microsoft Qlib Market Data Provider Adapter.
"""

import logging
from typing import Optional, Sequence

import pandas as pd

from .base import validate_market_data

logger = logging.getLogger(__name__)


class QlibMarketDataProvider:
    """MarketDataProvider adapter interfacing with Microsoft Qlib data layer."""

    def __init__(self, qlib_dir: Optional[str] = None):
        self.qlib_dir = qlib_dir

    def load_prices(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load market price data from Qlib data engine."""
        try:
            import qlib
            from qlib.data import D
        except ImportError as exc:
            raise ImportError(
                "Microsoft Qlib (pyqlib) package is not installed. "
                'Please install via `pip install "drr-framework[quant-ml]"` or `pip install pyqlib`.'
            ) from exc

        if not getattr(qlib, "_initialized", False):
            if self.qlib_dir:
                qlib.init(provider_uri=self.qlib_dir)
            else:
                qlib.init()

        df = D.features(
            instruments=list(symbols),
            fields=["$close"],
            start_time=start,
            end_time=end,
        )

        if isinstance(df.index, pd.MultiIndex):
            prices = df["$close"].unstack(level="instrument")
        else:
            prices = df

        prices = prices.dropna()
        validate_market_data(prices, symbols)
        return prices
