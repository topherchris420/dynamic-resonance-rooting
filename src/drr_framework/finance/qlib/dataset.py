"""
Qlib Dataset & DataHandler Adapter.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class QlibDatasetAdapter:
    """
    Adapter converting combined feature matrices into Qlib-compatible Dataset/DataHandler objects.
    Raises clean optional dependency error if pyqlib is not installed.
    """

    def __init__(self, target_symbol: str = "SPY", forward_horizon: int = 5):
        self.target_symbol = target_symbol
        self.forward_horizon = forward_horizon

    def build_qlib_dataset(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> Any:
        """
        Build Qlib-compatible DatasetH or pandas dataset with target label.

        Label defined as forward target_symbol return over forward_horizon.
        """
        if self.target_symbol not in returns_df.columns:
            raise ValueError(f"Target symbol '{self.target_symbol}' not found in returns DataFrame.")

        # Compute forward return target label
        fwd_return = (
            returns_df[self.target_symbol]
            .rolling(window=self.forward_horizon)
            .sum()
            .shift(-self.forward_horizon)
        )

        common_idx = features_df.index.intersection(fwd_return.dropna().index)
        if common_idx.empty:
            raise ValueError("No overlapping dates for Qlib feature matrix and forward target label.")

        X = features_df.loc[common_idx].copy()
        y = fwd_return.loc[common_idx].rename("LABEL0")

        # Try initializing Qlib DatasetH if Qlib is installed
        try:
            import qlib
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.handler import DataHandlerLP

            # Create Qlib multi-index format (datetime, instrument)
            X_qlib = X.copy()
            X_qlib["instrument"] = self.target_symbol
            dt_col = X_qlib.index.name or "datetime"
            X_qlib = X_qlib.reset_index()
            X_qlib = X_qlib.set_index([X_qlib.columns[0], "instrument"])

            y_qlib = pd.DataFrame(y)
            y_qlib["instrument"] = self.target_symbol
            y_qlib = y_qlib.reset_index()
            y_qlib = y_qlib.set_index([y_qlib.columns[0], "instrument"])

            data_df = pd.concat([X_qlib, y_qlib], axis=1)

            class SimpleDataHandler(DataHandlerLP):
                def __init__(self, df_data):
                    self._data = df_data
                    super().__init__()

                def _init_data(self):
                    pass

                def get_split_data(self, start_time=None, end_time=None):
                    return self._data

            handler = SimpleDataHandler(data_df)
            return DatasetH(handler=handler)

        except ImportError:
            logger.info("pyqlib not installed; returning standardized pandas (X, y) dataset tuple.")
            return X, y
