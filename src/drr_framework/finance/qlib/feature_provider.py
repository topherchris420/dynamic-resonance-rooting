"""
DRR Qlib Feature Provider Adapter.

Adapts DRR Market State Features into Microsoft Qlib feature format.
"""

import logging
from typing import Dict, Any, Optional, Sequence

import pandas as pd

from ..features.drr_features import build_drr_feature_matrix
from ..features.conventional import extract_conventional_features
from ..config import QuantResearchConfig

logger = logging.getLogger(__name__)


class DRRQlibFeatureProvider:
    """
    Feature provider preparing aligned conventional and DRR market state features for Qlib.
    """

    def __init__(self, config: Optional[QuantResearchConfig] = None):
        self.config = config or QuantResearchConfig()

    def generate_feature_dataset(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate combined conventional + DRR market state feature matrix.

        Returns:
            pd.DataFrame indexed by date containing conventional and DRR state features.
        """
        # 1. Conventional features
        df_conv = extract_conventional_features(prices=prices, returns=returns)

        # 2. DRR features
        df_drr = build_drr_feature_matrix(
            returns=returns,
            depth_window=self.config.depth_window,
            step_size=1,
            config=self.config,
        )

        # 3. Align features chronologically
        common_idx = df_conv.index.intersection(df_drr.index)
        if common_idx.empty:
            raise ValueError("No overlapping dates between conventional and DRR features.")

        df_combined = pd.concat([df_conv.loc[common_idx], df_drr.loc[common_idx]], axis=1)
        return df_combined.dropna()
