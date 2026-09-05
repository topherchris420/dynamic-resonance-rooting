"""
Primary Matched Experiment Engine (Control vs Experiment) for Qlib + DRR.

Isolates the incremental out-of-sample informational contribution of DRR features.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from ..config import QuantResearchConfig
from ..types import QuantExperimentResult, MarketData
from .feature_provider import DRRQlibFeatureProvider
from .dataset import QlibDatasetAdapter
from .evaluation import calculate_ic_metrics

logger = logging.getLogger(__name__)


class QlibDRRMatchedExperiment:
    """
    Primary Qlib Matched Experiment framework.

    CONTROL: Qlib model + conventional financial features
    EXPERIMENT: same Qlib model + conventional features + DRR structural state features
    """

    def __init__(
        self,
        config: Optional[QuantResearchConfig] = None,
        target_symbol: str = "SPY",
        forward_horizon: int = 5,
    ):
        self.config = config or QuantResearchConfig()
        self.target_symbol = target_symbol
        self.forward_horizon = forward_horizon
        self.feature_provider = DRRQlibFeatureProvider(config=self.config)
        self.dataset_adapter = QlibDatasetAdapter(
            target_symbol=target_symbol, forward_horizon=forward_horizon
        )

    def _select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only numeric feature columns and fill any NaNs."""
        num_df = df.select_dtypes(include=[np.number]).copy()
        return num_df.fillna(0.0)

    def run_matched_experiment(
        self,
        market_data: MarketData,
        model_family: str = "linear",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run strictly matched out-of-sample Control vs Experiment evaluation.

        Args:
            market_data: MarketData container
            model_family: 'linear', 'lightgbm', or 'mlp'

        Returns:
            Dict containing 'control' and 'experiment' evaluation results.
        """
        # 1. Generate features
        combined_features = self.feature_provider.generate_feature_dataset(
            prices=market_data.prices,
            returns=market_data.returns,
        )

        num_features = self._select_numeric_features(combined_features)

        # Separate feature sets
        conv_cols = [c for c in num_features.columns if not c.startswith("drr_")]
        drr_cols = [c for c in num_features.columns if c.startswith("drr_")]

        X_control = num_features[conv_cols]
        X_experiment = num_features[conv_cols + drr_cols]

        # Target label
        fwd_return = (
            market_data.returns[self.target_symbol]
            .rolling(window=self.forward_horizon)
            .sum()
            .shift(-self.forward_horizon)
        )

        common_idx = num_features.index.intersection(fwd_return.dropna().index)
        y = fwd_return.loc[common_idx]

        X_control = X_control.loc[common_idx]
        X_experiment = X_experiment.loc[common_idx]

        # Chronological Train / OOS Test Split (e.g., 60% train, 40% test)
        n_samples = len(common_idx)
        n_train = int(n_samples * self.config.train_ratio)

        X_ctrl_train, X_ctrl_test = X_control.iloc[:n_train], X_control.iloc[n_train:]
        X_exp_train, X_exp_test = X_experiment.iloc[:n_train], X_experiment.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        # Train Control Model
        model_ctrl = self._instantiate_model(model_family)
        model_ctrl.fit(X_ctrl_train, y_train)
        preds_ctrl = pd.Series(model_ctrl.predict(X_ctrl_test), index=X_ctrl_test.index)

        # Train Experiment Model
        model_exp = self._instantiate_model(model_family)
        model_exp.fit(X_exp_train, y_train)
        preds_exp = pd.Series(model_exp.predict(X_exp_test), index=X_exp_test.index)

        # Evaluate Out-of-Sample IC Metrics
        ctrl_metrics = calculate_ic_metrics(preds_ctrl, y_test)
        exp_metrics = calculate_ic_metrics(preds_exp, y_test)

        # Compute Incremental Deltas
        ic_delta = exp_metrics["ic"] - ctrl_metrics["ic"]
        rank_ic_delta = exp_metrics["rank_ic"] - ctrl_metrics["rank_ic"]

        logger.info(
            "Qlib Experiment Complete. Control IC: %.4f | Experiment IC: %.4f | Delta IC: %+.4f",
            ctrl_metrics["ic"],
            exp_metrics["ic"],
            ic_delta,
        )

        return {
            "control": {
                "metrics": ctrl_metrics,
                "predictions": preds_ctrl,
                "features_used": list(conv_cols),
            },
            "experiment": {
                "metrics": exp_metrics,
                "predictions": preds_exp,
                "features_used": list(conv_cols + drr_cols),
            },
            "summary_deltas": {
                "ic_delta": float(ic_delta),
                "rank_ic_delta": float(rank_ic_delta),
                "has_incremental_signal": bool(rank_ic_delta > 0.01),
            },
        }

    def run_feature_ablation_study(
        self,
        market_data: MarketData,
        model_family: str = "linear",
    ) -> Dict[str, Dict[str, float]]:
        """
        Run feature ablation study:
          Baseline
          Baseline + Depth
          Baseline + Network Density
          Baseline + Agent Belief
          Baseline + All DRR
        """
        combined_features = self.feature_provider.generate_feature_dataset(
            prices=market_data.prices,
            returns=market_data.returns,
        )

        num_features = self._select_numeric_features(combined_features)

        conv_cols = [c for c in num_features.columns if not c.startswith("drr_")]
        depth_cols = [c for c in num_features.columns if "depth" in c]
        net_cols = [c for c in num_features.columns if "network" in c or "edges" in c]
        agent_cols = [c for c in num_features.columns if "agent" in c or "rooted" in c]
        all_drr = [c for c in num_features.columns if c.startswith("drr_")]

        fwd_return = (
            market_data.returns[self.target_symbol]
            .rolling(window=self.forward_horizon)
            .sum()
            .shift(-self.forward_horizon)
        )

        common_idx = num_features.index.intersection(fwd_return.dropna().index)
        y = fwd_return.loc[common_idx]

        n_samples = len(common_idx)
        n_train = int(n_samples * self.config.train_ratio)
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        study_variants = {
            "Baseline": conv_cols,
            "Baseline + Depth": conv_cols + depth_cols,
            "Baseline + Network": conv_cols + net_cols,
            "Baseline + Agent": conv_cols + agent_cols,
            "Baseline + All DRR": conv_cols + all_drr,
        }

        ablation_results = {}

        for variant_name, feature_list in study_variants.items():
            X = num_features[feature_list].loc[common_idx]
            X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]

            mdl = self._instantiate_model(model_family)
            mdl.fit(X_tr, y_train)
            preds = pd.Series(mdl.predict(X_te), index=X_te.index)
            metrics = calculate_ic_metrics(preds, y_test)
            ablation_results[variant_name] = metrics

        return ablation_results

    def _instantiate_model(self, model_family: str):
        """Instantiate predictive model."""
        if model_family == "linear":
            return Ridge(alpha=1.0)
        elif model_family in ("lightgbm", "tree", "gradient_boosting"):
            return HistGradientBoostingRegressor(random_state=self.config.random_state, max_iter=50)
        elif model_family == "mlp":
            return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
