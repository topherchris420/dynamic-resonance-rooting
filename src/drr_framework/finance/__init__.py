"""
DRR Quant Lab - Quantitative Finance Application Layer for Dynamic Resonance Rooting Framework.

Exports core market data loading, DRR regime analysis, feature engineering,
Qlib/Riskfolio/VectorBT adapters, portfolio policy, walk-forward backtesting, and performance evaluation tools.
"""

from .config import QuantResearchConfig
from .types import MarketData, MarketResonanceState, QuantExperimentResult
from .data import (
    DEFAULT_OBSERVATION_UNIVERSE,
    DEFAULT_PORTFOLIO_UNIVERSE,
    DataFrameMarketDataProvider,
    MarketDataProvider,
    OpenBBMarketDataProvider,
    QlibMarketDataProvider,
    calculate_returns,
    validate_market_data,
)
from .features import (
    DRRMarketFeatureGenerator,
    aggregate_market_resonance_state,
    build_drr_feature_matrix,
    extract_conventional_features,
)
from .synthetic import generate_synthetic_market_data, load_market_data
from .regimes import PortfolioRegimePolicy, analyze_market_regime
from .portfolio import annual_to_daily_rf, optimize_portfolio
from .metrics import calculate_performance_metrics, validate_drr_predictive_signal
from .backtest import BacktestResult, QuantMacroConfig, WalkForwardBacktester

__all__ = [
    "QuantResearchConfig",
    "MarketData",
    "MarketResonanceState",
    "QuantExperimentResult",
    "DEFAULT_OBSERVATION_UNIVERSE",
    "DEFAULT_PORTFOLIO_UNIVERSE",
    "MarketDataProvider",
    "DataFrameMarketDataProvider",
    "OpenBBMarketDataProvider",
    "QlibMarketDataProvider",
    "validate_market_data",
    "calculate_returns",
    "generate_synthetic_market_data",
    "load_market_data",
    "extract_conventional_features",
    "aggregate_market_resonance_state",
    "DRRMarketFeatureGenerator",
    "build_drr_feature_matrix",
    "PortfolioRegimePolicy",
    "analyze_market_regime",
    "annual_to_daily_rf",
    "optimize_portfolio",
    "calculate_performance_metrics",
    "validate_drr_predictive_signal",
    "BacktestResult",
    "QuantMacroConfig",
    "WalkForwardBacktester",
]
