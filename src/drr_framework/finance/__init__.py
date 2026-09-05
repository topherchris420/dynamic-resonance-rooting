"""
DRR Quant Lab - Financial Application Layer for Dynamic Resonance Rooting Framework.

Exports core market data loading, DRR regime analysis, portfolio policy,
walk-forward backtesting, and performance evaluation tools.
"""

from .features import (
    DEFAULT_OBSERVATION_UNIVERSE,
    DEFAULT_PORTFOLIO_UNIVERSE,
    MarketData,
    calculate_returns,
    generate_synthetic_market_data,
    load_market_data,
    validate_market_data,
)
from .regimes import (
    MarketResonanceState,
    PortfolioRegimePolicy,
    analyze_market_regime,
)
from .portfolio import (
    annual_to_daily_rf,
    optimize_portfolio,
)
from .metrics import (
    calculate_performance_metrics,
    validate_drr_predictive_signal,
)
from .backtest import (
    BacktestResult,
    QuantMacroConfig,
    WalkForwardBacktester,
)

__all__ = [
    "DEFAULT_OBSERVATION_UNIVERSE",
    "DEFAULT_PORTFOLIO_UNIVERSE",
    "MarketData",
    "calculate_returns",
    "generate_synthetic_market_data",
    "load_market_data",
    "validate_market_data",
    "MarketResonanceState",
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
