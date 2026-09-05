"""
Portfolio Optimization Package Exports.
"""

from .policy import DRRRegimePortfolioPolicy, annual_to_daily_rf
from .riskfolio_adapter import RiskfolioAllocator


def optimize_portfolio(
    returns,
    policy_mode: str = "standard",
    annual_rf: float = 0.04,
    min_weight: float = 0.0,
    max_weight: float = 0.50,
    allow_short: bool = False,
    trading_days: int = 252,
):
    allocator = RiskfolioAllocator(
        annual_rf=annual_rf,
        trading_days=trading_days,
        min_weight=min_weight,
        max_weight=max_weight,
        allow_short=allow_short,
    )
    return allocator.optimize(returns=returns, policy_mode=policy_mode)


__all__ = [
    "annual_to_daily_rf",
    "DRRRegimePortfolioPolicy",
    "RiskfolioAllocator",
    "optimize_portfolio",
]
