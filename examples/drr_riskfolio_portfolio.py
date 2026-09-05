#!/usr/bin/env python3
"""
DRR-Conditioned Riskfolio Portfolio Optimization Example.

Compares static portfolio policies (Mean-Variance, CVaR, Equal Weight) against
DRR-conditioned dynamic regime switching.
"""

import argparse
import logging

from drr_framework.finance.synthetic import load_market_data
from drr_framework.finance.backtest import WalkForwardBacktester, QuantMacroConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drr_riskfolio_portfolio")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRR Riskfolio Portfolio Optimizer")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--offline", action="store_true", help="Force synthetic data")

    args = parser.parse_args()
    provider = "synthetic" if args.offline else "synthetic"

    mdata = load_market_data(start_date=args.start, end_date=args.end, provider=provider)
    cfg = QuantMacroConfig(depth_window=126, rooting_surrogates=99)
    tester = WalkForwardBacktester(config=cfg)

    results = {}
    for policy in ["equal_weight", "standard", "high_resonance", None]:
        name = policy.capitalize() if policy else "DRR Conditioned"
        logger.info("Evaluating portfolio policy: %s", name)
        res = tester.run(mdata, policy_override=policy)
        results[name] = res

    print("\nPORTFOLIO POLICY COMPARISON")
    print("-----------------------------------------------------------------------------------")
    print(
        f"{'Policy':<20} | {'Sharpe':<8} | {'Max DD (%)':<10} | {'CVaR 95% (%)':<12} | {'CAGR (%)':<10}"
    )
    print("-" * 70)

    for name, res in results.items():
        m = res.metrics
        print(
            f"{name:<20} | {m['sharpe_ratio']:>8.2f} | {m['max_drawdown']*100.0:>10.2f} | "
            f"{m['historical_cvar_95']*100.0:>12.2f} | {m['cagr']*100.0:>10.2f}"
        )


if __name__ == "__main__":
    main()
