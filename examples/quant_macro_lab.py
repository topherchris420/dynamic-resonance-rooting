#!/usr/bin/env python3
"""
DRR Quant Macro Lab - Example Script

Demonstrates Dynamic Resonance Rooting (DRR) applied to real/synthetic multivariate financial-market data
as a structural regime-detection layer that conditions downstream portfolio risk decisions.

Pipeline:
  Market Data -> DRR Structural State Inference -> Portfolio Policy -> Walk-Forward Evaluation
"""

import argparse
import logging
from pathlib import Path
import sys

from drr_framework.finance import (
    DEFAULT_OBSERVATION_UNIVERSE,
    DEFAULT_PORTFOLIO_UNIVERSE,
    QuantMacroConfig,
    WalkForwardBacktester,
    load_market_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quant_macro_lab")


def run_quant_macro_lab(
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    lookback: int = 252,
    rebalance: str = "monthly",
    provider: str = "synthetic",
    output_dir: str = "results/quant_macro_lab",
) -> None:
    """Run DRR Quant Macro Lab walk-forward evaluation across strategies."""
    print("==================================================")
    print("DRR Quant Macro Lab")
    print("==================================================")
    print(f"Observation Universe:  {' '.join(DEFAULT_OBSERVATION_UNIVERSE)}")
    print(f"Portfolio Universe:    {' '.join(DEFAULT_PORTFOLIO_UNIVERSE)}")
    print(f"Data Provider:         {provider}")
    print(f"Lookback Window:       {lookback} days")
    print(f"Rebalance Frequency:   {rebalance}")
    print(f"Transaction Cost:      5 bps")
    print("--------------------------------------------------\n")

    # 1. Load Market Data
    logger.info("Loading market data...")
    market_data = load_market_data(
        symbols=DEFAULT_OBSERVATION_UNIVERSE,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        portfolio_symbols=DEFAULT_PORTFOLIO_UNIVERSE,
    )

    # 2. Configure Backtest
    config = QuantMacroConfig(
        lookback=lookback,
        depth_window=126,
        rebalance_frequency=rebalance,
        transaction_cost_bps=5.0,
        threshold_type="expanding_percentile",
        percentile=80.0,
        rooting_method="transfer_entropy",
        rooting_surrogates=99,
        random_state=42,
    )

    backtester = WalkForwardBacktester(config=config)

    # 3. Execute Strategies
    logger.info("Running walk-forward DRR analysis and strategies...")
    strategies = {
        "Equal Weight": "equal_weight",
        "Mean Variance": "standard",
        "Static CVaR": "high_resonance",
        "DRR Conditioned": None,  # Dynamic DRR policy selection
        "SPY Buy-Hold": "spy",
    }

    results = {}
    for name, policy_override in strategies.items():
        logger.info("Evaluating strategy: %s", name)
        res = backtester.run(market_data=market_data, policy_override=policy_override)
        results[name] = res

    # 4. Print Comparison Table
    print("\nRESULTS SUMMARY")
    print(
        "---------------------------------------------------------------------------------------------------"
    )
    header = f"{'Strategy':<20} | {'CAGR (%)':<10} | {'Vol (%)':<10} | {'Sharpe':<8} | {'Max DD (%)':<10} | {'CVaR 95%':<10}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        m = res.metrics
        cagr_pct = m["cagr"] * 100.0
        vol_pct = m["annualized_volatility"] * 100.0
        sharpe = m["sharpe_ratio"]
        max_dd_pct = m["max_drawdown"] * 100.0
        cvar_pct = m["historical_cvar_95"] * 100.0

        print(
            f"{name:<20} | {cagr_pct:>10.2f} | {vol_pct:>10.2f} | {sharpe:>8.2f} | {max_dd_pct:>10.2f} | {cvar_pct:>10.2f}"
        )

    # 5. Print DRR Diagnostics
    drr_res = results["DRR Conditioned"]
    drr_m = drr_res.metrics
    states = drr_res.resonance_states

    print("\nDRR DIAGNOSTICS")
    print("--------------------------------------------------")
    print(f"Regime switches:        {drr_m['regime_switches']}")
    print(f"Avg holding period:     {drr_m['average_holding_period_days']:.1f} days")
    print(f"Avg turnover:           {drr_m['average_turnover']*100.0:.2f}% per rebalance")
    print(f"Mean resonance depth:   {states['mean_depth'].mean():.4f}")
    print(f"Avg network density:    {states['network_density'].mean():.4f}")
    print(f"High-resonance ratio:   {(drr_res.regimes == 'high_resonance').mean()*100.0:.1f}%")

    # 6. Export Results
    out_path = Path(output_dir)
    drr_res.export(out_path)
    print(f"\nExported DRR Conditioned strategy artifacts to '{out_path.resolve()}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DRR Quant Macro Lab Walk-Forward Strategy Evaluator"
    )
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window in days")
    parser.add_argument(
        "--rebalance",
        type=str,
        default="monthly",
        choices=["daily", "weekly", "monthly", "quarterly"],
        help="Rebalance frequency",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="synthetic",
        choices=["synthetic", "yfinance", "openbb"],
        help="Market data provider",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Force synthetic offline data provider"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/quant_macro_lab",
        help="Directory for exported results",
    )

    args = parser.parse_args()

    provider = "synthetic" if args.offline else args.provider

    run_quant_macro_lab(
        start_date=args.start,
        end_date=args.end,
        lookback=args.lookback,
        rebalance=args.rebalance,
        provider=provider,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
