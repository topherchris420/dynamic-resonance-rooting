#!/usr/bin/env python3
"""
DRR VectorBT Robustness & Parameter Sweeps Example.

Executes parameter sweeps over lookback windows, depth windows, and regime thresholds to evaluate strategy stability.
"""

import argparse
import logging

from drr_framework.finance.synthetic import load_market_data
from drr_framework.finance.validation import VectorBTAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drr_vectorbt_robustness")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRR VectorBT Parameter Robustness Sweeper")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--offline", action="store_true", help="Force synthetic data")

    args = parser.parse_args()

    mdata = load_market_data(start_date=args.start, end_date=args.end, provider="synthetic")
    adapter = VectorBTAdapter(transaction_cost_bps=5.0)

    logger.info("Executing parameter robustness sweep...")
    sweep_df = adapter.run_parameter_robustness_sweep(
        market_data=mdata,
        lookbacks=(126, 252),
        percentiles=(70.0, 80.0, 90.0),
        depth_windows=(63, 126),
    )

    print("\nPARAMETER ROBUSTNESS SWEEP RESULTS")
    print("----------------------------------------------------------------------------------")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
