#!/usr/bin/env python3
"""
Primary Qlib + DRR Matched Experiment Script.

Compares Control (Qlib model + conventional features) vs Experiment (Qlib model + conventional + DRR features)
and runs a feature ablation study.
"""

import argparse
import logging
from pathlib import Path

from drr_framework.finance.synthetic import load_market_data
from drr_framework.finance.config import QuantResearchConfig
from drr_framework.finance.qlib import QlibDRRMatchedExperiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qlib_drr_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qlib + DRR Matched Experiment Engine")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "lightgbm", "mlp"])
    parser.add_argument("--provider", type=str, default="synthetic", choices=["synthetic", "openbb"])
    parser.add_argument("--offline", action="store_true", help="Force offline synthetic data")

    args = parser.parse_args()
    provider = "synthetic" if args.offline else args.provider

    logger.info("Loading market data...")
    mdata = load_market_data(start_date=args.start, end_date=args.end, provider=provider)

    cfg = QuantResearchConfig(depth_window=126, rooting_surrogates=99)
    exp = QlibDRRMatchedExperiment(config=cfg, target_symbol="SPY", forward_horizon=5)

    logger.info("Running Matched Experiment (Model: %s)...", args.model)
    matched_res = exp.run_matched_experiment(mdata, model_family=args.model)

    print("\n==================================================")
    print("QLIB + DRR MATCHED EXPERIMENT RESULTS")
    print("==================================================")
    print(f"Control IC:        {matched_res['control']['metrics']['ic']:.4f}")
    print(f"Experiment IC:     {matched_res['experiment']['metrics']['ic']:.4f}")
    print(f"Delta IC:          {matched_res['summary_deltas']['ic_delta']:+.4f}")
    print(f"Control Rank IC:   {matched_res['control']['metrics']['rank_ic']:.4f}")
    print(f"Experiment Rank IC:{matched_res['experiment']['metrics']['rank_ic']:.4f}")
    print(f"Delta Rank IC:     {matched_res['summary_deltas']['rank_ic_delta']:+.4f}")
    print("--------------------------------------------------\n")

    logger.info("Running Feature Ablation Study...")
    ablation = exp.run_feature_ablation_study(mdata, model_family=args.model)

    print("FEATURE ABLATION STUDY")
    print("--------------------------------------------------")
    print(f"{'Variant':<25} | {'IC':<8} | {'Rank IC':<8} | {'RMSE':<8}")
    print("-" * 55)
    for variant, metrics in ablation.items():
        print(f"{variant:<25} | {metrics['ic']:>8.4f} | {metrics['rank_ic']:>8.4f} | {metrics['rmse']:>8.4f}")


if __name__ == "__main__":
    main()
