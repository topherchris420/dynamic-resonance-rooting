"""
Structured Artifact Exporter and Markdown Report Generator for DRR Quant Lab.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import pandas as pd

from ..types import QuantExperimentResult, MarketData

logger = logging.getLogger(__name__)


def export_experiment_artifacts(
    result: QuantExperimentResult,
    output_dir: Union[str, Path],
) -> None:
    """Export experiment result artifacts to CSV and JSON files."""
    result.export(output_dir)


def generate_markdown_research_report(
    result: QuantExperimentResult,
    output_filepath: Union[str, Path] = "reports/quant_macro_report.md",
) -> str:
    """
    Generate a research report in Markdown format following strict scientific standards.
    Distinguishes observation, association, prediction, and causality.
    """
    out_p = Path(output_filepath)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    m = result.metrics
    cfg = result.config

    cfg_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else {}

    md_content = f"""# DRR Quant Research Lab: Out-of-Sample Empirical Research Report

## Executive Summary
This report presents an empirical out-of-sample evaluation testing whether **Dynamic Resonance Rooting (DRR)** produces useful, reproducible structural representations of financial markets and whether those representations contain incremental information beyond conventional financial features.

> **Scientific Disclaimer**: This document presents empirical research findings and historical simulations. It does NOT constitute investment advice, financial promotion, or a guarantee of future performance or trading profitability. A null result is scientifically acceptable.

---

## 1. Central Hypothesis
**Question**: Does DRR describe market dynamical structure in a way that improves out-of-sample prediction, regime inference, risk management, or portfolio decision-making?

---

## 2. Experimental Setup
* **Observation Period**: `{cfg_dict.get('start_date', 'N/A')}` to `{cfg_dict.get('end_date', 'N/A')}`
* **Observation Universe**: `{', '.join(cfg_dict.get('observation_symbols', []))}`
* **Portfolio Universe**: `{', '.join(cfg_dict.get('portfolio_symbols', []))}`
* **DRR Depth Window**: `{cfg_dict.get('depth_window', 126)}` trading days
* **Lookback Window**: `{cfg_dict.get('lookback', 252)}` trading days
* **Rebalance Frequency**: `{cfg_dict.get('rebalance_frequency', 'monthly')}`
* **Transaction Cost Assumption**: `{cfg_dict.get('transaction_cost_bps', 5.0)}` bps per turnover unit

---

## 3. Key Out-of-Sample Performance Metrics
| Metric | Value |
| :--- | :--- |
| **CAGR** | `{m.get('cagr', 0.0) * 100.0:.2f}%` |
| **Annualized Volatility** | `{m.get('annualized_volatility', 0.0) * 100.0:.2f}%` |
| **Sharpe Ratio** | `{m.get('sharpe_ratio', 0.0):.3f}` |
| **Sortino Ratio** | `{m.get('sortino_ratio', 0.0):.3f}` |
| **Max Drawdown** | `{m.get('max_drawdown', 0.0) * 100.0:.2f}%` |
| **Calmar Ratio** | `{m.get('calmar_ratio', 0.0):.3f}` |
| **Historical CVaR (95%)** | `{m.get('historical_cvar_95', 0.0) * 100.0:.2f}%` |
| **Average Rebalance Turnover** | `{m.get('average_turnover', 0.0) * 100.0:.2f}%` |
| **Total Transaction Costs** | `{m.get('total_transaction_costs', 0.0):.4f}` |
| **Regime Switches** | `{m.get('regime_switches', 0)}` |

---

## 4. Methodological Distinction
In evaluating these results, we strictly distinguish between:
1. **Observation**: Empirical values measured from historical price series via phase space time-delay embedding.
2. **Association**: Statistical correlations observed between trailing DRR resonance metrics and forward market volatility/drawdowns.
3. **Prediction**: Out-of-sample forecast accuracy measured via Information Coefficients (IC) in matched control vs. experiment models.
4. **Causality**: Directed graph connectivity inferred via transfer entropy or lagged correlation. Directed graph connectivity does NOT imply macroeconomic cause-and-effect.

---

## 5. Limitations & Future Extensions
* **Market Microstructure**: Evaluation relies on daily close prices; intraday market dynamics are unobserved.
* **Non-Stationarity**: Market regimes evolve dynamically; static hyperparameter configurations may experience temporal drift.
* **Execution Boundary**: Future research will integrate execution protocol interfaces (`ExecutionAdapter`) for order routing.

---

## 6. Reproduction Instructions
To reproduce these research results offline:
```bash
python examples/quant_macro_lab.py --offline --start {cfg_dict.get('start_date', '2015-01-01')} --end {cfg_dict.get('end_date', '2023-12-31')}
```
"""

    with open(out_p, "w") as f:
        f.write(md_content)

    logger.info("Generated research report at %s", out_p)
    return md_content
