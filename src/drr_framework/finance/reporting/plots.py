"""
DRR Quant Research Graphics & Plotting Utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_quant_lab_summary(
    drr_states: pd.DataFrame,
    returns_series: pd.Series,
    output_filepath: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    Generate professional quantitative research summary graphics.

    Plots:
      1. Mean DRR Resonance Depth over time
      2. Causal Network Density over time
      3. Cumulative Net Strategy Returns vs Benchmarks
      4. Strategy Drawdowns over time
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1. DRR Mean Depth
    if "mean_depth" in drr_states.columns:
        axes[0].plot(drr_states.index, drr_states["mean_depth"], color="navy", label="Mean Depth")
        axes[0].axhline(drr_states["mean_depth"].mean(), color="gray", linestyle="--", alpha=0.7)
        axes[0].set_ylabel("Mean Depth")
        axes[0].set_title("DRR Structural Market Resonance Depth")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

    # 2. Network Density
    if "network_density" in drr_states.columns:
        axes[1].plot(drr_states.index, drr_states["network_density"], color="darkgreen", label="Network Density")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Causal Network Topology Density")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left")

    # 3. Cumulative Strategy Performance
    cum_rets = (1.0 + returns_series).cumprod()
    axes[2].plot(cum_rets.index, cum_rets, color="darkred", label="Strategy Net Equity")
    axes[2].set_ylabel("Growth ($1)")
    axes[2].set_title("Out-of-Sample Cumulative Net Return")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    # 4. Drawdowns
    peak = cum_rets.cummax()
    dd = (cum_rets - peak) / peak
    axes[3].fill_between(dd.index, dd, 0, color="crimson", alpha=0.4, label="Drawdown")
    axes[3].set_ylabel("Drawdown")
    axes[3].set_xlabel("Date")
    axes[3].set_title("Strategy Drawdown Profile")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="lower left")

    plt.tight_layout()

    if output_filepath:
        out_p = Path(output_filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        logger.info("Saved summary graphic to %s", out_p)

    if show:
        plt.show()
    else:
        plt.close(fig)
