"""
Anti-Leakage Utilities & Lookahead Invariant Tests.

Defines explicit verification utilities and invariant tests ensuring zero future information leakage.
"""

import logging
from typing import Callable, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assert_no_lookahead_leakage(
    run_pipeline_fn: Callable[[pd.DataFrame], pd.Series],
    full_dataset: pd.DataFrame,
    cutoff_date: str = "2021-12-31",
) -> None:
    """
    Assert lookahead invariant:
      Running the pipeline on full_dataset (e.g. through 2025) and truncated_dataset (through cutoff_date)
      must produce identical outputs for all dates <= cutoff_date.

    Args:
        run_pipeline_fn: Callable taking market DataFrame and returning a Series of signals/decisions.
        full_dataset: Full historical market dataset.
        cutoff_date: String YYYY-MM-DD cutoff date.

    Raises:
        AssertionError if signals dated <= cutoff_date differ.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    truncated_dataset = full_dataset[full_dataset.index <= cutoff_ts].copy()

    # Run on full dataset
    full_signals = run_pipeline_fn(full_dataset)

    # Run on truncated dataset
    truncated_signals = run_pipeline_fn(truncated_dataset)

    # Slice full signals up to cutoff date
    full_signals_sub = full_signals[full_signals.index <= cutoff_ts]

    # Align indexes
    common_idx = full_signals_sub.index.intersection(truncated_signals.index)
    if common_idx.empty:
        raise AssertionError("No overlapping signal dates between full and truncated runs.")

    sig_full = full_signals_sub.loc[common_idx]
    sig_trunc = truncated_signals.loc[common_idx]

    if isinstance(sig_full.iloc[0], (float, int, np.number)):
        diff = (sig_full - sig_trunc).abs().max()
        if diff > 1e-6:
            raise AssertionError(f"Lookahead leakage detected! Max numerical difference: {diff:.8f}")
    else:
        mismatches = (sig_full != sig_trunc).sum()
        if mismatches > 0:
            raise AssertionError(f"Lookahead leakage detected! {mismatches} mismatching signal decisions.")

    logger.info("Anti-leakage invariant test PASSED. Zero lookahead leakage detected.")
