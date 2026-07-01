"""Lightweight visualization helpers for DRR analysis outputs."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: Mapping[str, Any], data: np.ndarray) -> None:
    """Plot input data for a compact visual smoke check.

    The richer ``DynamicResonanceRooting.plot_results`` method remains the main
    report-style visualization API. This helper is intentionally minimal for
    notebooks and quick experiments.
    """

    _ = results
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title("DRR Analysis Input Trace")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
