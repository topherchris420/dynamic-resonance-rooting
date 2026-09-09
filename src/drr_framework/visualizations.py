"""Lightweight visualization helpers for DRR analysis outputs."""

from __future__ import annotations

from typing import Any, Mapping, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from ._spectral import welch as signal_welch

logger = logging.getLogger(__name__)


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


def plot_analysis_results(
    results: Mapping[str, Any],
    data: np.ndarray,
    *,
    sampling_rate: float,
    embedding_dim: int,
    tau: int,
    phase_space: Optional[np.ndarray] = None,
    save_plots: bool = False,
    show: bool = True,
) -> None:
    """Render the comprehensive DRR report from results and explicit context.

    This function does not depend on an analysis instance. Saving writes
    ``drr_analysis_results.png`` in the current directory; ``show=False``
    closes the figure after any save for headless use.
    """
    if not results:
        logger.warning("No results to plot")
        return

    # Determine number of subplots needed
    n_plots = 3 if "influence_network" in results else 2
    fig, axes = plt.subplots(n_plots, 2, figsize=(15, 5 * n_plots))

    if n_plots == 2:
        axes = axes.reshape(2, 2)

    # Plot 1: Original time series
    if data.ndim == 1:
        t = np.arange(len(data)) / sampling_rate
        axes[0, 0].plot(t, data, "b-", alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title("Original Time Series")
    else:
        t = np.arange(len(data)) / sampling_rate
        for i in range(min(3, data.shape[1])):
            axes[0, 0].plot(t, data[:, i], alpha=0.7, linewidth=0.5, label=f"Dim {i}")
        axes[0, 0].set_title("Original Time Series (First 3 Dimensions)")
        axes[0, 0].legend()

    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Phase space (if embedded)
    if phase_space is not None and phase_space.shape[1] >= 2:
        ps_x = phase_space[:, 0]
        ps_y = phase_space[:, 1]
        # Downsample for large datasets to keep plotting fast
        max_plot_points = 10_000
        if len(ps_x) > max_plot_points:
            step = len(ps_x) // max_plot_points
            ps_x = ps_x[::step]
            ps_y = ps_y[::step]
        axes[0, 1].plot(ps_x, ps_y, "r-", alpha=0.6, linewidth=0.3, rasterized=True)
        axes[0, 1].set_title("Phase Space Reconstruction")
        axes[0, 1].set_xlabel("X(t)")
        axes[0, 1].set_ylabel("X(t + tau)")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Phase space\nnot available",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Phase Space")

    # Plot 3: Power spectrum
    if "resonances" in results and results["resonances"]:
        dim_key = list(results["resonances"].keys())[0]
        resonance_data = results["resonances"][dim_key]

        if "frequencies" in resonance_data and len(resonance_data["frequencies"]) > 0:
            # Plot power spectrum
            series = phase_space[:, 0] if phase_space is not None else data.flatten()
            freqs, psd = signal_welch(series, fs=sampling_rate, nperseg=min(256, len(series) // 4))

            axes[1, 0].semilogy(freqs, psd, "b-", alpha=0.7)

            # Mark detected resonances
            res_freqs = resonance_data["frequencies"]
            res_power = resonance_data["power"]
            if len(res_freqs) > 0:
                axes[1, 0].scatter(res_freqs, res_power, color="red", s=50, zorder=5)

            axes[1, 0].set_title("Power Spectrum with Detected Resonances")
            axes[1, 0].set_xlabel("Frequency (Hz)")
            axes[1, 0].set_ylabel("Power Spectral Density")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No resonances\ndetected",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Power Spectrum")

    # Plot 4: Resonance depths
    if "resonance_depths" in results and results["resonance_depths"]:
        depths = results["resonance_depths"]
        dims = list(depths.keys())
        values = list(depths.values())

        bars = axes[1, 1].bar(dims, values, alpha=0.7, color="green")
        axes[1, 1].set_title("Resonance Depths by Dimension")
        axes[1, 1].set_xlabel("Dimension")
        axes[1, 1].set_ylabel("Resonance Depth")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

    # Plot 5: Influence network (if available)
    if "influence_network" in results and n_plots == 3:
        G = results["influence_network"]
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, iterations=50)
            nx.draw(
                G,
                pos,
                ax=axes[2, 0],
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                edge_color="gray",
                arrows=True,
            )
            axes[2, 0].set_title("Influence Network")
        else:
            axes[2, 0].text(
                0.5,
                0.5,
                "No significant\ninfluences detected",
                ha="center",
                va="center",
                transform=axes[2, 0].transAxes,
            )
            axes[2, 0].set_title("Influence Network")

        # Summary statistics
        summary_text = f"Analysis Summary:\n\n"
        summary_text += f"Sampling Rate: {sampling_rate} Hz\n"
        summary_text += f"Embedding Dim: {embedding_dim}\n"
        summary_text += f"Time Delay: {tau}\n\n"

        if results["resonance_depths"]:
            avg_depth = np.mean(list(results["resonance_depths"].values()))
            summary_text += f"Avg Resonance Depth: {avg_depth:.4f}\n"

        if "influence_network" in results:
            G = results["influence_network"]
            summary_text += f"Network Nodes: {G.number_of_nodes()}\n"
            summary_text += f"Network Edges: {G.number_of_edges()}\n"

        axes[2, 1].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[2, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[2, 1].set_title("Analysis Summary")
        axes[2, 1].axis("off")

    plt.tight_layout()

    if save_plots:
        plt.savefig("drr_analysis_results.png", dpi=300, bbox_inches="tight")
        logger.info("Plot saved as 'drr_analysis_results.png'")

    if show:
        plt.show()
    else:
        plt.close(fig)
