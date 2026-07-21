"""
Dynamic Resonance Rooting (DRR) Framework
Author: Christopher Woodyard (2025)
"""

from .qbism_agent import QBistAgent
import numpy as np
import matplotlib.pyplot as plt
from ._spectral import welch as signal_welch
import networkx as nx
from typing import Dict, Optional
import logging

from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator
from .state_space import analyze_resonance_state_space
from .benchmarks import BenchmarkSystems

logger = logging.getLogger(__name__)


class DynamicResonanceRooting:
    """
    Dynamic Resonance Rooting framework for Complex Adaptive Systems analysis.

    This framework integrates three core modules:
    1. Dynamic Resonance Detection: Identifies dominant oscillatory patterns
    2. Rooting Analysis: Maps causal dependencies using transfer entropy
    3. Resonance Depth Calculation: Quantifies the stability of resonances
    """

    def __init__(self, embedding_dim: int = 3, tau: int = 1, sampling_rate: float = 100.0):
        """
        Initialize the DRR framework.

        Args:
            embedding_dim (int): Dimension for phase space reconstruction
            tau (int): Time delay for time-delay embedding
            sampling_rate (float): Sampling rate of input data in Hz
        """
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if tau < 1:
            raise ValueError("tau must be at least 1")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        self.embedding_dim = embedding_dim
        self.tau = tau
        self.sampling_rate = sampling_rate

        # Initialize component modules
        self.resonance_detector = ResonanceDetector()
        self.rooting_analyzer = RootingAnalyzer()
        self.depth_calculator = DepthCalculator()

        # Initialize QBist Agent
        self.agent = QBistAgent(prior=0.5, learning_rate=0.1)

        # Storage for analysis results
        self.phase_space: Optional[np.ndarray] = None
        self.resonances: Dict[str, Dict[str, object]] = {}
        self.influence_network: Optional[nx.DiGraph] = None
        self.resonance_depths: Dict[str, float] = {}
        self.resonance_depth_details: Dict[str, dict] = {}
        self.rooting_results: Dict[str, object] = {}
        self.belief_states: Dict[str, float] = {}

    def time_delay_embedding(self, data: np.ndarray) -> np.ndarray:
        """
        Perform time-delay embedding for phase space reconstruction.

        Args:
            data (np.ndarray): 1D time series data

        Returns:
            np.ndarray: Embedded phase space coordinates
        """
        if data.ndim != 1:
            raise ValueError("Input data must be 1-dimensional for embedding")

        n = len(data)
        m = self.embedding_dim

        if n <= (m - 1) * self.tau:
            raise ValueError("Data too short for specified embedding parameters")

        embedded = np.zeros((n - (m - 1) * self.tau, m))
        for i in range(m):
            start_idx = i * self.tau
            end_idx = n - (m - 1 - i) * self.tau
            embedded[:, i] = data[start_idx:end_idx]

        return embedded

    def detect_resonances(
        self, data: np.ndarray, method: str = "fft", peak_height_ratio: float = 0.1
    ) -> Dict:
        """
        Identify dominant resonances in time series data.

        Args:
            data (np.ndarray): Time series data (1D or multivariate)
            method (str): Method for spectral analysis
                ('fft', 'welch', 'wavelet', or 'markov')
            peak_height_ratio (float): Ratio of max power for peak detection

        Returns:
            Dict: Dictionary containing resonance information
        """
        if data.ndim == 1:
            self.phase_space = self.time_delay_embedding(data)
        else:
            self.phase_space = data

        phase_space = self.phase_space
        if phase_space is None:
            raise RuntimeError("phase_space was not initialized")

        resonances: Dict[str, Dict[str, object]] = {}

        for dim in range(phase_space.shape[1]):
            series = phase_space[:, dim]
            key = f"dim_{dim}"

            detector_results = self.resonance_detector.detect(
                data=series,
                method=method,
                sampling_rate=self.sampling_rate,
                peak_height_ratio=peak_height_ratio,
            )

            freqs = detector_results.get("dominant_freq", np.array([]))
            resonances[key] = {
                "frequencies": freqs,
                "power": detector_results.get("peak_magnitude", np.array([])),
                "dominant_freq": freqs[0] if len(freqs) > 0 else 0,
            }

        self.resonances = resonances
        return resonances

    def calculate_resonance_depths(self, window_size: int = 100) -> Dict[str, float]:
        """
        Calculate normalized resonance depth for each dimension.

        The public return value stays backward-compatible as ``{dimension: float}``.
        Full component details are stored in ``self.resonance_depth_details`` and
        included by ``analyze_system``.

        The QBist agent is reset to its prior on every call, so repeated
        analyses on the same instance produce identical belief states.
        """
        if self.phase_space is None:
            raise ValueError("Must detect resonances first")

        depths = {}
        details = {}
        self.agent.reset()
        self.belief_states = {}

        for dim in range(self.phase_space.shape[1]):
            key = f"dim_{dim}"
            series = self.phase_space[:, dim]
            raw_frequencies = self.resonances.get(key, {}).get("frequencies", np.array([]))
            resonance_frequencies = (
                raw_frequencies if isinstance(raw_frequencies, np.ndarray) else np.array([])
            )

            result = self.depth_calculator.calculate(
                series,
                window_size,
                sampling_rate=self.sampling_rate,
                resonance_frequencies=resonance_frequencies,
            )
            resonance_depth = result["resonance_depth"]
            depths[key] = resonance_depth
            details[key] = result

            belief_state = self.agent.update_belief(resonance_depth)
            self.belief_states[key] = belief_state

        self.resonance_depths = depths
        self.resonance_depth_details = details
        return depths

    def analyze_influence_network(self) -> Optional[nx.DiGraph]:
        """
        Analyze directed relationships between system components.

        Returns:
            Optional[nx.DiGraph]: Directed graph representing significant influence edges.
        """
        if self.phase_space is None or self.phase_space.shape[1] < 2:
            logger.warning("Multivariate data required for influence network analysis")
            return None

        try:
            rooting_results = self.rooting_analyzer.analyze(
                self.phase_space,
                max_lag=max(1, self.tau),
                n_surrogates=25,
                random_state=0,
            )
            self.rooting_results = rooting_results
            te_matrix = rooting_results["transfer_entropy"]

            G = nx.DiGraph()
            n_dims = te_matrix.shape[0]

            for i in range(n_dims):
                G.add_node(f"dim_{i}")

            if rooting_results.get("significant_edges"):
                for edge in rooting_results["significant_edges"]:
                    G.add_edge(
                        edge["source"],
                        edge["target"],
                        weight=edge["weight"],
                        p_value=edge["p_value"],
                        lag=edge["lag"],
                    )
            else:
                threshold = rooting_results.get(
                    "edge_threshold", float(np.mean(te_matrix) + np.std(te_matrix))
                )
                for i in range(n_dims):
                    for j in range(n_dims):
                        if i != j and te_matrix[i, j] > threshold:
                            G.add_edge(f"dim_{i}", f"dim_{j}", weight=te_matrix[i, j])

            self.influence_network = G
            return G

        except Exception as e:
            logger.error("Error in influence network analysis: %s", e)
            return None

    def analyze_system(
        self,
        data: np.ndarray,
        multivariate: bool = False,
        window_size: int = 100,
        state_space: bool = True,
        state_space_horizon: int = 12,
        method: str = "fft",
        peak_height_ratio: float = 0.1,
    ) -> Dict[str, object]:
        """
        Perform complete DRR analysis on system data.

        Args:
            data (np.ndarray): Input time series data
            multivariate (bool): Whether data is multivariate
            window_size (int): Window size for depth calculation
            state_space (bool): Whether to attach DSGE-inspired state-space diagnostics
            state_space_horizon (int): Horizon for impulse-response diagnostics
            method (str): Spectral method for resonance detection
                ('fft', 'welch', 'wavelet', or 'markov')
            peak_height_ratio (float): Ratio of max power for peak detection

        Returns:
            Dict: Complete analysis results
        """
        results: Dict[str, object] = {}

        try:
            # Step 1: Detect resonances
            logger.info("Detecting resonances...")
            resonances = self.detect_resonances(
                data, method=method, peak_height_ratio=peak_height_ratio
            )
            results["resonances"] = resonances

            # Step 2: Calculate resonance depths
            logger.info("Calculating resonance depths...")
            depths = self.calculate_resonance_depths(window_size)
            results["resonance_depths"] = depths
            results["resonance_depth_details"] = self.resonance_depth_details
            results["agent_belief"] = self.belief_states

            # Epistemic rooting: the agent accumulates evidence across all
            # dimensions, so its final belief reflects the whole system.
            results["is_rooted"] = self.agent.belief > 0.65

            # Step 3: Analyze influence network (if multivariate)
            if multivariate and data.ndim > 1:
                logger.info("Analyzing influence network...")
                network = self.analyze_influence_network()
                if network is not None:
                    results["influence_network"] = network
                    results["rooting_analysis"] = self.rooting_results

            # Step 4: Fit DSGE-inspired state-space diagnostics
            if state_space and self.phase_space is not None and self.phase_space.shape[0] >= 3:
                logger.info("Fitting DSGE-inspired state-space diagnostics...")
                state_names = [f"dim_{idx}" for idx in range(self.phase_space.shape[1])]
                try:
                    results["state_space_analysis"] = analyze_resonance_state_space(
                        self.phase_space,
                        impulse_horizon=state_space_horizon,
                        state_names=state_names,
                        observable_names=state_names,
                    )
                except Exception as exc:
                    logger.warning("State-space diagnostics failed: %s", exc)
                    results["state_space_analysis"] = {"error": str(exc)}
            logger.info("DRR analysis complete.")
            return results

        except Exception:
            logger.exception("Error in system analysis")
            raise

    def plot_results(
        self, results: Dict, data: np.ndarray, save_plots: bool = False, show: bool = True
    ):
        """
        Create comprehensive visualization of DRR analysis results.

        Args:
            results (Dict): Analysis results from analyze_system
            data (np.ndarray): Original input data
            save_plots (bool): Whether to save plots to files
            show (bool): Whether to display the figure; pass False in headless
                or batch environments (the figure is closed after saving)
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
            t = np.arange(len(data)) / self.sampling_rate
            axes[0, 0].plot(t, data, "b-", alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title("Original Time Series")
        else:
            t = np.arange(len(data)) / self.sampling_rate
            for i in range(min(3, data.shape[1])):
                axes[0, 0].plot(t, data[:, i], alpha=0.7, linewidth=0.5, label=f"Dim {i}")
            axes[0, 0].set_title("Original Time Series (First 3 Dimensions)")
            axes[0, 0].legend()

        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Phase space (if embedded)
        if self.phase_space is not None and self.phase_space.shape[1] >= 2:
            ps_x = self.phase_space[:, 0]
            ps_y = self.phase_space[:, 1]
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
                series = self.phase_space[:, 0] if self.phase_space is not None else data.flatten()
                freqs, psd = signal_welch(
                    series, fs=self.sampling_rate, nperseg=min(256, len(series) // 4)
                )

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
            summary_text += f"Sampling Rate: {self.sampling_rate} Hz\n"
            summary_text += f"Embedding Dim: {self.embedding_dim}\n"
            summary_text += f"Time Delay: {self.tau}\n\n"

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


# Make the class available for import
__all__ = ["DynamicResonanceRooting", "BenchmarkSystems"]
