"""
Dynamic Resonance Rooting (DRR) Framework
Author: Christopher Woodyard (2025)
"""

from .qbism_agent import QBistAgent
import numpy as np
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

    def analyze_influence_network(
        self,
        *,
        rooting_method: str = "lagged_correlation",
        rooting_max_lag: Optional[int] = None,
        rooting_n_surrogates: int = 25,
        rooting_random_state: Optional[int] = 0,
        rooting_alpha: float = 0.05,
        rooting_surrogate_method: str = "circular_shift",
        rooting_correction: str = "max_statistic",
    ) -> Optional[nx.DiGraph]:
        """
        Analyze directed relationships between system components.

        Args:
            rooting_method: Rooting backend to use. ``"lagged_correlation"`` is
                the deterministic default. ``"transfer_entropy"`` requests
                transfer entropy, uses it when ``pyinform`` is available, and
                otherwise falls back to lagged correlation.
            rooting_max_lag: Maximum lag to search. ``None`` resolves to
                ``max(1, tau)``.
            rooting_n_surrogates: Number of surrogate draws to use for
                significance testing. Set to ``0`` to disable inference.
            rooting_random_state: Seed for reproducible surrogate generation.
            rooting_alpha: Threshold applied to the selected p-value.
            rooting_surrogate_method: Null model for surrogate generation,
                either ``"circular_shift"`` or ``"permutation"``.
            rooting_correction: P-value selection rule, either
                ``"max_statistic"`` for adjusted p-values or ``"none"`` for raw
                p-values.

        Returns:
            Optional[nx.DiGraph]: Directed graph representing significant
            influence edges only. The full rooting result remains available in
            ``self.rooting_results`` and includes ``score_matrix`` with the
            compatibility alias ``transfer_entropy``, raw and adjusted p-values,
            exploratory ``candidate_edges``, selected ``significant_edges``, the
            effective correction, the surrogate method, and
            ``minimum_attainable_p_value``. The returned ``method`` names the
            effective backend actually used. When ``n_surrogates == 0``, the
            off-diagonal p-values are ``NaN``, ``inference_available`` is false,
            and the graph may be empty even if candidate edges exist.
        """
        if self.phase_space is None or self.phase_space.shape[1] < 2:
            logger.warning("Multivariate data required for influence network analysis")
            return None

        try:
            max_lag = max(1, self.tau) if rooting_max_lag is None else rooting_max_lag
            rooting_results = self.rooting_analyzer.analyze(
                self.phase_space,
                max_lag=max_lag,
                n_surrogates=rooting_n_surrogates,
                random_state=rooting_random_state,
                alpha=rooting_alpha,
                method=rooting_method,
                surrogate_method=rooting_surrogate_method,
                correction=rooting_correction,
            )
            self.rooting_results = rooting_results
            score_matrix = rooting_results["score_matrix"]

            G = nx.DiGraph()
            n_dims = score_matrix.shape[0]

            for i in range(n_dims):
                G.add_node(f"dim_{i}")

            for edge in rooting_results["significant_edges"]:
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["weight"],
                    p_value=edge["p_value"],
                    adjusted_p_value=edge["adjusted_p_value"],
                    lag=edge["lag"],
                )

            self.influence_network = G
            return G

        except Exception as e:
            self.rooting_results = {"error": str(e), "method": rooting_method}
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
        rooting_method: str = "lagged_correlation",
        rooting_max_lag: Optional[int] = None,
        rooting_n_surrogates: int = 25,
        rooting_random_state: Optional[int] = 0,
        rooting_alpha: float = 0.05,
        rooting_surrogate_method: str = "circular_shift",
        rooting_correction: str = "max_statistic",
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
            rooting_method (str): Rooting backend to use for multivariate runs.
                ``"lagged_correlation"`` is the default. ``"transfer_entropy"``
                requests transfer entropy, uses it when ``pyinform`` is
                available, and otherwise falls back to lagged correlation.
            rooting_max_lag (Optional[int]): Maximum lag to search for rooting
                analysis. ``None`` resolves to ``max(1, tau)``.
            rooting_n_surrogates (int): Number of surrogate draws used for
                significance testing. ``0`` disables inference.
            rooting_random_state (Optional[int]): Seed for reproducible
                surrogate generation.
            rooting_alpha (float): Threshold applied to the selected p-value.
            rooting_surrogate_method (str): Surrogate null model, either
                ``"circular_shift"`` or ``"permutation"``.
            rooting_correction (str): P-value selection rule, either
                ``"max_statistic"`` for adjusted p-values or ``"none"`` for raw
                p-values.

        Example:
            >>> results = drr.analyze_system(
            ...     data,
            ...     multivariate=True,
            ...     window_size=256,
            ...     rooting_method="lagged_correlation",
            ...     rooting_n_surrogates=25,
            ...     rooting_random_state=42,
            ... )

        Returns:
            Dict: Complete analysis results. For multivariate inputs, the
            returned ``rooting_analysis`` payload exposes ``score_matrix`` as
            the canonical matrix and preserves ``transfer_entropy`` as a legacy
            alias. It also includes raw and adjusted p-values, exploratory
            ``candidate_edges``, selected ``significant_edges``, the correction
            mode, surrogate method, surrogate count, and
            ``minimum_attainable_p_value``. The returned ``method`` names the
            effective backend actually used. If rooting fails inside the
            facade, ``rooting_analysis`` contains a structured error record
            instead of being omitted.
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
                self.rooting_results = {}
                network = self.analyze_influence_network(
                    rooting_method=rooting_method,
                    rooting_max_lag=rooting_max_lag,
                    rooting_n_surrogates=rooting_n_surrogates,
                    rooting_random_state=rooting_random_state,
                    rooting_alpha=rooting_alpha,
                    rooting_surrogate_method=rooting_surrogate_method,
                    rooting_correction=rooting_correction,
                )
                if network is not None:
                    results["influence_network"] = network
                if self.rooting_results:
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

        from .visualizations import plot_analysis_results

        plot_analysis_results(
            results,
            data,
            sampling_rate=self.sampling_rate,
            embedding_dim=self.embedding_dim,
            tau=self.tau,
            phase_space=self.phase_space,
            save_plots=save_plots,
            show=show,
        )


# Make the class available for import
__all__ = ["DynamicResonanceRooting", "BenchmarkSystems"]
