"""
Dynamic Resonance Rooting (DRR) Framework 
Author: Christopher Woodyard (2025)
"""
from .qbism_agent import QBistAgent
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import networkx as nx
from typing import Dict, List, Optional, Union, Tuple
import warnings
from sklearn.preprocessing import StandardScaler

# Import your existing modules
from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator
from .benchmarks import BenchmarkSystems

warnings.filterwarnings('ignore')


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
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.sampling_rate = sampling_rate
        
        # Initialize component modules
        self.resonance_detector = ResonanceDetector()
        self.rooting_analyzer = RootingAnalyzer()
        self.depth_calculator = DepthCalculator()
        
        # Initialize QBist Agent
        self.agent = QBistAgent(
            prior=0.5,
            learning_rate=0.1
        )

        # Storage for analysis results
        self.phase_space = None
        self.resonances = {}
        self.influence_network = None
        self.resonance_depths = {}
        self.belief_states = {}
        
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
        self, 
        data: np.ndarray, 
        method: str = 'fft',
        peak_height_ratio: float = 0.1
    ) -> Dict:
        """
        Identify dominant resonances in time series data.
        
        Args:
            data (np.ndarray): Time series data (1D or multivariate)
            method (str): Method for spectral analysis ('fft' or 'wavelet')
            peak_height_ratio (float): Ratio of max power for peak detection
            
        Returns:
            Dict: Dictionary containing resonance information
        """
        if data.ndim == 1:
            # Single time series - perform embedding
            self.phase_space = self.time_delay_embedding(data)
        else:
            # Multivariate data - use directly
            self.phase_space = data
            
        resonances = {}
        
        for dim in range(self.phase_space.shape[1]):
            series = self.phase_space[:, dim]
            key = f'dim_{dim}'
            
            if method == 'fft':
                # FFT-based resonance detection
                # Optimization: Use rfft for real-valued input (approx 2x faster)
                fft_vals = rfft(series)
                freqs = rfftfreq(len(series), 1 / self.sampling_rate)
                power_spectrum = np.abs(fft_vals)**2
                
                # Only consider positive frequencies
                # rfftfreq returns [0, ..., nyquist] (all non-negative)
                # We filter out 0 (DC component)
                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                power_pos = power_spectrum[pos_mask]
                
                # Find peaks
                if len(power_pos) > 0:
                    peaks, properties = signal.find_peaks(
                        power_pos, 
                        height=np.max(power_pos) * peak_height_ratio
                    )
                    
                    # For rfft, fft_vals has same indices as freqs (length N/2 + 1)
                    # So pos_mask aligns correctly with fft_vals
                    resonances[key] = {
                        'frequencies': freqs_pos[peaks],
                        'amplitudes': np.abs(fft_vals[pos_mask][peaks]),
                        'power': power_pos[peaks],
                        'dominant_freq': freqs_pos[peaks[np.argmax(power_pos[peaks])]] if len(peaks) > 0 else 0
                    }
                else:
                    resonances[key] = {
                        'frequencies': np.array([]),
                        'amplitudes': np.array([]),
                        'power': np.array([]),
                        'dominant_freq': 0
                    }
                    
            elif method == 'wavelet':
                # Wavelet-based analysis (simplified)
                try:
                    from scipy.signal import cwt, morlet2
                    scales = np.logspace(0, 3, 50)
                    coefficients = cwt(series, morlet2, scales)
                    frequencies = self.sampling_rate / scales
                    
                    # Average power across time for each frequency
                    power_freq = np.mean(np.abs(coefficients)**2, axis=1)
                    peaks, _ = signal.find_peaks(power_freq, height=np.max(power_freq) * peak_height_ratio)
                    
                    resonances[key] = {
                        'frequencies': frequencies[peaks],
                        'power': power_freq[peaks],
                        'cwt_matrix': coefficients,
                        'dominant_freq': frequencies[peaks[np.argmax(power_freq[peaks])]] if len(peaks) > 0 else 0
                    }
                except ImportError:
                    print("Wavelet analysis requires scipy. Falling back to FFT.")
                    return self.detect_resonances(data, method='fft', peak_height_ratio=peak_height_ratio)
        
        self.resonances = resonances
        return resonances
    
    def calculate_resonance_depths(self, window_size: int = 100) -> Dict[str, float]:
        """
        Calculate resonance depth for each dimension.
        
        Args:
            window_size (int): Window size for depth calculation
            
        Returns:
            Dict[str, float]: Resonance depths for each dimension
        """
        if self.phase_space is None:
            raise ValueError("Must detect resonances first")
            
        depths = {}
        self.belief_states = {}
        
        for dim in range(self.phase_space.shape[1]):
            key = f'dim_{dim}'
            series = self.phase_space[:, dim]
            
            # Calculate depth using rolling statistics
            result = self.depth_calculator.calculate(series, window_size)
            resonance_depth = result['resonance_depth']
            depths[key] = resonance_depth

            # Update agent belief
            belief_state = self.agent.update_belief(resonance_depth)
            self.belief_states[key] = belief_state
            
        self.resonance_depths = depths
        return depths
    
    def analyze_influence_network(self) -> Optional[nx.DiGraph]:
        """
        Analyze causal relationships between system components.
        
        Returns:
            Optional[nx.DiGraph]: Directed graph representing influence network
        """
        if self.phase_space is None or self.phase_space.shape[1] < 2:
            print("Multivariate data required for influence network analysis")
            return None
            
        try:
            # Use the rooting analyzer
            rooting_results = self.rooting_analyzer.analyze(self.phase_space)
            te_matrix = rooting_results['transfer_entropy']
            
            # Create directed graph
            G = nx.DiGraph()
            n_dims = te_matrix.shape[0]
            
            # Add nodes
            for i in range(n_dims):
                G.add_node(f'dim_{i}')
            
            # Add edges based on transfer entropy
            threshold = np.mean(te_matrix) + np.std(te_matrix)
            for i in range(n_dims):
                for j in range(n_dims):
                    if i != j and te_matrix[i, j] > threshold:
                        G.add_edge(f'dim_{i}', f'dim_{j}', weight=te_matrix[i, j])
            
            self.influence_network = G
            return G
            
        except Exception as e:
            print(f"Error in influence network analysis: {e}")
            return None
    
    def analyze_system(
        self, 
        data: np.ndarray, 
        multivariate: bool = False,
        window_size: int = 100
    ) -> Dict:
        """
        Perform complete DRR analysis on system data.
        
        Args:
            data (np.ndarray): Input time series data
            multivariate (bool): Whether data is multivariate
            window_size (int): Window size for depth calculation
            
        Returns:
            Dict: Complete analysis results
        """
        results = {}
        
        try:
            # Step 1: Detect resonances
            print("Detecting resonances...")
            resonances = self.detect_resonances(data)
            results['resonances'] = resonances
            
            # Step 2: Calculate resonance depths
            print("Calculating resonance depths...")
            depths = self.calculate_resonance_depths(window_size)
            results['resonance_depths'] = depths
            results['agent_belief'] = self.belief_states

            # Determine rooted status (epistemic rooting)
            # Use the mean belief if multivariate, or check if any dimension is rooted?
            # User instructions: "rooted = belief_state > 0.65"
            # We'll calculate a system-wide belief or check if all/any are rooted.
            # Assuming system rootedness based on the final belief state of the agent
            # (which accumulates across dimensions in the current implementation,
            # but wait, the agent accumulates history, but self.belief is the CURRENT belief).
            # If we iterate dimensions, the final belief reflects all dimensions.

            results['is_rooted'] = self.agent.belief > 0.65
            
            # Step 3: Analyze influence network (if multivariate)
            if multivariate and data.ndim > 1:
                print("Analyzing influence network...")
                network = self.analyze_influence_network()
                if network:
                    results['influence_network'] = network
            
            print("DRR analysis complete!")
            return results
            
        except Exception as e:
            print(f"Error in system analysis: {e}")
            return {}
    
    def plot_results(self, results: Dict, data: np.ndarray, save_plots: bool = False):
        """
        Create comprehensive visualization of DRR analysis results.
        
        Args:
            results (Dict): Analysis results from analyze_system
            data (np.ndarray): Original input data
            save_plots (bool): Whether to save plots to files
        """
        if not results:
            print("No results to plot")
            return
            
        # Determine number of subplots needed
        n_plots = 3 if 'influence_network' in results else 2
        fig, axes = plt.subplots(n_plots, 2, figsize=(15, 5 * n_plots))
        
        if n_plots == 2:
            axes = axes.reshape(2, 2)
        
        # Plot 1: Original time series
        if data.ndim == 1:
            t = np.arange(len(data)) / self.sampling_rate
            axes[0, 0].plot(t, data, 'b-', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title('Original Time Series')
        else:
            t = np.arange(len(data)) / self.sampling_rate
            for i in range(min(3, data.shape[1])):
                axes[0, 0].plot(t, data[:, i], alpha=0.7, linewidth=0.5, label=f'Dim {i}')
            axes[0, 0].set_title('Original Time Series (First 3 Dimensions)')
            axes[0, 0].legend()
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Phase space (if embedded)
        if self.phase_space is not None and self.phase_space.shape[1] >= 2:
            axes[0, 1].plot(self.phase_space[:, 0], self.phase_space[:, 1], 
                           'r-', alpha=0.6, linewidth=0.3)
            axes[0, 1].set_title('Phase Space Reconstruction')
            axes[0, 1].set_xlabel('X(t)')
            axes[0, 1].set_ylabel('X(t+τ)')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Phase space\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Phase Space')
        
        # Plot 3: Power spectrum
        if 'resonances' in results and results['resonances']:
            dim_key = list(results['resonances'].keys())[0]
            resonance_data = results['resonances'][dim_key]
            
            if 'frequencies' in resonance_data and len(resonance_data['frequencies']) > 0:
                # Plot power spectrum
                series = self.phase_space[:, 0] if self.phase_space is not None else data.flatten()
                freqs, psd = signal.welch(series, fs=self.sampling_rate, nperseg=min(256, len(series)//4))
                
                axes[1, 0].semilogy(freqs, psd, 'b-', alpha=0.7)
                
                # Mark detected resonances
                res_freqs = resonance_data['frequencies']
                res_power = resonance_data['power']
                if len(res_freqs) > 0:
                    axes[1, 0].scatter(res_freqs, res_power, color='red', s=50, zorder=5)
                
                axes[1, 0].set_title('Power Spectrum with Detected Resonances')
                axes[1, 0].set_xlabel('Frequency (Hz)')
                axes[1, 0].set_ylabel('Power Spectral Density')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No resonances\ndetected', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Power Spectrum')
        
        # Plot 4: Resonance depths
        if 'resonance_depths' in results and results['resonance_depths']:
            depths = results['resonance_depths']
            dims = list(depths.keys())
            values = list(depths.values())
            
            bars = axes[1, 1].bar(dims, values, alpha=0.7, color='green')
            axes[1, 1].set_title('Resonance Depths by Dimension')
            axes[1, 1].set_xlabel('Dimension')
            axes[1, 1].set_ylabel('Resonance Depth')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 5: Influence network (if available)
        if 'influence_network' in results and n_plots == 3:
            G = results['influence_network']
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G)
                nx.draw(G, pos, ax=axes[2, 0], with_labels=True, 
                       node_color='lightblue', node_size=500, 
                       edge_color='gray', arrows=True)
                axes[2, 0].set_title('Influence Network')
            else:
                axes[2, 0].text(0.5, 0.5, 'No significant\ninfluences detected',
                               ha='center', va='center', transform=axes[2, 0].transAxes)
                axes[2, 0].set_title('Influence Network')
            
            # Summary statistics
            summary_text = f"Analysis Summary:\n\n"
            summary_text += f"Sampling Rate: {self.sampling_rate} Hz\n"
            summary_text += f"Embedding Dim: {self.embedding_dim}\n"
            summary_text += f"Time Delay: {self.tau}\n\n"
            
            if results['resonance_depths']:
                avg_depth = np.mean(list(results['resonance_depths'].values()))
                summary_text += f"Avg Resonance Depth: {avg_depth:.4f}\n"
            
            if 'influence_network' in results:
                G = results['influence_network']
                summary_text += f"Network Nodes: {G.number_of_nodes()}\n"
                summary_text += f"Network Edges: {G.number_of_edges()}\n"
            
            axes[2, 1].text(0.1, 0.9, summary_text, transform=axes[2, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2, 1].set_title('Analysis Summary')
            axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('drr_analysis_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'drr_analysis_results.png'")
        
        plt.show()


# Make the class available for import
__all__ = ['DynamicResonanceRooting', 'BenchmarkSystems']
