"""
Dynamic Resonance Rooting (DRR) Framework

A computational framework for analyzing Complex Adaptive Systems through
Dynamic Resonance Detection, Rooting Analysis, and Resonance Depth Calculation.

Author: Christopher Woodyard
License: MIT
"""

# Only import modules that actually exist in your current structure
from .benchmarks import BenchmarkSystems
from .datasets import PolicyResonanceDataset, load_policy_dataset
from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator, AnomalyDetector
from .reporting import (
    render_markdown_report,
    serialize_analysis_results,
    summarize_analysis_results,
    write_analysis_report,
)
from .realtime import RealTimeDRR
from .state_space import (
    KalmanResult,
    Measurement,
    StateSpaceSystem,
    Transition,
    analyze_resonance_state_space,
    fit_resonance_state_space,
    impulse_response,
    kalman_filter,
)

from .analysis import DynamicResonanceRooting
from .generative_design_suite import GenerativeDesignSuite
from .validation import generate_coupled_oscillator, run_reproduction_experiment

__version__ = "0.2.0"
__author__ = "Christopher Woodyard"
__email__ = "ciao_chris@example.com"

__all__ = [
    'BenchmarkSystems', 
    'PolicyResonanceDataset',
    'load_policy_dataset',
    'ResonanceDetector',
    'RootingAnalyzer',
    'DepthCalculator',
    'AnomalyDetector',
    'RealTimeDRR',
    'Transition',
    'Measurement',
    'StateSpaceSystem',
    'KalmanResult',
    'fit_resonance_state_space',
    'kalman_filter',
    'impulse_response',
    'analyze_resonance_state_space',
    'serialize_analysis_results',
    'summarize_analysis_results',
    'render_markdown_report',
    'write_analysis_report',
    'DynamicResonanceRooting',
    'GenerativeDesignSuite',
    'generate_coupled_oscillator',
    'run_reproduction_experiment',
]
