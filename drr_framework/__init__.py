"""
Dynamic Resonance Rooting (DRR) Framework

A computational framework for analyzing Complex Adaptive Systems through
Dynamic Resonance Detection, Rooting Analysis, and Resonance Depth Calculation.

Author: Christopher Woodyard
License: MIT
"""

# Only import modules that actually exist in your current structure
from .benchmarks import BenchmarkSystems
from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator, AnomalyDetector
from .realtime import RealTimeDRR

from .analysis import DynamicResonanceRooting

__version__ = "0.1.0"
__author__ = "Christopher Woodyard"
__email__ = "ciao_chris@example.com"

__all__ = [
    'BenchmarkSystems', 
    'ResonanceDetector',
    'RootingAnalyzer',
    'DepthCalculator',
    'AnomalyDetector',
    'RealTimeDRR',
    'DynamicResonanceRooting',
]
