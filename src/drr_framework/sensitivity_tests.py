"""Automated Sensitivity, Placebo, and Synthetic Null Testing Suite for DRR."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analysis import DynamicResonanceRooting
from .modules import DepthCalculator, ResonanceDetector, RootingAnalyzer

logger = logging.getLogger(__name__)


def run_parameter_sensitivity_experiment(
    data: np.ndarray,
    sampling_rate: float = 100.0,
    window_sizes: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
    tau_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Perform automated sensitivity experiment over parameter variations."""
    if window_sizes is None:
        window_sizes = [32, 64, 128, 256]
    if methods is None:
        methods = ["fft", "welch", "wavelet"]
    if tau_values is None:
        tau_values = [1, 2, 4]

    results = []
    base_depths = []

    for w in window_sizes:
        for m in methods:
            for tau in tau_values:
                try:
                    drr = DynamicResonanceRooting(tau=tau, sampling_rate=sampling_rate)
                    res = drr.analyze_system(data, multivariate=(data.ndim > 1), window_size=w, method=m)
                    depths = res.get("resonance_depths", {})
                    avg_depth = float(np.mean(list(depths.values()))) if depths else 0.0
                    base_depths.append(avg_depth)
                    results.append({
                        "window_size": w,
                        "method": m,
                        "tau": tau,
                        "avg_resonance_depth": avg_depth,
                        "is_rooted": bool(res.get("is_rooted", False)),
                    })
                except Exception as exc:
                    results.append({
                        "window_size": w,
                        "method": m,
                        "tau": tau,
                        "error": str(exc),
                    })

    depth_arr = np.array(base_depths)
    fragility_score = float(np.std(depth_arr) / (np.mean(depth_arr) + 1e-8)) if len(depth_arr) > 0 else 0.0

    return {
        "sensitivity_trials": results,
        "mean_resonance_depth": float(np.mean(depth_arr)) if len(depth_arr) > 0 else 0.0,
        "std_resonance_depth": float(np.std(depth_arr)) if len(depth_arr) > 0 else 0.0,
        "fragility_score": fragility_score,
        "is_fragile": fragility_score > 0.5,
    }


def run_placebo_and_null_tests(
    n_samples: int = 500,
    sampling_rate: float = 100.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute null tests against Gaussian noise, phase-shuffled signals, and decorrelated series."""
    rng = np.random.default_rng(random_state)
    detector = ResonanceDetector()
    depth_calc = DepthCalculator()
    rooting = RootingAnalyzer()

    # Null 1: Pure i.i.d. Gaussian Noise
    gaussian_noise = rng.normal(size=n_samples)
    g_det = detector.detect(gaussian_noise, method="welch", sampling_rate=sampling_rate)
    g_depth = depth_calc.calculate(gaussian_noise, window_size=128, sampling_rate=sampling_rate)

    # Null 2: Phase-Shuffled Signal (Destroys non-linear dynamic structure)
    t = np.arange(n_samples) / sampling_rate
    sine_signal = np.sin(2 * np.pi * 10 * t) + rng.normal(scale=0.1, size=n_samples)
    fft_val = np.fft.fft(sine_signal)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples)
    shuffled_fft = np.abs(fft_val) * np.exp(1j * phases)
    phase_shuffled = np.real(np.fft.ifft(shuffled_fft))

    ps_det = detector.detect(phase_shuffled, method="welch", sampling_rate=sampling_rate)
    ps_depth = depth_calc.calculate(phase_shuffled, window_size=128, sampling_rate=sampling_rate)

    # Null 3: Decorrelated Multivariate Financial Noise (Independent Series)
    multivariate_null = rng.normal(size=(n_samples, 3))
    rooting_null = rooting.analyze(multivariate_null, max_lag=2, n_surrogates=20, random_state=random_state)

    return {
        "gaussian_noise": {
            "dominant_freq": [float(f) for f in g_det.get("dominant_freq", [])[:2]],
            "resonance_depth": float(g_depth["resonance_depth"]),
            "false_positive_detected": float(g_depth["resonance_depth"]) > 0.4,
        },
        "phase_shuffled": {
            "dominant_freq": [float(f) for f in ps_det.get("dominant_freq", [])[:2]],
            "resonance_depth": float(ps_depth["resonance_depth"]),
            "depth_reduction_from_shuffle": float(ps_depth["resonance_depth"]),
        },
        "decorrelated_multivariate": {
            "significant_edges_count": len(rooting_null["significant_edges"]),
            "spurious_edges_detected": len(rooting_null["significant_edges"]) > 0,
        },
    }
