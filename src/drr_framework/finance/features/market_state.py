"""
Global Resonance Aggregation & Market State Extraction.

Helper utilities to aggregate per-dimension DRR metrics into system-level MarketResonanceState objects.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..types import MarketResonanceState


def aggregate_market_resonance_state(
    analysis_results: Dict[str, Any],
    timestamp: Optional[pd.Timestamp] = None,
    requested_rooting_method: str = "transfer_entropy",
    n_vars: int = 1,
) -> MarketResonanceState:
    """
    Aggregate raw DRR system analysis outputs into a canonical MarketResonanceState object.

    Aggregates:
      - mean_depth, max_depth, min_depth, depth_dispersion across dimensions
      - network_density = E / [N * (N - 1)] for directed graph without self-edges
      - effective rooting backend provenance (lagged_correlation vs transfer_entropy)
      - agent belief and is_rooted flag
      - state-space stability diagnostics if present
    """
    resonance_depths = analysis_results.get("resonance_depths", {})
    depth_vals = list(resonance_depths.values())

    if depth_vals:
        mean_depth = float(np.mean(depth_vals))
        max_depth = float(np.max(depth_vals))
        min_depth = float(np.min(depth_vals))
        depth_dispersion = float(np.std(depth_vals))
    else:
        mean_depth, max_depth, min_depth, depth_dispersion = 0.0, 0.0, 0.0, 0.0

    # Extract component details for dynamical properties
    details = analysis_results.get("resonance_depth_details", {})
    spectral_concs, temp_pers, phase_cohs, amp_stabs = [], [], [], []
    for dim_k, d in details.items():
        comps = d.get("components", {}) if isinstance(d, dict) else {}
        spectral_concs.append(comps.get("spectral_concentration", 0.0))
        temp_pers.append(comps.get("temporal_persistence", 0.0))
        phase_cohs.append(comps.get("phase_coherence", 0.0))
        amp_stabs.append(comps.get("amplitude_stability", 0.0))

    spectral_concentration = float(np.mean(spectral_concs)) if spectral_concs else 0.0
    temporal_persistence = float(np.mean(temp_pers)) if temp_pers else 0.0
    phase_coherence = float(np.mean(phase_cohs)) if phase_cohs else 0.0
    amplitude_stability = float(np.mean(amp_stabs)) if amp_stabs else 0.0

    # Network density calculation
    rooting_analysis = analysis_results.get("rooting_analysis", {})
    sig_edges = rooting_analysis.get("significant_edges", [])
    sig_edge_count = len(sig_edges)
    effective_method = rooting_analysis.get("method", requested_rooting_method)

    if n_vars > 1:
        max_possible_edges = n_vars * (n_vars - 1)
        network_density = (
            float(sig_edge_count / max_possible_edges) if max_possible_edges > 0 else 0.0
        )
    else:
        network_density = 0.0

    agent_beliefs = analysis_results.get("agent_belief", {})
    if agent_beliefs is None:
        agent_belief = 0.5
    elif isinstance(agent_beliefs, dict):
        agent_belief = float(np.mean(list(agent_beliefs.values()))) if agent_beliefs else 0.5
    else:
        agent_belief = float(agent_beliefs)

    is_rooted = bool(analysis_results.get("is_rooted", False))
    state_space_diag = analysis_results.get("state_space_analysis", None)

    return MarketResonanceState(
        timestamp=timestamp,
        mean_depth=mean_depth,
        max_depth=max_depth,
        min_depth=min_depth,
        depth_dispersion=depth_dispersion,
        spectral_concentration=spectral_concentration,
        temporal_persistence=temporal_persistence,
        phase_coherence=phase_coherence,
        amplitude_stability=amplitude_stability,
        network_density=network_density,
        significant_edge_count=sig_edge_count,
        agent_belief=agent_belief,
        is_rooted=is_rooted,
        effective_rooting_method=effective_method,
        requested_rooting_method=requested_rooting_method,
        resonance_depths=resonance_depths,
        state_space_diagnostics=state_space_diag,
        provenance={"n_vars": n_vars, "n_edges": sig_edge_count},
    )
