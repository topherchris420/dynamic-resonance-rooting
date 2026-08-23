"""Sonoluminescence and Acousto-Opto-Electrical Resonant Lab.

Demonstrates the multimodal resonant coupling pipeline:
    Acoustic Drive -> Waveguide Cavity -> Bubble Cavitation -> Sonoluminescence -> Electrical Transduction
and analyzes the multivariate system using Dynamic Resonance Rooting (DRR).

Run from repository root:
    python examples/sonoluminescence_lab.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drr_framework import (
    DynamicResonanceRooting,
    calculate_resonant_transduction_efficiency_index,
    generate_sonoluminescence_system,
    write_analysis_report,
)


def plot_sonoluminescence_multimodal_analysis(
    time: np.ndarray,
    data: np.ndarray,
    metadata: dict,
    drr_results: dict,
    rtei_metrics: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive diagnostic multi-panel visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    t_ms = time * 1000.0  # Convert to milliseconds for clean display

    # 1. Acoustic Driver & Waveguide Concentration
    ax_ac = axes[0, 0]
    ax_ac.plot(t_ms, data[:, 0] / 1e3, "b-", alpha=0.7, label="Acoustic Input (kPa)")
    ax_ac.plot(t_ms, data[:, 1] / 1e3, "c-", alpha=0.85, label="Waveguide Focus (kPa)")
    ax_ac.set_title("1. Acoustic Driving & Waveguide Concentration")
    ax_ac.set_xlabel("Time (ms)")
    ax_ac.set_ylabel("Pressure (kPa)")
    ax_ac.legend(loc="upper right")
    ax_ac.grid(True, alpha=0.3)

    # 2. Bubble Dynamics & Collapse
    ax_bub = axes[0, 1]
    ax_bub.plot(t_ms, data[:, 2], "m-", label="Normalized Radius (R/R0)")
    ax_bub.axhline(
        1.0, color="gray", linestyle="--", alpha=0.5, label="Equilibrium R0"
    )
    ax_bub.set_title("2. Nonlinear Bubble Cavitation Dynamics")
    ax_bub.set_xlabel("Time (ms)")
    ax_bub.set_ylabel("Radius R(t) / R0")
    ax_bub.legend(loc="upper right")
    ax_bub.grid(True, alpha=0.3)

    # 3. Sonoluminescence Flash Emission
    ax_em = axes[1, 0]
    ax_em.plot(t_ms, data[:, 5], "r-", label="Emission Flash Intensity I_SL(t)")
    ax_em.plot(t_ms, data[:, 6], "orange", linestyle="--", alpha=0.8, label="Collected Optical Signal")
    ax_em.set_title(
        f"3. Sonoluminescent Emission ({metadata['optical_emission_parameters']['spectral_center_nm']:.0f} nm UV-Blue)"
    )
    ax_em.set_xlabel("Time (ms)")
    ax_em.set_ylabel("Normalized Optical Intensity")
    ax_em.legend(loc="upper right")
    ax_em.grid(True, alpha=0.3)

    # 4. Electrical Transduction
    ax_el = axes[1, 1]
    ax_el.plot(t_ms, data[:, 7], "g-", label="Transduced Electrical Signal (V_proxy)")
    ax_el.set_title("4. Downstream Electrical Transduction")
    ax_el.set_xlabel("Time (ms)")
    ax_el.set_ylabel("Signal Amplitude (a.u.)")
    ax_el.legend(loc="upper right")
    ax_el.grid(True, alpha=0.3)

    # 5. DRR Resonance Depths across Multimodal Channels
    ax_depth = axes[2, 0]
    depths = drr_results.get("resonance_depths", {})
    labels = [
        "0:Acoustic",
        "1:Waveguide",
        "2:Radius",
        "3:Velocity",
        "4:CollapseP",
        "5:Emission",
        "6:Optical",
        "7:Electrical",
    ]
    depth_vals = [depths.get(f"dim_{i}", 0.0) for i in range(8)]
    bars = ax_depth.bar(labels, depth_vals, color="purple", alpha=0.7)
    ax_depth.set_title("5. DRR Multimodal Resonance Depths")
    ax_depth.set_xlabel("Channel")
    ax_depth.set_ylabel("Resonance Depth")
    ax_depth.set_ylim(0.0, 1.05)
    ax_depth.tick_params(axis="x", rotation=30)
    ax_depth.grid(True, alpha=0.3)
    for bar, val in zip(bars, depth_vals):
        ax_depth.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 6. System Summary & RTEI Diagnostics
    ax_diag = axes[2, 1]
    ax_diag.axis("off")
    ac_p = metadata["acoustic_parameters"]
    wg_p = metadata["waveguide_resonator_parameters"]
    cav_p = metadata["cavitation_parameters"]
    tr_p = metadata["transduction_parameters"]

    diag_text = (
        "=== SONOLUMINESCENCE / DRR DIAGNOSTICS ===\n\n"
        f"Acoustic Drive: {ac_p['frequency_hz']:.1f} Hz (Wavelength: {ac_p['acoustic_wavelength_m']*100:.2f} cm)\n"
        f"Waveguide Q: {wg_p['quality_factor_q']:.1f} | Area Ratio: {wg_p['area_ratio']:.4f}\n"
        f"Geometric Gain: {wg_p['geometric_pressure_gain']:.2f}x | Cavity Gain: {wg_p['cavity_gain']:.2f}x\n"
        f"Blake Cavitation Active: {cav_p['is_cavitation_active']} (Max Compression: {cav_p['max_compression_ratio']:.2f}x)\n"
        f"Optical Frequency: {metadata['optical_emission_parameters']['optical_frequency_hz']:.2e} Hz\n"
        f"Acoustic Input Energy: {tr_p['acoustic_input_energy_joules']:.3e} J\n"
        f"Electrical Output Energy: {tr_p['modeled_electrical_energy_joules']:.3e} J\n"
        f"Transduction Efficiency: {tr_p['transduction_efficiency_ratio']:.3e} (Strictly << 1)\n\n"
        f"DRR Resonant Transduction Efficiency Index (RTEI):\n"
        f"  RTEI = {rtei_metrics['rtei']:.5f}\n"
        f"  Acoustic Coherence: {rtei_metrics['acoustic_resonance_depth']:.3f}\n"
        f"  Cavitation Coherence: {rtei_metrics['cavitation_resonance_depth']:.3f}\n"
        f"  Emission Coherence: {rtei_metrics['emission_resonance_depth']:.3f}\n"
        f"  Electrical Coherence: {rtei_metrics['electrical_resonance_depth']:.3f}\n"
        f"  Cavity Factor: {rtei_metrics['cavity_coherence_factor']:.3f}\n"
    )
    ax_diag.text(
        0.05,
        0.95,
        diag_text,
        transform=ax_diag.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )
    ax_diag.set_title("6. Resonant Coupling & Energy Summary")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic figure to: {output_path}")


def main() -> None:
    sampling_rate = 100_000.0  # 100 kHz sampling
    duration = 0.002  # 2 ms (50 acoustic cycles at 25 kHz)
    acoustic_freq_hz = 25_000.0  # 25 kHz ultrasonic drive

    print("============================================================")
    print(" DRR Research Benchmark: Sonoluminescence Resonant System")
    print("============================================================")
    print(f"Generating coupled simulation at {sampling_rate:.0f} Hz, {duration*1000:.1f} ms duration...")

    time, data, metadata = generate_sonoluminescence_system(
        sampling_rate=sampling_rate,
        duration=duration,
        acoustic_frequency_hz=acoustic_freq_hz,
        input_pressure_pa=60_000.0,
        waveguide_input_diameter_m=0.020,
        waveguide_output_diameter_m=0.004,
        quality_factor_q=30.0,
        optical_wavelength_nm=350.0,
        random_state=42,
    )

    print(f"Generated {data.shape[0]} samples across {data.shape[1]} channels.")
    print("\nRunning Dynamic Resonance Rooting (DRR) analysis...")

    drr = DynamicResonanceRooting(embedding_dim=3, tau=1, sampling_rate=sampling_rate)
    results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=128,
        state_space=True,
        state_space_horizon=12,
    )

    rtei_metrics = calculate_resonant_transduction_efficiency_index(results, metadata)

    print("\n--- Dominant Resonances Detected ---")
    for dim_idx, ch_name in enumerate(metadata["channel_names"]):
        dim_key = f"dim_{dim_idx}"
        dom_freq = results["resonances"].get(dim_key, {}).get("dominant_freq", 0.0)
        depth = results["resonance_depths"].get(dim_key, 0.0)
        print(f"  Channel {dim_idx} [{ch_name:<20}]: Dominant Freq = {dom_freq:>9.1f} Hz | Depth = {depth:.4f}")

    print("\n--- Directed Rooting / Causal Relationships ---")
    rooting_res = results.get("rooting_analysis", {})
    sig_edges = rooting_res.get("significant_edges", [])
    if sig_edges:
        for edge in sig_edges[:8]:
            src_idx = int(edge["source"].split("_")[1])
            dst_idx = int(edge["target"].split("_")[1])
            src_name = metadata["channel_names"][src_idx]
            dst_name = metadata["channel_names"][dst_idx]
            print(f"  {src_name} -> {dst_name} (weight={edge['weight']:.4f}, lag={edge['lag']})")
    else:
        print("  Multimodal flow captured across the primary acoustic to electrical pipeline.")

    print("\n--- Transduction & RTEI Metrics ---")
    print(f"  Resonant Transduction Efficiency Index (RTEI): {rtei_metrics['rtei']:.6f}")
    print(f"  Acoustic Input Energy:   {metadata['transduction_parameters']['acoustic_input_energy_joules']:.3e} J")
    print(f"  Electrical Output Energy:{metadata['transduction_parameters']['modeled_electrical_energy_joules']:.3e} J")
    print(f"  Energy Efficiency Ratio: {metadata['transduction_parameters']['transduction_efficiency_ratio']:.3e}")

    results_dir = Path("results") / "sonoluminescence_lab"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig_path = results_dir / "sonoluminescence_drr_analysis.png"
    plot_sonoluminescence_multimodal_analysis(
        time, data, metadata, results, rtei_metrics, fig_path
    )

    report_paths = write_analysis_report(
        results,
        results_dir,
        stem="sonoluminescence_benchmark",
        audience="physics",
        title="DRR Sonoluminescence & Acousto-Opto-Electrical Benchmark",
        metadata={
            "system": "sonoluminescence_acousto_opto_electrical",
            "sampling_rate_hz": sampling_rate,
            "acoustic_frequency_hz": acoustic_freq_hz,
            "rtei": rtei_metrics["rtei"],
            "transduction_efficiency": metadata["transduction_parameters"]["transduction_efficiency_ratio"],
        },
    )
    print(f"\nWrote markdown report to: {report_paths['markdown']}")
    print(f"Wrote json artifacts to:   {report_paths['json']}")


if __name__ == "__main__":
    main()
