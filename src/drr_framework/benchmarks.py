"""Synthetic benchmark systems used by DRR examples and tests."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_micro_doppler_analog(
    duration: float = 30.0,
    sampling_rate: float = 100.0,
    n_channels: int = 3,
    target_frequency_hz: float = 0.3,
    clutter_frequency_hz: float = 0.02,
    lag: int = 2,
    regime_change_time: Optional[float] = None,
    noise_scale: float = 0.05,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate multi-channel micro-Doppler radar DSP analog time series.

    Simulates range-gated radar returns as a cross-domain stress test for
    DRR diagnostics (resonance detection, rooting analysis, resonance depth,
    and state-space change detection).

    Features:
    - Strong quasi-static clutter near 0 Hz.
    - Weak low-frequency target oscillator (e.g., breathing scale) with known lag across channels.
    - Injected regime change partway through (e.g., standing still -> walking).

    Args:
        duration: Total duration of signal in seconds.
        sampling_rate: Sampling frequency in Hz.
        n_channels: Number of range-bin / sensor channels.
        target_frequency_hz: Weak target oscillation frequency in Hz (e.g., 0.3 Hz breathing).
        clutter_frequency_hz: Quasi-static clutter drift frequency near 0 Hz (e.g., 0.02 Hz).
        lag: Inter-channel sample lag from channel 0 to successive channels.
        regime_change_time: Time in seconds where regime transition occurs (default: duration / 2).
        noise_scale: Standard deviation of additive Gaussian noise.
        random_state: Random seed for reproducible noise generation.

    Returns:
        Tuple of (t, data, metadata):
            t: 1D NumPy array of time stamps in seconds.
            data: (N, n_channels) 2D NumPy array of channel measurements.
            metadata: Dictionary containing ground-truth parameters.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if n_channels < 1:
        raise ValueError("n_channels must be at least 1")

    if regime_change_time is None:
        regime_change_time = duration / 2.0

    rng = np.random.default_rng(random_state)
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    # Generate source breathing target signal
    source_target = 0.4 * np.sin(2 * np.pi * target_frequency_hz * t)

    # Injected regime change: standing still -> walking (stride frequency ~1.8 Hz)
    walking_mask = (t >= regime_change_time).astype(float)
    walking_component = 1.2 * np.sin(2 * np.pi * 1.8 * t) * walking_mask

    data = np.zeros((n_samples, n_channels))

    for c in range(n_channels):
        # Quasi-static clutter (strong offset/drift near 0 Hz)
        clutter = 3.0 * np.cos(2 * np.pi * clutter_frequency_hz * t + (c * 0.1)) + 1.5 * (c + 1)

        # Lagged target signal for channel c
        c_lag = c * lag
        if c_lag > 0:
            target_c = np.roll(source_target, c_lag)
            target_c[:c_lag] = 0.0
            walking_c = np.roll(walking_component, c_lag)
            walking_c[:c_lag] = 0.0
        else:
            target_c = source_target
            walking_c = walking_component

        # Add Gaussian noise
        noise = rng.normal(scale=noise_scale, size=n_samples)

        data[:, c] = clutter + target_c + walking_c + noise

    metadata = {
        "benchmark_system": "micro_doppler_sensing_analog",
        "sampling_rate_hz": sampling_rate,
        "duration_seconds": duration,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "target_frequency_hz": target_frequency_hz,
        "clutter_frequency_hz": clutter_frequency_hz,
        "walking_stride_frequency_hz": 1.8,
        "inter_channel_lag_samples": lag,
        "regime_change_time_seconds": regime_change_time,
        "noise_scale": noise_scale,
        "channel_names": [f"channel_{c}" for c in range(n_channels)],
        "regimes": ["standing_still", "walking"],
    }

    return t, data, metadata


class BenchmarkSystems:
    """Factory methods for canonical dynamical-system benchmark data."""

    @staticmethod
    def generate_lorenz_data(
        duration: float = 30,
        dt: float = 0.01,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Lorenz-system data with deterministic Euler integration."""

        logger.info("Generating Lorenz benchmark data")
        state = np.asarray(initial_state or [1.0, 1.0, 1.0], dtype=float)
        n_steps = int(duration / dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = state
        sigma, rho, beta = 10, 28, 8 / 3
        for i in range(n_steps - 1):
            x, y, z = xyz[i]
            xyz[i + 1] = [
                x + sigma * (y - x) * dt,
                y + (x * (rho - z) - y) * dt,
                z + (x * y - beta * z) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz

    @staticmethod
    def generate_heston_data(
        duration: float = 252,
        dt: float = 1 / 252,
        initial_state: Optional[dict] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stochastic-volatility data from a compact Heston model.

        Args:
            duration: Number of model years or periods represented by the run.
            dt: Step size.
            initial_state: Optional mapping with ``s0`` and ``v0`` keys.
            random_state: Optional seed for reproducible stochastic draws.
        """

        logger.info("Generating Heston benchmark data")
        state = initial_state or {"s0": 100, "v0": 0.04}
        rng = np.random.default_rng(random_state)
        n_steps = int(duration * (1 / dt))
        s = np.zeros(n_steps)
        v = np.zeros(n_steps)
        s[0] = state["s0"]
        v[0] = state["v0"]

        kappa, theta, sigma, rho = 2.0, 0.04, 0.2, -0.7

        # Pre-draw the correlated Brownian increments; only the state recursion
        # needs to stay sequential.
        w_s = rng.normal(size=n_steps - 1)
        w_v = rho * w_s + np.sqrt(1 - rho**2) * rng.normal(size=n_steps - 1)

        for i in range(1, n_steps):
            s[i] = s[i - 1] * np.exp(
                (0.05 - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * w_s[i - 1]
            )
            v[i] = np.maximum(
                0,
                v[i - 1]
                + kappa * (theta - v[i - 1]) * dt
                + sigma * np.sqrt(v[i - 1] * dt) * w_v[i - 1],
            )

        t = np.linspace(0, duration, n_steps)
        return t, np.vstack((s, v)).T

    @staticmethod
    def generate_fitzhugh_nagumo_data(
        duration: float = 500,
        dt: float = 0.1,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate FitzHugh-Nagumo excitable-system benchmark data."""

        logger.info("Generating FitzHugh-Nagumo benchmark data")
        state = np.asarray(initial_state or [0.1, 0.1], dtype=float)
        n_steps = int(duration / dt)
        xy = np.zeros((n_steps, 2))
        xy[0] = state
        a, b, c = 0.7, 0.8, 0.08

        for i in range(n_steps - 1):
            x, y = xy[i]
            xy[i + 1] = [
                x + (x - x**3 / 3 - y) * dt,
                y + c * (x + a - b * y) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xy

    @staticmethod
    def generate_rossler_data(
        duration: float = 30,
        dt: float = 0.01,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Roessler-system data with deterministic Euler integration."""

        logger.info("Generating Roessler benchmark data")
        state = np.asarray(initial_state or [1.0, 1.0, 1.0], dtype=float)
        n_steps = int(duration / dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = state
        a, b, c = 0.2, 0.2, 5.7
        for i in range(n_steps - 1):
            x, y, z = xyz[i]
            xyz[i + 1] = [
                x + (-y - z) * dt,
                y + (x + a * y) * dt,
                z + (b + z * (x - c)) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz

    @staticmethod
    def generate_sonoluminescence_data(
        sampling_rate: float = 100_000.0,
        duration: float = 0.002,
        acoustic_frequency_hz: float = 25_000.0,
        sound_speed_m_s: float = 1482.0,
        input_pressure_pa: float = 60_000.0,
        resonator_length_m: float = 0.02964,
        waveguide_input_diameter_m: float = 0.020,
        waveguide_output_diameter_m: float = 0.004,
        bubble_radius_m: float = 5.0e-6,
        quality_factor_q: float = 30.0,
        optical_wavelength_nm: float = 350.0,
        optical_collection_efficiency: float = 0.15,
        conversion_efficiency: float = 0.25,
        detector_gain: float = 10.0,
        noise_scale: float = 0.005,
        waveguide_material: Optional[Any] = None,
        copper_solute_fraction: float = 0.0,
        boron_solute_fraction: float = 0.0,
        noble_gas_fraction: float = 0.01,
        noble_gas_species: str = "argon",
        dopant_mixture: Optional[Any] = None,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate sonoluminescence / acousto-opto-electrical benchmark data."""
        logger.info("Generating Sonoluminescence benchmark data")
        from .sonoluminescence import generate_sonoluminescence_system

        return generate_sonoluminescence_system(
            sampling_rate=sampling_rate,
            duration=duration,
            acoustic_frequency_hz=acoustic_frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
            input_pressure_pa=input_pressure_pa,
            resonator_length_m=resonator_length_m,
            waveguide_input_diameter_m=waveguide_input_diameter_m,
            waveguide_output_diameter_m=waveguide_output_diameter_m,
            bubble_radius_m=bubble_radius_m,
            quality_factor_q=quality_factor_q,
            optical_wavelength_nm=optical_wavelength_nm,
            optical_collection_efficiency=optical_collection_efficiency,
            conversion_efficiency=conversion_efficiency,
            detector_gain=detector_gain,
            noise_scale=noise_scale,
            waveguide_material=waveguide_material,
            copper_solute_fraction=copper_solute_fraction,
            boron_solute_fraction=boron_solute_fraction,
            noble_gas_fraction=noble_gas_fraction,
            noble_gas_species=noble_gas_species,  # type: ignore[arg-type]
            dopant_mixture=dopant_mixture,
            random_state=random_state,
        )

    @staticmethod
    def generate_micro_doppler_analog(
        duration: float = 30.0,
        sampling_rate: float = 100.0,
        n_channels: int = 3,
        target_frequency_hz: float = 0.3,
        clutter_frequency_hz: float = 0.02,
        lag: int = 2,
        regime_change_time: Optional[float] = None,
        noise_scale: float = 0.05,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate micro-Doppler sensing analog benchmark data."""
        logger.info("Generating Micro-Doppler sensing analog benchmark data")
        return generate_micro_doppler_analog(
            duration=duration,
            sampling_rate=sampling_rate,
            n_channels=n_channels,
            target_frequency_hz=target_frequency_hz,
            clutter_frequency_hz=clutter_frequency_hz,
            lag=lag,
            regime_change_time=regime_change_time,
            noise_scale=noise_scale,
            random_state=random_state,
        )
