"""Tests for Sonoluminescence and Acousto-Opto-Electrical Resonant Benchmark."""

import numpy as np
import pytest

from drr_framework import (
    SPEED_OF_LIGHT,
    AcousticDriver,
    AcousticResonator,
    AcousticWaveguide,
    BubbleDynamics,
    CavitationModel,
    DynamicResonanceRooting,
    OpticalElectricalTransducer,
    SonoluminescenceModel,
    SonoluminescenceSystem,
    calculate_resonant_transduction_efficiency_index,
    generate_sonoluminescence_system,
)
from drr_framework.benchmarks import BenchmarkSystems


def test_acoustic_wavelength_calculation():
    """Verify acoustic wavelength formula lambda_a = c_s / f_a."""
    driver = AcousticDriver(frequency_hz=25_000.0, sound_speed_m_s=1482.0)
    expected_wavelength = 1482.0 / 25_000.0  # 0.05928 m = 5.928 cm
    assert np.isclose(driver.acoustic_wavelength, expected_wavelength)

    # Higher frequency -> shorter wavelength
    driver_hf = AcousticDriver(frequency_hz=50_000.0, sound_speed_m_s=1482.0)
    assert np.isclose(driver_hf.acoustic_wavelength, 1482.0 / 50_000.0)


def test_waveguide_area_ratio_and_geometric_gain():
    """Verify geometric area ratio and idealized pressure amplification."""
    d_in = 0.020  # 20 mm
    d_out = 0.004  # 4 mm
    resonator = AcousticResonator(
        input_diameter_m=d_in,
        output_diameter_m=d_out,
        length_m=0.03,
        taper_profile="conical",
    )

    expected_area_ratio = (d_out / d_in) ** 2  # (4/20)^2 = 0.04
    assert np.isclose(resonator.area_ratio, expected_area_ratio)

    expected_geom_gain = d_in / d_out  # 20/4 = 5.0
    assert np.isclose(resonator.geometric_pressure_gain, expected_geom_gain)

    # Cylindrical waveguide has area ratio 1 and geom gain 1
    cylindrical = AcousticResonator(
        input_diameter_m=d_in,
        output_diameter_m=d_in,
        length_m=0.03,
        taper_profile="cylindrical",
    )
    assert np.isclose(cylindrical.area_ratio, 1.0)
    assert np.isclose(cylindrical.geometric_pressure_gain, 1.0)


def test_resonator_resonance_response_and_detuning():
    """Verify cavity standing-wave response at and off resonance."""
    sound_speed = 1482.0
    f_res = 25_000.0
    # Length tuned to half-wavelength: L = c_s / (2 * f_res)
    l_resonant = sound_speed / (2.0 * f_res)

    resonator = AcousticResonator(
        input_diameter_m=0.02,
        output_diameter_m=0.005,
        length_m=l_resonant,
        quality_factor_q=40.0,
    )

    # Exactly at resonance
    gain_res, detuning_res, length_ratio_res = resonator.cavity_response(
        f_res, sound_speed
    )
    assert np.isclose(detuning_res, 0.0, atol=1e-10)
    assert np.isclose(gain_res, 40.0)  # At resonance, cavity gain is Q
    assert np.isclose(length_ratio_res, 0.5)

    # Off-resonance detuned by 10%
    f_off = f_res * 1.10
    gain_off, detuning_off, _ = resonator.cavity_response(f_off, sound_speed)
    assert np.isclose(detuning_off, 0.10)
    assert gain_off < gain_res
    assert gain_off >= 1.0


def test_cavitation_blake_threshold():
    """Verify Blake threshold formula and cavitation activation check."""
    r0 = 5.0e-6
    sigma = 0.0728
    p0 = 101325.0
    cav = CavitationModel(
        equilibrium_radius_m=r0,
        surface_tension_n_m=sigma,
        ambient_pressure_pa=p0,
    )

    expected_blake_tension = 0.77 * (sigma / r0)
    assert np.isclose(cav.blake_acoustic_threshold_pa, expected_blake_tension)
    assert np.isclose(cav.blake_threshold_pressure, p0 + expected_blake_tension)

    # Below Blake threshold
    assert not cav.is_cavitation_active(expected_blake_tension * 0.5)
    # Above Blake threshold
    assert cav.is_cavitation_active(expected_blake_tension * 1.5)


def test_bubble_dynamics_numerical_stability():
    """Verify Rayleigh-Plesset ODE solver runs stably without NaN/Inf."""
    cav = CavitationModel(equilibrium_radius_m=5.0e-6)
    bubble = BubbleDynamics(cavitation_model=cav)

    sampling_rate = 100_000.0
    duration = 0.001  # 1 ms
    t = np.linspace(0.0, duration, int(sampling_rate * duration))
    # 25 kHz acoustic driving pressure with 120 kPa amplitude
    p_drive = 120_000.0 * np.sin(2 * np.pi * 25_000.0 * t)

    res = bubble.simulate(t, p_drive, sub_steps=10)

    assert np.all(np.isfinite(res["radius"]))
    assert np.all(np.isfinite(res["velocity"]))
    assert np.all(np.isfinite(res["collapse_pressure"]))
    assert np.all(res["radius"] > 0.0)
    assert res["max_compression_ratio"] >= 1.0
    assert len(res["collapse_indices"]) > 0


def test_sonoluminescence_emission_activation():
    """Verify light emission occurs only during violent collapse."""
    emission = SonoluminescenceModel(
        spectral_center_nm=350.0,
        emission_threshold_compression=2.5,
    )

    # Case 1: Active violent collapse
    r0 = 5.0e-6
    n = 100
    t = np.linspace(0, 1e-4, n)
    r_arr = np.full(n, r0)
    v_arr = np.zeros(n)
    # Inject a sharp collapse at index 50
    r_arr[50] = r0 / 4.0  # Compression ratio = 4.0 > 2.5
    v_arr[50] = -800.0

    bubble_active = {
        "radius": r_arr,
        "velocity": v_arr,
        "collapse_indices": np.array([50]),
    }
    res_active = emission.simulate_emission(t, bubble_active)
    assert res_active["peak_emission"] > 0.0
    assert len(res_active["emission_event_times"]) == 1

    # Case 2: Weak oscillation below emission threshold
    r_arr_weak = np.full(n, r0)
    r_arr_weak[50] = r0 / 1.5  # Compression ratio = 1.5 < 2.5
    bubble_weak = {
        "radius": r_arr_weak,
        "velocity": v_arr,
        "collapse_indices": np.array([50]),
    }
    res_weak = emission.simulate_emission(t, bubble_weak)
    assert np.isclose(res_weak["peak_emission"], 0.0)
    assert len(res_weak["emission_event_times"]) == 0


def test_optical_wavelength_and_frequency():
    """Verify optical frequency calculation f = c / lambda."""
    emission = SonoluminescenceModel(spectral_center_nm=350.0)
    expected_freq = SPEED_OF_LIGHT / (350.0 * 1e-9)
    assert np.isclose(emission.optical_frequency_hz, expected_freq)


def test_electrical_transduction_and_energy_bookkeeping():
    """Verify optical-to-electrical transduction and energy conservation."""
    transducer = OpticalElectricalTransducer(
        optical_collection_efficiency=0.20,
        conversion_efficiency=0.50,
        detector_gain=5.0,
    )
    driver = AcousticDriver(input_pressure_pa=50_000.0)
    resonator = AcousticResonator()

    t = np.linspace(0, 0.001, 100)
    emission_pulse = np.zeros(100)
    emission_pulse[45:55] = 1.0  # Normalized flash

    res = transducer.transduce(t, emission_pulse, driver, resonator)

    assert "optical_signal" in res
    assert "electrical_signal" in res
    assert np.all(np.isfinite(res["electrical_signal"]))
    assert np.all(res["electrical_signal"] >= 0.0)
    assert res["acoustic_energy_joules"] > 0.0
    assert res["electrical_energy_joules"] >= 0.0
    assert res["transduction_efficiency"] >= 0.0
    # Physical consistency: electrical output energy strictly less than acoustic input energy
    assert res["transduction_efficiency"] < 1.0


def test_benchmark_reproducibility_with_random_state():
    """Verify deterministic output reproducibility with random_state."""
    t1, d1, m1 = generate_sonoluminescence_system(
        sampling_rate=50_000,
        duration=0.001,
        acoustic_frequency_hz=25_000,
        random_state=123,
    )
    t2, d2, m2 = generate_sonoluminescence_system(
        sampling_rate=50_000,
        duration=0.001,
        acoustic_frequency_hz=25_000,
        random_state=123,
    )

    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(d1, d2)
    assert m1["n_samples"] == m2["n_samples"]


def test_benchmark_system_dimensions_and_metadata():
    """Verify output data shape (N, 8) and rich metadata schema."""
    sampling_rate = 80_000.0
    duration = 0.001
    time, data, meta = generate_sonoluminescence_system(
        sampling_rate=sampling_rate,
        duration=duration,
        acoustic_frequency_hz=20_000.0,
        noise_scale=0.0,
        random_state=42,
    )

    n_expected = int(sampling_rate * duration)
    assert data.shape == (n_expected, 8)
    assert len(time) == n_expected
    assert meta["channel_names"] == SonoluminescenceSystem.CHANNEL_NAMES
    assert "acoustic_parameters" in meta
    assert "waveguide_resonator_parameters" in meta
    assert "cavitation_parameters" in meta
    assert "optical_emission_parameters" in meta
    assert "transduction_parameters" in meta


def test_benchmark_systems_class_method_integration():
    """Verify BenchmarkSystems.generate_sonoluminescence_data class method."""
    t, data, meta = BenchmarkSystems.generate_sonoluminescence_data(
        sampling_rate=60_000,
        duration=0.001,
        acoustic_frequency_hz=20_000,
        random_state=42,
    )
    assert data.shape == (60, 8)
    assert meta["benchmark_system"] == "sonoluminescence_acousto_opto_electrical"


def test_drr_system_analysis_compatibility():
    """Verify that generated multivariate benchmark seamlessly runs through DRR."""
    sampling_rate = 100_000.0
    time, data, meta = generate_sonoluminescence_system(
        sampling_rate=sampling_rate,
        duration=0.0015,
        acoustic_frequency_hz=25_000.0,
        random_state=42,
    )

    drr = DynamicResonanceRooting(
        embedding_dim=3, tau=1, sampling_rate=sampling_rate
    )
    results = drr.analyze_system(
        data,
        multivariate=True,
        window_size=64,
        state_space=True,
        state_space_horizon=8,
    )

    assert "resonances" in results
    assert "resonance_depths" in results
    assert "influence_network" in results
    assert "state_space_analysis" in results

    # Verify RTEI metric computation
    rtei_res = calculate_resonant_transduction_efficiency_index(results, meta)
    assert 0.0 <= rtei_res["rtei"] <= 1.0
    assert 0.0 <= rtei_res["acoustic_resonance_depth"] <= 1.0
    assert 0.0 <= rtei_res["cavitation_resonance_depth"] <= 1.0
    assert 0.0 <= rtei_res["emission_resonance_depth"] <= 1.0
    assert 0.0 <= rtei_res["electrical_resonance_depth"] <= 1.0


def test_parameter_validation_and_edge_cases():
    """Verify clear ValueErrors on physically or numerically invalid inputs."""
    # Driver validation
    with pytest.raises(ValueError, match="frequency_hz must be positive"):
        AcousticDriver(frequency_hz=0.0)
    with pytest.raises(ValueError, match="frequency_hz must be positive"):
        AcousticDriver(frequency_hz=-1000.0)
    with pytest.raises(ValueError, match="sound_speed_m_s must be positive"):
        AcousticDriver(sound_speed_m_s=0.0)
    with pytest.raises(ValueError, match="input_pressure_pa must be non-negative"):
        AcousticDriver(input_pressure_pa=-50.0)

    # Resonator validation
    with pytest.raises(ValueError, match="input_diameter_m must be positive"):
        AcousticResonator(input_diameter_m=0.0)
    with pytest.raises(ValueError, match="output_diameter_m must be positive"):
        AcousticResonator(output_diameter_m=-0.01)
    with pytest.raises(ValueError, match="length_m must be positive"):
        AcousticResonator(length_m=0.0)
    with pytest.raises(ValueError, match="quality_factor_q must be >= 1.0"):
        AcousticResonator(quality_factor_q=0.5)
    with pytest.raises(ValueError, match="Unsupported taper_profile"):
        AcousticResonator(taper_profile="parabolic")

    # Cavitation validation
    with pytest.raises(ValueError, match="equilibrium_radius_m must be positive"):
        CavitationModel(equilibrium_radius_m=0.0)
    with pytest.raises(ValueError, match="liquid_density_kg_m3 must be positive"):
        CavitationModel(liquid_density_kg_m3=-100.0)
    with pytest.raises(ValueError, match="ambient_pressure_pa must be positive"):
        CavitationModel(ambient_pressure_pa=0.0)
    with pytest.raises(ValueError, match="polytropic_index must be >= 1.0"):
        CavitationModel(polytropic_index=0.8)

    # Emission validation
    with pytest.raises(ValueError, match="spectral_center_nm must be positive"):
        SonoluminescenceModel(spectral_center_nm=-350.0)
    with pytest.raises(ValueError, match="emission_threshold_compression must be > 1.0"):
        SonoluminescenceModel(emission_threshold_compression=0.5)

    # Transducer validation
    with pytest.raises(ValueError, match="optical_collection_efficiency must be in"):
        OpticalElectricalTransducer(optical_collection_efficiency=1.5)
    with pytest.raises(ValueError, match="conversion_efficiency must be in"):
        OpticalElectricalTransducer(conversion_efficiency=-0.1)
    with pytest.raises(ValueError, match="detector_gain must be positive"):
        OpticalElectricalTransducer(detector_gain=0.0)

    # Benchmark generator validation
    with pytest.raises(ValueError, match="sampling_rate .* must be at least twice"):
        generate_sonoluminescence_system(
            sampling_rate=30_000, acoustic_frequency_hz=25_000
        )
    with pytest.raises(ValueError, match="duration must be positive"):
        generate_sonoluminescence_system(duration=-0.01)
