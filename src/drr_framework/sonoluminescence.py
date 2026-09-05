"""Sonoluminescence and Acousto-Opto-Electrical Resonant Benchmark for DRR.

This module provides a research-grade computational model of a multi-stage
coupled resonant physical system:

    Acoustic Driver
          ↓
    Acoustic Resonator / Waveguide (Copper, Boron, or Custom Alloys)
          ↓
    Acoustic Impedance Transformation
          ↓
    Cavitation / Bubble Dynamics (Doped with Noble Gases, Cu, B, Electrolytes)
          ↓
    Nonlinear Bubble Collapse
          ↓
    Sonoluminescent Emission (Multispectral Continuum + Atomic/Excimer Lines)
          ↓
    Optical / Electromagnetic Coupling
          ↓
    Electrical Transduction

Physical vs. Phenomenological Modeling Disclosures:
--------------------------------------------------
1. Acoustic Domain & Waveguide Metallurgy:
   - Governed by fluid acoustic wave speed c_s (~1482 m/s in water) and ultrasonic
     frequencies f_a (~20-50 kHz), with acoustic wavelength lambda_a = c_s / f_a (~cm).
   - Resonator horns can be composed of specific metallurgical alloys (e.g., OFHC Copper,
     Copper-Boron alloys, Beryllium-Copper, Titanium, or Aluminum).
   - Solid acoustic impedance Z_mat = rho_mat * c_mat and horn-to-liquid interface
     transmission coefficient T = 4 * Z_mat * Z_fluid / (Z_mat + Z_fluid)^2 are modeled
     from continuum acoustic boundary physics.
   - Waveguide horn pressure concentration (d_in / d_out) is an idealized 1D
     geometric approximation assuming lossless energy flux conservation.
     A continuous-field model would additionally require complex boundary layer
     viscous/thermal dissipation, horn cut-off frequencies, and radiation impedance.

2. Cavitation Fluid & Solute/Dopant Mixtures:
   - Supports gas doping (e.g. Argon, Xenon, Helium, Nitrogen) modulating effective
     gas heat capacity ratio / polytropic index gamma_mix.
   - Supports dissolved / colloidal metal and metalloid solutes (Copper Cu, Boron B,
     Alkali electrolytes) modulating mixture density rho_mix, viscosity mu_mix,
     and surface tension sigma_mix.
   - Bubble oscillation is solved via the modified Rayleigh-Plesset equation with
     van der Waals excluded volume hard core (R_core ≈ R_0 / 8.5) and Blake threshold.

3. Optical / Electromagnetic Domain & Multispectral Lines:
   - Ultrafast sonoluminescent emission flashes (~100-300 ps) are triggered during
     violent collapse rebounds when gas compression exceeds threshold.
   - Solute species generate characteristic atomic/molecular emission lines:
     * Continuum baseline: UV-blue bremsstrahlung/blackbody (~350 nm)
     * Copper lines: Cu I atomic transitions (324.7 nm & 327.4 nm)
     * Boron lines: BO* excimer/oxide green bands (518.0 nm)
     * Alkali lines: Na D-line (589.0 nm)
   - Optical wavelength lambda_EM (~320-600 nm) and optical frequency
     f_EM = c / lambda_EM (~10^14-10^15 Hz) are strictly distinct from acoustic
     frequencies and wavelengths (separated by ~10 orders of magnitude).

4. Electrical Transduction & Energy Bookkeeping:
   - Transduction represents downstream photodetector responsivity.
   - This computational benchmark explicitly DOES NOT claim net energy amplification.
     Total transduction efficiency (electrical / acoustic energy) is strictly << 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Fundamental Physical Constants
SPEED_OF_LIGHT: float = 299_792_458.0  # Vacuum speed of light in m/s (exact SI)

# Standard Reference Medium Constants (Pure Water at 20 °C, 1 atm)
DEFAULT_SOUND_SPEED: float = 1482.0  # m/s
DEFAULT_WATER_DENSITY: float = 998.2  # kg/m^3
DEFAULT_SURFACE_TENSION: float = 0.0728  # N/m
DEFAULT_LIQUID_VISCOSITY: float = 0.001002  # Pa*s
DEFAULT_VAPOR_PRESSURE: float = 2330.0  # Pa
DEFAULT_AMBIENT_PRESSURE: float = 101325.0  # Pa (1 atm)
DEFAULT_POLYTROPIC_INDEX: float = 1.4  # Diatomic gas ratio of specific heats


@dataclass(frozen=True)
class WaveguideMaterial:
    """Solid material properties of the acoustic waveguide resonator horn.

    Parameters:
        name: Material or alloy identifier.
        density_kg_m3: Solid density in kg/m^3 (> 0).
        sound_speed_m_s: Longitudinal sound velocity in the solid in m/s (> 0).
        quality_factor_q: Mechanical quality factor Q of the solid material (>= 1.0).
        copper_fraction: Mass/volume fraction of Copper (Cu) in [0.0, 1.0].
        boron_fraction: Mass/volume fraction of Boron (B) in [0.0, 1.0].
        other_fractions: Dictionary of additional alloying element fractions.
    """

    name: str = "copper_boron_alloy"
    density_kg_m3: float = 8920.0
    sound_speed_m_s: float = 4850.0
    quality_factor_q: float = 75.0
    copper_fraction: float = 0.98
    boron_fraction: float = 0.02
    other_fractions: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.density_kg_m3 <= 0:
            raise ValueError(f"density_kg_m3 must be positive, got {self.density_kg_m3}")
        if self.sound_speed_m_s <= 0:
            raise ValueError(f"sound_speed_m_s must be positive, got {self.sound_speed_m_s}")
        if self.quality_factor_q < 1.0:
            raise ValueError(f"quality_factor_q must be >= 1.0, got {self.quality_factor_q}")
        if not (0.0 <= self.copper_fraction <= 1.0):
            raise ValueError(f"copper_fraction must be in [0, 1], got {self.copper_fraction}")
        if not (0.0 <= self.boron_fraction <= 1.0):
            raise ValueError(f"boron_fraction must be in [0, 1], got {self.boron_fraction}")

    @property
    def acoustic_impedance_rayl(self) -> float:
        """Characteristic acoustic impedance Z_mat = rho_mat * c_mat in Rayls (Pa*s/m)."""
        return float(self.density_kg_m3 * self.sound_speed_m_s)

    def interface_transmission_coefficient(
        self,
        fluid_density_kg_m3: float = DEFAULT_WATER_DENSITY,
        fluid_sound_speed_m_s: float = DEFAULT_SOUND_SPEED,
    ) -> float:
        """Power transmission coefficient T across normal solid-fluid interface.

        T = 4 * Z_solid * Z_fluid / (Z_solid + Z_fluid)^2.
        """
        z_solid = self.acoustic_impedance_rayl
        z_fluid = fluid_density_kg_m3 * fluid_sound_speed_m_s
        return float(4.0 * z_solid * z_fluid / ((z_solid + z_fluid) ** 2))


# Canonical Material Presets
WAVEGUIDE_MATERIALS: Dict[str, WaveguideMaterial] = {
    "ofhc_copper": WaveguideMaterial(
        name="ofhc_copper",
        density_kg_m3=8960.0,
        sound_speed_m_s=4700.0,
        quality_factor_q=50.0,
        copper_fraction=1.0,
        boron_fraction=0.0,
    ),
    "copper_boron_alloy": WaveguideMaterial(
        name="copper_boron_alloy",
        density_kg_m3=8920.0,
        sound_speed_m_s=4850.0,
        quality_factor_q=75.0,
        copper_fraction=0.98,
        boron_fraction=0.02,
    ),
    "beryllium_copper": WaveguideMaterial(
        name="beryllium_copper",
        density_kg_m3=8250.0,
        sound_speed_m_s=5000.0,
        quality_factor_q=80.0,
        copper_fraction=0.98,
        boron_fraction=0.0,
        other_fractions={"Be": 0.02},
    ),
    "titanium_ti6al4v": WaveguideMaterial(
        name="titanium_ti6al4v",
        density_kg_m3=4430.0,
        sound_speed_m_s=6070.0,
        quality_factor_q=100.0,
        copper_fraction=0.0,
        boron_fraction=0.0,
        other_fractions={"Ti": 0.90, "Al": 0.06, "V": 0.04},
    ),
    "aluminum_6061": WaveguideMaterial(
        name="aluminum_6061",
        density_kg_m3=2700.0,
        sound_speed_m_s=6320.0,
        quality_factor_q=90.0,
        copper_fraction=0.003,
        boron_fraction=0.0,
        other_fractions={"Al": 0.98, "Mg": 0.01, "Si": 0.006},
    ),
    "boron_carbide": WaveguideMaterial(
        name="boron_carbide",
        density_kg_m3=2520.0,
        sound_speed_m_s=11000.0,
        quality_factor_q=120.0,
        copper_fraction=0.0,
        boron_fraction=0.78,
        other_fractions={"C": 0.22},
    ),
}


@dataclass(frozen=True)
class DopantMixture:
    """Thermodynamic, fluid mechanical, and spectroscopic properties of liquid/gas mixtures.

    Parameters:
        carrier_liquid: Base solvent name (e.g. 'water').
        gas_species: Dissolved gas species ('argon', 'xenon', 'helium', 'nitrogen', 'air').
        gas_fraction: Dissolved noble/active gas fraction in [0.0, 0.1].
        copper_solute_fraction: Dissolved or colloidal copper fraction (e.g. 0.001 = 1000 ppm).
        boron_solute_fraction: Dissolved or colloidal boron fraction (e.g. 0.001 = 1000 ppm).
        alkali_solute_fraction: Dissolved alkali salt fraction (e.g. NaCl / KCl).
        carrier_density_kg_m3: Density of carrier liquid in kg/m^3.
        carrier_viscosity_pa_s: Viscosity of carrier liquid in Pa*s.
        carrier_surface_tension_n_m: Surface tension of carrier liquid in N/m.
        carrier_vapor_pressure_pa: Vapor pressure of carrier liquid in Pa.
    """

    carrier_liquid: str = "water"
    gas_species: Literal["argon", "xenon", "helium", "nitrogen", "air"] = "argon"
    gas_fraction: float = 0.01  # 1% Argon dissolved (typical SBSL optimum)
    copper_solute_fraction: float = 0.0
    boron_solute_fraction: float = 0.0
    alkali_solute_fraction: float = 0.0
    carrier_density_kg_m3: float = DEFAULT_WATER_DENSITY
    carrier_viscosity_pa_s: float = DEFAULT_LIQUID_VISCOSITY
    carrier_surface_tension_n_m: float = DEFAULT_SURFACE_TENSION
    carrier_vapor_pressure_pa: float = DEFAULT_VAPOR_PRESSURE

    def __post_init__(self) -> None:
        if not (0.0 <= self.gas_fraction <= 0.20):
            raise ValueError(f"gas_fraction must be in [0.0, 0.20], got {self.gas_fraction}")
        if not (0.0 <= self.copper_solute_fraction <= 0.10):
            raise ValueError(
                f"copper_solute_fraction must be in [0.0, 0.10], got {self.copper_solute_fraction}"
            )
        if not (0.0 <= self.boron_solute_fraction <= 0.10):
            raise ValueError(
                f"boron_solute_fraction must be in [0.0, 0.10], got {self.boron_solute_fraction}"
            )
        if not (0.0 <= self.alkali_solute_fraction <= 0.10):
            raise ValueError(
                f"alkali_solute_fraction must be in [0.0, 0.10], got {self.alkali_solute_fraction}"
            )
        if self.gas_species not in ("argon", "xenon", "helium", "nitrogen", "air"):
            raise ValueError(f"Unsupported gas_species: {self.gas_species}")

    @property
    def effective_polytropic_index(self) -> float:
        """Effective gas polytropic index gamma_mix.

        Monatomic noble gases (Ar, Xe, He) have gamma = 5/3 ≈ 1.667.
        Diatomic gases (N2, air) have gamma = 7/5 = 1.40.
        """
        gamma_gas = 1.667 if self.gas_species in ("argon", "xenon", "helium") else 1.40
        # Weighted mix between base vapor/diatomic background and dissolved gas
        return float((1.0 - self.gas_fraction) * 1.40 + self.gas_fraction * gamma_gas)

    @property
    def effective_density_kg_m3(self) -> float:
        """Liquid mixture mass density rho_mix accounting for dissolved/colloidal species."""
        return float(
            self.carrier_density_kg_m3
            * (
                1.0
                + 0.8 * self.copper_solute_fraction
                + 0.2 * self.boron_solute_fraction
                + 0.1 * self.alkali_solute_fraction
            )
        )

    @property
    def effective_viscosity_pa_s(self) -> float:
        """Liquid dynamic viscosity mu_mix accounting for colloidal solutes."""
        return float(
            self.carrier_viscosity_pa_s
            * (
                1.0
                + 2.5 * (self.copper_solute_fraction + self.boron_solute_fraction)
                + 1.5 * self.alkali_solute_fraction
            )
        )

    @property
    def effective_surface_tension_n_m(self) -> float:
        """Liquid surface tension sigma_mix in N/m."""
        return float(
            self.carrier_surface_tension_n_m
            * (1.0 - 0.02 * self.copper_solute_fraction + 0.05 * self.alkali_solute_fraction)
        )

    @property
    def effective_vapor_pressure_pa(self) -> float:
        """Solvent vapor pressure P_v modified by solute mole fraction."""
        solute_total = (
            self.copper_solute_fraction + self.boron_solute_fraction + self.alkali_solute_fraction
        )
        return float(self.carrier_vapor_pressure_pa * max(0.5, 1.0 - solute_total))

    @property
    def active_spectral_lines(self) -> List[Dict[str, Any]]:
        """List of active spectroscopic emission lines for this mixture.

        Includes thermal continuum plus element-specific atomic and molecular transitions:
        - Continuum: Bremsstrahlung / blackbody (~350 nm UV-blue)
        - Copper (Cu I): 324.7 nm & 327.4 nm resonance doublet
        - Boron (BO*): 518.0 nm excimer green band
        - Alkali (Na I): 589.0 nm D-line doublet
        """
        lines: List[Dict[str, Any]] = [
            {
                "species": "Continuum",
                "name": "Bremsstrahlung / Blackbody Continuum",
                "wavelength_nm": 350.0,
                "bandwidth_nm": 150.0,
                "relative_weight": 1.0,
            }
        ]

        if self.copper_solute_fraction > 0:
            w_cu1 = float(np.clip(self.copper_solute_fraction * 150.0, 0.01, 0.6))
            w_cu2 = float(np.clip(self.copper_solute_fraction * 120.0, 0.01, 0.5))
            lines.append(
                {
                    "species": "Copper",
                    "name": "Cu I (324.7 nm)",
                    "wavelength_nm": 324.7,
                    "bandwidth_nm": 3.0,
                    "relative_weight": w_cu1,
                }
            )
            lines.append(
                {
                    "species": "Copper",
                    "name": "Cu I (327.4 nm)",
                    "wavelength_nm": 327.4,
                    "bandwidth_nm": 3.0,
                    "relative_weight": w_cu2,
                }
            )

        if self.boron_solute_fraction > 0:
            w_b = float(np.clip(self.boron_solute_fraction * 180.0, 0.01, 0.6))
            lines.append(
                {
                    "species": "Boron",
                    "name": "BO* Excimer Green Band (518.0 nm)",
                    "wavelength_nm": 518.0,
                    "bandwidth_nm": 15.0,
                    "relative_weight": w_b,
                }
            )

        if self.alkali_solute_fraction > 0:
            w_alkali = float(np.clip(self.alkali_solute_fraction * 200.0, 0.01, 0.6))
            lines.append(
                {
                    "species": "Alkali",
                    "name": "Na D-line (589.0 nm)",
                    "wavelength_nm": 589.0,
                    "bandwidth_nm": 5.0,
                    "relative_weight": w_alkali,
                }
            )

        return lines


@dataclass(frozen=True)
class AcousticDriver:
    """Acoustic excitation source driving the resonant system.

    Parameters:
        frequency_hz: Driving frequency f_a in Hertz (must be > 0).
        sound_speed_m_s: Speed of sound in the fluid medium in m/s (must be > 0).
        input_pressure_pa: Input acoustic pressure amplitude P_in in Pascals (>= 0).
        phase_rad: Initial phase angle in radians.
    """

    frequency_hz: float = 25_000.0
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED
    input_pressure_pa: float = 60_000.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {self.frequency_hz}")
        if self.sound_speed_m_s <= 0:
            raise ValueError(f"sound_speed_m_s must be positive, got {self.sound_speed_m_s}")
        if self.input_pressure_pa < 0:
            raise ValueError(
                f"input_pressure_pa must be non-negative, got {self.input_pressure_pa}"
            )

    @property
    def acoustic_wavelength(self) -> float:
        """Acoustic wavelength lambda_a = c_s / f_a in meters."""
        return self.sound_speed_m_s / self.frequency_hz

    def waveform(self, time: np.ndarray) -> np.ndarray:
        """Calculate input acoustic pressure time series P_in(t) in Pascals."""
        return self.input_pressure_pa * np.sin(
            2.0 * np.pi * self.frequency_hz * time + self.phase_rad
        )


@dataclass(frozen=True)
class AcousticResonator:
    """Acoustic resonator and geometric waveguide horn.

    Acts as an acoustic impedance transformer and spatial pressure concentrator.
    This is an acoustic structure, NOT an electrical transformer.

    Parameters:
        input_diameter_m: Waveguide input aperture diameter in meters (must be > 0).
        output_diameter_m: Waveguide output / throat diameter in meters (must be > 0).
        length_m: Resonator longitudinal length in meters (must be > 0).
        taper_profile: Horn profile ('conical', 'exponential', or 'cylindrical').
        quality_factor_q: Acoustic cavity quality factor Q (must be >= 1.0).
        material: Solid metallurgical horn material (e.g. Copper-Boron alloy, OFHC Cu).
    """

    input_diameter_m: float = 0.020  # 20 mm
    output_diameter_m: float = 0.004  # 4 mm
    length_m: float = 0.02964  # Half-wavelength in water at 25 kHz (1482 / (2 * 25000))
    taper_profile: Literal["conical", "exponential", "cylindrical"] = "conical"
    quality_factor_q: float = 30.0
    material: Optional[WaveguideMaterial] = None

    def __post_init__(self) -> None:
        if self.input_diameter_m <= 0:
            raise ValueError(f"input_diameter_m must be positive, got {self.input_diameter_m}")
        if self.output_diameter_m <= 0:
            raise ValueError(f"output_diameter_m must be positive, got {self.output_diameter_m}")
        if self.length_m <= 0:
            raise ValueError(f"length_m must be positive, got {self.length_m}")
        if self.quality_factor_q < 1.0:
            raise ValueError(f"quality_factor_q must be >= 1.0, got {self.quality_factor_q}")
        if self.taper_profile not in ("conical", "exponential", "cylindrical"):
            raise ValueError(f"Unsupported taper_profile: {self.taper_profile}")

    @property
    def input_area_m2(self) -> float:
        """Input cross-sectional area A_in in m^2."""
        return float(np.pi * (self.input_diameter_m / 2.0) ** 2)

    @property
    def output_area_m2(self) -> float:
        """Output cross-sectional area A_out in m^2."""
        return float(np.pi * (self.output_diameter_m / 2.0) ** 2)

    @property
    def area_ratio(self) -> float:
        """Geometric area ratio A_out / A_in = (d_out / d_in)^2."""
        return float((self.output_diameter_m / self.input_diameter_m) ** 2)

    @property
    def geometric_pressure_gain(self) -> float:
        """Idealized acoustic pressure concentration factor G_geom.

        Approximated via 1D lossless energy flux conservation across the horn:
            P_out / P_in ≈ sqrt(A_in / A_out) = d_in / d_out.
        """
        if self.taper_profile == "cylindrical":
            return 1.0
        return float(self.input_diameter_m / self.output_diameter_m)

    @property
    def effective_quality_factor(self) -> float:
        """Effective quality factor Q incorporating material damping if material is specified."""
        if self.material is not None:
            # Composite cavity Q influenced by solid horn Q
            return float(max(self.quality_factor_q, self.material.quality_factor_q))
        return float(self.quality_factor_q)

    def resonant_harmonics(
        self, sound_speed_m_s: float = DEFAULT_SOUND_SPEED, n_modes: int = 5
    ) -> np.ndarray:
        """Calculate the first n longitudinal half-wave resonant frequencies in Hz."""
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1")
        modes = np.arange(1, n_modes + 1, dtype=float)
        return modes * sound_speed_m_s / (2.0 * self.length_m)

    def cavity_response(
        self, frequency_hz: float, sound_speed_m_s: float = DEFAULT_SOUND_SPEED
    ) -> Tuple[float, float, float]:
        """Compute standing-wave cavity resonance response."""
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        acoustic_wavelength = sound_speed_m_s / frequency_hz
        resonator_length_ratio = self.length_m / acoustic_wavelength

        mode_n = max(1, int(np.round(2.0 * self.length_m * frequency_hz / sound_speed_m_s)))
        resonant_freq = mode_n * sound_speed_m_s / (2.0 * self.length_m)

        detuning_error = (frequency_hz - resonant_freq) / resonant_freq

        q = self.effective_quality_factor
        gain = float(1.0 + (q - 1.0) / np.sqrt(1.0 + 4.0 * (q**2) * (detuning_error**2)))

        return gain, float(detuning_error), float(resonator_length_ratio)

    def effective_pressure_amplitude(self, driver: AcousticDriver) -> float:
        """Compute net acoustic pressure amplitude at the waveguide focus in Pascals."""
        gain_cav, _, _ = self.cavity_response(driver.frequency_hz, driver.sound_speed_m_s)
        return float(driver.input_pressure_pa * self.geometric_pressure_gain * gain_cav)


# Alias for flexible nomenclature
AcousticWaveguide = AcousticResonator


@dataclass(frozen=True)
class CavitationModel:
    """Thermodynamic and fluid mechanics cavitation properties.

    Parameters:
        equilibrium_radius_m: Undisturbed bubble equilibrium radius R_0 in meters (> 0).
        liquid_density_kg_m3: Liquid mass density rho_L in kg/m^3 (> 0).
        surface_tension_n_m: Gas-liquid surface tension sigma in N/m (>= 0).
        liquid_viscosity_pa_s: Dynamic shear viscosity mu_L in Pa*s (>= 0).
        vapor_pressure_pa: Saturated vapor pressure P_v in Pascals (>= 0).
        ambient_pressure_pa: Static ambient atmospheric pressure P_0 in Pascals (> 0).
        polytropic_index: Polytropic gas exponent gamma (>= 1.0).
        dopant_mixture: Optional dopant mixture defining solute & gas modifications.
    """

    equilibrium_radius_m: float = 5.0e-6  # 5 micrometers
    liquid_density_kg_m3: float = DEFAULT_WATER_DENSITY
    surface_tension_n_m: float = DEFAULT_SURFACE_TENSION
    liquid_viscosity_pa_s: float = DEFAULT_LIQUID_VISCOSITY
    vapor_pressure_pa: float = DEFAULT_VAPOR_PRESSURE
    ambient_pressure_pa: float = DEFAULT_AMBIENT_PRESSURE
    polytropic_index: float = DEFAULT_POLYTROPIC_INDEX
    dopant_mixture: Optional[DopantMixture] = None

    def __post_init__(self) -> None:
        if self.equilibrium_radius_m <= 0:
            raise ValueError(
                f"equilibrium_radius_m must be positive, got {self.equilibrium_radius_m}"
            )
        if self.liquid_density_kg_m3 <= 0:
            raise ValueError(
                f"liquid_density_kg_m3 must be positive, got {self.liquid_density_kg_m3}"
            )
        if self.surface_tension_n_m < 0:
            raise ValueError(
                f"surface_tension_n_m must be non-negative, got {self.surface_tension_n_m}"
            )
        if self.liquid_viscosity_pa_s < 0:
            raise ValueError(
                f"liquid_viscosity_pa_s must be non-negative, got {self.liquid_viscosity_pa_s}"
            )
        if self.vapor_pressure_pa < 0:
            raise ValueError(
                f"vapor_pressure_pa must be non-negative, got {self.vapor_pressure_pa}"
            )
        if self.ambient_pressure_pa <= 0:
            raise ValueError(
                f"ambient_pressure_pa must be positive, got {self.ambient_pressure_pa}"
            )
        if self.polytropic_index < 1.0:
            raise ValueError(f"polytropic_index must be >= 1.0, got {self.polytropic_index}")

    @property
    def density(self) -> float:
        """Effective fluid density in kg/m^3."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.effective_density_kg_m3
        return self.liquid_density_kg_m3

    @property
    def viscosity(self) -> float:
        """Effective fluid viscosity in Pa*s."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.effective_viscosity_pa_s
        return self.liquid_viscosity_pa_s

    @property
    def surface_tension(self) -> float:
        """Effective surface tension in N/m."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.effective_surface_tension_n_m
        return self.surface_tension_n_m

    @property
    def vapor_pressure(self) -> float:
        """Effective vapor pressure in Pa."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.effective_vapor_pressure_pa
        return self.vapor_pressure_pa

    @property
    def gamma(self) -> float:
        """Effective polytropic index gamma."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.effective_polytropic_index
        return self.polytropic_index

    @property
    def blake_threshold_pressure(self) -> float:
        """Blake critical cavitation threshold pressure in Pascals.

        P_Blake = P_0 + 0.77 * (sigma / R_0).
        """
        return float(
            self.ambient_pressure_pa + 0.77 * (self.surface_tension / self.equilibrium_radius_m)
        )

    @property
    def blake_acoustic_threshold_pa(self) -> float:
        """Acoustic pressure amplitude threshold delta P_a = 0.77 * (sigma / R_0) in Pa."""
        return float(0.77 * (self.surface_tension / self.equilibrium_radius_m))

    def is_cavitation_active(self, acoustic_pressure_amplitude_pa: float) -> bool:
        """Evaluate whether driving pressure amplitude exceeds the Blake threshold."""
        return bool(acoustic_pressure_amplitude_pa >= self.blake_acoustic_threshold_pa)


class BubbleDynamics:
    """Nonlinear radial bubble dynamics solver (modified Rayleigh-Plesset oscillator).

    Integrates the radial motion R(t) and wall velocity v(t) = dR/dt of an acoustically
    driven spherical bubble, including van der Waals gas excluded volume, liquid viscosity,
    and surface tension:

        R * R'' + (3/2) * (R')^2 =
            (1 / rho_L) * [ P_gas(R) + P_v - (2 * sigma / R) - (4 * mu_L * R' / R) - P_inf(t) ]

    where P_gas(R) = (P_0 + 2*sigma/R_0 - P_v) * ((R_0^3 - h^3) / (R^3 - h^3))^gamma
    and h = R_0 / 8.5 is the van der Waals hard-core radius.
    """

    def __init__(
        self,
        cavitation_model: Optional[CavitationModel] = None,
        hard_core_ratio: float = 1.0 / 8.5,
    ) -> None:
        self.cavitation = cavitation_model or CavitationModel()
        if not (0.0 < hard_core_ratio < 0.5):
            raise ValueError(f"hard_core_ratio must be in (0.0, 0.5), got {hard_core_ratio}")
        self.hard_core_ratio = hard_core_ratio

    def _rk4_step(
        self,
        r: float,
        v: float,
        p_driving: float,
        dt: float,
        r0: float,
        h_core: float,
        p0: float,
        pv: float,
        sigma: float,
        mu: float,
        rho: float,
        gamma: float,
        p_gas0: float,
    ) -> Tuple[float, float]:
        """Single 4th-order Runge-Kutta step for Rayleigh-Plesset ODE."""

        def deriv(r_val: float, v_val: float) -> Tuple[float, float]:
            r_eff = max(r_val, h_core * 1.05)
            # Van der Waals gas pressure with physical upper bound
            vol_ratio = min(1e6, (r0**3 - h_core**3) / max(r_eff**3 - h_core**3, 1e-20))
            p_gas = min(1e11, p_gas0 * (vol_ratio**gamma))

            # Surface tension & viscous damping terms
            p_surface = 2.0 * sigma / r_eff
            p_viscous = 4.0 * mu * v_val / r_eff

            # Liquid pressure at infinity: P_inf = P_0 - P_driving
            p_inf = p0 - p_driving
            delta_p = p_gas + pv - p_surface - p_viscous - p_inf

            # Rayleigh-Plesset acceleration: R'' = (delta_p / rho - 1.5 * v^2) / R
            dv_dt = (delta_p / rho - 1.5 * (v_val**2)) / r_eff
            dv_dt = float(np.clip(dv_dt, -1e12, 1e12))
            return v_val, dv_dt

        # RK4 stages
        k1_r, k1_v = deriv(r, v)
        k2_r, k2_v = deriv(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)
        k3_r, k3_v = deriv(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)
        k4_r, k4_v = deriv(r + dt * k3_r, v + dt * k3_v)

        r_next = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Enforce physical positivity bounded by hard core
        r_next = max(r_next, h_core * 1.05)
        v_next = float(np.clip(v_next, -2500.0, 2500.0))
        return r_next, v_next

    def simulate(
        self,
        time: np.ndarray,
        driving_pressure_pa: np.ndarray,
        sub_steps: int = 25,
    ) -> Dict[str, Any]:
        """Integrate bubble dynamics over time.

        Args:
            time: 1D array of simulation time points in seconds.
            driving_pressure_pa: 1D array of acoustic driving pressure in Pascals.
            sub_steps: Number of internal RK4 sub-steps per sampling interval for stability.

        Returns:
            Dictionary containing:
                radius: Bubble radius R(t) in meters.
                normalized_radius: Dimensionless radius R(t) / R_0.
                velocity: Bubble wall velocity dR/dt in m/s.
                collapse_pressure: Internal gas collapse pressure P_collapse(t) in Pascals.
                max_compression_ratio: Maximum compression ratio R_0 / R_min.
                collapse_indices: Timestep indices of detected collapse events.
                collapse_times: Time points of collapse events in seconds.
                collapse_intensities: Peak collapse wall velocities at collapse events.
        """
        n_steps = len(time)
        if n_steps != len(driving_pressure_pa):
            raise ValueError("time and driving_pressure_pa must have identical length")
        if n_steps < 2:
            raise ValueError("Time series must have at least 2 points")
        if sub_steps < 1:
            raise ValueError("sub_steps must be at least 1")

        dt = float(time[1] - time[0])
        dt_sub = dt / sub_steps

        r0 = self.cavitation.equilibrium_radius_m
        h_core = r0 * self.hard_core_ratio
        p0 = self.cavitation.ambient_pressure_pa
        pv = self.cavitation.vapor_pressure
        sigma = self.cavitation.surface_tension
        mu = self.cavitation.viscosity
        rho = self.cavitation.density
        gamma = self.cavitation.gamma
        p_gas0 = p0 + (2.0 * sigma / r0) - pv

        r_arr = np.zeros(n_steps, dtype=float)
        v_arr = np.zeros(n_steps, dtype=float)
        p_collapse_arr = np.zeros(n_steps, dtype=float)

        r_curr = r0
        v_curr = 0.0
        r_arr[0] = r_curr
        v_arr[0] = v_curr
        p_collapse_arr[0] = 0.0

        for i in range(1, n_steps):
            p_drive_start = driving_pressure_pa[i - 1]
            p_drive_end = driving_pressure_pa[i]

            for s in range(sub_steps):
                frac = (s + 0.5) / sub_steps
                p_sub = p_drive_start + frac * (p_drive_end - p_drive_start)
                r_curr, v_curr = self._rk4_step(
                    r_curr,
                    v_curr,
                    p_sub,
                    dt_sub,
                    r0,
                    h_core,
                    p0,
                    pv,
                    sigma,
                    mu,
                    rho,
                    gamma,
                    p_gas0,
                )

            r_arr[i] = r_curr
            v_arr[i] = v_curr

            # Internal gas pressure calculation
            vol_ratio = min(1e6, (r0**3 - h_core**3) / max(r_curr**3 - h_core**3, 1e-20))
            p_gas = min(1e11, p_gas0 * (vol_ratio**gamma))
            p_collapse_arr[i] = max(0.0, p_gas - p0)

        # Detect collapse event indices: local radius minima where R < 0.85 * R_0
        collapse_mask = (
            (r_arr[1:-1] <= r_arr[:-2]) & (r_arr[1:-1] <= r_arr[2:]) & (r_arr[1:-1] < 0.85 * r0)
        )
        collapse_indices = np.where(collapse_mask)[0] + 1

        r_min = float(np.min(r_arr))
        max_compression_ratio = float(r0 / max(r_min, h_core))

        collapse_times = time[collapse_indices] if len(collapse_indices) > 0 else np.array([])
        collapse_intensities = (
            np.abs(v_arr[collapse_indices]) if len(collapse_indices) > 0 else np.array([])
        )

        return {
            "radius": r_arr,
            "normalized_radius": r_arr / r0,
            "velocity": v_arr,
            "collapse_pressure": p_collapse_arr,
            "max_compression_ratio": max_compression_ratio,
            "collapse_indices": collapse_indices,
            "collapse_times": collapse_times,
            "collapse_intensities": collapse_intensities,
        }


@dataclass(frozen=True)
class SonoluminescenceModel:
    """Phenomenological sonoluminescence flash emission model.

    Generates optical emission flashes during violent bubble collapse rebounds,
    including multispectral atomic lines from dissolved metal/metalloid solutes (Cu, B, Na).

    Parameters:
        spectral_center_nm: Center optical emission wavelength in nanometers (> 0).
        spectral_bandwidth_nm: Optical emission bandwidth in nanometers (> 0).
        pulse_duration_s: Modeled physical emission flash duration in seconds (> 0).
        emission_threshold_compression: Minimum compression ratio R_0 / R required for light emission.
        intensity_scaling: Dimensionless scaling factor for peak normalized intensity.
        dopant_mixture: Optional dopant mixture generating multispectral lines.
    """

    spectral_center_nm: float = 350.0  # UV-blue (~350 nm center for SBSL in water)
    spectral_bandwidth_nm: float = 150.0  # Broadband continuum
    pulse_duration_s: float = 2.0e-10  # 200 picoseconds physical duration
    emission_threshold_compression: float = 2.2
    intensity_scaling: float = 1.0
    dopant_mixture: Optional[DopantMixture] = None

    def __post_init__(self) -> None:
        if self.spectral_center_nm <= 0:
            raise ValueError(f"spectral_center_nm must be positive, got {self.spectral_center_nm}")
        if self.spectral_bandwidth_nm <= 0:
            raise ValueError(
                f"spectral_bandwidth_nm must be positive, got {self.spectral_bandwidth_nm}"
            )
        if self.pulse_duration_s <= 0:
            raise ValueError(f"pulse_duration_s must be positive, got {self.pulse_duration_s}")
        if self.emission_threshold_compression <= 1.0:
            raise ValueError(
                f"emission_threshold_compression must be > 1.0, got {self.emission_threshold_compression}"
            )
        if self.intensity_scaling < 0:
            raise ValueError(
                f"intensity_scaling must be non-negative, got {self.intensity_scaling}"
            )

    @property
    def spectral_lines(self) -> List[Dict[str, Any]]:
        """Active spectral lines from continuum and dopants."""
        if self.dopant_mixture is not None:
            return self.dopant_mixture.active_spectral_lines
        return [
            {
                "species": "Continuum",
                "name": "Bremsstrahlung Continuum",
                "wavelength_nm": self.spectral_center_nm,
                "bandwidth_nm": self.spectral_bandwidth_nm,
                "relative_weight": 1.0,
            }
        ]

    @property
    def effective_spectral_center_nm(self) -> float:
        """Weighted effective optical center wavelength in nm across all emission lines."""
        lines = self.spectral_lines
        total_w = sum(line["relative_weight"] for line in lines)
        if total_w <= 0:
            return self.spectral_center_nm
        return float(
            sum(line["wavelength_nm"] * line["relative_weight"] for line in lines) / total_w
        )

    @property
    def optical_frequency_hz(self) -> float:
        """Optical electromagnetic frequency f_EM = c / lambda_EM in Hertz."""
        return float(SPEED_OF_LIGHT / (self.effective_spectral_center_nm * 1e-9))

    def simulate_emission(
        self,
        time: np.ndarray,
        bubble_results: Dict[str, Any],
        sound_speed_m_s: float = DEFAULT_SOUND_SPEED,
    ) -> Dict[str, Any]:
        """Compute sonoluminescent emission intensity time series."""
        n_steps = len(time)
        dt = float(time[1] - time[0]) if n_steps > 1 else 1e-6
        emission_arr = np.zeros(n_steps, dtype=float)

        radius = bubble_results["radius"]
        r0 = float(radius[0]) if len(radius) > 0 else 5e-6
        velocity = bubble_results["velocity"]
        collapse_indices = bubble_results["collapse_indices"]

        active_collapse_times: List[float] = []
        sigma_dt = max(dt * 0.75, self.pulse_duration_s)

        # Total intensity multiplier from noble gas and dopants
        dopant_boost = 1.0
        if self.dopant_mixture is not None:
            # Noble gas (e.g. Argon 1%) significantly enhances peak SBSL emission
            dopant_boost = 1.0 + 5.0 * self.dopant_mixture.gas_fraction

        for idx in collapse_indices:
            r_val = radius[idx]
            compression = r0 / max(r_val, 1e-12)

            if compression >= self.emission_threshold_compression:
                v_wall = abs(velocity[idx])
                mach = v_wall / sound_speed_m_s
                peak_i = float(
                    self.intensity_scaling
                    * dopant_boost
                    * ((compression / self.emission_threshold_compression) ** 2.5)
                    * (mach**1.5)
                )
                t_event = time[idx]
                active_collapse_times.append(float(t_event))

                window_samples = max(3, int(np.ceil(3.0 * sigma_dt / dt)))
                start_i = max(0, idx - window_samples)
                end_i = min(n_steps, idx + window_samples + 1)

                t_slice = time[start_i:end_i]
                pulse = peak_i * np.exp(-0.5 * ((t_slice - t_event) / sigma_dt) ** 2)
                emission_arr[start_i:end_i] += pulse

        max_val = float(np.max(emission_arr)) if np.max(emission_arr) > 0 else 1.0
        normalized_emission = np.clip(emission_arr / max(max_val, 1.0), 0.0, 1.0)

        return {
            "emission_intensity": normalized_emission,
            "peak_emission": float(np.max(normalized_emission)),
            "emission_duration_s": self.pulse_duration_s,
            "emission_event_times": np.array(active_collapse_times, dtype=float),
            "spectral_center_nm": self.effective_spectral_center_nm,
            "spectral_bandwidth_nm": self.spectral_bandwidth_nm,
            "optical_frequency_hz": self.optical_frequency_hz,
            "spectral_lines": self.spectral_lines,
        }


@dataclass(frozen=True)
class OpticalElectricalTransducer:
    """Downstream optical-to-electrical transducer (photodiode / detector model).

    Parameters:
        optical_collection_efficiency: Fraction of emitted optical flux collected (0 < eta_col <= 1).
        conversion_efficiency: Photodetector quantum responsivity (0 <= eta_e <= 1).
        detector_gain: Amplifier / detector transimpedance gain G_det (> 0).
        detector_time_constant_s: Detector RC response time in seconds (> 0).
        load_resistance_ohms: Output circuit load resistance in Ohms (> 0).
    """

    optical_collection_efficiency: float = 0.15
    conversion_efficiency: float = 0.25
    detector_gain: float = 10.0
    detector_time_constant_s: float = 1.0e-7  # 100 ns
    load_resistance_ohms: float = 50.0

    def __post_init__(self) -> None:
        if not (0.0 < self.optical_collection_efficiency <= 1.0):
            raise ValueError(
                "optical_collection_efficiency must be in (0.0, 1.0], "
                f"got {self.optical_collection_efficiency}"
            )
        if not (0.0 <= self.conversion_efficiency <= 1.0):
            raise ValueError(
                "conversion_efficiency must be in [0.0, 1.0], " f"got {self.conversion_efficiency}"
            )
        if self.detector_gain <= 0:
            raise ValueError(f"detector_gain must be positive, got {self.detector_gain}")
        if self.detector_time_constant_s <= 0:
            raise ValueError(
                f"detector_time_constant_s must be positive, got {self.detector_time_constant_s}"
            )
        if self.load_resistance_ohms <= 0:
            raise ValueError(
                f"load_resistance_ohms must be positive, got {self.load_resistance_ohms}"
            )

    def transduce(
        self,
        time: np.ndarray,
        emission_intensity: np.ndarray,
        acoustic_driver: AcousticDriver,
        acoustic_resonator: AcousticResonator,
    ) -> Dict[str, Any]:
        """Calculate optical and electrical signals with energy bookkeeping."""
        n_steps = len(time)
        dt = float(time[1] - time[0]) if n_steps > 1 else 1e-6

        # Optical signal at detector aperture
        optical_signal = self.optical_collection_efficiency * emission_intensity

        # 1st-order low-pass filter representing detector bandwidth (RC time constant)
        alpha = dt / (dt + self.detector_time_constant_s)
        electrical_signal = np.zeros(n_steps, dtype=float)
        v_prev = 0.0
        for i in range(n_steps):
            v_target = self.detector_gain * self.conversion_efficiency * optical_signal[i]
            v_curr = v_prev + alpha * (v_target - v_prev)
            electrical_signal[i] = v_curr
            v_prev = v_curr

        # Electrical power P_e(t) = V^2 / R_load (scaled proxy)
        electrical_power = (electrical_signal**2) / self.load_resistance_ohms

        # Energy Accounting:
        # 1. Acoustic driving energy: E_ac = Integral( (P_driver(t)^2 / (rho * c_s)) * A_in ) dt
        rho = DEFAULT_WATER_DENSITY
        c_s = acoustic_driver.sound_speed_m_s
        p_driver = acoustic_driver.waveform(time)
        acoustic_power = (p_driver**2 / (rho * c_s)) * acoustic_resonator.input_area_m2
        acoustic_energy_joules = (
            float(np.trapezoid(acoustic_power, time))
            if hasattr(np, "trapezoid")
            else float(np.trapz(acoustic_power, time))
        )

        # 2. Modeled optical emission energy proxy (scaled for consistency)
        integrated_emission = (
            float(np.trapezoid(emission_intensity, time))
            if hasattr(np, "trapezoid")
            else float(np.trapz(emission_intensity, time))
        )
        optical_energy_joules = 1e-10 * integrated_emission

        # 3. Modeled electrical energy
        electrical_energy_joules = (
            float(np.trapezoid(electrical_power, time))
            if hasattr(np, "trapezoid")
            else float(np.trapz(electrical_power, time))
        )

        # Overall efficiency ratio
        transduction_efficiency = (
            electrical_energy_joules / max(acoustic_energy_joules, 1e-12)
            if acoustic_energy_joules > 0
            else 0.0
        )

        return {
            "optical_signal": optical_signal,
            "electrical_signal": electrical_signal,
            "electrical_power": electrical_power,
            "acoustic_energy_joules": acoustic_energy_joules,
            "emission_energy_joules": optical_energy_joules,
            "electrical_energy_joules": electrical_energy_joules,
            "transduction_efficiency": float(transduction_efficiency),
        }


class SonoluminescenceSystem:
    """Composite resonant acousto-opto-electrical benchmark system for DRR.

    Encapsulates the acoustic driver, geometric waveguide resonator, cavitation
    model, nonlinear bubble oscillator, sonoluminescence emission, and electrical
    transduction into a unified reproducible simulation pipeline.
    """

    CHANNEL_NAMES: List[str] = [
        "acoustic_pressure",
        "waveguide_pressure",
        "bubble_radius",
        "bubble_velocity",
        "collapse_pressure",
        "emission_intensity",
        "optical_signal",
        "electrical_signal",
    ]

    def __init__(
        self,
        driver: Optional[AcousticDriver] = None,
        resonator: Optional[AcousticResonator] = None,
        cavitation: Optional[CavitationModel] = None,
        bubble_dynamics: Optional[BubbleDynamics] = None,
        emission_model: Optional[SonoluminescenceModel] = None,
        transducer: Optional[OpticalElectricalTransducer] = None,
        waveguide_material: Optional[WaveguideMaterial] = None,
        dopant_mixture: Optional[DopantMixture] = None,
    ) -> None:
        self.material = waveguide_material or (
            resonator.material if resonator else WAVEGUIDE_MATERIALS["copper_boron_alloy"]
        )
        self.driver = driver or AcousticDriver()
        self.resonator = resonator or AcousticResonator(material=self.material)
        self.dopant_mixture = dopant_mixture or DopantMixture()
        self.cavitation = cavitation or CavitationModel(dopant_mixture=self.dopant_mixture)
        self.bubble_dynamics = bubble_dynamics or BubbleDynamics(cavitation_model=self.cavitation)
        self.emission_model = emission_model or SonoluminescenceModel(
            dopant_mixture=self.dopant_mixture
        )
        self.transducer = transducer or OpticalElectricalTransducer()

    def simulate(
        self,
        duration: float = 0.002,
        sampling_rate: float = 100_000.0,
        noise_scale: float = 0.005,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Simulate the complete coupled acousto-opto-electrical benchmark."""
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        if sampling_rate < 2.0 * self.driver.frequency_hz:
            raise ValueError(
                f"sampling_rate ({sampling_rate} Hz) must be at least twice the acoustic "
                f"frequency ({self.driver.frequency_hz} Hz) for Nyquist-Shannon criteria."
            )
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")

        rng = np.random.default_rng(random_state)
        n_steps = int(np.round(duration * sampling_rate))
        if n_steps < 4:
            raise ValueError("Duration and sampling_rate produce too few samples (< 4)")

        time = np.linspace(0.0, duration, n_steps, endpoint=False)

        # 1. Acoustic excitation and waveguide cavity response
        acoustic_pressure = self.driver.waveform(time)
        cavity_gain, detuning, length_ratio = self.resonator.cavity_response(
            self.driver.frequency_hz, self.driver.sound_speed_m_s
        )
        geom_gain = self.resonator.geometric_pressure_gain
        waveguide_pressure = acoustic_pressure * (geom_gain * cavity_gain)

        # 2. Bubble dynamics & cavitation
        bubble_res = self.bubble_dynamics.simulate(time, waveguide_pressure)
        bubble_radius = bubble_res["normalized_radius"]
        bubble_velocity = bubble_res["velocity"]
        collapse_pressure = bubble_res["collapse_pressure"]

        # 3. Sonoluminescent emission
        emission_res = self.emission_model.simulate_emission(
            time, bubble_res, sound_speed_m_s=self.driver.sound_speed_m_s
        )
        emission_intensity = emission_res["emission_intensity"]

        # 4. Electrical transduction & energy accounting
        trans_res = self.transducer.transduce(time, emission_intensity, self.driver, self.resonator)
        optical_signal = trans_res["optical_signal"]
        electrical_signal = trans_res["electrical_signal"]

        # Optional observational noise
        if noise_scale > 0:
            noise_matrix = rng.normal(scale=noise_scale, size=(n_steps, 8))
            acoustic_pressure_noisy = acoustic_pressure + noise_matrix[:, 0] * (
                self.driver.input_pressure_pa * 0.05
            )
            waveguide_pressure_noisy = waveguide_pressure + noise_matrix[:, 1] * (
                self.driver.input_pressure_pa * geom_gain * 0.05
            )
            bubble_radius_noisy = np.maximum(0.05, bubble_radius + noise_matrix[:, 2] * 0.02)
            bubble_velocity_noisy = bubble_velocity + noise_matrix[:, 3] * 10.0
            collapse_pressure_noisy = np.maximum(0.0, collapse_pressure + noise_matrix[:, 4] * 1e4)
            emission_intensity_noisy = np.clip(
                emission_intensity + noise_matrix[:, 5] * 0.01, 0.0, 1.0
            )
            optical_signal_noisy = np.maximum(0.0, optical_signal + noise_matrix[:, 6] * 0.01)
            electrical_signal_noisy = np.maximum(0.0, electrical_signal + noise_matrix[:, 7] * 0.01)
        else:
            acoustic_pressure_noisy = acoustic_pressure
            waveguide_pressure_noisy = waveguide_pressure
            bubble_radius_noisy = bubble_radius
            bubble_velocity_noisy = bubble_velocity
            collapse_pressure_noisy = collapse_pressure
            emission_intensity_noisy = emission_intensity
            optical_signal_noisy = optical_signal
            electrical_signal_noisy = electrical_signal

        data = np.column_stack(
            [
                acoustic_pressure_noisy,
                waveguide_pressure_noisy,
                bubble_radius_noisy,
                bubble_velocity_noisy,
                collapse_pressure_noisy,
                emission_intensity_noisy,
                optical_signal_noisy,
                electrical_signal_noisy,
            ]
        )

        metadata: Dict[str, Any] = {
            "benchmark_system": "sonoluminescence_acousto_opto_electrical",
            "version": "1.1.0",
            "sampling_rate_hz": float(sampling_rate),
            "duration_s": float(duration),
            "n_samples": int(n_steps),
            "channel_names": list(self.CHANNEL_NAMES),
            "waveguide_material_properties": {
                "name": self.material.name if self.material else "custom",
                "copper_fraction": float(self.material.copper_fraction) if self.material else 0.0,
                "boron_fraction": float(self.material.boron_fraction) if self.material else 0.0,
                "solid_density_kg_m3": float(self.material.density_kg_m3) if self.material else 0.0,
                "solid_sound_speed_m_s": (
                    float(self.material.sound_speed_m_s) if self.material else 0.0
                ),
                "acoustic_impedance_rayl": (
                    float(self.material.acoustic_impedance_rayl) if self.material else 0.0
                ),
                "interface_transmission_coefficient": (
                    float(
                        self.material.interface_transmission_coefficient(
                            self.cavitation.density, self.driver.sound_speed_m_s
                        )
                    )
                    if self.material
                    else 1.0
                ),
            },
            "dopant_mixture_properties": {
                "carrier_liquid": self.dopant_mixture.carrier_liquid,
                "gas_species": self.dopant_mixture.gas_species,
                "gas_fraction": float(self.dopant_mixture.gas_fraction),
                "copper_solute_fraction": float(self.dopant_mixture.copper_solute_fraction),
                "boron_solute_fraction": float(self.dopant_mixture.boron_solute_fraction),
                "alkali_solute_fraction": float(self.dopant_mixture.alkali_solute_fraction),
                "effective_density_kg_m3": float(self.dopant_mixture.effective_density_kg_m3),
                "effective_viscosity_pa_s": float(self.dopant_mixture.effective_viscosity_pa_s),
                "effective_surface_tension_n_m": float(
                    self.dopant_mixture.effective_surface_tension_n_m
                ),
                "effective_polytropic_index": float(self.dopant_mixture.effective_polytropic_index),
            },
            "acoustic_parameters": {
                "frequency_hz": float(self.driver.frequency_hz),
                "sound_speed_m_s": float(self.driver.sound_speed_m_s),
                "acoustic_wavelength_m": float(self.driver.acoustic_wavelength),
                "input_pressure_pa": float(self.driver.input_pressure_pa),
                "is_physical": True,
            },
            "waveguide_resonator_parameters": {
                "input_diameter_m": float(self.resonator.input_diameter_m),
                "output_diameter_m": float(self.resonator.output_diameter_m),
                "length_m": float(self.resonator.length_m),
                "area_ratio": float(self.resonator.area_ratio),
                "geometric_pressure_gain": float(geom_gain),
                "quality_factor_q": float(self.resonator.effective_quality_factor),
                "cavity_gain": float(cavity_gain),
                "detuning_error": float(detuning),
                "length_ratio_in_acoustic_wavelengths": float(length_ratio),
                "is_physical_idealized": True,
            },
            "cavitation_parameters": {
                "equilibrium_radius_m": float(self.cavitation.equilibrium_radius_m),
                "blake_threshold_pressure_pa": float(self.cavitation.blake_threshold_pressure),
                "blake_acoustic_threshold_pa": float(self.cavitation.blake_acoustic_threshold_pa),
                "is_cavitation_active": bool(
                    self.cavitation.is_cavitation_active(
                        self.resonator.effective_pressure_amplitude(self.driver)
                    )
                ),
                "max_compression_ratio": float(bubble_res["max_compression_ratio"]),
                "collapse_event_count": int(len(bubble_res["collapse_indices"])),
                "is_physical_formulation": True,
            },
            "optical_emission_parameters": {
                "spectral_center_nm": float(emission_res["spectral_center_nm"]),
                "spectral_bandwidth_nm": float(emission_res["spectral_bandwidth_nm"]),
                "optical_frequency_hz": float(emission_res["optical_frequency_hz"]),
                "pulse_duration_s": float(emission_res["emission_duration_s"]),
                "peak_emission": float(emission_res["peak_emission"]),
                "emission_event_count": int(len(emission_res["emission_event_times"])),
                "spectral_lines": emission_res.get("spectral_lines", []),
                "is_phenomenological": True,
            },
            "transduction_parameters": {
                "optical_collection_efficiency": float(
                    self.transducer.optical_collection_efficiency
                ),
                "conversion_efficiency": float(self.transducer.conversion_efficiency),
                "detector_gain": float(self.transducer.detector_gain),
                "load_resistance_ohms": float(self.transducer.load_resistance_ohms),
                "acoustic_input_energy_joules": float(trans_res["acoustic_energy_joules"]),
                "modeled_emission_energy_joules": float(trans_res["emission_energy_joules"]),
                "modeled_electrical_energy_joules": float(trans_res["electrical_energy_joules"]),
                "transduction_efficiency_ratio": float(trans_res["transduction_efficiency"]),
                "is_downstream_transduction_model": True,
            },
            "notes": (
                "Sonoluminescence computational benchmark for DRR multimodal resonance "
                "and causal rooting analysis with copper/boron waveguide metallurgy and fluid doping. "
                "No net energy amplification is claimed."
            ),
        }

        return time, data, metadata


def generate_sonoluminescence_system(
    sampling_rate: float = 100_000.0,
    duration: float = 0.002,
    acoustic_frequency_hz: float = 25_000.0,
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED,
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
    waveguide_material: Optional[Union[str, WaveguideMaterial]] = None,
    copper_solute_fraction: float = 0.0,
    boron_solute_fraction: float = 0.0,
    noble_gas_fraction: float = 0.01,
    noble_gas_species: Literal["argon", "xenon", "helium", "nitrogen", "air"] = "argon",
    dopant_mixture: Optional[DopantMixture] = None,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate multivariate acousto-opto-electrical sonoluminescence benchmark data.

    Returns:
        time: 1D array of timestamps (seconds).
        data: 2D array of shape (N, 8) with aligned channels:
            0: acoustic_pressure (Pa)
            1: waveguide_pressure (Pa)
            2: bubble_radius (normalized R/R0)
            3: bubble_velocity (m/s)
            4: collapse_pressure (Pa)
            5: emission_intensity (normalized [0, 1])
            6: optical_signal (a.u.)
            7: electrical_signal (a.u.)
        metadata: Comprehensive metadata documenting physical parameters, mixtures, and units.
    """
    # Resolve waveguide material
    mat_obj: Optional[WaveguideMaterial] = None
    if isinstance(waveguide_material, str):
        if waveguide_material in WAVEGUIDE_MATERIALS:
            mat_obj = WAVEGUIDE_MATERIALS[waveguide_material]
        else:
            raise ValueError(
                f"Unknown waveguide_material preset '{waveguide_material}'. "
                f"Available presets: {list(WAVEGUIDE_MATERIALS.keys())}"
            )
    elif isinstance(waveguide_material, WaveguideMaterial):
        mat_obj = waveguide_material

    # Resolve dopant mixture
    mix_obj = dopant_mixture or DopantMixture(
        gas_species=noble_gas_species,
        gas_fraction=noble_gas_fraction,
        copper_solute_fraction=copper_solute_fraction,
        boron_solute_fraction=boron_solute_fraction,
    )

    driver = AcousticDriver(
        frequency_hz=acoustic_frequency_hz,
        sound_speed_m_s=sound_speed_m_s,
        input_pressure_pa=input_pressure_pa,
    )
    resonator = AcousticResonator(
        input_diameter_m=waveguide_input_diameter_m,
        output_diameter_m=waveguide_output_diameter_m,
        length_m=resonator_length_m,
        quality_factor_q=quality_factor_q,
        material=mat_obj,
    )
    cavitation = CavitationModel(
        equilibrium_radius_m=bubble_radius_m,
        dopant_mixture=mix_obj,
    )
    bubble = BubbleDynamics(cavitation_model=cavitation)
    emission = SonoluminescenceModel(
        spectral_center_nm=optical_wavelength_nm,
        dopant_mixture=mix_obj,
    )
    transducer = OpticalElectricalTransducer(
        optical_collection_efficiency=optical_collection_efficiency,
        conversion_efficiency=conversion_efficiency,
        detector_gain=detector_gain,
    )

    system = SonoluminescenceSystem(
        driver=driver,
        resonator=resonator,
        cavitation=cavitation,
        bubble_dynamics=bubble,
        emission_model=emission,
        transducer=transducer,
        waveguide_material=mat_obj,
        dopant_mixture=mix_obj,
    )

    return system.simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        noise_scale=noise_scale,
        random_state=random_state,
    )


def calculate_resonant_transduction_efficiency_index(
    drr_results: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, float]:
    """Compute the Resonant Transduction Efficiency Index (RTEI).

    RTEI = D_acoustic * D_cavitation * D_emission * D_electrical * (G_cavity / Q)
    """
    depths = drr_results.get("resonance_depths", {})
    d_acoustic = float(depths.get("dim_0", depths.get("dim_1", 0.5)))
    d_waveguide = float(depths.get("dim_1", 0.5))
    d_cavitation = float(depths.get("dim_2", depths.get("dim_3", 0.5)))
    d_emission = float(depths.get("dim_5", 0.5))
    d_electrical = float(depths.get("dim_7", 0.5))

    q = float(metadata.get("waveguide_resonator_parameters", {}).get("quality_factor_q", 30.0))
    cav_gain = float(metadata.get("waveguide_resonator_parameters", {}).get("cavity_gain", 1.0))
    cavity_coherence_factor = float(np.clip(cav_gain / max(q, 1.0), 0.0, 1.0))

    stage_coherence = (
        (d_acoustic * 0.5 + d_waveguide * 0.5) * d_cavitation * d_emission * d_electrical
    )
    rtei = float(np.clip(stage_coherence * cavity_coherence_factor, 0.0, 1.0))

    return {
        "rtei": rtei,
        "acoustic_resonance_depth": float(d_acoustic),
        "waveguide_resonance_depth": float(d_waveguide),
        "cavitation_resonance_depth": float(d_cavitation),
        "emission_resonance_depth": float(d_emission),
        "electrical_resonance_depth": float(d_electrical),
        "cavity_coherence_factor": float(cavity_coherence_factor),
    }


__all__ = [
    "SPEED_OF_LIGHT",
    "DEFAULT_SOUND_SPEED",
    "DEFAULT_WATER_DENSITY",
    "DEFAULT_SURFACE_TENSION",
    "DEFAULT_LIQUID_VISCOSITY",
    "DEFAULT_VAPOR_PRESSURE",
    "DEFAULT_AMBIENT_PRESSURE",
    "DEFAULT_POLYTROPIC_INDEX",
    "WAVEGUIDE_MATERIALS",
    "WaveguideMaterial",
    "DopantMixture",
    "AcousticDriver",
    "AcousticResonator",
    "AcousticWaveguide",
    "CavitationModel",
    "BubbleDynamics",
    "SonoluminescenceModel",
    "OpticalElectricalTransducer",
    "SonoluminescenceSystem",
    "generate_sonoluminescence_system",
    "calculate_resonant_transduction_efficiency_index",
]
