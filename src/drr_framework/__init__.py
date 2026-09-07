"""
Dynamic Resonance Rooting (DRR) Framework

A computational framework for analyzing Complex Adaptive Systems through
Dynamic Resonance Detection, Rooting Analysis, and Resonance Depth Calculation.

Author: Christopher Woodyard
License: MIT
"""

# Lazy import pattern for optional dependencies
# Core modules that work without optional dependencies
from .modules import ResonanceDetector, RootingAnalyzer, DepthCalculator, AnomalyDetector
from .benchmarks import BenchmarkSystems, generate_micro_doppler_analog
from .datasets import (
    PolicyResonanceDataset,
    SupervisoryPanelDataset,
    load_policy_dataset,
    load_policy_dataset_from_sql,
    load_supervisory_panel,
    load_supervisory_panel_from_sql,
)
from .realtime import RealTimeDRR
from .state_space import (
    ChandrasekharResult,
    KalmanResult,
    Measurement,
    StateSpaceSystem,
    Transition,
    analyze_resonance_state_space,
    chandrasekhar_recursion,
    fit_resonance_state_space,
    impulse_response,
    kalman_filter,
    stationary_initialization,
)
from .smoothers import (
    SimulationSmootherResult,
    SmootherResult,
    carter_kohn_smoother,
    durbin_koopman_smoother,
    hamilton_smoother,
    kalman_smoother,
    koopman_smoother,
)
from .particle_filter import (
    NonlinearStateSpaceModel,
    ParticleFilterResult,
    linear_gaussian_model,
    tempered_particle_filter,
)
from .resonance_kernel import resonance_regularized_covariance, resonance_risk_kernel
from .structural_surprise import StructuralSurpriseResult, structural_surprise
from .topology_dynamics import (
    RootingTopologySummary,
    root_distribution,
    root_migration,
    summarize_topology,
    topology_drift,
)
from .sonoluminescence import (
    DEFAULT_AMBIENT_PRESSURE,
    DEFAULT_LIQUID_VISCOSITY,
    DEFAULT_POLYTROPIC_INDEX,
    DEFAULT_SOUND_SPEED,
    DEFAULT_SURFACE_TENSION,
    DEFAULT_VAPOR_PRESSURE,
    DEFAULT_WATER_DENSITY,
    SPEED_OF_LIGHT,
    WAVEGUIDE_MATERIALS,
    AcousticDriver,
    AcousticResonator,
    AcousticWaveguide,
    BubbleDynamics,
    CavitationModel,
    DopantMixture,
    OpticalElectricalTransducer,
    SonoluminescenceModel,
    SonoluminescenceSystem,
    WaveguideMaterial,
    calculate_resonant_transduction_efficiency_index,
    generate_sonoluminescence_system,
)


# Lazy imports for modules with heavy optional dependencies
def __getattr__(name):
    if name in (
        "ControlIntervention",
        "NavigationMetrics",
        "ResonanceControlExperimentSuite",
        "ResonanceNavigationEngine",
        "ResonanceState",
        "ResonanceTarget",
    ):
        from . import control_engine

        return getattr(control_engine, name)
    if name in (
        "CollectiveResonanceModes",
        "CrossResonanceTensor",
        "collective_modes",
        "estimate_cross_resonance",
    ):
        from . import cross_resonance

        return getattr(cross_resonance, name)
    if name in (
        "DynamicResonanceGeometry",
        "ResonanceGeometryResult",
        "StructuralResonanceFingerprint",
    ):
        from . import resonance_geometry

        return getattr(resonance_geometry, name)
    if name in (
        "render_markdown_report",
        "serialize_analysis_results",
        "summarize_analysis_results",
        "write_analysis_report",
        "write_tableau_artifacts",
    ):
        from . import reporting

        return getattr(reporting, name)
    if name in (
        "FED_SUPERVISORY_REFERENCE_BASIS",
        "SUPERVISORY_INSTITUTION_PROFILES",
        "SUPERVISORY_RISK_DOMAINS",
        "SupervisoryInstitutionProfile",
        "SupervisoryRiskDomain",
        "build_supervisory_alignment_metadata",
        "supervisory_profile_options",
        "supervisory_risk_domain_options",
    ):
        from . import supervision

        return getattr(supervision, name)
    if name in ("DynamicResonanceRooting",):
        from . import analysis

        return getattr(analysis, name)
    if name in ("GenerativeDesignSuite",):
        from . import generative_design_suite

        return getattr(generative_design_suite, name)
    if name in ("generate_coupled_oscillator", "run_reproduction_experiment"):
        from . import validation

        return getattr(validation, name)
    if name in (
        "VALIDATION_READINESS_CHECKLIST",
        "VALIDATION_REFERENCE_BASIS",
        "append_shadow_review_record",
        "build_model_risk_card",
        "build_validation_readiness_packet",
        "create_shadow_review_record",
        "explain_supervisory_signal",
        "run_event_backtest",
    ):
        from . import validation_readiness

        return getattr(validation_readiness, name)
    if name == "finance":
        from . import finance

        return finance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Version is single-sourced from the installed package metadata (pyproject.toml).
try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _package_version

    try:
        __version__ = _package_version("drr-framework")
    except _PackageNotFoundError:
        __version__ = "4.3.0"
except ImportError:  # pragma: no cover - Python < 3.8
    __version__ = "4.3.0"

__author__ = "Christopher Woodyard"
__email__ = "ciao_chris@example.com"

__all__ = [
    "BenchmarkSystems",
    "generate_micro_doppler_analog",
    "PolicyResonanceDataset",
    "SupervisoryPanelDataset",
    "load_policy_dataset",
    "load_policy_dataset_from_sql",
    "load_supervisory_panel",
    "load_supervisory_panel_from_sql",
    "ResonanceDetector",
    "RootingAnalyzer",
    "DepthCalculator",
    "AnomalyDetector",
    "RealTimeDRR",
    "SupervisoryInstitutionProfile",
    "SupervisoryRiskDomain",
    "SUPERVISORY_INSTITUTION_PROFILES",
    "SUPERVISORY_RISK_DOMAINS",
    "FED_SUPERVISORY_REFERENCE_BASIS",
    "build_supervisory_alignment_metadata",
    "supervisory_profile_options",
    "supervisory_risk_domain_options",
    "Transition",
    "Measurement",
    "StateSpaceSystem",
    "KalmanResult",
    "ChandrasekharResult",
    "fit_resonance_state_space",
    "kalman_filter",
    "impulse_response",
    "analyze_resonance_state_space",
    "chandrasekhar_recursion",
    "stationary_initialization",
    "SmootherResult",
    "SimulationSmootherResult",
    "kalman_smoother",
    "hamilton_smoother",
    "koopman_smoother",
    "durbin_koopman_smoother",
    "carter_kohn_smoother",
    "NonlinearStateSpaceModel",
    "ParticleFilterResult",
    "tempered_particle_filter",
    "linear_gaussian_model",
    "CrossResonanceTensor",
    "CollectiveResonanceModes",
    "estimate_cross_resonance",
    "collective_modes",
    "DynamicResonanceGeometry",
    "ResonanceGeometryResult",
    "StructuralResonanceFingerprint",
    "RootingTopologySummary",
    "root_distribution",
    "root_migration",
    "topology_drift",
    "summarize_topology",
    "StructuralSurpriseResult",
    "structural_surprise",
    "resonance_risk_kernel",
    "resonance_regularized_covariance",
    "serialize_analysis_results",
    "summarize_analysis_results",
    "render_markdown_report",
    "write_analysis_report",
    "write_tableau_artifacts",
    "DynamicResonanceRooting",
    "GenerativeDesignSuite",
    "generate_coupled_oscillator",
    "run_reproduction_experiment",
    "VALIDATION_READINESS_CHECKLIST",
    "VALIDATION_REFERENCE_BASIS",
    "append_shadow_review_record",
    "build_model_risk_card",
    "build_validation_readiness_packet",
    "create_shadow_review_record",
    "explain_supervisory_signal",
    "run_event_backtest",
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
    "ResonanceState",
    "ResonanceTarget",
    "ControlIntervention",
    "NavigationMetrics",
    "ResonanceNavigationEngine",
    "ResonanceControlExperimentSuite",
]
