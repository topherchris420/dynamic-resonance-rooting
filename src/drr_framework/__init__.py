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
from .benchmarks import BenchmarkSystems
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
    KalmanResult,
    Measurement,
    StateSpaceSystem,
    Transition,
    analyze_resonance_state_space,
    fit_resonance_state_space,
    impulse_response,
    kalman_filter,
)

# Lazy imports for modules with heavy optional dependencies
def __getattr__(name):
    if name in ("render_markdown_report", "serialize_analysis_results", "summarize_analysis_results",
                "write_analysis_report", "write_tableau_artifacts"):
        from . import reporting
        return getattr(reporting, name)
    if name in ("FED_SUPERVISORY_REFERENCE_BASIS", "SUPERVISORY_INSTITUTION_PROFILES",
                "SUPERVISORY_RISK_DOMAINS", "SupervisoryInstitutionProfile", "SupervisoryRiskDomain",
                "build_supervisory_alignment_metadata", "supervisory_profile_options",
                "supervisory_risk_domain_options"):
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
    if name in ("VALIDATION_READINESS_CHECKLIST", "VALIDATION_REFERENCE_BASIS",
                "append_shadow_review_record", "build_model_risk_card",
                "build_validation_readiness_packet", "create_shadow_review_record",
                "explain_supervisory_signal", "run_event_backtest"):
        from . import validation_readiness
        return getattr(validation_readiness, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.0"
__author__ = "Christopher Woodyard"
__email__ = "ciao_chris@example.com"

__all__ = [
    "BenchmarkSystems",
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
    "fit_resonance_state_space",
    "kalman_filter",
    "impulse_response",
    "analyze_resonance_state_space",
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
]
