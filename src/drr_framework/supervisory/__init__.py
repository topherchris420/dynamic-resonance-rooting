"""Local, evidence-first regulatory monitoring. No network calls on import."""

from .analyzer import DRRConfig, LFBORegimeAnalyzer
from .backtesting import walk_forward_validate
from .baselines import (
    BaselineConfig,
    BaselineResult,
    lagged_correlations,
    matched_evaluation,
    run_baselines,
)
from .briefs import generate_lfbo_monitoring_brief, generate_morning_brief
from .change_detection import MaterialChange, attach_peer_context, detect_material_changes
from .common import DECISION_BOUNDARY
from .entity_graph import EntityGraph, EntityRelationship, PublicEntity
from .evidence_ledger import (
    AuditEvent,
    AnalystDisposition,
    AnalystReview,
    EvidenceEntry,
    EvidenceLedger,
)
from .falsification import (
    SignalRobustnessReport,
    SignalSpecification,
    SpecificationResult,
    falsify_drr_signal,
    falsify_material_change,
    run_falsification,
)
from .feedback import AnalystFeedbackMetrics, evaluate_signal_usefulness
from .ingestion import FilingContext, ingest_wide_csv, ingest_wide_filing
from .issues import IssueTracker, MonitoringIssue
from .model_risk import (
    MODEL_RISK_REFERENCE_BASIS,
    MODEL_RISK_SECTIONS,
    ModelRiskProfile,
    ModelRiskTier,
    ModelUseClassification,
    ValidationEvidence,
    ValidationStatus,
)
from .monitoring import (
    AttentionBudget,
    AttentionItem,
    AttentionSelection,
    MonitoringDelta,
    MonitoringSignal,
    ReviewState,
    compare_review_states,
    build_review_activity,
)
from .passport import (
    AnalysisPassport,
    dependency_inventory,
    software_identity,
    verify_monitoring_snapshot,
)
from .revisions import build_revision_audit
from .peer_analysis import PeerAnalysis, PeerGroupDefinition, SignalContext, analyze_peers
from .policy_context import ApplicabilityBasis, PolicyContext, PolicyEvent
from .reconciliation import (
    DataQualityException,
    ReconciliationRule,
    reconcile_dataset,
    reconcile_store,
)
from .semantics import MetricDefinition, SemanticRegistry, VerificationStatus, bundled_registry
from .snc import PublicSNCAggregate, analyze_public_snc
from .vintage import (
    CalculationLineage,
    FilingRevision,
    ObservationProvenance,
    RegulatoryObservation,
    VintageStore,
)
from .workbench import (
    DEFAULT_MODEL_RISK_PROFILE,
    MonitoringWorkbench,
    WorkbenchConfig,
    review_state_from_dict,
)
from .perspectives import (
    PerspectiveKind,
    ScopedPerspective,
    DocumentedDisagreement,
    PerspectiveInventory,
)

__all__ = [
    "PerspectiveKind",
    "ScopedPerspective",
    "DocumentedDisagreement",
    "PerspectiveInventory",
    "DECISION_BOUNDARY",
    "DEFAULT_MODEL_RISK_PROFILE",
    "MODEL_RISK_REFERENCE_BASIS",
    "MODEL_RISK_SECTIONS",
    "AnalysisPassport",
    "AnalystDisposition",
    "AnalystFeedbackMetrics",
    "AnalystReview",
    "ApplicabilityBasis",
    "AttentionBudget",
    "AttentionItem",
    "AttentionSelection",
    "AuditEvent",
    "BaselineConfig",
    "BaselineResult",
    "CalculationLineage",
    "DRRConfig",
    "DataQualityException",
    "EntityGraph",
    "EntityRelationship",
    "EvidenceEntry",
    "EvidenceLedger",
    "FilingContext",
    "FilingRevision",
    "IssueTracker",
    "LFBORegimeAnalyzer",
    "MaterialChange",
    "MetricDefinition",
    "ModelRiskProfile",
    "ModelRiskTier",
    "ModelUseClassification",
    "MonitoringDelta",
    "MonitoringIssue",
    "MonitoringSignal",
    "MonitoringWorkbench",
    "ObservationProvenance",
    "PeerAnalysis",
    "PeerGroupDefinition",
    "PolicyContext",
    "PolicyEvent",
    "PublicEntity",
    "PublicSNCAggregate",
    "ReconciliationRule",
    "RegulatoryObservation",
    "ReviewState",
    "SemanticRegistry",
    "SignalContext",
    "SignalRobustnessReport",
    "SignalSpecification",
    "SpecificationResult",
    "ValidationEvidence",
    "ValidationStatus",
    "VerificationStatus",
    "VintageStore",
    "WorkbenchConfig",
    "analyze_peers",
    "analyze_public_snc",
    "attach_peer_context",
    "bundled_registry",
    "compare_review_states",
    "build_review_activity",
    "build_revision_audit",
    "verify_monitoring_snapshot",
    "dependency_inventory",
    "detect_material_changes",
    "evaluate_signal_usefulness",
    "falsify_drr_signal",
    "falsify_material_change",
    "generate_lfbo_monitoring_brief",
    "generate_morning_brief",
    "ingest_wide_csv",
    "ingest_wide_filing",
    "lagged_correlations",
    "matched_evaluation",
    "reconcile_dataset",
    "reconcile_store",
    "review_state_from_dict",
    "run_baselines",
    "run_falsification",
    "software_identity",
    "walk_forward_validate",
]
