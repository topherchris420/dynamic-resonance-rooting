"""Selected SR 26-2 principles, not a compliance certification or bank risk rating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .common import instant, sha256_hex

MODEL_RISK_REFERENCE_BASIS = (
    {
        "label": "SR 26-2 Revised Guidance on Model Risk Management",
        "status": "current",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm",
        "verified_on": "2026-09-12",
    },
    {
        "label": "SR 26-2 attachment: model risk management principles",
        "status": "current",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf",
        "verified_on": "2026-09-12",
    },
    {
        "label": "SR 11-7 model risk management guidance (superseded April 17, 2026)",
        "status": "superseded",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm",
    },
)

MODEL_RISK_SECTIONS = (
    "scope_and_use_classification",
    "inherent_risk",
    "exposure_and_purpose",
    "materiality",
    "intended_use_and_foreseeable_misuse",
    "conceptual_soundness",
    "implementation_verification",
    "outcomes_analysis",
    "ongoing_monitoring",
    "effective_challenge",
    "aggregate_model_dependencies",
    "governance_and_change_control",
    "third_party_products",
)


class ModelUseClassification(str, Enum):
    DETERMINISTIC_TOOL = "deterministic_tool_outside_model_definition"
    RESEARCH_DIAGNOSTIC = "research_diagnostic"
    SHADOW_MONITORING = "shadow_monitoring"


class ModelRiskTier(str, Enum):
    UNASSESSED = "unassessed"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ValidationStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    DEVELOPMENT_TESTED = "development_tested"
    INDEPENDENT_REVIEW_PENDING = "independent_review_pending"
    INDEPENDENTLY_REVIEWED = "independently_reviewed_for_scoped_use"


@dataclass(frozen=True)
class ValidationEvidence:
    area: str
    evidence_ids: Tuple[str, ...]
    reviewer: str
    reviewed_at: str
    independent: bool = False
    limitations: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not isinstance(self.area, str) or not self.area.strip():
            raise ValueError("Validation evidence requires an area")
        if not self.evidence_ids or any(
            not isinstance(value, str) or not value.strip() for value in self.evidence_ids
        ):
            raise ValueError("Validation evidence requires immutable evidence IDs")
        if not isinstance(self.reviewer, str) or not self.reviewer.strip():
            raise ValueError("Validation evidence requires an attributed reviewer")
        object.__setattr__(self, "reviewed_at", instant(self.reviewed_at).isoformat())
        if not isinstance(self.independent, bool):
            raise ValueError("Validation evidence independence must be explicit")
        if any(not isinstance(value, str) or not value.strip() for value in self.limitations):
            raise ValueError("Validation evidence limitations must be nonempty text")

    @property
    def content_addressed(self):
        """Whether every referenced artifact is a SHA-256 content identifier."""
        return all(_is_sha256_identifier(value) for value in self.evidence_ids)


@dataclass(frozen=True)
class ModelRiskProfile:
    model_name: str
    intended_use: str
    foreseeable_misuse: Tuple[str, ...]
    use_classification: ModelUseClassification = ModelUseClassification.RESEARCH_DIAGNOSTIC
    risk_tier: ModelRiskTier = ModelRiskTier.UNASSESSED
    complexity: str = "not assessed"
    data_quality: str = "not assessed"
    exposure: str = "not assessed"
    purpose: str = "analytical research"
    materiality: str = "not assessed"
    tier_rationale: str = ""
    validation_status: ValidationStatus = ValidationStatus.NOT_REVIEWED
    evidence: Tuple[ValidationEvidence, ...] = ()
    limitations: Tuple[str, ...] = ()
    use_boundaries: Tuple[str, ...] = ("human interpretation required",)
    monitoring_plan: Tuple[str, ...] = ()
    change_control: Tuple[str, ...] = ()
    dependencies: Tuple[str, ...] = ()
    third_party_products: Tuple[str, ...] = ()

    def __post_init__(self):
        for name, enum in (
            ("use_classification", ModelUseClassification),
            ("risk_tier", ModelRiskTier),
            ("validation_status", ValidationStatus),
        ):
            object.__setattr__(self, name, enum(getattr(self, name)))
        for name in (
            "foreseeable_misuse",
            "evidence",
            "limitations",
            "use_boundaries",
            "monitoring_plan",
            "change_control",
            "dependencies",
            "third_party_products",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "evidence",
            tuple(
                ValidationEvidence(**item) if isinstance(item, dict) else item
                for item in self.evidence
            ),
        )
        if any(not isinstance(item, ValidationEvidence) for item in self.evidence):
            raise ValueError("Model-risk evidence must use ValidationEvidence records")
        if not self.model_name or not self.intended_use or not self.foreseeable_misuse:
            raise ValueError("Intended use and foreseeable misuse must be explicit")
        if self.risk_tier != ModelRiskTier.UNASSESSED and not self.tier_rationale:
            raise ValueError("A human-assigned model risk tier needs a rationale")
        if self.validation_status == ValidationStatus.INDEPENDENTLY_REVIEWED and not any(
            e.independent and e.content_addressed for e in self.evidence
        ):
            raise ValueError(
                "Independent review status requires independent, content-addressed evidence"
            )


def _is_sha256_identifier(value):
    try:
        sha256_hex(value)
    except ValueError:
        return False
    return True
