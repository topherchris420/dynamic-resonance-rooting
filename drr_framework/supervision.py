"""Federal Reserve supervisory alignment helpers for DRR reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SupervisoryInstitutionProfile:
    """Institution segment preset from the Federal Reserve supervision taxonomy."""

    key: str
    label: str
    description: str
    rating_frameworks: Tuple[str, ...]
    review_focus: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "rating_frameworks": list(self.rating_frameworks),
            "review_focus": list(self.review_focus),
        }


@dataclass(frozen=True)
class SupervisoryRiskDomain:
    """Risk-domain tag used to keep DRR outputs in supervisory language."""

    key: str
    label: str
    description: str
    typical_metrics: Tuple[str, ...]
    rating_alignment: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "typical_metrics": list(self.typical_metrics),
            "rating_alignment": list(self.rating_alignment),
        }


SUPERVISORY_INSTITUTION_PROFILES: Mapping[str, SupervisoryInstitutionProfile] = {
    "community_bank": SupervisoryInstitutionProfile(
        key="community_bank",
        label="Community bank",
        description="Smaller banking organization supervised with emphasis on local business model and risk profile.",
        rating_frameworks=("CAMELS",),
        review_focus=("local credit conditions", "capital adequacy", "liquidity", "management controls"),
    ),
    "regional_or_foreign_bank_under_100b": SupervisoryInstitutionProfile(
        key="regional_or_foreign_bank_under_100b",
        label="Regional bank or foreign bank under $100B",
        description="Regional bank or foreign banking organization with U.S. assets below $100 billion.",
        rating_frameworks=("CAMELS", "RFI", "ROCA"),
        review_focus=("asset quality", "funding profile", "governance", "cross-border controls"),
    ),
    "large_bank_or_large_fbo": SupervisoryInstitutionProfile(
        key="large_bank_or_large_fbo",
        label="Large bank or large FBO",
        description="Large domestic banking organization or large foreign banking organization.",
        rating_frameworks=("LFI",),
        review_focus=("capital planning", "liquidity risk", "governance and controls", "recovery readiness"),
    ),
    "g_sib": SupervisoryInstitutionProfile(
        key="g_sib",
        label="Global systemically important bank",
        description="Systemically important banking organization with heightened monitoring expectations.",
        rating_frameworks=("LFI", "G-SIB surcharge"),
        review_focus=("resolvability", "interconnectedness", "market and counterparty risk", "liquidity resilience"),
    ),
    "financial_market_utility": SupervisoryInstitutionProfile(
        key="financial_market_utility",
        label="Financial market utility",
        description="Designated financial market utility or critical financial market infrastructure.",
        rating_frameworks=("Regulation HH", "PFMI"),
        review_focus=("operational resilience", "settlement risk", "participant risk", "recovery planning"),
    ),
}


SUPERVISORY_RISK_DOMAINS: Mapping[str, SupervisoryRiskDomain] = {
    "capital": SupervisoryRiskDomain(
        key="capital",
        label="Capital adequacy",
        description="Capital position, planning, and sensitivity to losses or stress paths.",
        typical_metrics=("capital ratio", "risk-weighted assets", "stress capital buffer"),
        rating_alignment=("CAMELS capital", "LFI capital"),
    ),
    "liquidity": SupervisoryRiskDomain(
        key="liquidity",
        label="Liquidity and funding",
        description="Liquidity reserves, funding mix, deposit behavior, and market funding pressure.",
        typical_metrics=("liquidity ratio", "funding spread", "deposit outflow proxy"),
        rating_alignment=("CAMELS liquidity", "LFI liquidity"),
    ),
    "asset_quality": SupervisoryRiskDomain(
        key="asset_quality",
        label="Asset quality and credit",
        description="Credit performance, delinquency, charge-off, concentration, and portfolio quality signals.",
        typical_metrics=("nonperforming loans", "charge-offs", "CRE concentration"),
        rating_alignment=("CAMELS asset quality",),
    ),
    "earnings": SupervisoryRiskDomain(
        key="earnings",
        label="Earnings and profitability",
        description="Earnings durability, margin pressure, and recurring income quality.",
        typical_metrics=("net interest margin", "return on assets", "provision expense"),
        rating_alignment=("CAMELS earnings",),
    ),
    "market_risk": SupervisoryRiskDomain(
        key="market_risk",
        label="Market and rate risk",
        description="Sensitivity to rate, spread, market-liquidity, and trading-book movements.",
        typical_metrics=("duration gap", "trading VaR proxy", "securities unrealized loss proxy"),
        rating_alignment=("CAMELS sensitivity", "LFI capital"),
    ),
    "operational_resilience": SupervisoryRiskDomain(
        key="operational_resilience",
        label="Operational and cyber resilience",
        description="Technology, cyber, third-party, and operational continuity monitoring signals.",
        typical_metrics=("incident count", "critical-service outage", "cyber control exception rate"),
        rating_alignment=("management", "LFI governance and controls"),
    ),
    "governance_controls": SupervisoryRiskDomain(
        key="governance_controls",
        label="Governance and controls",
        description="Board, management, audit, risk-management, and control environment signals.",
        typical_metrics=("controls risk proxy", "audit issue aging", "model risk exception count"),
        rating_alignment=("CAMELS management", "LFI governance and controls", "RFI management"),
    ),
    "interconnectedness": SupervisoryRiskDomain(
        key="interconnectedness",
        label="Interconnectedness and counterparty exposure",
        description="Exposure to nonbank financial intermediation, counterparties, and cross-market channels.",
        typical_metrics=("counterparty exposure", "NDFI exposure proxy", "cross-border funding"),
        rating_alignment=("LFI capital", "LFI liquidity", "financial stability monitoring"),
    ),
}


FED_SUPERVISORY_REFERENCE_BASIS: Tuple[Dict[str, str], ...] = (
    {
        "label": "Federal Reserve supervision and regulation",
        "url": "https://www.federalreserve.gov/supervisionreg.htm",
    },
    {
        "label": "Updated Statement of Supervisory Operating Principles",
        "url": "https://www.federalreserve.gov/supervisionreg/updated-statement-of-supervisory-operating-principles.htm",
    },
    {
        "label": "Federal Reserve Supervision and Regulation Report",
        "url": "https://www.federalreserve.gov/publications/files/202606-supervision-and-regulation-report.pdf",
    },
    {
        "label": "Micro Data Reference Manual",
        "url": "https://www.federalreserve.gov/apps/mdrm/",
    },
    {
        "label": "National Information Center",
        "url": "https://www.ffiec.gov/NPW",
    },
    {
        "label": "SR 11-7 model risk management guidance",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm",
    },
)


DEFAULT_SUPERVISORY_ALIGNMENT_CHECKLIST: Mapping[str, str] = {
    "materiality": "Review metric magnitude and trend before escalation.",
    "proportionality": "Interpret the signal relative to institution size, complexity, business model, and risk profile.",
    "timeliness": "Confirm the data vintage and rerun cadence are appropriate for the monitoring use case.",
    "source_data_lineage": "Document reporting form, MDRM codes, FFIEC or NIC sources, and analyst-owned transformations.",
    "peer_group_fit": "Confirm peer institutions are comparable before using peer aggregate diagnostics.",
    "diagnostic_boundary": "DRR output is not a supervisory conclusion, rating, examination finding, MRA, MRIA, enforcement recommendation, or policy decision.",
}


def build_supervisory_alignment_metadata(
    *,
    institution_type: str,
    institution_label: str,
    risk_domains: Sequence[str],
    peer_group: Optional[str] = None,
    data_vintage: Optional[str] = None,
    reporting_form: Optional[str] = None,
    mdrm_codes: Optional[Sequence[str]] = None,
    ffiec_source: Optional[str] = None,
    nic_identifier: Optional[str] = None,
    review_owner: Optional[str] = None,
    supervisory_scope: str = "monitoring diagnostic",
    checklist_overrides: Optional[Mapping[str, str]] = None,
    reference_basis: Optional[Sequence[Mapping[str, str]]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build report metadata that aligns DRR outputs to Fed supervisory review habits."""

    if institution_type not in SUPERVISORY_INSTITUTION_PROFILES:
        allowed = ", ".join(sorted(SUPERVISORY_INSTITUTION_PROFILES))
        raise ValueError(f"institution_type must be one of: {allowed}")

    missing_domains = [domain for domain in risk_domains if domain not in SUPERVISORY_RISK_DOMAINS]
    if missing_domains:
        allowed = ", ".join(sorted(SUPERVISORY_RISK_DOMAINS))
        raise ValueError(f"risk_domains not found: {missing_domains}. Expected one of: {allowed}")

    checklist = dict(DEFAULT_SUPERVISORY_ALIGNMENT_CHECKLIST)
    if checklist_overrides:
        checklist.update(dict(checklist_overrides))

    data_lineage = _drop_empty(
        {
            "data_vintage": data_vintage,
            "reporting_form": reporting_form,
            "mdrm_codes": list(mdrm_codes or ()),
            "ffiec_source": ffiec_source,
            "nic_identifier": nic_identifier,
        }
    )

    alignment = {
        "institution_label": institution_label,
        "peer_group": peer_group,
        "supervisory_scope": supervisory_scope,
        "review_owner": review_owner,
        "validation_status": "research diagnostic; not validated supervisory methodology",
        "institution_profile": SUPERVISORY_INSTITUTION_PROFILES[institution_type].to_dict(),
        "risk_domains": [SUPERVISORY_RISK_DOMAINS[domain].to_dict() for domain in risk_domains],
        "data_lineage": data_lineage,
        "checklist": checklist,
        "reference_basis": [dict(reference) for reference in (reference_basis or FED_SUPERVISORY_REFERENCE_BASIS)],
    }

    metadata: Dict[str, Any] = {"supervisory_alignment": _drop_empty(alignment)}
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return metadata


def supervisory_profile_options() -> Dict[str, Dict[str, Any]]:
    """Return available institution profile presets as serializable records."""

    return {key: profile.to_dict() for key, profile in SUPERVISORY_INSTITUTION_PROFILES.items()}


def supervisory_risk_domain_options() -> Dict[str, Dict[str, Any]]:
    """Return available supervisory risk-domain tags as serializable records."""

    return {key: domain.to_dict() for key, domain in SUPERVISORY_RISK_DOMAINS.items()}


def _drop_empty(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }
