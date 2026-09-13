"""Deterministic synthetic monitoring lab, with filing and amendment ground truth."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .entity_graph import PublicEntity, EntityRelationship, EntityGraph
from .peer_analysis import PeerGroupDefinition
from .policy_context import PolicyContext, PolicyEvent
from .semantics import MetricDefinition, SemanticRegistry
from .vintage import RegulatoryObservation, VintageStore, ObservationProvenance
from .perspectives import (
    DocumentedDisagreement,
    PerspectiveInventory,
    PerspectiveKind,
    ScopedPerspective,
)

PREVIOUS_REVIEW = "2026-08-06"
CURRENT_REVIEW = "2026-08-10"


def synthetic_monitoring_lab():
    metrics = (
        ("SYN_ASSETS", "Total assets", "USD millions", "balance_sheet"),
        ("SYN_FUNDING", "Wholesale funding", "USD millions", "funding"),
        ("SYN_CAPITAL", "Capital ratio", "percent", "capital"),
    )
    definitions = tuple(
        MetricDefinition(
            code,
            label,
            "FR Y-9C",
            "synthetic",
            "synthetic",
            f"Deterministic synthetic {label}; not an MDRM mapping",
            unit,
            "2010-01-01",
            "synthetic-v1",
            "synthetic:lfbo-lab",
            "2010-01-01",
            status="synthetic",
            domain=domain,
        )
        for code, label, unit, domain in metrics
    )
    institutions = ("DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E")
    rng = np.random.default_rng(730)
    rows = []
    periods = pd.period_range("2018Q3", periods=32, freq="Q")
    for j, institution in enumerate(institutions):
        for i, p in enumerate(periods):
            period = p.end_time.date().isoformat()
            filing = (p.end_time.normalize() + pd.Timedelta(days=35)).date().isoformat()
            if institution == "DEMO-A" and i == 31:
                filing = "2026-08-09"
            values = (
                1000 + 30 * j + 2 * i + 5 * np.sin(i),
                100 + 2 * j + 0.5 * i + 3 * np.sin(i / 2),
                12 + 0.1 * j + 0.2 * np.sin(i / 3),
            )
            for (code, label, unit, domain), value in zip(metrics, values):
                value = round(value + rng.normal(0, 0.025), 6)
                if institution == "DEMO-A" and i == 31 and code == "SYN_FUNDING":
                    value += 55
                if institution == "DEMO-B" and i == 31 and code == "SYN_CAPITAL":
                    value = 24.0
                row = RegulatoryObservation(
                    institution,
                    f"Synthetic Institution {institution[-1]}",
                    "FR Y-9C",
                    code,
                    period,
                    value,
                    unit,
                    filing,
                    filing,
                    f"{institution}-{p}-original",
                    filing,
                    "synthetic-v1",
                    "synthetic:lfbo-lab",
                    ObservationProvenance.SYNTHETIC_TEST,
                    perimeter_version="synthetic-us-perimeter-v1",
                )
                rows.append(row)
                if institution == "DEMO-B" and i == 31 and code == "SYN_CAPITAL":
                    rows.append(
                        replace(
                            row,
                            value=12.2,
                            amendment_date="2026-08-08",
                            ingestion_date="2026-08-08",
                            available_as_of="2026-08-08",
                            source_vintage=f"{institution}-{p}-amended",
                            supersedes=row.observation_id,
                        )
                    )
    cohort = PeerGroupDefinition(
        "Synthetic LFBO comparison cohort",
        institutions,
        "Fixed five-firm synthetic panel; each target is excluded from its peer distribution",
        "2018-01-01",
        "2018-01-01",
    )
    entities = [
        PublicEntity(
            "SYN-PARENT",
            "Synthetic foreign parent",
            "foreign_parent",
            "GB",
            "synthetic:lfbo-lab",
            "2018-01-01",
            provenance="synthetic_test",
        )
    ]
    entities += [
        PublicEntity(
            i,
            f"Synthetic Institution {i[-1]}",
            "ihc",
            "US",
            "synthetic:lfbo-lab",
            "2018-01-01",
            provenance="synthetic_test",
        )
        for i in institutions
    ]
    relationships = tuple(
        EntityRelationship(
            "SYN-PARENT",
            i,
            "synthetic ownership",
            "2018-01-01",
            "2018-01-01",
            "synthetic:lfbo-lab",
            provenance="synthetic_test",
        )
        for i in institutions
    )
    policy = PolicyContext(
        (
            PolicyEvent(
                "SR 26-2",
                "Federal Reserve / FDIC / OCC",
                "Revised Guidance on Model Risk Management",
                "2026-04-17",
                "2026-04-17",
                "issued",
                (),
                (),
                (),
                "model risk management",
                "https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm",
                "2026-04-17",
                analyst_interpretation="Research context: selected principles inform the workbench's model-risk records. Institution-specific applicability is not inferred.",
            ),
        )
    )
    return (
        VintageStore(rows),
        SemanticRegistry(definitions),
        cohort,
        policy,
        EntityGraph(entities, relationships),
    )


def synthetic_perspectives(store):
    """Four scoped fixture perspectives used by the offline UI demonstration."""
    records = {
        o.metric: o
        for o in store.as_of(CURRENT_REVIEW)
        if o.institution_id == "DEMO-A" and o.reporting_period == "2026-06-30"
    }
    common = dict(
        system="Synthetic banking economy",
        author="Synthetic fixture contributor",
        geography="Synthetic United States",
        period_start="2018-09-30",
        period_end="2026-06-30",
        horizon="quarterly",
        available_as_of=CURRENT_REVIEW,
        limitations=("Deterministic fixture; not a real economic assessment",),
        support_assessor="Synthetic review panel",
    )
    perspectives = (
        ScopedPerspective(
            **common,
            kind=PerspectiveKind.INSTITUTIONAL_ASSESSMENT,
            claim="Aggregate capitalization is stable within the evaluated banking panel",
            population="All synthetic institutions",
            scale="aggregate",
            dimension="bank capitalization",
            evidence_ids=(records["SYN_CAPITAL"].observation_id,),
            method="Institutional ratio review",
            outside_scope=("Household purchasing power and affordability",),
            support_assessment="supported",
            interpretation="Supported within the aggregate capital scope.",
            confidence=0.9,
            institution_ids=("DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E"),
        ),
        ScopedPerspective(
            **common,
            kind=PerspectiveKind.QUANTITATIVE_MODEL,
            claim="Funding movement is unusual relative to the institution's trailing history",
            population="DEMO-A",
            scale="institution",
            dimension="wholesale funding",
            evidence_ids=(records["SYN_FUNDING"].observation_id,),
            method="Transparent baseline and DRR candidate diagnostics",
            outside_scope=("Causal economic attribution",),
            support_assessment="supported",
            interpretation="The diagnostic identifies an unusual movement for review.",
            confidence=0.75,
            institution_ids=("DEMO-A",),
        ),
        ScopedPerspective(
            **common,
            kind=PerspectiveKind.EXPERT_JUDGMENT,
            claim="The funding movement may reflect a reporting or business-context explanation",
            population="DEMO-A reviewers",
            scale="institution",
            dimension="contextual explanation",
            evidence_ids=(records["SYN_FUNDING"].observation_id,),
            method="Attributed analyst hypothesis",
            outside_scope=("Population-wide affordability outcomes",),
            support_assessment="mixed",
            interpretation="Requires investigation; this is not a machine conclusion.",
            confidence=0.5,
            institution_ids=("DEMO-A",),
        ),
        ScopedPerspective(
            **{
                **common,
                "limitations": (
                    "Deterministic fixture; not a real economic assessment",
                    "No household survey, interview, or distributional affordability series is supplied",
                ),
            },
            kind=PerspectiveKind.LIVED_EXPERIENCE,
            claim="Material affordability may diverge from aggregate banking indicators; no household series is supplied in this fixture",
            population="Households outside the synthetic bank panel",
            scale="local and distribution-sensitive",
            dimension="material affordability",
            evidence_ids=(records["SYN_ASSETS"].observation_id,),
            method="Lived-material perspective placeholder linked to contextual fixture evidence",
            outside_scope=("Bank-level solvency and liquidity measurement",),
            support_assessment="not_evaluated",
            interpretation="Retained as a distinct observational slot. A real local or material assessment must provide its own sourced evidence before the claim can be supported.",
            institution_ids=("DEMO-A",),
        ),
    )
    disagreement = DocumentedDisagreement(
        tuple(p.perspective_id for p in perspectives),
        "Aggregate institutional indicators and local material context use different populations, scales, and dimensions; interpret them together.",
        "Synthetic review panel",
        CURRENT_REVIEW,
    )
    return PerspectiveInventory(perspectives, (disagreement,))
