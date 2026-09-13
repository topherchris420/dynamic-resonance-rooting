from dataclasses import replace

import pytest

from test_lfbo_evidence_workflow import entry
from test_regulatory_foundation import observation
from drr_framework.supervisory import (
    EvidenceEntry,
    EvidenceLedger,
    ScopedPerspective,
    DocumentedDisagreement,
    PerspectiveInventory,
)


def test_scope_divergence_preserves_both_attributed_assessments(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    oid = ledger.append(entry())
    # Synthetic fixture demonstrates scope behavior, not real economic findings.
    institutional = ScopedPerspective(
        system="Synthetic economy",
        kind="institutional_assessment",
        claim="Synthetic aggregate stabilization",
        author="Synthetic analyst",
        population="All banks",
        geography="Synthetic country",
        scale="aggregate",
        dimension="bank capitalization",
        period_start="2025-01-01",
        period_end="2025-03-31",
        horizon="one quarter",
        available_as_of="2025-09-01",
        evidence_ids=(oid,),
        method="Synthetic assessment",
        limitations=("Fixture only",),
        outside_scope=("Household affordability",),
        support_assessment="supported",
        support_assessor="Synthetic reviewer",
        confidence=0.99,
    )
    local = replace(
        institutional,
        kind="lived_experience",
        claim="Synthetic material deterioration",
        population="Evaluated households",
        scale="local",
        dimension="housing burden",
        outside_scope=("Bank solvency",),
        confidence=0.6,
    )
    disagreement = DocumentedDisagreement(
        (institutional.perspective_id, local.perspective_id),
        "Different populations and dimensions; interpret together",
        "Synthetic reviewer",
        "2025-09-02",
    )
    inventory = PerspectiveInventory((institutional, local), (disagreement,))
    snapshot = inventory.snapshot(ledger, as_of="2025-09-02")
    assert len(snapshot["perspectives"]) == 2
    assert {p["support_assessment"] for p in snapshot["perspectives"]} == {"supported"}
    assert snapshot["disagreements"][0]["comparison_status"] == "different_observational_scopes"
    assert inventory.snapshot(ledger, as_of="2025-08-01")["perspectives"] == []
    assert (
        PerspectiveInventory((local, institutional), (disagreement,)).snapshot(
            ledger, as_of="2025-09-02"
        )
        == snapshot
    )
    with pytest.raises(ValueError, match="future"):
        PerspectiveInventory((replace(local, available_as_of="2025-07-01"),)).snapshot(
            ledger, as_of="2025-09-02"
        )
    with pytest.raises(ValueError, match="missing"):
        PerspectiveInventory((institutional,), (disagreement,))


def test_raw_observation_ids_are_valid_scoped_evidence():
    raw = observation()
    perspective = ScopedPerspective(
        system="Synthetic economy",
        kind="lived_experience",
        claim="Local condition is recorded",
        author="Analyst",
        population="Local households",
        geography="Synthetic city",
        scale="local",
        dimension="affordability",
        period_start="2025-01-01",
        period_end="2025-03-31",
        horizon="quarterly",
        available_as_of="2025-09-01",
        evidence_ids=(raw.observation_id,),
        method="Recorded observation",
        limitations=("Synthetic fixture",),
        outside_scope=("Aggregate bank capital",),
        support_assessment="not_evaluated",
        support_assessor="Analyst",
    )
    snapshot = PerspectiveInventory((perspective,)).snapshot(
        as_of="2025-09-01", observations=(raw,)
    )
    assert snapshot["perspectives"][0]["evidence_ids"] == [raw.observation_id]


def test_perspective_rejects_later_created_evidence_with_old_source_facts(tmp_path):
    payload = entry().payload
    payload["evidence_available_as_of"] = "2025-10-01"
    later_entry = EvidenceEntry.create(**payload)
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    ledger.append(later_entry)
    perspective = ScopedPerspective(
        system="Synthetic economy",
        kind="quantitative_model",
        claim="A later-created packet must not enter an earlier perspective",
        author="Analyst",
        population="Synthetic banks",
        geography="Synthetic country",
        scale="aggregate",
        dimension="assets",
        period_start="2025-01-01",
        period_end="2025-03-31",
        horizon="quarterly",
        available_as_of="2025-09-01",
        evidence_ids=(later_entry.evidence_id,),
        method="Point-in-time evidence check",
        limitations=("Synthetic fixture",),
        outside_scope=("Household conditions",),
        support_assessment="not_evaluated",
        support_assessor="Analyst",
    )
    with pytest.raises(ValueError, match="future"):
        PerspectiveInventory((perspective,)).snapshot(ledger, as_of="2025-10-01")
