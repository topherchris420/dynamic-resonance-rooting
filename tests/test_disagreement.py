"""
Unit tests for the Disagreement Principle and Representation Scope Registry.
"""

from __future__ import annotations

import pytest
from drr_framework import (
    DRR_ScopeResolver,
    ObservationalPerspective,
    ObservationalScale,
)


def test_observational_perspective_creation() -> None:
    """Test creation and property coercion/validation of ObservationalPerspective."""
    perspective = ObservationalPerspective(
        source_id="Fed_Macroprudential_Model",
        scale="MACRO",
        confidence_score=0.95,
        indicators={"inflation_pressure": "moderating", "systemic_risk": "stabilizing"},
        evidence_provenance="hash_fed_2026_q1",
        dissent_logged=False,
    )

    assert perspective.source_id == "Fed_Macroprudential_Model"
    assert perspective.scale == ObservationalScale.MACRO
    assert perspective.confidence_score == 0.95
    assert perspective.indicators == {
        "inflation_pressure": "moderating",
        "systemic_risk": "stabilizing",
    }
    assert perspective.evidence_provenance == "hash_fed_2026_q1"
    assert perspective.dissent_logged is False

    d = perspective.to_dict()
    assert d["scale"] == "MACRO"
    assert d["confidence_score"] == 0.95


def test_observational_perspective_invalid_values() -> None:
    """Test validation errors for invalid confidence scores or scales."""
    with pytest.raises(ValueError, match="confidence_score must be in range"):
        ObservationalPerspective(
            source_id="Test_Model",
            scale=ObservationalScale.LOCAL,
            confidence_score=1.5,
            indicators={"power": "deteriorating"},
            evidence_provenance="provenance_ref",
        )

    with pytest.raises(ValueError, match="Invalid scale"):
        ObservationalPerspective(
            source_id="Test_Model",
            scale="UNIVERSE",
            confidence_score=0.8,
            indicators={"power": "deteriorating"},
            evidence_provenance="provenance_ref",
        )


def test_disagreement_principle_non_erasure_and_conclusion() -> None:
    """
    Test that high MACRO confidence score does NOT silently overwrite, filter, or smooth out
    conflicting metrics from LOCAL/MICRO scale perspectives (Samira & Christopher archetypes).
    """
    resolver = DRR_ScopeResolver()

    # Samira's profile archetype: High-confidence macroprudential model
    samira_macro = ObservationalPerspective(
        source_id="Fed_Macroprudential_Model",
        scale=ObservationalScale.MACRO,
        confidence_score=0.98,
        indicators={
            "inflation_pressure": "moderating",
            "banking_liquidity": "stabilizing",
        },
        evidence_provenance="vintage_hash_samira_macro_001",
        dissent_logged=False,
    )

    # Christopher's profile archetype: Distribution-sensitive local tracker
    christopher_local = ObservationalPerspective(
        source_id="Local_Material_Survey",
        scale=ObservationalScale.LOCAL,
        confidence_score=0.72,
        indicators={
            "purchasing_power": "deteriorating",
            "household_debt_stress": "fragile",
        },
        evidence_provenance="vintage_hash_christopher_local_999",
        dissent_logged=True,
    )

    resolver.register_perspective(samira_macro)
    resolver.register_perspective(christopher_local)

    # Invariant check: Both perspectives survive without mutual erasure
    assert len(resolver.perspectives) == 2
    assert resolver.perspectives[0].source_id == "Fed_Macroprudential_Model"
    assert resolver.perspectives[1].source_id == "Local_Material_Survey"

    result = resolver.generate_drr_conclusion()

    # Verify structured payload
    assert result["status"] == "DIVERGENCE_DETECTED"
    assert result["forced_consensus_rejected"] is True

    # Natural language template verification
    expected_template = (
        "Aggregate financial-system indicators support moderating, stabilizing, "
        "while material indicators for the evaluated population support deteriorating, fragile. "
        "These findings operate at different observational scopes and should be "
        "interpreted together rather than collapsed into a single state."
    )
    assert result["conclusion"] == expected_template

    # Metadata verification for intellectual humility
    metadata = result["metadata"]
    assert metadata["disagreement_principle_applied"] is True
    assert "accuracy_within_representation" in metadata
    assert "completeness_of_representation" in metadata
    assert metadata["completeness_of_representation"]["is_complete"] is False


def test_consensus_scenario() -> None:
    """Test that aligned perspectives result in CONSENSUS status without rejection."""
    resolver = DRR_ScopeResolver()

    macro_p = ObservationalPerspective(
        source_id="Fed_Macroprudential_Model",
        scale=ObservationalScale.MACRO,
        confidence_score=0.90,
        indicators={"economic_trend": "stabilizing"},
        evidence_provenance="prov_macro",
    )

    local_p = ObservationalPerspective(
        source_id="Local_Material_Survey",
        scale=ObservationalScale.LOCAL,
        confidence_score=0.85,
        indicators={"local_trend": "recovering"},
        evidence_provenance="prov_local",
    )

    resolver.register_perspective(macro_p)
    resolver.register_perspective(local_p)

    result = resolver.generate_drr_conclusion()

    assert result["status"] == "CONSENSUS"
    assert result["forced_consensus_rejected"] is False
    assert "consistent with" in result["conclusion"]
    assert result["metadata"]["disagreement_principle_applied"] is False
