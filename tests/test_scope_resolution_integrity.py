from copy import deepcopy

import pytest

from drr_framework.disagreement import DRR_ScopeResolver, ObservationalPerspective


def perspective(scale="MACRO", **kwargs):
    return ObservationalPerspective(
        **dict(
            dict(
                source_id=scale,
                scale=scale,
                confidence_score=0.9,
                indicators={"state": "stable"},
                evidence_provenance="fixture",
            ),
            **kwargs,
        )
    )


def test_absent_scopes_are_never_fabricated():
    empty = DRR_ScopeResolver().generate_drr_conclusion()
    assert empty["status"] == "no_perspectives"
    assert empty["macro_state"] == empty["local_state"] == "undetermined"
    result = DRR_ScopeResolver(
        [perspective("MESO"), perspective("MICRO", indicators={"state": "declining"})]
    ).generate_drr_conclusion()
    assert result["status"] == "divergent_scopes_preserved"
    assert result["macro_state"] == "undetermined"
    assert "MACRO:" not in result["conclusion"]
    assert "MESO: stable" in result["conclusion"]
    assert "MICRO: declining" in result["conclusion"]


@pytest.mark.parametrize(
    "value",
    [
        42,
        {},
        [],
        "not stable",
        "not weak",
        "possibly strong",
        "strong-42",
        "stable but declining",
        "unmeasured",
    ],
)
def test_unclassified_or_negated_indicators_do_not_establish_consensus(value):
    result = DRR_ScopeResolver(
        [perspective(), perspective("LOCAL", indicators={"state": value})]
    ).generate_drr_conclusion()
    assert result["status"] == "indeterminate"
    assert not result["has_divergence"]
    assert result["scope_coverage"]["unclassified_indicators"] == 1


def test_single_scale_and_empty_records_are_not_cross_scope_agreement():
    assert DRR_ScopeResolver([perspective()]).generate_drr_conclusion()["status"] == "indeterminate"
    result = DRR_ScopeResolver(
        [perspective(), perspective("LOCAL", indicators={})]
    ).generate_drr_conclusion()
    assert result["status"] == "indeterminate"
    assert result["scope_coverage"]["empty_perspectives"] == 1


def test_mixed_scale_preserves_individual_conflicts_and_primary_state_does_not_hide_them():
    records = [
        perspective(indicators={"state": "stable", "local_measure": "declining"}),
        perspective("LOCAL"),
    ]
    result = DRR_ScopeResolver(records).generate_drr_conclusion()
    assert result["has_divergence"]
    assert len(result["divergence_pairs"]) == 2
    assert len(result["indicator_assessments"]) == 3
    assert "MACRO: stable, declining" in result["conclusion"]
    assert {p["cross_scale"] for p in result["divergence_pairs"]} == {True, False}
    same_scale = DRR_ScopeResolver(records[:1]).generate_drr_conclusion()
    assert same_scale["status"] == "divergent_perspectives_preserved"


def test_confidence_and_duplicate_source_names_cannot_erase_dissent():
    result = DRR_ScopeResolver(
        [
            perspective(source_id="same", confidence_score=1.0, dissent_logged=True),
            perspective(
                "LOCAL", source_id="same", confidence_score=0.0, indicators={"state": "weak"}
            ),
        ]
    ).generate_drr_conclusion()
    assert result["has_divergence"]
    humility = result["intellectual_humility"]
    assert humility["dissent_logged_summary"] == {"same": True}
    assert len(humility["dissent_records"]) == 2
    assert "Accuracy is not verified" in humility["accuracy_within_representation"]


def test_nested_caller_and_output_mutations_cannot_change_registered_evidence():
    data = {"state": "stable", "nested": {"samples": [1, 2]}}
    record = perspective(indicators=data)
    resolver = DRR_ScopeResolver([record])
    expected = deepcopy(resolver.generate_drr_conclusion())
    data["nested"]["samples"].append(3)
    record.indicators["nested"]["samples"].append(4)
    resolver.perspectives[0].indicators["nested"]["samples"].append(5)
    returned = resolver.generate_drr_conclusion()
    returned["registered_perspectives"][0]["indicators"]["nested"]["samples"].append(6)
    returned["indicator_assessments"][1]["value"]["samples"].append(7)
    assert resolver.generate_drr_conclusion() == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_id": " "},
        {"evidence_provenance": ""},
        {"confidence_score": True},
        {"confidence_score": float("nan")},
        {"dissent_logged": "yes"},
        {"indicators": {1: "stable"}},
    ],
)
def test_invalid_evidence_metadata_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        perspective(**kwargs)
