import pytest

from drr_framework import (
    build_supervisory_alignment_metadata,
    write_analysis_report,
)


def _minimal_results():
    return {
        "resonances": {
            "dim_0": {"dominant_freq": 0.25, "frequencies": [0.25]},
        },
        "resonance_depths": {"dim_0": 0.72},
        "is_rooted": False,
        "rooting_analysis": {"significant_edges": []},
        "state_space_analysis": {
            "diagnostics": {
                "is_stable": True,
                "spectral_radius": 0.82,
                "mean_log_likelihood": -4.2,
                "state_count": 1,
                "shock_count": 1,
            }
        },
    }


def test_alignment_metadata_captures_supervisory_context_and_lineage():
    metadata = build_supervisory_alignment_metadata(
        institution_type="large_bank_or_large_fbo",
        institution_label="FBO-Atlas",
        peer_group="LFBO",
        risk_domains=["capital", "liquidity", "operational_resilience"],
        data_vintage="2026Q2",
        reporting_form="FR Y-15",
        mdrm_codes=["BHCK8274", "BHCK0081"],
        ffiec_source="FFIEC 031/041",
        nic_identifier="123456",
        review_owner="Supervisory analytics team",
    )

    alignment = metadata["supervisory_alignment"]

    assert alignment["institution_profile"]["label"] == "Large bank or large FBO"
    assert alignment["institution_profile"]["rating_frameworks"] == ["LFI"]
    assert alignment["data_lineage"]["reporting_form"] == "FR Y-15"
    assert alignment["data_lineage"]["mdrm_codes"] == ["BHCK8274", "BHCK0081"]
    assert alignment["data_lineage"]["data_vintage"] == "2026Q2"
    assert [domain["key"] for domain in alignment["risk_domains"]] == [
        "capital",
        "liquidity",
        "operational_resilience",
    ]
    assert (
        alignment["checklist"]["materiality"]
        == "Review metric magnitude and trend before escalation."
    )
    assert alignment["checklist"]["diagnostic_boundary"].startswith("DRR output is not")
    assert any(
        reference["url"] == "https://www.federalreserve.gov/supervisionreg.htm"
        for reference in alignment["reference_basis"]
    )


def test_alignment_metadata_rejects_unknown_segments_and_domains():
    with pytest.raises(ValueError, match="institution_type"):
        build_supervisory_alignment_metadata(
            institution_type="unknown_bank",
            institution_label="Example",
            risk_domains=["capital"],
        )

    with pytest.raises(ValueError, match="risk_domains"):
        build_supervisory_alignment_metadata(
            institution_type="community_bank",
            institution_label="Example",
            risk_domains=["moon_risk"],
        )


def test_supervision_report_renders_alignment_checklist_and_references(tmp_path):
    metadata = build_supervisory_alignment_metadata(
        institution_type="large_bank_or_large_fbo",
        institution_label="FBO-Atlas",
        peer_group="LFBO",
        risk_domains=["capital", "liquidity", "governance_controls"],
        data_vintage="2026Q2",
        reporting_form="FR Y-15",
        review_owner="Supervisory analytics team",
    )

    paths = write_analysis_report(
        _minimal_results(),
        tmp_path,
        stem="supervision_alignment",
        audience="supervision",
        metadata=metadata,
    )

    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert "## Supervisory Alignment" in markdown
    assert "Large bank or large FBO" in markdown
    assert "Capital adequacy" in markdown
    assert "Governance and controls" in markdown
    assert "Materiality" in markdown
    assert "Proportionality" in markdown
    assert "Federal Reserve supervision and regulation" in markdown
    assert "https://www.federalreserve.gov/supervisionreg.htm" in markdown
    assert "DRR output is not a supervisory conclusion" in markdown
