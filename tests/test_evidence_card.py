"""Unit tests for DRR Evidence Card schema."""

import pytest
from drr_framework.evidence_card import create_drr_evidence_card, DRREvidenceCard


def test_create_drr_evidence_card():
    card = create_drr_evidence_card(
        signal_id="sig_test_001",
        variables=["dim_0", "dim_1"],
        methodology="welch_transfer_entropy",
        parameter_configuration={"window_size": 128, "tau": 1},
        p_value=0.02,
        effect_size_dict={"resonance_depth": 0.65},
        robustness_score=0.90,
        benchmark_comparison={"vs_rolling_vol": 0.15},
        confidence_interval=(0.60, 0.70),
        data_provenance={"source": "Call_Report_FFIEC002"},
        detection_statement="Dominant peak at 12.5 Hz detected with p=0.02",
        interpretation_statement="Potential dynamic funding mismatch across institutions",
    )

    assert card.signal_id == "sig_test_001"
    assert card.reproducibility_hash is not None
    assert len(card.reproducibility_hash) == 64  # SHA-256 length
    assert card.detection_vs_interpretation["detection"] == "Dominant peak at 12.5 Hz detected with p=0.02"
    assert "caveat" in card.detection_vs_interpretation
