import json

import pandas as pd

from drr_framework import (
    build_model_risk_card,
    build_validation_readiness_packet,
    explain_supervisory_signal,
    append_shadow_review_record,
    create_shadow_review_record,
    run_event_backtest,
    write_analysis_report,
)


def _minimal_results():
    return {
        "resonances": {
            "dim_0": {"dominant_freq": 0.25, "frequencies": [0.25, 0.5]},
            "dim_1": {"dominant_freq": 0.5, "frequencies": [0.5]},
        },
        "resonance_depths": {"dim_0": 0.81, "dim_1": 0.64},
        "is_rooted": True,
        "rooting_analysis": {
            "significant_edges": [
                {"source": "dim_0", "target": "dim_1", "weight": 0.74, "lag": 2, "p_value": 0.01}
            ]
        },
        "state_space_analysis": {
            "diagnostics": {
                "is_stable": True,
                "spectral_radius": 0.76,
                "mean_log_likelihood": -3.2,
                "state_count": 2,
                "shock_count": 2,
            }
        },
    }


def test_model_risk_card_and_packet_make_drr_validation_ready_not_validated():
    model_card = build_model_risk_card(
        model_name="Dynamic Resonance Rooting supervisory diagnostic",
        version="0.3.0-candidate",
        intended_use="Shadow-mode monitoring of LFBO supervisory metrics.",
        prohibited_uses=["ratings", "findings", "MRAs", "MRIAs", "enforcement recommendations"],
        owners=["Supervisory analytics team"],
        data_lineage={"reporting_form": "FR Y-15", "data_vintage": "2026Q2"},
        assumptions=["Quarterly metrics are comparable after transformation."],
        limitations=["No independent supervisory validation has approved this methodology."],
    )
    packet = build_validation_readiness_packet(
        model_card=model_card,
        analysis_results=_minimal_results(),
        benchmark_results={"precision": 0.75, "recall": 0.60},
        shadow_mode_summary={"records_reviewed": 12, "useful_signals": 7},
    )

    assert model_card["validation_status"] == "candidate; not validated supervisory methodology"
    assert "ratings" in model_card["prohibited_uses"]
    assert (
        packet["validation_readiness"]["status"]
        == "validation-ready candidate; not validated methodology"
    )
    assert packet["validation_readiness"]["sr_11_7_sections"] == [
        "intended_use",
        "conceptual_soundness",
        "implementation_verification",
        "outcomes_analysis",
        "ongoing_monitoring",
        "governance_and_change_control",
    ]
    assert (
        packet["validation_readiness"]["checklist"]["independent_validation"]
        == "Required before validated use"
    )
    assert any(
        "srletters/sr1107" in item["url"]
        for item in packet["validation_readiness"]["reference_basis"]
    )


def test_event_backtest_measures_alert_quality_and_lead_time():
    frame = pd.DataFrame(
        {
            "period": pd.date_range("2020-01-01", periods=10, freq="QS"),
            "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.0, 0.85, 0.1, 0.0, 0.0],
            "event": [False, False, False, False, True, False, False, False, True, False],
        }
    )

    result = run_event_backtest(
        frame,
        date_column="period",
        score_column="score",
        event_column="event",
        threshold=0.75,
        lead_window=2,
    )

    assert result["alert_count"] == 3
    assert result["event_count"] == 2
    assert result["true_positive_alerts"] == 2
    assert result["false_positive_alerts"] == 1
    assert result["detected_events"] == 2
    assert result["missed_events"] == 0
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["mean_lead_time_periods"] == 1.5


def test_shadow_review_records_are_append_only_jsonl(tmp_path):
    record = create_shadow_review_record(
        signal_id="FBO-A-2026Q2-liquidity",
        reviewer="Analyst A",
        disposition="useful",
        rationale="Matched an independently reviewed funding pressure narrative.",
        analyst_action="route to source-data owner",
    )

    path = append_shadow_review_record(tmp_path / "shadow_reviews.jsonl", record)
    path = append_shadow_review_record(path, {**record, "signal_id": "FBO-A-2026Q3-capital"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2
    assert rows[0]["mode"] == "shadow"
    assert rows[0]["disposition"] == "useful"
    assert rows[0]["decision_boundary"] == "review feedback only; no supervisory action"


def test_explainability_summarizes_drivers_and_caveats():
    explanation = explain_supervisory_signal(
        _minimal_results(),
        variable_names=("liquidity_ratio", "funding_spread"),
        data_lineage={"reporting_form": "FR Y-15", "data_vintage": "2026Q2"},
    )

    assert explanation["dominant_modes"][0]["variable"] == "liquidity_ratio"
    assert explanation["top_edges"][0]["source_variable"] == "liquidity_ratio"
    assert explanation["state_space_summary"]["is_stable"] is True
    assert explanation["data_lineage"]["reporting_form"] == "FR Y-15"
    assert any("not a finding" in caveat for caveat in explanation["caveats"])
    assert explanation["analyst_questions"]


def test_supervision_report_renders_validation_readiness_packet(tmp_path):
    model_card = build_model_risk_card(
        model_name="DRR supervisory diagnostic",
        version="0.3.0-candidate",
        intended_use="Shadow-mode monitoring.",
        prohibited_uses=["ratings", "findings"],
    )
    packet = build_validation_readiness_packet(
        model_card=model_card, analysis_results=_minimal_results()
    )
    paths = write_analysis_report(
        _minimal_results(),
        tmp_path,
        stem="validation_ready",
        audience="supervision",
        metadata={"validation_readiness": packet["validation_readiness"]},
    )

    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert "## Validation Readiness" in markdown
    assert "candidate; not validated supervisory methodology" in markdown
    assert "SR 11-7 model risk management guidance" in markdown
    assert "Independent Validation" in markdown
    assert "Shadow mode" in markdown
