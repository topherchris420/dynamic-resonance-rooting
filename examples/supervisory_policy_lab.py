"""Supervisory analytics DRR workflow.

Run from the repository root:

    python examples/supervisory_policy_lab.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drr_framework import (
    DynamicResonanceRooting,
    SupervisoryPanelDataset,
    append_shadow_review_record,
    build_model_risk_card,
    build_supervisory_alignment_metadata,
    build_validation_readiness_packet,
    create_shadow_review_record,
    explain_supervisory_signal,
    run_event_backtest,
    write_analysis_report,
    write_tableau_artifacts,
)


METRICS = [
    "liquidity_ratio",
    "capital_ratio",
    "funding_spread",
    "asset_growth",
    "controls_risk_proxy",
]


def synthetic_supervisory_panel(periods: int = 96, random_state: int = 730) -> pd.DataFrame:
    """Create deterministic institution panel data for supervisory demos."""

    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2002-01-01", periods=periods, freq="QS")
    cycle = np.sin(np.linspace(0, 10 * np.pi, periods))
    credit_cycle = np.sin(np.linspace(0.4, 8.5 * np.pi, periods))
    stress = np.clip(0.55 + 0.28 * cycle + rng.normal(scale=0.04, size=periods), 0.05, None)

    institutions = [
        ("FBO-Atlas", "LFBO", 0.10, 0.00),
        ("FBO-Cascade", "LFBO", -0.03, 0.22),
        ("FBO-Meridian", "LFBO", 0.04, -0.16),
        ("Domestic-Ridge", "Domestic peer", -0.08, 0.08),
    ]

    rows = []
    for institution_id, peer_group, balance_shift, phase_shift in institutions:
        shifted_cycle = np.sin(np.linspace(phase_shift, 10 * np.pi + phase_shift, periods))
        local_noise = rng.normal(scale=0.035, size=(periods, len(METRICS)))
        funding_spread = 58 + 18 * stress + 7 * np.roll(shifted_cycle, 1) + local_noise[:, 0]
        asset_growth = 2.4 + 0.95 * np.roll(credit_cycle, 2) - 0.015 * funding_spread + local_noise[:, 1]
        liquidity_ratio = 118 + 4.5 * balance_shift - 8.5 * stress - 1.2 * asset_growth + local_noise[:, 2]
        capital_ratio = 12.5 + 1.2 * balance_shift - 0.55 * np.roll(stress, 2) - 0.06 * asset_growth + local_noise[:, 3]
        controls_risk_proxy = 0.42 + 0.18 * stress + 0.05 * np.roll(funding_spread, 1) / 100
        controls_risk_proxy += local_noise[:, 4]

        for index, date in enumerate(dates):
            rows.append(
                {
                    "institution_id": institution_id,
                    "peer_group": peer_group,
                    "date": date,
                    "liquidity_ratio": liquidity_ratio[index],
                    "capital_ratio": capital_ratio[index],
                    "funding_spread": funding_spread[index],
                    "asset_growth": asset_growth[index],
                    "controls_risk_proxy": controls_risk_proxy[index],
                }
            )

    return pd.DataFrame(rows)


def analyze_dataset(dataset, *, window_size: int = 48):
    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=dataset.sampling_rate)
    return drr.analyze_system(
        dataset.to_drr_input(),
        multivariate=True,
        window_size=window_size,
        state_space=True,
        state_space_horizon=8,
    )


def build_validation_artifacts(results, dataset, *, scope_label: str, lineage: dict, output_dir: Path) -> dict:
    """Create synthetic validation-readiness artifacts for the demo workflow."""

    scores = np.mean(np.abs(dataset.values), axis=1)
    threshold = float(np.quantile(scores, 0.75))
    events = np.zeros(len(scores), dtype=bool)
    alert_indices = np.flatnonzero(scores >= threshold)
    for index in alert_indices[:3]:
        if index + 2 < len(events):
            events[index + 2] = True

    dates = dataset.dates or tuple(str(index) for index in range(len(scores)))
    backtest_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "drr_score": scores,
            "review_event": events,
        }
    )
    backtest = run_event_backtest(
        backtest_frame,
        date_column="date",
        score_column="drr_score",
        event_column="review_event",
        threshold=threshold,
        lead_window=2,
    )

    validation_dir = output_dir / "validation"
    shadow_record = create_shadow_review_record(
        signal_id=f"{scope_label}-synthetic-shadow-signal",
        reviewer="supervisory analytics demo owner",
        disposition="needs_review",
        rationale="Synthetic shadow-mode record showing how analyst feedback is captured without supervisory effect.",
        analyst_action="review source-data lineage and peer-group fit",
        metadata={"scope": scope_label},
    )
    append_shadow_review_record(validation_dir / "shadow_reviews.jsonl", shadow_record)

    explanation = explain_supervisory_signal(
        results,
        variable_names=dataset.variable_names,
        data_lineage=lineage,
    )
    model_card = build_model_risk_card(
        model_name="DRR supervisory diagnostic",
        version="0.3.0-candidate",
        intended_use="Shadow-mode monitoring of synthetic LFBO supervisory metrics.",
        prohibited_uses=[
            "ratings",
            "findings",
            "MRAs",
            "MRIAs",
            "enforcement recommendations",
            "policy decisions",
        ],
        owners=["supervisory analytics demo owner"],
        data_lineage=lineage,
        assumptions=["Synthetic quarterly metrics are comparable after differencing and standardization."],
        limitations=["Synthetic example only; no independent supervisory validation has approved this methodology."],
    )
    packet = build_validation_readiness_packet(
        model_card=model_card,
        analysis_results=results,
        benchmark_results=backtest,
        shadow_mode_summary={
            "records_reviewed": 1,
            "useful_signals": 0,
            "needs_review_signals": 1,
            "mode": "Shadow mode",
        },
        explanations=explanation,
    )
    return {
        "validation_readiness": packet["validation_readiness"],
        "backtest": backtest,
        "explanation": explanation,
    }


def main() -> None:
    panel = SupervisoryPanelDataset.from_frame(
        synthetic_supervisory_panel(),
        institution_column="institution_id",
        date_column="date",
        metric_columns=METRICS,
        peer_group_column="peer_group",
        sampling_rate=4.0,
        transform="diff",
        interpolate=True,
        standardize=True,
        source="synthetic quarterly supervisory metrics",
    )

    focus_institution = "FBO-Atlas"
    focus_peer_group = "LFBO"
    institution_dataset = panel.dataset_for_institution(focus_institution)
    peer_dataset = panel.dataset_for_peer_group(focus_peer_group)

    output_dir = Path("results") / "supervisory_policy_lab"
    tableau_dir = output_dir / "tableau"
    tableau_dir.mkdir(parents=True, exist_ok=True)

    panel.to_tableau_frame().to_csv(
        tableau_dir / "synthetic_supervisory_panel_tableau.csv",
        index=False,
    )

    institution_results = analyze_dataset(institution_dataset)
    institution_lineage = {
        "reporting_form": "synthetic FR Y-15-style panel",
        "data_vintage": "synthetic 2002Q1-2025Q4",
        "ffiec_source": "synthetic FFIEC-style supervisory panel",
        "nic_identifier": "synthetic-fbo-atlas",
    }
    institution_validation = build_validation_artifacts(
        institution_results,
        institution_dataset,
        scope_label=focus_institution,
        lineage=institution_lineage,
        output_dir=output_dir,
    )
    institution_metadata = {
        **panel.metadata,
        **build_supervisory_alignment_metadata(
            institution_type="large_bank_or_large_fbo",
            institution_label=focus_institution,
            peer_group=focus_peer_group,
            risk_domains=[
                "capital",
                "liquidity",
                "asset_quality",
                "operational_resilience",
                "governance_controls",
            ],
            data_vintage=institution_lineage["data_vintage"],
            reporting_form=institution_lineage["reporting_form"],
            mdrm_codes=["synthetic_liquidity", "synthetic_capital", "synthetic_controls"],
            ffiec_source=institution_lineage["ffiec_source"],
            nic_identifier=institution_lineage["nic_identifier"],
            review_owner="supervisory analytics demo owner",
        ),
        **institution_validation,
        "analysis_scope": "institution",
        "metrics": panel.metric_columns,
        "note": "Synthetic example for supervisory workflow demonstration only.",
    }
    institution_paths = write_analysis_report(
        institution_results,
        output_dir,
        stem="fbo_atlas_supervisory_diagnostic",
        audience="supervision",
        title="DRR Supervisory Diagnostic: FBO-Atlas",
        metadata=institution_metadata,
    )
    write_tableau_artifacts(
        institution_results,
        tableau_dir,
        stem="fbo_atlas",
        metadata=institution_metadata,
    )

    peer_results = analyze_dataset(peer_dataset)
    peer_lineage = {
        "reporting_form": "synthetic FR Y-15-style panel",
        "data_vintage": "synthetic 2002Q1-2025Q4",
        "ffiec_source": "synthetic FFIEC-style peer aggregation",
    }
    peer_validation = build_validation_artifacts(
        peer_results,
        peer_dataset,
        scope_label=focus_peer_group,
        lineage=peer_lineage,
        output_dir=output_dir,
    )
    peer_metadata = {
        **panel.metadata,
        **build_supervisory_alignment_metadata(
            institution_type="large_bank_or_large_fbo",
            institution_label=f"{focus_peer_group} peer-group aggregate",
            peer_group=focus_peer_group,
            risk_domains=["capital", "liquidity", "asset_quality", "governance_controls"],
            data_vintage=peer_lineage["data_vintage"],
            reporting_form=peer_lineage["reporting_form"],
            mdrm_codes=["synthetic_peer_capital", "synthetic_peer_liquidity"],
            ffiec_source=peer_lineage["ffiec_source"],
            review_owner="supervisory analytics demo owner",
        ),
        **peer_validation,
        "analysis_scope": "peer_group",
        "metrics": panel.metric_columns,
        "note": "Synthetic peer-group aggregate for workflow demonstration only.",
    }
    peer_paths = write_analysis_report(
        peer_results,
        output_dir,
        stem="lfbo_peer_group_diagnostic",
        audience="supervision",
        title="DRR Supervisory Diagnostic: LFBO Peer Group",
        metadata=peer_metadata,
    )
    write_tableau_artifacts(peer_results, tableau_dir, stem="lfbo_peer_group", metadata=peer_metadata)

    print(f"Wrote institution report: {institution_paths['markdown']}")
    print(f"Wrote peer-group report: {peer_paths['markdown']}")
    print(f"Wrote Tableau CSV directory: {tableau_dir}")
    print(f"Wrote validation shadow log: {output_dir / 'validation' / 'shadow_reviews.jsonl'}")


if __name__ == "__main__":
    main()
