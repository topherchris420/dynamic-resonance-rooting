import json
import sqlite3

import numpy as np
import pandas as pd

from drr_framework import (
    DynamicResonanceRooting,
    SupervisoryPanelDataset,
    load_policy_dataset_from_sql,
    load_supervisory_panel_from_sql,
    write_analysis_report,
    write_tableau_artifacts,
)


METRICS = (
    "liquidity_ratio",
    "capital_ratio",
    "funding_spread",
    "asset_growth",
    "controls_risk_proxy",
)


def _supervisory_panel_frame(periods=72):
    rng = np.random.default_rng(407)
    dates = pd.date_range("2008-01-01", periods=periods, freq="QS")
    cycle = np.sin(np.linspace(0, 7 * np.pi, periods))
    stress = 0.6 + 0.25 * cycle + rng.normal(scale=0.03, size=periods)
    institutions = (
        ("FBO-A", "LFBO", 0.10),
        ("FBO-B", "LFBO", -0.02),
        ("Domestic-C", "Domestic peer", 0.04),
    )

    rows = []
    for institution_id, peer_group, shift in institutions:
        local = rng.normal(scale=0.025, size=(periods, len(METRICS)))
        funding_spread = 65 + 15 * stress + 4 * np.roll(cycle, 1) + local[:, 0]
        asset_growth = 2.0 + 0.7 * np.roll(cycle, 2) - 0.01 * funding_spread + local[:, 1]
        liquidity_ratio = 112 + 5 * shift - 7 * stress - asset_growth + local[:, 2]
        capital_ratio = 12.0 + shift - 0.35 * np.roll(stress, 2) + local[:, 3]
        controls_risk_proxy = 0.4 + 0.17 * stress + 0.03 * np.roll(funding_spread, 1) / 100
        controls_risk_proxy += local[:, 4]

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


def _panel(frame=None):
    return SupervisoryPanelDataset.from_frame(
        _supervisory_panel_frame() if frame is None else frame,
        institution_column="institution_id",
        date_column="date",
        metric_columns=METRICS,
        peer_group_column="peer_group",
        sampling_rate=4.0,
        transform="diff",
        interpolate=True,
        standardize=True,
    )


def test_supervisory_panel_prepares_institution_and_peer_datasets():
    frame = _supervisory_panel_frame()
    frame.loc[4, "capital_ratio"] = np.nan
    panel = _panel(frame)

    assert panel.institutions == ("FBO-A", "FBO-B", "Domestic-C")
    assert panel.peer_groups == ("LFBO", "Domestic peer")
    assert panel.metadata["institution_count"] == 3
    assert panel.metadata["transform"] == "diff"

    institution_dataset = panel.dataset_for_institution("FBO-A")
    peer_dataset = panel.dataset_for_peer_group("LFBO")

    assert institution_dataset.variable_names == METRICS
    assert institution_dataset.values.shape[1] == len(METRICS)
    assert peer_dataset.values.shape[1] == len(METRICS)
    assert np.isfinite(institution_dataset.values).all()
    assert np.isfinite(peer_dataset.values).all()

    tidy = panel.to_tableau_frame()
    assert {"institution_id", "peer_group", "date", "metric", "value"}.issubset(tidy.columns)
    assert len(tidy) == len(panel.frame) * len(METRICS)


def test_supervisory_sql_loaders_prepare_panel_and_policy_views():
    frame = _supervisory_panel_frame(periods=24)
    with sqlite3.connect(":memory:") as connection:
        frame.to_sql("supervisory_metrics", connection, index=False)
        panel = load_supervisory_panel_from_sql(
            "select * from supervisory_metrics",
            connection,
            institution_column="institution_id",
            date_column="date",
            metric_columns=METRICS,
            peer_group_column="peer_group",
            sampling_rate=4.0,
            transform="diff",
        )

        policy_frame = frame[frame["institution_id"] == "FBO-A"][["date", *METRICS]]
        policy_frame.to_sql("policy_metrics", connection, index=False)
        policy_dataset = load_policy_dataset_from_sql(
            "select * from policy_metrics",
            connection,
            date_column="date",
            value_columns=METRICS,
            sampling_rate=4.0,
            transform="diff",
        )

    assert panel.metadata["source"] == "sql_query"
    assert panel.dataset_for_institution("FBO-B").values.shape[1] == len(METRICS)
    assert policy_dataset.metadata["source"] == "sql_query"
    assert policy_dataset.values.shape[1] == len(METRICS)


def test_supervision_report_and_tableau_exports_are_caveated_and_stable(tmp_path):
    panel = _panel()
    dataset = panel.dataset_for_institution("FBO-A")
    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=dataset.sampling_rate)
    results = drr.analyze_system(
        dataset.to_drr_input(),
        multivariate=True,
        window_size=32,
        state_space=True,
        state_space_horizon=6,
    )

    metadata = {"focus_institution": "FBO-A", "peer_group": "LFBO", "metrics": METRICS}
    report_paths = write_analysis_report(
        results,
        tmp_path,
        stem="supervision",
        audience="supervision",
        metadata=metadata,
    )
    tableau_paths = write_tableau_artifacts(results, tmp_path / "tableau", stem="supervision", metadata=metadata)

    payload = json.loads(report_paths["json"].read_text(encoding="utf-8"))
    markdown = report_paths["markdown"].read_text(encoding="utf-8")

    assert payload["audience"] == "supervision"
    assert "not an examination finding" in payload["summary"]["caveat"]
    assert "DRR Supervisory Analytics Report" in markdown
    assert "not as a supervisory rating" in markdown

    expected_exports = {"summary", "resonance_map", "rooting_edges", "state_space_diagnostics"}
    assert expected_exports == set(tableau_paths)
    for path in tableau_paths.values():
        assert path.exists()

    summary_frame = pd.read_csv(tableau_paths["summary"])
    resonance_frame = pd.read_csv(tableau_paths["resonance_map"])
    edges_frame = pd.read_csv(tableau_paths["rooting_edges"])

    assert {"metric", "value", "meta_focus_institution"}.issubset(summary_frame.columns)
    assert {"dimension", "dominant_frequency", "meta_focus_institution"}.issubset(
        resonance_frame.columns
    )
    assert {"source", "target", "weight", "lag", "p_value", "meta_focus_institution"}.issubset(
        edges_frame.columns
    )
