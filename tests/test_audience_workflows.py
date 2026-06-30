import json

import numpy as np
import pandas as pd

from drr_framework import (
    DynamicResonanceRooting,
    PolicyResonanceDataset,
    generate_coupled_oscillator,
    write_analysis_report,
)


def _policy_frame(periods=72):
    rng = np.random.default_rng(99)
    dates = pd.date_range("2000-01-01", periods=periods, freq="QS")
    cycle = np.sin(np.linspace(0, 6 * np.pi, periods))
    output_gap = 0.8 * cycle + rng.normal(scale=0.08, size=periods)
    inflation = 2.0 + 0.25 * np.roll(output_gap, 2) + rng.normal(scale=0.05, size=periods)
    policy_rate = 2.3 + 0.4 * np.roll(inflation - 2.0, 1) + 0.2 * output_gap
    return pd.DataFrame(
        {
            "date": dates,
            "inflation": inflation,
            "output_gap": output_gap,
            "policy_rate": policy_rate,
        }
    )


def test_policy_resonance_dataset_prepares_transformed_observables():
    frame = _policy_frame()
    frame.loc[5, "inflation"] = np.nan

    dataset = PolicyResonanceDataset.from_frame(
        frame,
        date_column="date",
        value_columns=["inflation", "output_gap", "policy_rate"],
        sampling_rate=4.0,
        transform="diff",
        interpolate=True,
        standardize=True,
    )

    assert dataset.values.shape[1] == 3
    assert dataset.variable_names == ("inflation", "output_gap", "policy_rate")
    assert dataset.sampling_rate == 4.0
    assert np.isfinite(dataset.values).all()
    assert dataset.metadata["transform"] == "diff"


def test_policy_report_writes_json_and_markdown(tmp_path):
    dataset = PolicyResonanceDataset.from_frame(
        _policy_frame(),
        date_column="date",
        value_columns=["inflation", "output_gap", "policy_rate"],
        sampling_rate=4.0,
        transform="diff",
    )
    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=dataset.sampling_rate)

    results = drr.analyze_system(
        dataset.to_drr_input(),
        multivariate=True,
        window_size=32,
        state_space=True,
        state_space_horizon=6,
    )
    paths = write_analysis_report(
        results,
        tmp_path,
        stem="policy",
        audience="policy",
        metadata={"variables": dataset.variable_names},
    )

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert payload["audience"] == "policy"
    assert payload["summary"]["caveat"].startswith("This report is a research diagnostic")
    assert "state_space_analysis" in payload["results"]
    assert "DRR Policy Lab Report" in markdown
    assert "not a forecast" in markdown


def test_physics_report_serializes_coupled_oscillator(tmp_path):
    sampling_rate = 200.0
    _, data = generate_coupled_oscillator(sampling_rate=sampling_rate, random_state=321)
    drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)
    results = drr.analyze_system(data, multivariate=True, window_size=128)

    paths = write_analysis_report(results, tmp_path, stem="physics", audience="physics")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert payload["audience"] == "physics"
    assert payload["summary"]["mean_resonance_depth"] >= 0.0
    assert "impulse_responses" in payload["results"]["state_space_analysis"]
    assert paths["markdown"].exists()
