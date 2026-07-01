import json

from examples.quickstart_resonance_export import (
    analyze_sample,
    export_summary,
    load_sample,
    save_trace_plot,
)


def test_quickstart_load_analyze_and_export(tmp_path):
    sampling_rate, data = load_sample()

    summary = analyze_sample(data, sampling_rate)
    paths = export_summary(summary, tmp_path)
    figure_path = save_trace_plot(data, tmp_path)

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["dataset"] == "data/raw/coupled_oscillator_sample.csv"
    assert abs(payload["detected_frequency_hz"] - 12.5) < 1.0
    assert 0.0 <= payload["resonance_depth"] <= 1.0
    assert any(
        edge["source"] == "dim_0" and edge["target"] == "dim_1"
        for edge in payload["significant_edges"]
    )
    assert paths["csv"].exists()
    assert figure_path.exists()
