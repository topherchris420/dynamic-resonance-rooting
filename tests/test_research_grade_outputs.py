import json

import numpy as np

from drr_framework.modules import DepthCalculator, ResonanceDetector, RootingAnalyzer


def test_welch_detector_identifies_noisy_dominant_frequency_with_confidence():
    sampling_rate = 200.0
    duration = 4.0
    target_freq = 12.5
    rng = np.random.default_rng(42)
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    data = np.sin(2 * np.pi * target_freq * t) + 0.05 * rng.normal(size=t.size)

    result = ResonanceDetector().detect(
        data,
        method="welch",
        sampling_rate=sampling_rate,
        peak_height_ratio=0.2,
    )

    assert result["method"] == "welch"
    assert np.isclose(result["dominant_freq"][0], target_freq, atol=1.0)
    assert 0.0 <= result["confidence"][0] <= 1.0
    assert result["noise_floor"] > 0.0


def test_depth_calculator_reports_composite_metric_components():
    sampling_rate = 200.0
    t = np.arange(int(sampling_rate * 4.0)) / sampling_rate
    data = np.sin(2 * np.pi * 8.0 * t) + 0.2 * np.sin(2 * np.pi * 16.0 * t)

    result = DepthCalculator().calculate(
        data,
        window_size=256,
        sampling_rate=sampling_rate,
        resonance_frequencies=np.array([8.0]),
    )

    assert result["method"] == "drr_composite_v1"
    assert 0.0 <= result["resonance_depth"] <= 1.0
    assert set(result["components"]) == {
        "spectral_concentration",
        "temporal_persistence",
        "phase_coherence",
        "amplitude_stability",
    }
    lower, upper = result["confidence_interval"]
    assert lower <= result["resonance_depth"] <= upper


def test_rooting_analyzer_reports_method_lags_p_values_and_edges():
    rng = np.random.default_rng(7)
    n = 500
    source = rng.normal(size=n)
    target = np.roll(source, 2) + 0.15 * rng.normal(size=n)
    target[:2] = rng.normal(size=2)
    independent = rng.normal(size=n)
    data = np.column_stack([source, target, independent])

    result = RootingAnalyzer().analyze(
        data,
        max_lag=4,
        n_surrogates=25,
        random_state=7,
        alpha=0.05,
    )

    assert result["method"] in {"transfer_entropy", "lagged_correlation"}
    assert result["transfer_entropy"].shape == (3, 3)
    assert result["p_values"].shape == (3, 3)
    assert result["effective_lag"].shape == (3, 3)
    assert any(
        edge["source"] == "dim_0" and edge["target"] == "dim_1"
        for edge in result["significant_edges"]
    )


def test_reproduction_experiment_writes_research_artifacts(tmp_path):
    from drr_framework.validation import run_reproduction_experiment

    output_dir = tmp_path / "reproduction"

    summary = run_reproduction_experiment(
        output_dir=output_dir,
        random_state=123,
        save_artifacts=True,
    )

    assert summary["frequency_error_hz"] < 1.0
    assert 0.0 <= summary["resonance_depth"] <= 1.0
    assert summary["rooting_method"] in {"transfer_entropy", "lagged_correlation"}
    assert (output_dir / "drr_reproduction_summary.json").exists()
    assert (output_dir / "drr_reproduction_metrics.csv").exists()

    payload = json.loads((output_dir / "drr_reproduction_summary.json").read_text())
    assert payload["target_frequency_hz"] == summary["target_frequency_hz"]
    assert payload["detected_frequency_hz"] == summary["detected_frequency_hz"]
