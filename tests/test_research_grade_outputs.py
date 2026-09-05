import json

import numpy as np
import pytest

from drr_framework.modules import DepthCalculator, ResonanceDetector, RootingAnalyzer


def _infer_circular_offset(original: np.ndarray, shifted: np.ndarray) -> int:
    matches = np.flatnonzero(shifted == original[0])
    assert matches.size == 1
    return int(matches[0])


def _circular_distance(offset_a: int, offset_b: int, n_samples: int) -> int:
    return min((offset_a - offset_b) % n_samples, (offset_b - offset_a) % n_samples)


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


def test_rooting_without_surrogates_does_not_fabricate_significance():
    rng = np.random.default_rng(11)
    source = rng.normal(size=200)
    target = np.roll(source, 2)
    result = RootingAnalyzer().analyze(np.column_stack([source, target]), n_surrogates=0)

    assert result["inference_available"] is False
    assert np.isnan(result["p_values"][0, 1])
    assert np.isnan(result["adjusted_p_values"][0, 1])
    assert result["significant_edges"] == []
    assert result["candidate_edges"]
    assert result["score_matrix"] is result["transfer_entropy"]


def test_rooting_analyzer_reports_corrected_surrogate_inference():
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
        n_surrogates=49,
        random_state=7,
        alpha=0.05,
        surrogate_method="circular_shift",
        correction="max_statistic",
    )

    assert result["method"] in {"transfer_entropy", "lagged_correlation"}
    assert result["score_matrix"].shape == (3, 3)
    assert result["transfer_entropy"].shape == (3, 3)
    assert result["p_values"].shape == (3, 3)
    assert result["adjusted_p_values"].shape == (3, 3)
    assert result["effective_lag"].shape == (3, 3)
    assert result["inference_available"] is True
    assert result["surrogate_method"] == "circular_shift"
    assert result["correction"] == "max_statistic"
    assert result["n_surrogates"] == 49
    assert result["minimum_attainable_p_value"] == 1.0 / 50.0
    assert result["effective_lag"][0, 1] == 2

    finite_off_diagonal = ~np.eye(3, dtype=bool) & np.isfinite(result["p_values"])
    assert np.all(
        result["adjusted_p_values"][finite_off_diagonal] >= result["p_values"][finite_off_diagonal]
    )
    assert any(
        edge["source"] == "dim_0"
        and edge["target"] == "dim_1"
        and edge["lag"] == 2
        and edge["adjusted_p_value"] <= 0.05
        for edge in result["significant_edges"]
    )


def test_circular_shift_surrogates_keep_all_column_offsets_farther_than_max_lag():
    analyzer = RootingAnalyzer()
    n_samples = 30
    max_lag = 4
    data = np.column_stack([np.arange(n_samples) + (1000 * index) for index in range(4)])

    surrogate = analyzer._generate_surrogate(
        data,
        max_lag=max_lag,
        rng=np.random.default_rng(0),
        surrogate_method="circular_shift",
    )

    offsets = [
        _infer_circular_offset(data[:, index], surrogate[:, index])
        for index in range(data.shape[1])
    ]

    for left in range(len(offsets)):
        for right in range(left + 1, len(offsets)):
            assert _circular_distance(offsets[left], offsets[right], n_samples) > max_lag


def test_rooting_analyzer_rejects_impossible_circular_shift_configuration():
    analyzer = RootingAnalyzer()
    data = np.column_stack([np.arange(3, dtype=float), np.arange(10.0, 13.0)])

    with pytest.raises(
        ValueError,
        match="more samples, fewer variables, or surrogate_method='permutation'",
    ):
        analyzer.analyze(
            data,
            max_lag=1,
            n_surrogates=1,
            random_state=0,
            surrogate_method="circular_shift",
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
