"""Which null assumptions survive long memory, and the committed study says so."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework import DynamicResonanceRooting
from drr_framework.dependent_noise import (
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    davies_harte_eigenvalues,
    describe_process,
    evaluate_hypotheses,
    fgn_autocovariance,
    integrity_checks,
    make_null_process,
    null_false_alarms,
    render_dependent_noise_report,
    run_dependent_noise_study,
    simulate_fractional_gaussian_noise,
    simulate_garch,
    simulate_long_memory_pair,
)
from drr_framework.modules import RootingAnalyzer, phase_randomized_surrogate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "expected" / "dependent_noise_null_study.json"
REPORT = ROOT / "results" / "expected" / "dependent_noise_null_study.md"


# ------------------------------------------------------------------ simulators


@pytest.mark.parametrize("hurst", [0.5, 0.7, 0.85, 0.95])
def test_fractional_gaussian_noise_has_the_fgn_autocovariance(hurst):
    """Averaged over independent paths without demeaning, so the estimate is unbiased."""
    paths = simulate_fractional_gaussian_noise(512, 2000, hurst, np.random.default_rng(1))
    lags = np.array([0, 1, 2, 5, 20])
    measured = np.array([np.mean(paths[: 512 - k] * paths[k:]) for k in lags])
    assert measured == pytest.approx(fgn_autocovariance(lags, hurst), abs=0.04)


def test_fractional_gaussian_noise_paths_are_independent_and_white_at_half():
    rng = np.random.default_rng(2)
    paths = simulate_fractional_gaussian_noise(20_000, 2, 0.7, rng)
    assert abs(np.corrcoef(paths[:, 0], paths[:, 1])[0, 1]) < 0.05
    white = simulate_fractional_gaussian_noise(20_000, 1, 0.5, rng)[:, 0]
    assert abs(np.corrcoef(white[:-1], white[1:])[0, 1]) < 0.03
    assert fgn_autocovariance([1, 2, 3], 0.5) == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("hurst", [0.5, 0.7, 0.85, 0.95, 0.99])
def test_circulant_embedding_is_a_valid_covariance(hurst):
    eigenvalues = davies_harte_eigenvalues(512, hurst)
    assert eigenvalues.shape == (1024,)
    assert eigenvalues.min() >= -1e-9 * eigenvalues.max()


def test_long_memory_autocovariance_is_not_summable_but_short_memory_is():
    lags = np.arange(1, 100_000)
    assert fgn_autocovariance(lags, 0.95).sum() > 50
    assert fgn_autocovariance(lags, 0.5).sum() == pytest.approx(0.0, abs=1e-9)


def test_garch_is_uncorrelated_but_dependent_with_unit_variance():
    series = simulate_garch(100_000, 1, 0.1, 0.85, np.random.default_rng(3))[:, 0]
    assert series.var() == pytest.approx(1.0, abs=0.1)
    assert abs(np.corrcoef(series[:-1], series[1:])[0, 1]) < 0.02
    squares = series**2
    assert np.corrcoef(squares[:-1], squares[1:])[0, 1] > 0.1
    assert np.mean((series / series.std()) ** 4) > 3.3


def test_long_memory_pair_has_the_requested_coupling_lag_and_scale():
    data = simulate_long_memory_pair(20_000, 0.3, 2, 0.7, np.random.default_rng(4))
    assert data.shape == (20_000, 3)
    assert np.corrcoef(data[:-2, 0], data[2:, 1])[0, 1] == pytest.approx(0.3, abs=0.03)
    assert np.std(data, axis=0) == pytest.approx(np.ones(3), abs=0.08)


def test_process_specs_build_draws_with_the_configured_shape():
    for spec in DEFAULT_CONFIG["size"]["processes"]:
        draw = make_null_process(spec, 64, 3)
        assert draw(np.random.default_rng(0)).shape == (64, 3)
        assert describe_process(spec)
    with pytest.raises(ValueError, match="unknown process family"):
        make_null_process({"family": "pink"}, 64, 3)


# ------------------------------------------------------------------ the new null


@pytest.mark.parametrize("n_samples", [128, 129])
def test_phase_randomization_keeps_each_spectrum_and_mean_and_is_real(n_samples):
    rng = np.random.default_rng(5)
    data = rng.standard_normal((n_samples, 3)) + np.array([0.0, 2.0, -1.0])
    surrogate = phase_randomized_surrogate(data, rng)
    assert surrogate.shape == data.shape
    assert np.isrealobj(surrogate)
    assert np.abs(np.fft.rfft(surrogate, axis=0)) == pytest.approx(
        np.abs(np.fft.rfft(data, axis=0)), rel=1e-9, abs=1e-9
    )
    assert surrogate.mean(axis=0) == pytest.approx(data.mean(axis=0), abs=1e-9)
    assert not np.allclose(surrogate, data)


def test_phase_randomization_destroys_alignment_between_series():
    rng = np.random.default_rng(6)
    source = rng.standard_normal(512)
    data = np.column_stack((source, 0.95 * source + 0.05 * rng.standard_normal(512)))
    assert np.corrcoef(data[:, 0], data[:, 1])[0, 1] > 0.9
    correlations = [np.corrcoef(*phase_randomized_surrogate(data, rng).T)[0, 1] for _ in range(200)]
    assert np.mean(np.abs(correlations)) < 0.15


def test_rooting_analyzer_accepts_phase_randomization_end_to_end():
    rng = np.random.default_rng(7)
    data = simulate_long_memory_pair(512, 0.6, 2, 0.7, rng)
    result = RootingAnalyzer().analyze(
        data, max_lag=4, n_surrogates=49, random_state=1, surrogate_method="phase_randomization"
    )
    assert result["surrogate_method"] == "phase_randomization"
    assert [(e["source"], e["target"], e["lag"]) for e in result["significant_edges"]] == [
        ("dim_0", "dim_1", 2)
    ]
    facade = DynamicResonanceRooting(tau=2, sampling_rate=1.0).analyze_system(
        data,
        multivariate=True,
        window_size=128,
        state_space=False,
        rooting_max_lag=4,
        rooting_n_surrogates=19,
        rooting_random_state=1,
        rooting_surrogate_method="phase_randomization",
    )
    assert facade["rooting_analysis"]["surrogate_method"] == "phase_randomization"
    with pytest.raises(ValueError, match="surrogate_method"):
        RootingAnalyzer().analyze(data, surrogate_method="bootstrap")


# ------------------------------------------------------------------ the study


def _tiny_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["n_samples"] = 128
    config["n_surrogates"] = 19
    keep = {"fgn_0.50", "fgn_0.95", "random_walk"}
    config["size"]["processes"] = [p for p in config["size"]["processes"] if p["id"] in keep]
    config["size"]["n_trials"] = 6
    config["coupled"].update(n_trials=4, hurst=[0.5, 0.95])
    return config


def test_study_is_deterministic_renders_and_records_every_hypothesis():
    first = run_dependent_noise_study(_tiny_config())
    second = run_dependent_noise_study(_tiny_config())
    assert first == second
    assert len(first["size"]) == 3 * 3
    assert len(first["coupled"]) == 2 * 2
    assert [h["id"] for h in first["hypotheses"]] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert {h["outcome"] for h in first["hypotheses"]} <= {"supported", "not_supported"}
    report = render_dependent_noise_report(first)
    for heading in ("## The nulls", "## Size", "## Power", "## Preregistered", "## Integrity"):
        assert heading in report
    # Table rows start with the process label, never with a bare number, so the
    # claim-boundary test can tell these rows from the calibration study's rows.
    for line in report.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            assert not line[2].isdigit(), line


def test_committed_study_is_complete_passes_its_checks_and_matches_its_report():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["config"] == DEFAULT_CONFIG
    assert all(artifact["checks"].values()), artifact["checks"]
    n_processes = len(DEFAULT_CONFIG["size"]["processes"])
    assert len(artifact["size"]) == n_processes * len(DEFAULT_CONFIG["surrogate_methods"])
    assert len(artifact["coupled"]) == len(DEFAULT_CONFIG["coupled"]["hurst"]) * len(
        DEFAULT_CONFIG["coupled"]["surrogate_methods"]
    )
    assert REPORT.read_text(encoding="utf-8") == render_dependent_noise_report(artifact)


def test_committed_hypothesis_outcomes_are_recomputed_from_the_rows():
    """An outcome cannot be edited by hand without the rows disagreeing."""
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    recomputed = evaluate_hypotheses(artifact["size"], artifact["coupled"], artifact["config"])
    assert artifact["hypotheses"] == recomputed
    assert artifact["checks"] == integrity_checks(
        artifact["size"], artifact["coupled"], artifact["embedding"], artifact["config"]
    )
    for row in artifact["size"]:
        assert row["family_wise_error"]["count"] == len(row["false_positive_trials"])


@pytest.mark.parametrize("row_index", [0, 7, 17])
def test_committed_size_rows_reproduce_exactly_on_a_prefix(row_index):
    """Rerun the first trials of a committed row with its seed schedule and compare."""
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = artifact["config"]
    committed = artifact["size"][row_index]
    process_index = [p["id"] for p in config["size"]["processes"]].index(committed["process"])
    method_index = config["surrogate_methods"].index(committed["surrogate_method"])
    spec = config["size"]["processes"][process_index]
    prefix = 25
    fresh = null_false_alarms(
        make_null_process(spec, config["n_samples"], config["n_variables"]),
        n_trials=prefix,
        max_lag=config["max_lag"],
        n_surrogates=config["n_surrogates"],
        alpha=config["alpha"],
        surrogate_method=committed["surrogate_method"],
        seed=config["seed"] + 10_000 * (process_index + 1) + 1_000 * method_index,
    )
    expected = [trial for trial in committed["false_positive_trials"] if trial < prefix]
    assert fresh["false_positive_trials"] == expected
