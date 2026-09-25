"""The rooting test keeps its nominal size, and the committed study says so."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.calibration import (
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    depth_reference,
    render_calibration_report,
    rooting_size,
    run_calibration_study,
    simulate_directed_pair,
    wilson_interval,
)
from drr_framework.modules import RootingAnalyzer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "expected" / "rooting_calibration_study.json"
REPORT = ROOT / "results" / "expected" / "rooting_calibration_study.md"


def _count_configurations(n_samples, n_variables, separation, draws, seed=0):
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(draws):
        key = tuple(
            RootingAnalyzer._circular_shift_offsets(n_samples, n_variables, separation, rng)
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _valid_configurations(n_samples, n_variables, separation):
    def distance(a, b):
        d = abs(a - b) % n_samples
        return min(d, n_samples - d)

    grids = np.array(np.meshgrid(*[range(n_samples)] * (n_variables - 1))).reshape(
        n_variables - 1, -1
    )
    valid = set()
    for column in grids.T:
        offsets = (0, *column.tolist())
        pairs = [(a, b) for i, a in enumerate(offsets) for b in offsets[i + 1 :]]
        if all(distance(a, b) >= separation for a, b in pairs):
            valid.add(offsets)
    return valid


@pytest.mark.parametrize("n_samples, n_variables, separation", [(20, 2, 3), (12, 3, 3), (14, 4, 2)])
def test_circular_shift_offsets_are_uniform_over_every_valid_configuration(
    n_samples, n_variables, separation
):
    valid = _valid_configurations(n_samples, n_variables, separation)
    draws = 150 * len(valid)
    counts = _count_configurations(n_samples, n_variables, separation, draws)
    assert set(counts) == valid
    expected = draws / len(valid)
    chi_square = sum((count - expected) ** 2 / expected for count in counts.values())
    # 99.9th percentile of chi-square with len(valid) - 1 degrees of freedom, bounded loosely.
    assert chi_square < len(valid) - 1 + 5 * np.sqrt(2 * (len(valid) - 1)) + 10


def test_offsets_fill_the_tight_case_exactly():
    offsets = RootingAnalyzer._circular_shift_offsets(9, 3, 3, np.random.default_rng(1))
    assert sorted(offsets.tolist()) == [0, 3, 6]


def test_circular_shift_null_holds_its_size_on_red_noise():
    """v4.3 split the slack with a multinomial draw and rejected 30% of true nulls here.

    This is a hard case (effective sample size about 13), where the fixed test
    runs near 0.06 over 3200 trials. The bound separates the two cleanly.
    """
    row = rooting_size(
        n_trials=160,
        n_samples=256,
        n_variables=3,
        ar_coefficient=0.9,
        max_lag=4,
        n_surrogates=49,
        alpha=0.05,
        surrogate_method="circular_shift",
        seed=7,
    )
    assert row["family_wise_error"]["rate"] <= 0.15


def test_permutation_null_is_invalid_for_autocorrelated_series():
    row = rooting_size(
        n_trials=60,
        n_samples=256,
        n_variables=3,
        ar_coefficient=0.9,
        max_lag=4,
        n_surrogates=49,
        alpha=0.05,
        surrogate_method="permutation",
        seed=7,
    )
    assert row["family_wise_error"]["ci95"][0] > 0.05


def test_directed_pair_has_the_requested_coupling_and_lag():
    data = simulate_directed_pair(20_000, 0.4, 3, 0.5, np.random.default_rng(0))
    source, target = data[:, 0], data[:, 1]
    lagged = np.corrcoef(source[:-3], target[3:])[0, 1]
    assert lagged == pytest.approx(0.4, abs=0.03)
    assert np.std(source) == pytest.approx(1.0, abs=0.05)
    assert np.std(target) == pytest.approx(1.0, abs=0.05)


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(5, 100)
    assert low == pytest.approx(0.02154, abs=1e-4)
    assert high == pytest.approx(0.11175, abs=1e-4)


def _tiny_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["n_samples"] = 128
    config["n_surrogates"] = 19
    config["size"].update(n_trials=6, ar_coefficients=[0.0, 0.9])
    config["power"].update(n_trials=6, couplings=[0.0, 0.6])
    config["depth"].update(n_trials=4, window_size=64, tone_amplitudes=[1.0])
    return config


def test_study_is_deterministic_and_renders():
    first = run_calibration_study(_tiny_config())
    second = run_calibration_study(_tiny_config())
    assert first == second
    report = render_calibration_report(first)
    assert "## Size" in report and "## Power" in report and "## Checks" in report


def test_committed_study_is_complete_and_passes_its_checks():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["config"] == DEFAULT_CONFIG
    assert all(artifact["checks"].values()), artifact["checks"]
    size_cfg = DEFAULT_CONFIG["size"]
    assert len(artifact["size"]) == len(size_cfg["ar_coefficients"]) * len(
        size_cfg["surrogate_methods"]
    )
    assert len(artifact["power"]) == len(DEFAULT_CONFIG["power"]["couplings"])
    assert REPORT.read_text(encoding="utf-8") == render_calibration_report(artifact)


@pytest.mark.parametrize("row_index", [0, 5])
def test_committed_size_rows_reproduce_exactly_on_a_prefix(row_index):
    """Rerun the first trials of a committed row with its seed schedule and compare."""
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = artifact["config"]
    committed = artifact["size"][row_index]
    methods = config["size"]["surrogate_methods"]
    coefficient_index = config["size"]["ar_coefficients"].index(committed["ar_coefficient"])
    method_index = methods.index(committed["surrogate_method"])
    prefix = 40
    fresh = rooting_size(
        n_trials=prefix,
        n_samples=config["n_samples"],
        n_variables=config["size"]["n_variables"],
        ar_coefficient=committed["ar_coefficient"],
        max_lag=config["max_lag"],
        n_surrogates=config["n_surrogates"],
        alpha=config["alpha"],
        surrogate_method=committed["surrogate_method"],
        seed=config["seed"] + 10_000 * (coefficient_index + 1) + 1_000 * method_index,
    )
    expected = [trial for trial in committed["false_positive_trials"] if trial < prefix]
    assert fresh["false_positive_trials"] == expected
    assert committed["family_wise_error"]["count"] == len(committed["false_positive_trials"])


def test_committed_depth_row_reproduces_exactly():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = artifact["config"]
    depth = config["depth"]
    fresh = depth_reference(
        n_trials=depth["n_trials"],
        n_samples=config["n_samples"],
        window_size=depth["window_size"],
        noise_ar_coefficients=depth["noise_ar_coefficients"][:1],
        tone_cycles_per_sample=depth["tone_cycles_per_sample"],
        tone_amplitudes=[],
        seed=config["seed"] + 200_000,
    )
    assert fresh[0] == artifact["depth"][0]
