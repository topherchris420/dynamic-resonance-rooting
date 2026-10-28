"""The phase and harmonic benchmark: simulators, determinism, and the committed artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.phase_sync import PhaseSynchronyAnalyzer, dominant_frequency
from drr_framework.resonance_math_benchmark import (
    DEFAULT_CONFIG,
    FROZEN_PREREGISTRATION_SHA256,
    METHODS,
    SCHEMA_VERSION,
    declared_rate,
    evaluate_hypotheses,
    integrity_checks,
    level_code,
    preregistration_sha256,
    render_benchmark_report,
    run_benchmark,
    run_condition,
    simulate_condition,
    summarize_row,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "expected" / "resonance_math_benchmark.json"
REPORT = ROOT / "results" / "expected" / "resonance_math_benchmark.md"


def _condition(split: str, condition_id: str):
    return next(c for c in DEFAULT_CONFIG[split]["conditions"] if c["id"] == condition_id)


def _draw(split: str, condition_id: str, seed: int = 0):
    return simulate_condition(_condition(split, condition_id), np.random.default_rng(seed))


# ------------------------------------------------------------------ simulators


def test_strong_coupling_locks_the_phases_and_none_leaves_them_free():
    spec = dict(_condition("development", "dev_coupled_1to1"), coupling=40.0, phase_diffusion=0.5)
    data, _ = simulate_condition(spec, np.random.default_rng(1))
    locked = PhaseSynchronyAnalyzer(spec["sampling_rate"]).analyze(data)["pairs"][0]
    assert locked["plv"] > 0.9
    free, _ = _draw("development", "dev_independent_similar", seed=2)
    unlocked = PhaseSynchronyAnalyzer(spec["sampling_rate"]).analyze(free)["pairs"][0]
    assert unlocked["plv"] < 0.5


def test_harmonic_coupling_puts_the_response_at_the_ratio_frequency():
    data, _ = _draw("holdout", "harmonic_2to3", seed=3)
    fs = _condition("holdout", "harmonic_2to3")["sampling_rate"]
    ratio = dominant_frequency(data[:, 1], fs) / dominant_frequency(data[:, 0], fs)
    assert ratio == pytest.approx(1.5, abs=0.05)


def test_missing_samples_and_jittered_timestamps_are_as_configured():
    data, stamps = _draw("holdout", "coupled_missing", seed=4)
    assert stamps is None
    assert np.isnan(data).mean() == pytest.approx(0.15, abs=0.02)
    values, stamps = _draw("holdout", "coupled_irregular", seed=5)
    spec = _condition("holdout", "coupled_irregular")
    step = 1.0 / spec["sampling_rate"]
    assert np.all(np.diff(stamps) > 0)
    offsets = stamps - np.arange(stamps.size) * step
    assert np.abs(offsets).max() <= 0.5 * spec["timing_jitter"] * step + 1e-12
    assert np.isfinite(values).all()


def test_a_sweep_raises_the_frequency_across_the_record():
    data, _ = _draw("development", "dev_common_chirp", seed=6)
    fs = _condition("development", "dev_common_chirp")["sampling_rate"]
    quarter = data.shape[0] // 4
    for column in data.T:
        early, late = dominant_frequency(column[:quarter], fs), dominant_frequency(
            column[-quarter:], fs
        )
        assert late - early > 2.0


def test_other_families_and_the_negative_control_have_the_right_shape():
    for condition in ("dev_ar1", "dev_random_walk", "dev_phase_randomized"):
        data, stamps = _draw("development", condition, seed=7)
        assert data.shape == (1024, 2) and stamps is None and np.isfinite(data).all()
    with pytest.raises(ValueError, match="unknown condition family"):
        simulate_condition({"family": "pink", "n_samples": 8, "sampling_rate": 1.0}, None)


# ------------------------------------------------------------------ codes and rates


def test_level_codes_and_declared_rates():
    assert [level_code(p) for p in (0.005, 0.03, 0.08, 0.5, None)] == ["1", "5", "T", ".", "-"]
    assert level_code(0.001, abstain=True) == "A"
    codes = "15T.-A"
    assert declared_rate(codes, 0.01)["count"] == 1
    assert declared_rate(codes, 0.05)["count"] == 2
    assert declared_rate(codes, 0.10)["count"] == 3


def _tiny_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["transfer_entropy"] = False
    keep_dev = {"dev_coupled_1to1", "dev_random_walk"}
    keep_hold = {"harmonic_1to2", "independent_coherent", "common_chirp", "short_coupled_1to1"}
    for split, keep in (("development", keep_dev), ("holdout", keep_hold)):
        config[split]["conditions"] = [c for c in config[split]["conditions"] if c["id"] in keep]
        config[split]["n_trials"] = {"related": 2, "unrelated": 2}
    return config


def test_benchmark_is_deterministic_for_any_worker_count_and_renders():
    config = _tiny_config()
    first = run_benchmark(config)
    second = run_benchmark(config, workers=2)
    assert {k: v for k, v in first.items() if k != "runtime"} == {
        k: v for k, v in second.items() if k != "runtime"
    }
    assert [h["id"] for h in first["hypotheses"]] == [f"H{k}" for k in range(1, 10)]
    assert {h["outcome"] for h in first["hypotheses"]} <= {"supported", "not_supported"}
    assert first["checks"]["every_holdout_row_complete"]
    assert not first["matches_frozen_preregistration"]
    report = render_benchmark_report(first)
    for heading in (
        "## Methods",
        "## Held-out: false relationships",
        "## Preregistered",
        "## Integrity",
    ):
        assert heading in report
    for line in report.splitlines():
        # Rows begin with a label, never a number or a surrogate-null name, so the
        # claim-boundary tests cannot mistake them for another study's rows.
        if line.startswith("| ") and not line.startswith("| ---"):
            assert not line[2].isdigit(), line


def test_frozen_fingerprint_is_the_fingerprint_of_the_default_config():
    assert preregistration_sha256(DEFAULT_CONFIG) == FROZEN_PREREGISTRATION_SHA256
    altered = json.loads(json.dumps(DEFAULT_CONFIG))
    altered["preregistration"]["hypotheses"][0]["statement"] += " Edited."
    assert preregistration_sha256(altered) != FROZEN_PREREGISTRATION_SHA256


# ------------------------------------------------------------------ the committed artifact


def _artifact():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_artifact_is_complete_frozen_and_matches_its_report():
    artifact = _artifact()
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["config"] == json.loads(json.dumps(DEFAULT_CONFIG))
    assert artifact["preregistration_sha256"] == FROZEN_PREREGISTRATION_SHA256
    assert artifact["matches_frozen_preregistration"] is True
    assert all(artifact["checks"].values()), artifact["checks"]
    assert set(artifact["methods"]) == set(METHODS)
    assert REPORT.read_text(encoding="utf-8") == render_benchmark_report(artifact)


def test_committed_rates_and_outcomes_are_recomputed_from_the_trial_codes():
    """Rates are not stored, and outcomes and checks must equal their recomputation."""
    artifact = _artifact()
    alpha = artifact["config"]["alpha"]
    for row in artifact["development"] + artifact["holdout"]:
        assert "summary" not in row, "rates are derived from the codes, never stored"
        assert all(len(codes) == row["n_trials"] for codes in row["levels"].values())
        assert summarize_row(row, alpha)["phase_sync"]["declared"]["trials"] == row["n_trials"]
    assert artifact["hypotheses"] == evaluate_hypotheses(artifact["holdout"], artifact["config"])
    assert artifact["checks"] == integrity_checks(
        artifact["development"], artifact["holdout"], artifact["config"]
    )


@pytest.mark.parametrize(
    "split, condition_id",
    [
        ("holdout", "harmonic_1to2"),
        ("holdout", "independent_coherent"),
        ("holdout", "coupled_missing"),
        ("development", "dev_common_chirp"),
    ],
)
def test_committed_rows_reproduce_exactly_on_a_prefix(split, condition_id):
    """Rerun the first trials of a committed row with its seed schedule and compare every code."""
    artifact = _artifact()
    committed = next(r for r in artifact[split] if r["condition"] == condition_id)
    prefix = 3
    fresh, _ = run_condition(_condition(split, condition_id), artifact["config"], prefix)
    for field in ("levels", "directions", "statuses"):
        for method, codes in fresh[field].items():
            assert codes == committed[field][method][:prefix], (field, method)
    assert fresh["gate"]["abstain"] == committed["gate"]["abstain"][:prefix]


def test_documents_quote_artifact_rows_and_outcomes():
    """Every quoted held-out row is a line of the rendered report; outcomes are the artifact's."""
    artifact = _artifact()
    rendered = set(render_benchmark_report(artifact).splitlines())
    labels = {row["label"] for row in artifact["holdout"]}
    for name in ("DRR_BENCHMARKS.md", "docs/resonance-math-report.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "resonance_math_benchmark.json" in text, name
        rows = [
            line
            for line in text.splitlines()
            if line.startswith("| ") and line.split("|")[1].strip() in labels
        ]
        assert len(rows) >= 22, name
        for row in rows:
            assert row in rendered, (name, row)
    report = (ROOT / "docs" / "resonance-math-report.md").read_text(encoding="utf-8")
    for hypothesis in artifact["hypotheses"]:
        line = next(row for row in report.splitlines() if row.startswith(f"| {hypothesis['id']} |"))
        assert f"**{hypothesis['outcome']}**" in line, hypothesis["id"]
    validation = (ROOT / "DRR_VALIDATION_REPORT.md").read_text(encoding="utf-8")
    assert "resonance_math_benchmark.json" in validation
