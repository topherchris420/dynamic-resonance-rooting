"""Independent timing, censoring, sequential baseline and evidence checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drr_framework.finance.breakdown.detectors import (
    bocpd,
    first_alarm,
    score_stream,
    simulate,
    threshold,
    var_errors,
)
from drr_framework.finance.breakdown.market import conditional_effect, load_market, warning_weights
from drr_framework.finance.breakdown.study import simulation_summary, verify_lock, wilson_upper
from drr_framework.finance.holdout.data import digest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/lead_lag_breakdown"


@pytest.fixture
def protocol():
    return json.loads((STUDY / "protocol.json").read_text())


def test_detector_and_var_predictions_are_prefix_invariant(protocol):
    data = simulate(protocol, 7, "unchanged")
    t, scores, errors, _ = score_stream(data, protocol)
    cutoff = 600
    short_t, short_scores, short_errors, _ = score_stream(data[:cutoff], protocol)
    poisoned = data.copy()
    poisoned[cutoff:] *= 10000
    _, poisoned_scores, poisoned_errors, _ = score_stream(poisoned, protocol)
    np.testing.assert_array_equal(short_t, t[t < cutoff])
    for method in protocol["methods"]:
        np.testing.assert_allclose(short_scores[method], scores[method][t < cutoff], atol=1e-12)
        np.testing.assert_allclose(
            scores[method][t < cutoff], poisoned_scores[method][t < cutoff], atol=1e-12
        )
    np.testing.assert_allclose(errors[:cutoff], short_errors, equal_nan=True)
    np.testing.assert_allclose(errors[:cutoff], poisoned_errors[:cutoff], equal_nan=True)


def test_seeds_and_prebreak_histories_are_paired_and_disjoint(protocol):
    spec = protocol["simulation"]
    seeds = set(
        range(spec["calibration_seed"], spec["calibration_seed"] + spec["calibration_replicates"])
    )
    assert seeds.isdisjoint(range(spec["test_seed"], spec["test_seed"] + spec["test_replicates"]))
    control = simulate(protocol, 7, "unchanged")
    for name in spec["changes"] + spec["nulls"] + spec["challenges"]:
        changed = simulate(protocol, 7, name)
        np.testing.assert_array_equal(changed[: spec["change"]], control[: spec["change"]])


def test_absolute_rooting_cannot_identify_a_clean_sign_flip(protocol):
    from drr_framework.finance.breakdown.detectors import fingerprint

    data = simulate(protocol, 11, "unchanged")[-64:]
    flipped = data * [1, -1]
    original, signed = fingerprint(data)
    reverse, reverse_signed = fingerprint(flipped)
    np.testing.assert_allclose(original[:4], reverse[:4], atol=1e-12)
    np.testing.assert_allclose(signed, -reverse_signed, atol=1e-12)


def test_bocpd_uses_short_run_probability_not_constant_reset_mass():
    rng = np.random.default_rng(5)
    stream = np.r_[rng.normal(size=300), rng.normal(5, 1, 30)]
    p = bocpd(stream)
    assert np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1 + 1e-12)
    assert np.ptp(p[256:]) > 0.01
    np.testing.assert_allclose(bocpd(stream[:300]), p[:300])


def test_bocpd_training_warmup_avoids_certain_startup_alarm(protocol):
    _, scores, _, _ = score_stream(simulate(protocol, 9, "unchanged"), protocol)
    assert scores["var_bocpd"][0] < 0.99


def test_threshold_rank_and_strict_crossing():
    assert threshold(np.arange(64)) == 61
    times = np.array([511, 519, 527])
    result = first_alarm(times, np.array([1, 1, 2]), 1)
    assert result["first_alarm"] == 527 and result["delay"] == 15
    with pytest.raises(ValueError):
        threshold([1])


@pytest.mark.parametrize("scores", [[2, 0, 2], [0, 0, 0]])
def test_early_alarms_and_misses_receive_full_delay_penalty(scores):
    result = first_alarm(np.array([511, 519, 527]), np.array(scores), 1)
    assert result["penalized_delay"] == 256
    assert not result["hit"] and result["delay"] is None


def test_var_zero_reference_is_zero_in_original_units():
    rng = np.random.default_rng(8)
    data = rng.normal(size=(400, 2)) + 3
    _, zero, _ = var_errors(data)
    np.testing.assert_allclose(zero, data / data[:256].std(axis=0))


def test_warning_timing_duration_overlap_and_matched_exposure():
    index = pd.date_range("2026-04-01", periods=60, freq="h")
    w = warning_weights(index, [index[0], index[20]])
    assert np.all(w[:2] == 0.5)
    assert np.all(w[2:46] == 0)
    assert np.all(w[46:] == 0.5)
    assert np.full_like(w, w.mean()).sum(axis=1).mean() == pytest.approx(w.sum(axis=1).mean())


def test_empty_warning_effect_is_not_zero_evidence():
    assert conditional_effect(np.zeros(10, bool), np.ones(10)) is None
    assert wilson_upper(0, 96) > 0


def test_failed_and_missed_episodes_are_not_dropped(protocol):
    rows = []
    spec = protocol["simulation"]
    for scenario in spec["nulls"] + spec["changes"] + spec["challenges"]:
        for rep in range(10):
            for method in protocol["methods"]:
                rows.append(
                    {
                        "scenario": scenario,
                        "replicate": rep,
                        "method": method,
                        "first_alarm": None,
                        "pre_change_alarm": False,
                        "hit": False,
                        "delay": None,
                        "penalized_delay": 256,
                    }
                )
    result = simulation_summary(pd.DataFrame(rows), protocol)
    assert result["verdict"] == "NOT SUPPORTED"
    assert len(result["tests"]) == 15
    assert len(result["outcomes"]) == 42
    assert all(t["p_holm"] == 1 for t in result["tests"])


def test_market_vintage_has_exact_hourly_calendar(protocol):
    if not (STUDY / "data/bars.csv.gz").exists():
        pytest.skip("Data not yet fetched")
    frame = load_market(STUDY / "data", protocol)
    assert len(frame) == 13104


def test_sealed_evidence_and_market_labels_are_auditable(protocol):
    output = ROOT / "results/expected/lead_lag_breakdown"
    if not (output / "provenance.json").exists():
        pytest.skip("Held-out evaluation not yet run")
    lock = verify_lock(ROOT, STUDY, STUDY / "calibration")
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["inputs"] == lock["inputs"]
    assert provenance["lock_sha256"] == digest(STUDY / "calibration/lock.json")
    for name, expected in provenance["artifacts"].items():
        assert digest(output / name) == expected
    episodes = pd.read_csv(output / "simulation_episodes.csv.gz")
    assert len(episodes) == 96 * 7 * 6
    rows = pd.read_csv(
        output / "market_scores.csv.gz", parse_dates=["time", "fit_last_time", "target_exit_time"]
    )
    assert (rows.fit_last_time < rows.time).all()
    valid = rows.future_error_ratio.notna()
    assert (
        rows.loc[valid, "target_exit_time"] - rows.loc[valid, "time"] == pd.Timedelta(hours=24)
    ).all()
    assert all(
        rows[name + "_warning"].equals(rows[name] > lock["market_thresholds"][name])
        for name in protocol["methods"]
    )


def test_published_market_targets_reconcile_independently_to_raw_prices(protocol):
    output = ROOT / "results/expected/lead_lag_breakdown"
    if not (output / "market_scores.csv.gz").exists():
        pytest.skip("Held-out evaluation not yet run")
    bars = pd.read_csv(STUDY / "data/bars.csv.gz", parse_dates=["time"])
    prices = bars.pivot(index="time", columns="symbol", values="close")[
        protocol["market"]["symbols"]
    ]
    returns = np.log(prices / prices.shift()).iloc[1:]
    ledger = pd.read_csv(output / "market_scores.csv.gz", parse_dates=["time"])
    for month, rows in ledger.groupby("month"):
        start = pd.Timestamp(month)
        training = returns.loc[returns.index < start].tail(256).to_numpy()
        means, scales = training.mean(axis=0), training.std(axis=0)
        standardized = (training - means) / scales
        design = np.array(
            [
                np.r_[1, standardized[t - 1], standardized[t - 2], standardized[t - 3]]
                for t in range(3, 256)
            ]
        )
        penalty = np.diag([0] + [0.001] * 6)
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ standardized[3:])
        for row in rows.loc[rows.future_error_ratio.notna()].iloc[[0, -1]].itertuples():
            future = pd.date_range(row.time + pd.Timedelta(hours=1), periods=24, freq="h")
            numerator, denominator = 0.0, 0.0
            for time in future:
                z = (returns.loc[time].to_numpy() - means) / scales
                regressors = np.concatenate(
                    [np.ones(1)]
                    + [
                        (returns.loc[time - pd.Timedelta(hours=k)].to_numpy() - means) / scales
                        for k in [1, 2, 3]
                    ]
                )
                numerator += float(np.sum((z - regressors @ beta) ** 2))
                denominator += float(np.sum((returns.loc[time].to_numpy() / scales) ** 2))
            assert row.future_error_ratio == pytest.approx(numerator / denominator, rel=1e-10)


def test_published_risk_exposure_controls_match_and_returns_reconcile():
    output = ROOT / "results/expected/lead_lag_breakdown"
    if not (output / "risk_ledger.csv.gz").exists():
        pytest.skip("Held-out evaluation not yet run")
    ledger = pd.read_csv(output / "risk_ledger.csv.gz")
    summary = json.loads((output / "summary.json").read_text())
    for record in summary["market"]["risk"]:
        rows = ledger.loc[
            (ledger.strategy == record["strategy"]) & (ledger.cost_bps == record["cost_bps"])
        ]
        assert len(rows) == 4391
        assert np.prod(1 + rows.net_return) - 1 == pytest.approx(record["total_return"], abs=1e-12)
        assert rows.exposure.mean() == pytest.approx(record["mean_exposure"])
        if record["strategy"].endswith("_matched_exposure"):
            original = record["strategy"][: -len("_matched_exposure")]
            matched = ledger.loc[
                (ledger.strategy == original) & (ledger.cost_bps == record["cost_bps"])
            ]
            assert rows.exposure.mean() == pytest.approx(matched.exposure.mean())
