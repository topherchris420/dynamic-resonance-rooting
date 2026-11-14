"""Adversarial timing, accounting, inference and evidence-integrity checks."""

from __future__ import annotations

import copy
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drr_framework.finance.holdout.data import build_features, digest, validate_bars
from drr_framework.finance.holdout.models import (
    calibration_quantile,
    split_training,
    walk_forward,
)
from drr_framework.finance.holdout.statistics import (
    block_inference,
    holm,
    performance,
    portfolio,
)
from drr_framework.finance.holdout.study import (
    new_output,
    source_hashes,
    summarize,
    verify_replay,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT / "experiments/financial_holdout/protocol.json").read_text())


def bars_fixture(n=850):
    rng = np.random.default_rng(71)
    dates = pd.date_range("2018-01-01", periods=n)
    rows = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        closes = opens * np.exp(rng.normal(0, 0.005, n))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": opens,
                    "close": closes,
                    "high": np.maximum(opens, closes) * 1.01,
                    "low": np.minimum(opens, closes) * 0.99,
                    "volume": rng.uniform(1, 100, n),
                }
            )
        )
    return pd.concat(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def synthetic_features():
    protocol = json.loads((ROOT / "experiments/financial_holdout/protocol.json").read_text())
    return build_features(bars_fixture(), protocol)


def test_target_is_delayed_open_to_open(protocol):
    bars = bars_fixture(100)
    features, targets = build_features(bars, protocol)
    date = targets.index[0]
    opens = bars.loc[bars.symbol == "BTCUSDT"].set_index("date").open
    expected = opens.loc[date + pd.Timedelta(days=3)] / opens.loc[date + pd.Timedelta(days=2)] - 1
    assert targets.loc[date, "BTCUSDT"] == expected
    assert len(features) - len(targets) == 3


def test_features_are_prefix_invariant_even_when_future_prices_are_poisoned(protocol):
    bars = bars_fixture(130)
    cutoff = pd.Timestamp("2018-04-15")
    original, _ = build_features(bars, protocol)
    truncated, _ = build_features(bars.loc[bars.date <= cutoff].copy(), protocol)
    changed = bars.copy()
    changed.loc[changed.date > cutoff, ["open", "high", "low", "close"]] *= 1000
    poisoned, _ = build_features(changed, protocol)
    pd.testing.assert_frame_equal(original.loc[:cutoff], truncated)
    pd.testing.assert_frame_equal(original.loc[:cutoff], poisoned.loc[:cutoff])


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "nonfinite", "ohlc", "unsorted", "symbol"]
)
def test_bad_market_data_is_rejected(protocol, defect):
    bars = bars_fixture(100)
    if defect == "missing":
        bars = bars.drop(index=30)
    elif defect == "duplicate":
        bars = pd.concat([bars, bars.iloc[[0]]])
    elif defect == "nonfinite":
        bars.loc[30, "close"] = np.nan
    elif defect == "ohlc":
        bars.loc[30, "high"] = 0.01
    elif defect == "unsorted":
        bars = bars.iloc[::-1]
    else:
        bars.loc[0, "symbol"] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_bars(bars, protocol["data"]["symbols"])


def test_fit_calibration_and_forecast_have_strict_label_gaps(protocol):
    index = pd.date_range("2018-01-01", "2025-12-31")
    first = pd.Timestamp("2022-12-30")
    fit, calibration = split_training(index, first, protocol)
    assert len(fit) <= 1095 and len(calibration) == 180
    assert fit[-1] + pd.Timedelta(days=3) < calibration[0] - pd.Timedelta(days=7)
    assert calibration[-1] + pd.Timedelta(days=3) < first - pd.Timedelta(days=7)
    assert set(fit).isdisjoint(calibration)


@pytest.mark.parametrize(
    "model,params",
    [("ridge_drr", {"alpha": 100.0}), ("hgb_drr", {"max_leaf_nodes": 7, "l2_regularization": 1.0})],
)
def test_actual_predictions_ignore_all_future_labels_and_features(
    protocol, synthetic_features, model, params
):
    features, targets = synthetic_features
    start, end = "2020-01-01", "2020-03-31"
    original = walk_forward(features, targets, protocol, model, params, start, end)
    changed_x, changed_y = features.copy(), targets.copy()
    boundary = pd.Timestamp("2020-04-01")
    changed_x.loc[changed_x.index >= boundary] *= 100
    changed_y.loc[changed_y.index >= pd.Timestamp("2019-12-23")] = 999.0
    changed = walk_forward(changed_x, changed_y, protocol, model, params, start, end)
    for column in ["prediction", "lower", "upper", "fit_last_exit", "calibration_last_exit"]:
        pd.testing.assert_series_equal(original[column], changed[column])


def test_calibration_uses_finite_sample_order_statistic():
    assert calibration_quantile(np.arange(1, 181, dtype=float), 0.9) == 163.0
    with pytest.raises(ValueError):
        calibration_quantile(np.array([1.0]), 0.99)


def test_transaction_costs_include_entry_and_terminal_liquidation():
    result = portfolio(np.array([[0.0]]), np.array([[0.5]]), 100)
    expected_wealth = 0.995 / 1.005
    assert result["net"][0] == pytest.approx(expected_wealth - 1)
    assert result["cost"][0] == pytest.approx(1 - expected_wealth)
    assert result["turnover"][0] * 0.01 == pytest.approx(result["cost"][0])


def test_rebalance_charges_drift_even_without_signal_changes():
    returns = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    weights = np.full_like(returns, 0.5)
    result = portfolio(returns, weights, 0)
    assert result["turnover"][1] == pytest.approx(1 / 3)
    assert result["net"][0] == pytest.approx(0.5)


def test_cash_and_cost_monotonicity():
    rng = np.random.default_rng(12)
    returns = rng.normal(0, 0.02, (50, 2))
    weights = rng.integers(0, 2, (50, 2)) / 2
    values = [np.prod(1 + portfolio(returns, weights, c)["net"]) for c in [0, 15, 30]]
    assert values[0] > values[1] > values[2]
    cash = portfolio(returns, np.zeros_like(weights), 30)
    assert np.all(cash["net"] == 0) and np.all(cash["cost"] == 0)
    assert performance(cash, np.zeros_like(weights))["sharpe"] is None


def test_holm_matches_hand_calculation_and_keeps_order():
    np.testing.assert_allclose(holm([0.04, 0.001, 0.03, 0.2]), [0.09, 0.004, 0.09, 0.2])
    with pytest.raises(ValueError):
        holm([np.nan])


def test_block_bootstrap_null_direction_and_reproducibility():
    values = np.column_stack([np.zeros(180), np.ones(180) * 0.1, -np.ones(180) * 0.1])
    first = block_inference(values, ["null", "better", "worse"], 30, 999, 17)
    assert first == block_inference(values, ["null", "better", "worse"], 30, 999, 17)
    assert first[0]["p_holm"] == 1
    assert first[1]["p_holm"] <= 0.05
    assert first[2]["p_holm"] == 1
    assert first[1]["ci95_low"] == pytest.approx(0.1)


def test_serial_dependence_increases_block_uncertainty():
    values = np.repeat(np.array([-1.0, 1.0] * 6), 30)[:, None]
    short = block_inference(values, ["persistent"], 1, 999, 19)[0]
    long = block_inference(values, ["persistent"], 15, 999, 19)[0]
    assert long["ci95_high"] - long["ci95_low"] > 2 * (short["ci95_high"] - short["ci95_low"])


def test_replay_detects_changed_metrics_and_verdict():
    expected = {"verdict": "NOT SUPPORTED", "mse": 0.01}
    verify_replay(copy.deepcopy(expected), expected)
    with pytest.raises(ValueError):
        verify_replay({**expected, "mse": 0.02}, expected)
    with pytest.raises(ValueError):
        verify_replay({**expected, "verdict": "SUPPORTED"}, expected)


def test_selection_replay_checks_integrity_and_compares_every_trial(tmp_path):
    verify_selection_replay = runpy.run_path(str(ROOT / "scripts/run_financial_holdout.py"))[
        "verify_selection_replay"
    ]
    study = ROOT / "experiments/financial_holdout"
    expected = study / "selection"
    actual = tmp_path / "selection"
    shutil.copytree(expected, actual)
    trials_path = actual / "trials.json"
    lock_path = actual / "selection.json"
    trials = json.loads(trials_path.read_text())
    lock = json.loads(lock_path.read_text())

    trials[0]["mse"] += 1e-15
    trials_path.write_text(json.dumps(trials))
    with pytest.raises(ValueError, match="Selection lock mismatch: trials_sha256"):
        verify_selection_replay(ROOT, study, actual, expected)

    lock["trials_sha256"] = digest(trials_path)
    lock_path.write_text(json.dumps(lock))
    (actual / "selection.sha256").write_text(digest(lock_path) + "\n")
    verify_selection_replay(ROOT, study, actual, expected)

    trials[0]["mse"] += 0.01
    trials_path.write_text(json.dumps(trials))
    lock["trials_sha256"] = digest(trials_path)
    lock_path.write_text(json.dumps(lock))
    (actual / "selection.sha256").write_text(digest(lock_path) + "\n")
    with pytest.raises(ValueError, match=r"trials\[0\]\.mse"):
        verify_selection_replay(ROOT, study, actual, expected)


def test_existing_run_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError):
        new_output(tmp_path)


def test_complete_negative_study_retains_all_models_costs_and_nulls(protocol):
    from drr_framework.finance.holdout.study import model_names, render_report

    protocol["inference"]["bootstrap_replicates"] = 99
    rng = np.random.default_rng(3)
    dates = pd.date_range("2023-01-01", periods=120)
    target = rng.normal(0, 0.01, (120, 2))
    records = []
    for model in model_names(protocol):
        for i, date in enumerate(dates):
            for j, symbol in enumerate(protocol["data"]["symbols"]):
                records.append(
                    {
                        "model": model,
                        "entry_date": date,
                        "symbol": symbol,
                        "prediction": 0.0,
                        "target": target[i, j],
                        "lower": -0.02,
                        "upper": 0.02,
                        "year": 2023,
                        "regime": ["low_vol_up", "high_vol_up", "low_vol_down", "high_vol_down"][
                            i % 4
                        ],
                    }
                )
    lock = {
        "primary_candidate": "hgb_drr",
        "economic_comparator": "zero",
        "high_volatility_threshold": 0.01,
    }
    summary, paths = summarize(pd.DataFrame(records), protocol, lock)
    assert summary["verdict"] == "NOT SUPPORTED"
    assert len(summary["tests"]) == 42
    assert len(summary["economic_metrics"]) == 36
    assert set(paths.one_way_bps) == {0, 15, 30}
    assert set(summary["forecast_metrics"]) == set(model_names(protocol))
    assert "NOT SUPPORTED" in render_report(summary, lock)
    assert all(test["p_holm"] == 1 for test in summary["tests"])


def test_published_artifacts_have_intact_hashes_and_failed_gates_are_published():
    result = ROOT / "results/expected/financial_holdout"
    if not result.exists():
        pytest.skip("Evidence has not been generated yet")
    from drr_framework.finance.holdout.data import digest
    from drr_framework.finance.holdout.study import verify_lock

    study = ROOT / "experiments/financial_holdout"
    lock = verify_lock(ROOT, study, study / "selection")
    provenance = json.loads((result / "provenance.json").read_text())
    assert provenance["selection_sha256"] == digest(study / "selection/selection.json")
    assert provenance["source_sha256"] == source_hashes(ROOT)
    for name, expected in provenance["artifacts"].items():
        assert digest(result / name) == expected
    summary = json.loads((result / "summary.json").read_text())
    assert len(summary["tests"]) == 42
    assert summary["test_days"] == 1096
    expected_verdict = "SUPPORTED" if all(summary["gates"].values()) else "NOT SUPPORTED"
    assert summary["verdict"] == expected_verdict
    assert summary["primary_candidate"] == lock["primary_candidate"]
    predictions = pd.read_csv(result / "predictions.csv.gz")
    assert len(predictions) == 1096 * 2 * 10
    assert set(predictions.model) == set(summary["forecast_metrics"])
    assert summary["verdict"] in (result / "report.md").read_text()


def test_published_forecasts_independently_reconcile_to_raw_prices_and_timestamps(protocol):
    result = ROOT / "results/expected/financial_holdout"
    if not (result / "predictions.csv.gz").exists():
        pytest.skip("Evidence has not been generated yet")
    bars = pd.read_csv(ROOT / "experiments/financial_holdout/data/bars.csv", parse_dates=["date"])
    prices = bars.set_index(["date", "symbol"]).open
    rows = pd.read_csv(
        result / "predictions.csv.gz",
        parse_dates=[
            "feature_date",
            "entry_date",
            "exit_date",
            "fit_last_exit",
            "calibration_first_feature",
            "calibration_last_exit",
        ],
    )
    summary = json.loads((result / "summary.json").read_text())
    assert not rows.duplicated(["model", "entry_date", "symbol"]).any()
    entry_prices = prices.reindex(
        pd.MultiIndex.from_arrays([rows.entry_date, rows.symbol])
    ).to_numpy()
    exit_prices = prices.reindex(
        pd.MultiIndex.from_arrays([rows.exit_date, rows.symbol])
    ).to_numpy()
    np.testing.assert_allclose(rows.target, exit_prices / entry_prices - 1, rtol=1e-10, atol=1e-14)
    assert (rows.entry_date - rows.feature_date == pd.Timedelta(days=2)).all()
    assert (rows.exit_date - rows.entry_date == pd.Timedelta(days=1)).all()
    assert (rows.fit_last_exit < rows.calibration_first_feature - pd.Timedelta(days=7)).all()
    assert (rows.calibration_last_exit < rows.feature_date - pd.Timedelta(days=7)).all()
    assert (rows.loc[rows.model == "zero", "prediction"] == 0).all()
    for model, group in rows.groupby("model"):
        squared_errors = (group.prediction - group.target) ** 2
        observed_coverage = ((group.lower <= group.target) & (group.target <= group.upper)).mean()
        assert squared_errors.mean() == pytest.approx(
            summary["forecast_metrics"][model]["mse"], abs=1e-14
        )
        assert observed_coverage == pytest.approx(summary["forecast_metrics"][model]["coverage"])
    selection = json.loads(
        (ROOT / "experiments/financial_holdout/selection/selection.json").read_text()
    )
    trials = json.loads((ROOT / "experiments/financial_holdout/selection/trials.json").read_text())
    for model, params in selection["selected_parameters"].items():
        best = min(
            (trial for trial in trials if trial["model"] == model), key=lambda trial: trial["mse"]
        )
        assert params == best["params"]
    assert pd.Timestamp(selection["last_validation_entry"]) + pd.Timedelta(days=1) < pd.Timestamp(
        selection["selection_label_cutoff_exclusive"]
    )
