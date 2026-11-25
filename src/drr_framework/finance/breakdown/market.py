"""Separate external validation without pretending real break times are known."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..holdout.data import digest
from ..holdout.statistics import portfolio
from .detectors import score_stream, threshold


def load_market(directory, protocol):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "bars.csv.gz"
    if digest(path) != manifest["sha256"]:
        raise ValueError("Market checksum mismatch")
    frame = pd.read_csv(path, parse_dates=["time"])
    if len(frame) != manifest["rows"] or frame.duplicated(["time", "symbol"]).any():
        raise ValueError("Invalid market row count or duplicate")
    symbols = protocol["market"]["symbols"]
    if set(frame.symbol) != set(symbols):
        raise ValueError("Unexpected market universe")
    dates = pd.date_range(protocol["market"]["months"][0], periods=1)[0]
    end = pd.Timestamp(protocol["market"]["months"][-1]) + pd.offsets.MonthBegin(1)
    expected = pd.date_range(dates, end - pd.Timedelta(hours=1), freq="h")
    for symbol in symbols:
        if not pd.DatetimeIndex(frame.loc[frame.symbol == symbol, "time"]).equals(expected):
            raise ValueError("Missing, unsorted or unexpected hourly bars")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Invalid market values")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (
        frame.low > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Impossible OHLC range")
    return frame


def market_scores(bars, protocol, months):
    symbols = protocol["market"]["symbols"]
    # Truncate before computing any returns for calibration-only execution.
    end = pd.Timestamp(months[-1]) + pd.offsets.MonthBegin(1)
    close = bars.loc[bars.time < end].pivot(index="time", columns="symbol", values="close")[symbols]
    returns = np.log(close / close.shift(1)).iloc[1:]
    rows = []
    for month in months:
        start = pd.Timestamp(month)
        stop = start + pd.offsets.MonthBegin(1)
        context = returns.loc[returns.index < start].tail(protocol["detector"]["fit"])
        current = returns.loc[(returns.index >= start) & (returns.index < stop)]
        if len(context) != protocol["detector"]["fit"]:
            raise ValueError("Insufficient prior-month fit history")
        data = pd.concat([context, current])
        times, scores, errors, zero = score_stream(data.to_numpy(), protocol)
        for j, t in enumerate(times):
            complete = t + 24 < len(data)
            target = (
                float(
                    np.sum(errors[t + 1 : t + 25] ** 2)
                    / max(np.sum(zero[t + 1 : t + 25] ** 2), 1e-12)
                )
                if complete
                else None
            )
            rows.append(
                {
                    "time": data.index[t],
                    "month": month,
                    "fit_last_time": context.index[-1],
                    "target_exit_time": data.index[t + 24] if complete else pd.NaT,
                    "future_error_ratio": target,
                    **{name: float(values[j]) for name, values in scores.items()},
                }
            )
    return pd.DataFrame(rows)


def warning_thresholds(records, protocol):
    mature = records.loc[records.future_error_ratio.notna()]
    return {name: threshold(mature[name].to_numpy()) for name in protocol["methods"]}


def conditional_effect(warning, target):
    if np.sum(warning) == 0 or np.sum(~warning) == 0:
        return None
    return float(target[warning].mean() - target[~warning].mean())


def market_effects(records, cutoffs, protocol):
    rows = records.loc[records.future_error_ratio.notna()]
    y = rows.future_error_ratio.to_numpy()
    rng = np.random.default_rng(9183)
    n, block = len(y), 21
    starts = rng.integers(0, n, size=(1999, int(np.ceil(n / block))))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(1999, -1)[:, :n]
    results = []
    for name in protocol["methods"]:
        warning = rows[name].to_numpy() > cutoffs[name]
        effects = [conditional_effect(warning[index], y[index]) for index in indices]
        valid = np.array([v for v in effects if v is not None])
        eligible = min(int(warning.sum()), int((~warning).sum())) >= 20 and len(
            valid
        ) >= 0.95 * len(effects)
        results.append(
            {
                "method": name,
                "eligible_timestamps": n,
                "warnings": int(warning.sum()),
                "warning_rate": float(warning.mean()),
                "future_error_ratio_difference": conditional_effect(warning, y),
                "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if eligible else None,
                "inference_status": (
                    "descriptive" if eligible else "INSUFFICIENT WARNINGS OR CONTROLS"
                ),
                "undefined_bootstrap_draws": len(effects) - len(valid),
            }
        )
    return results


def warning_weights(index, alarm_times):
    weights = np.full((len(index), 2), 0.5)
    for time in alarm_times:
        entry = pd.Timestamp(time) + pd.Timedelta(hours=2)
        weights[(index >= entry) & (index < entry + pd.Timedelta(hours=24))] = 0
    return weights


def risk_summary(path, weights):
    net = path["net"]
    equity = np.r_[1, np.cumprod(1 + net)]
    tail = np.sort(net)[: max(1, int(np.ceil(len(net) * 0.05)))]
    return {
        "total_return": float(equity[-1] - 1),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "worst_5pct_mean_hourly_loss": float(-tail.mean()),
        "mean_exposure": float(weights.sum(axis=1).mean()),
        "turnover": float(path["turnover"].sum()),
    }


def market_risk(bars, records, cutoffs, protocol):
    prices = bars.pivot(index="time", columns="symbol", values="open")[
        protocol["market"]["symbols"]
    ]
    returns = prices.shift(-1) / prices - 1
    start = pd.Timestamp(protocol["market"]["test_months"][0])
    # The final observed open has no next open: do not invent a terminal fill.
    returns = returns.loc[returns.index >= start].iloc[:-1]
    strategies = {"always_long": np.full(returns.shape, 0.5)}
    for name in protocol["methods"]:
        alarm_times = records.loc[records[name] > cutoffs[name], "time"]
        weights = warning_weights(returns.index, alarm_times)
        strategies[name] = weights
        strategies[name + "_matched_exposure"] = np.full_like(weights, weights.mean())
    summaries, ledgers = [], []
    for name, weights in strategies.items():
        for bps in [0, 15, 30]:
            path = portfolio(returns.to_numpy(), weights, bps)
            summaries.append({"strategy": name, "cost_bps": bps, **risk_summary(path, weights)})
            ledgers.append(
                pd.DataFrame(
                    {
                        "time": returns.index,
                        "strategy": name,
                        "cost_bps": bps,
                        "net_return": path["net"],
                        "turnover": path["turnover"],
                        "exposure": weights.sum(axis=1),
                    }
                )
            )
    return summaries, pd.concat(ledgers, ignore_index=True)
