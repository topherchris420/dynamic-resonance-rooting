"""Fail-closed snapshot loading and strictly trailing feature construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...modules import DepthCalculator, RootingAnalyzer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bars(directory: Path, protocol: dict) -> pd.DataFrame:
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "bars.csv") != manifest["sha256"]:
        raise ValueError("Market data checksum mismatch")
    bars = pd.read_csv(directory / "bars.csv", parse_dates=["date"])
    if len(bars) != manifest["rows"]:
        raise ValueError("Market data row count mismatch")
    validate_bars(bars, protocol["data"]["symbols"])
    dates = pd.date_range(protocol["data"]["start"], protocol["data"]["end"])
    for symbol in protocol["data"]["symbols"]:
        if not pd.DatetimeIndex(bars.loc[bars.symbol == symbol, "date"]).equals(dates):
            raise ValueError("Data does not cover the exact registered daily calendar")
    return bars


def validate_bars(bars: pd.DataFrame, symbols: list) -> None:
    if set(bars.columns) != {"date", "symbol", "open", "high", "low", "close", "volume"}:
        raise ValueError("Unexpected bar schema")
    if set(bars.symbol.unique()) != set(symbols) or bars.duplicated(["date", "symbol"]).any():
        raise ValueError("Unexpected symbols or duplicate bars")
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Invalid or nonfinite OHLCV")
    if (bars.high < bars[["open", "close", "low"]].max(axis=1)).any() or (
        bars.low > bars[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC range")
    common = None
    for symbol in symbols:
        dates = pd.DatetimeIndex(bars.loc[bars.symbol == symbol, "date"])
        if dates.empty or not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
            raise ValueError("Missing, unsorted or non-daily bars")
        if common is not None and not dates.equals(common):
            raise ValueError("Asset calendars differ")
        common = dates


def build_features(bars: pd.DataFrame, protocol: dict) -> tuple:
    """No labels or fitted whole-sample transforms enter this function."""
    symbols = protocol["data"]["symbols"]
    validate_bars(bars, symbols)
    fields = {
        field: bars.pivot(index="date", columns="symbol", values=field)[symbols]
        for field in ["open", "high", "low", "close", "volume"]
    }
    returns = fields["close"].pct_change(fill_method=None)
    f = {}
    for symbol in symbols:
        r = returns[symbol]
        for lag in protocol["features"]["return_lags"]:
            f[f"{symbol}_lag{lag}"] = r.shift(lag)
        for window in protocol["features"]["windows"]:
            f[f"{symbol}_mean{window}"] = r.rolling(window).mean()
            f[f"{symbol}_vol{window}"] = r.rolling(window).std()
            f[f"{symbol}_mom{window}"] = fields["close"][symbol].pct_change(
                window, fill_method=None
            )
        f[f"{symbol}_range"] = np.log(fields["high"][symbol] / fields["low"][symbol])
        f[f"{symbol}_intraday"] = np.log(fields["close"][symbol] / fields["open"][symbol])
        volume = np.log1p(fields["volume"][symbol])
        f[f"{symbol}_volume"] = volume - volume.rolling(20).mean()
    for window in [20, 60]:
        f[f"corr{window}"] = returns.iloc[:, 0].rolling(window).corr(returns.iloc[:, 1])
    conventional = pd.DataFrame(f)
    rows = []
    window = protocol["features"]["drr_window"]
    depth, rooting = DepthCalculator(), RootingAnalyzer()
    for i in range(window, len(returns)):
        sample = returns.iloc[i - window + 1 : i + 1].to_numpy()
        row = {}
        for j, symbol in enumerate(symbols):
            result = depth.calculate(sample[:, j], window_size=32, sampling_rate=1.0)
            row[f"spectral_{symbol}_depth"] = result["resonance_depth"]
            row[f"spectral_{symbol}_frequency"] = result["target_frequency_hz"]
            for name, value in result["components"].items():
                row[f"spectral_{symbol}_{name}"] = value
        edges = rooting.analyze(sample, max_lag=3, n_surrogates=0, method="lagged_correlation")
        for source, target in [(0, 1), (1, 0)]:
            row[f"rooting_{source}_{target}_score"] = edges["score_matrix"][source, target]
            row[f"rooting_{source}_{target}_lag"] = edges["effective_lag"][source, target]
        rows.append(row)
    drr = pd.DataFrame(rows, index=returns.index[window:])
    features = conventional.loc[drr.index].join(drr)
    rng = np.random.default_rng(protocol["inference"]["seed"])
    noise = rng.normal(size=(len(features), 16))
    for i in range(16):
        features[f"noise_{i}"] = noise[:, i]
    if features.empty or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Nonfinite features: no dropping or imputation is permitted")
    entry = protocol["timing"]["entry_lag_days"]
    exit_lag = protocol["timing"]["exit_lag_days"]
    targets = fields["open"].shift(-exit_lag) / fields["open"].shift(-entry) - 1
    # The last exit_lag rows lack future outcomes, not usable market bars.
    targets = targets.loc[features.index[:-exit_lag]]
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError("Invalid target")
    return features, targets


def feature_columns(features: pd.DataFrame, model: str) -> list:
    conventional = [c for c in features if not c.startswith(("spectral_", "rooting_", "noise_"))]
    if model.endswith("_drr"):
        return conventional + [c for c in features if c.startswith(("spectral_", "rooting_"))]
    for group in ["spectral", "rooting", "noise"]:
        if model.endswith("_" + group):
            return conventional + [c for c in features if c.startswith(group + "_")]
    return conventional
