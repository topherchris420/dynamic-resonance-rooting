"""Purged, quarter-by-quarter prediction with disjoint residual calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import feature_columns


def configurations(model: str, protocol: dict) -> list:
    if model.startswith("ridge"):
        return [{"alpha": alpha} for alpha in protocol["models"]["ridge_alpha"]]
    if model.startswith("hgb"):
        return protocol["models"]["hgb_grid"]
    return [{}]


def split_training(index: pd.DatetimeIndex, first_feature: pd.Timestamp, protocol: dict) -> tuple:
    spec = protocol["timing"]
    gap = pd.Timedelta(days=spec["label_gap_days"])
    exits = index + pd.Timedelta(days=spec["exit_lag_days"])
    eligible = index[exits < first_feature - gap]
    n = spec["calibration_rows"]
    if len(eligible) < n + spec["minimum_fit_rows"]:
        raise ValueError("Insufficient mature training/calibration history")
    calibration = eligible[-n:]
    fit = index[exits < calibration[0] - gap][-spec["fit_window_days"] :]
    if len(fit) < spec["minimum_fit_rows"]:
        raise ValueError("Insufficient fit history after purging")
    return fit, calibration


def calibration_quantile(residuals: np.ndarray, level: float) -> float:
    values = np.asarray(residuals, float)
    if len(values) < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid calibration residuals")
    rank = int(np.ceil((len(values) + 1) * level))
    if not 1 <= rank <= len(values):
        raise ValueError("Too few residuals for the requested interval")
    return float(np.sort(values)[rank - 1])


def walk_forward(features, targets, protocol, model, params, start, end):
    entry_lag = pd.Timedelta(days=protocol["timing"]["entry_lag_days"])
    exit_lag = pd.Timedelta(days=protocol["timing"]["exit_lag_days"])
    entry_dates = targets.index + entry_lag
    evaluation = targets.index[
        (entry_dates >= pd.Timestamp(start)) & (entry_dates <= pd.Timestamp(end))
    ]
    if evaluation.empty:
        raise ValueError("Empty evaluation period")
    quarters = (evaluation + entry_lag).to_period("Q")
    columns = feature_columns(features, model)
    records = []
    for quarter in quarters.unique():
        prediction_dates = evaluation[quarters == quarter]
        fit, calibration = split_training(targets.index, prediction_dates[0], protocol)
        for symbol in protocol["data"]["symbols"]:

            def predict_simple(dates, fit=fit, symbol=symbol):
                if model == "zero":
                    return np.zeros(len(dates))
                if model == "historical_mean":
                    return np.full(len(dates), targets.loc[fit, symbol].mean())
                return features.loc[dates, f"{symbol}_mean20"].to_numpy()

            if model in {"zero", "historical_mean", "momentum20"}:
                cal_pred, predicted = predict_simple(calibration), predict_simple(prediction_dates)
            else:
                estimator = (
                    make_pipeline(StandardScaler(), Ridge(**params))
                    if model.startswith("ridge")
                    else HistGradientBoostingRegressor(**protocol["models"]["hgb_fixed"], **params)
                )
                with threadpool_limits(limits=1):
                    estimator.fit(
                        features.loc[fit, columns].to_numpy(), targets.loc[fit, symbol].to_numpy()
                    )
                    cal_pred = estimator.predict(features.loc[calibration, columns].to_numpy())
                    predicted = estimator.predict(
                        features.loc[prediction_dates, columns].to_numpy()
                    )
            scale = features.loc[calibration, f"{symbol}_vol20"].to_numpy().clip(0.0001)
            residuals = np.abs(targets.loc[calibration, symbol].to_numpy() - cal_pred) / scale
            q = calibration_quantile(residuals, protocol["uncertainty"]["interval_level"])
            width = q * features.loc[prediction_dates, f"{symbol}_vol20"].to_numpy().clip(0.0001)
            for i, date in enumerate(prediction_dates):
                records.append(
                    {
                        "feature_date": date,
                        "entry_date": date + entry_lag,
                        "exit_date": date + exit_lag,
                        "symbol": symbol,
                        "model": model,
                        "prediction": float(predicted[i]),
                        "target": float(targets.loc[date, symbol]),
                        "lower": float(predicted[i] - width[i]),
                        "upper": float(predicted[i] + width[i]),
                        "fit_first_feature": fit[0],
                        "fit_last_exit": fit[-1] + exit_lag,
                        "calibration_first_feature": calibration[0],
                        "calibration_last_exit": calibration[-1] + exit_lag,
                        "fit_rows": len(fit),
                        "calibration_rows": len(calibration),
                    }
                )
    return pd.DataFrame(records).sort_values(["entry_date", "symbol"]).reset_index(drop=True)
