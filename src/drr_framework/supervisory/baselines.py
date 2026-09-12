"""Simple, trailing statistical competitors and matched event evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .change_detection import percentile, robust_z


@dataclass(frozen=True)
class BaselineResult:
    metric: str
    method: str
    statistic: Optional[float]
    threshold: Optional[float]
    flagged: bool
    direction: int
    training_count: int
    formula: str
    limitation: Optional[str] = None


@dataclass(frozen=True)
class BaselineConfig:
    lookback: int = 12
    min_history: int = 8
    z_threshold: float = 3.0
    ewma_alpha: float = 0.3
    cusum_allowance: float = 0.5
    cusum_threshold: float = 5.0
    percentile_tail: float = 2.5

    def __post_init__(self):
        if (
            self.min_history < 4
            or self.lookback < self.min_history
            or self.z_threshold <= 0
            or not 0 < self.ewma_alpha <= 1
            or self.cusum_allowance < 0
            or self.cusum_threshold <= 0
            or not 0 < self.percentile_tail < 50
        ):
            raise ValueError("Invalid baseline configuration")


def run_baselines(dataset, config=None):
    config = config or BaselineConfig()
    results = []
    for metric in dataset.variable_names:
        values = dataset.frame[metric].to_numpy(dtype=float)[-config.lookback - 1 :]
        history, current = values[:-1], values[-1]
        complete = len(history) >= config.min_history and np.isfinite(values).all()
        if not complete:
            results.append(
                BaselineResult(
                    metric,
                    "availability",
                    None,
                    None,
                    False,
                    0,
                    len(history),
                    "finite trailing observations",
                    "Insufficient complete history; no zero filling or time compression",
                )
            )
            continue
        center, scale = float(history.mean()), float(history.std(ddof=1))
        direction = int(np.sign(current - center))

        def add(
            method,
            score,
            threshold,
            flag,
            formula,
            limitation=None,
            *,
            metric=metric,
            direction=direction,
            history=history,
        ):
            results.append(
                BaselineResult(
                    metric,
                    method,
                    score,
                    threshold,
                    bool(flag),
                    direction,
                    len(history),
                    formula,
                    limitation,
                )
            )

        z = robust_z(current, history)
        add(
            "robust_z",
            z,
            config.z_threshold,
            z is not None and abs(z) >= config.z_threshold,
            "(current - median(prior)) / (1.4826 * MAD(prior))",
            "Zero MAD; standardized deviation undefined" if z is None else None,
        )
        pct = percentile(current, history)
        add(
            "rolling_percentile",
            pct,
            config.percentile_tail,
            pct <= config.percentile_tail or pct >= 100 - config.percentile_tail,
            "midrank of current within prior observations",
            "Coarse ranks on short quarterly histories",
        )
        mean = float(history[0])
        for value in history[1:]:
            mean = config.ewma_alpha * value + (1 - config.ewma_alpha) * mean
        residual = (current - mean) / scale if scale > 0 else None
        add(
            "ewma",
            residual,
            config.z_threshold,
            residual is not None and abs(residual) >= config.z_threshold,
            "(current - EWMA(prior)) / sd(prior)",
        )
        reference = history[: config.min_history]
        cmean, cscale = reference.mean(), reference.std(ddof=1)
        pos = neg = 0.0
        if cscale > 0:
            for value in values[config.min_history :]:
                innovation = (value - cmean) / cscale
                pos = max(0.0, pos + innovation - config.cusum_allowance)
                neg = min(0.0, neg + innovation + config.cusum_allowance)
        cusum = max(pos, -neg) if cscale > 0 else None
        add(
            "cusum",
            cusum,
            config.cusum_threshold,
            cusum is not None and cusum >= config.cusum_threshold,
            "two-sided CUSUM against initial training mean/sd",
        )
        volatility = float(np.std(values[-4:], ddof=1) / scale) if scale > 0 else None
        add(
            "rolling_volatility",
            volatility,
            2.0,
            volatility is not None and volatility >= 2,
            "sd(last 4) / sd(prior history)",
        )
        split = len(history) // 2
        difference = (
            (np.mean(values[-split:]) - np.mean(history[:split])) / scale if scale > 0 else None
        )
        add(
            "mean_shift",
            float(difference) if difference is not None else None,
            config.z_threshold,
            difference is not None and abs(difference) >= config.z_threshold,
            "difference of trailing and initial subwindow means / sd(prior)",
            "Heuristic break candidate; no calibrated change-point probability",
        )
        design = np.column_stack((np.ones(len(history) - 1), history[:-1]))
        beta = np.linalg.lstsq(design, history[1:], rcond=None)[0]
        errors = history[1:] - design @ beta
        err_scale = float(np.std(errors, ddof=1))
        ar = (
            float((current - np.array([1, history[-1]]) @ beta) / err_scale)
            if err_scale > 1e-12 * max(abs(center), 1)
            else None
        )
        add(
            "autoregression",
            ar,
            config.z_threshold,
            ar is not None and abs(ar) >= config.z_threshold,
            "one-step AR(1) prediction residual / training residual sd",
            "Undefined when training residual variance is zero" if ar is None else None,
        )
    return tuple(results)


def lagged_correlations(dataset, max_lag=2):
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    rows = []
    for source in dataset.variable_names:
        for target in dataset.variable_names:
            if source == target:
                continue
            for lag in range(1, max_lag + 1):
                x, y = (
                    dataset.frame[source].to_numpy()[:-lag],
                    dataset.frame[target].to_numpy()[lag:],
                )
                mask = np.isfinite(x) & np.isfinite(y)
                value = (
                    float(np.corrcoef(x[mask], y[mask])[0, 1])
                    if mask.sum() >= 4 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0
                    else None
                )
                rows.append(
                    dict(
                        source=source,
                        target=target,
                        lag=lag,
                        correlation=value,
                        pair_count=int(mask.sum()),
                    )
                )
    return tuple(rows)


def matched_evaluation(rows, *, events=None, lead_window=2):
    """Compare identical review dates. Labels are supplied separately from model outputs.

    Unlabeled periods are not declared true negatives. With no independent event
    inventory, only alert burden and incremental alerts are estimable.
    """
    if lead_window < 0:
        raise ValueError("lead_window must be nonnegative")
    rows = tuple(rows)
    dates = [row["as_of"] for row in rows]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("Matched evaluation dates must be unique and increasing")
    baseline = np.asarray([bool(row["baseline_alert"]) for row in rows])
    drr = np.asarray([bool(row["drr_alert"]) for row in rows])
    combined = baseline | drr
    result = {
        "review_count": len(rows),
        "baseline_alert_count": int(baseline.sum()),
        "combined_alert_count": int(combined.sum()),
        "incremental_alert_count": int((drr & ~baseline).sum()),
        "incremental_detection_rate": None,
        "conclusion": "Incremental usefulness is unestablished without independent outcomes or analyst feedback.",
    }
    if events is None:
        result["outcomes_status"] = "not_estimable_no_independent_event_inventory"
        return result
    from ..validation_readiness import run_event_backtest

    if not rows:
        result["outcomes_status"] = "no_reviews"
        return result
    event_set = set(events)
    if not event_set <= set(dates):
        raise ValueError("Event labels must map to predefined review dates")
    # Right-censored alerts cannot count as false positives without a full future horizon.
    for label, flags in (("baseline_only", baseline), ("baseline_plus_drr", combined)):
        frame = pd.DataFrame(
            {"date": dates, "alert": flags.astype(float), "event": [d in event_set for d in dates]}
        )
        if lead_window:
            frame.loc[frame.index[-lead_window:], "alert"] = 0.0
        metrics = run_event_backtest(
            frame,
            date_column="date",
            score_column="alert",
            event_column="event",
            threshold=0.5,
            lead_window=lead_window,
        )
        eligible = max(0, len(rows) - lead_window)
        eligible_negative = sum(
            not any(dates[j] in event_set for j in range(i, min(i + lead_window + 1, len(rows))))
            for i in range(eligible)
        )
        metrics["false_positive_rate"] = (
            metrics["false_positive_alerts"] / eligible_negative if eligible_negative else None
        )
        metrics["right_censored_reviews"] = min(lead_window, len(rows))
        result[label] = metrics
    result["incremental_detection_rate"] = (
        result["baseline_plus_drr"]["recall"] - result["baseline_only"]["recall"]
        if events
        else None
    )
    result["outcomes_status"] = "evaluated_against_caller_supplied_event_inventory"
    result["conclusion"] = (
        "Compare detection gains with added alert burden; these metrics do not establish supervisory usefulness."
    )
    return result
