"""Fixed public comparison on the NOAA CPC QBO zonal-wind index.

The preregistration names the dataset, the disruption windows, the conventional
models, the DRR ablations, the false-alarm tolerance, and the decision rule.
This module executes that plan and writes the artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

from ..modules import DepthCalculator, RootingAnalyzer

from .protocol import (
    evaluate_panel,
    json_ready,
    month_span,
    render_report,
    validate_preregistration,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
PREREGISTRATION_PATH = DATA_DIR / "preregistration.json"
MISSING = -999.90
_YEAR_LINE = re.compile(r"^(\d{4})\b(.*)$")
_NUMBER = re.compile(r"-999\.90|-?\d+\.\d+")


def load_preregistration() -> Dict:
    """Load the frozen plan and reject a snapshot whose checksum has changed."""
    spec = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    validate_preregistration(spec)
    for name, expected in spec["dataset"]["sha256"].items():
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{name} does not match the preregistered checksum")
    return spec


def preregistration_sha256() -> str:
    return hashlib.sha256(PREREGISTRATION_PATH.read_bytes()).hexdigest()


def parse_original_monthly(text: str) -> List[Tuple[int, int, Optional[float]]]:
    """Read the ORIGINAL DATA year-month block and stop at the next section."""
    start = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "ORIGINAL" in line and "DATA" in line:
            start = index
            break
    if start is None:
        raise ValueError("ORIGINAL DATA section not found")
    rows: List[Tuple[int, int, Optional[float]]] = []
    for line in lines[start + 1 :]:
        match = _YEAR_LINE.match(line.strip())
        if match is None:
            if rows:
                break
            continue
        year = int(match.group(1))
        values = [float(token) for token in _NUMBER.findall(match.group(2))]
        if len(values) != 12:
            raise ValueError(f"{year} did not contain 12 monthly values")
        for month, value in enumerate(values, start=1):
            observed = None if abs(value - MISSING) < 1e-6 else value
            rows.append((year, month, observed))
    if not rows:
        raise ValueError("ORIGINAL DATA section was empty")
    return rows


def load_qbo_series(spec: Optional[Dict] = None) -> Dict:
    """Align the two vendored levels and keep months where both are finite."""
    spec = spec or load_preregistration()
    u30 = {
        (year, month): value
        for year, month, value in parse_original_monthly(
            (DATA_DIR / "qbo.u30.index").read_text(encoding="utf-8")
        )
    }
    u50 = {
        (year, month): value
        for year, month, value in parse_original_monthly(
            (DATA_DIR / "qbo.u50.index").read_text(encoding="utf-8")
        )
    }
    keys = sorted(set(u30) & set(u50))
    finite = [key for key in keys if u30[key] is not None and u50[key] is not None]
    if not finite:
        raise ValueError("The QBO snapshots have no overlapping finite months")
    for previous, current in zip(finite, finite[1:]):
        if _month_index(*current) != _month_index(*previous) + 1:
            raise ValueError("The finite QBO record has an internal gap")
    months = [f"{year:04d}-{month:02d}" for year, month in finite]
    values = np.column_stack(
        (
            [u30[key] for key in finite],
            [u50[key] for key in finite],
        )
    ).astype(float)
    dataset = spec["dataset"]
    if months[0] != dataset["first_month"] or months[-1] != dataset["last_month"]:
        raise ValueError("The finite QBO span does not match the preregistration")
    if len(months) != int(dataset["finite_months"]):
        raise ValueError("The finite QBO length does not match the preregistration")
    return {"months": months, "values": values, "levels": ("u30", "u50")}


def regularized_logistic_scores(
    features: np.ndarray,
    labels: np.ndarray,
    estimation: np.ndarray,
    regularization_c: float,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Fit an estimation-only logistic arm, or withhold it when labels are absent."""
    design = np.asarray(features, dtype=float)
    response = np.asarray(labels, dtype=float)
    mask = np.asarray(estimation, dtype=bool)
    rows = design[mask]
    target = response[mask]
    usable = np.all(np.isfinite(rows), axis=1) & np.isfinite(target)
    if usable.sum() == 0 or float(np.sum(target[usable])) == 0.0:
        return None, "no positive labels in the estimation window"
    if float(np.sum(target[usable] == 0)) == 0.0:
        return None, "no negative labels in the estimation window"

    model = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(rows[usable], target[usable].astype(int))
    scores = np.full(len(design), np.nan)
    predictable = np.all(np.isfinite(design), axis=1)
    scores[predictable] = model.predict_proba(design[predictable])[:, 1]
    return scores, None


def compute_feature_frame(
    values: np.ndarray,
    months: Sequence[str],
    spec: Dict,
) -> Dict:
    """Causal monthly scores. A later month does not enter an earlier score."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 2 or series.shape[1] != 2:
        raise ValueError("QBO features expect two aligned levels")
    if len(months) != len(series):
        raise ValueError("Months and values must have the same length")
    analysis = spec["analysis"]
    window = int(analysis["trailing_window_months"])
    max_lag = int(analysis["rooting_max_lag"])
    holdout_start = analysis["holdout_start"]
    n_rows = len(series)
    if n_rows <= window:
        raise ValueError("The series is shorter than the trailing window")

    split = ["holdout" if month >= holdout_start else "estimation" for month in months]
    estimation = np.asarray([item == "estimation" for item in split], dtype=bool)
    origin = window
    volatility = np.full(n_rows, np.nan)
    absolute_correlation = np.full(n_rows, np.nan)
    spectral = np.full(n_rows, np.nan)
    dominant_lag = np.full(n_rows, np.nan)
    depth = DepthCalculator()
    rooting = RootingAnalyzer()

    for index in range(origin, n_rows):
        block = series[index - window + 1 : index + 1]
        volatility[index] = float(np.mean(np.std(block, axis=0, ddof=1)))
        if np.std(block[:, 0]) > 0 and np.std(block[:, 1]) > 0:
            absolute_correlation[index] = abs(float(np.corrcoef(block[:, 0], block[:, 1])[0, 1]))
        concentrations = []
        for column in range(block.shape[1]):
            detected = depth.calculate(
                block[:, column],
                window_size=window,
                sampling_rate=float(analysis["sampling_rate_per_month"]),
            )
            concentrations.append(detected["components"]["spectral_concentration"])
        spectral[index] = 1.0 - float(np.mean(concentrations))
        rooted = rooting.analyze(
            block,
            max_lag=max_lag,
            n_surrogates=0,
            method="lagged_correlation",
        )
        pair_scores = (rooted["score_matrix"][0, 1], rooted["score_matrix"][1, 0])
        pair_lags = (rooted["effective_lag"][0, 1], rooted["effective_lag"][1, 0])
        dominant_lag[index] = float(pair_lags[int(np.argmax(pair_scores))])

    scale_rows = series[estimation]
    center = np.mean(scale_rows, axis=0)
    spread = np.std(scale_rows, axis=0, ddof=1)
    spread = np.where(spread > 0, spread, 1.0)
    residual = _expanding_var_rms((series - center) / spread, min_history=window)

    estimation_scores = estimation.copy()
    estimation_scores[:origin] = False
    correlation_level = np.nanmedian(absolute_correlation[estimation_scores])
    correlation = np.full(n_rows, np.nan)
    finite_correlation = np.isfinite(absolute_correlation)
    correlation[finite_correlation] = np.maximum(
        0.0, correlation_level - absolute_correlation[finite_correlation]
    )
    lag_level = np.nanmedian(dominant_lag[estimation_scores])
    rooting_score = np.full(n_rows, np.nan)
    finite_lag = np.isfinite(dominant_lag)
    rooting_score[finite_lag] = np.abs(dominant_lag[finite_lag] - lag_level)
    spectral_standard = _standardize(spectral, estimation_scores)
    rooting_standard = _standardize(rooting_score, estimation_scores)
    both_finite = np.isfinite(spectral_standard) & np.isfinite(rooting_standard)
    combined = np.full(n_rows, np.nan)
    combined[both_finite] = np.maximum(
        spectral_standard[both_finite], rooting_standard[both_finite]
    )

    event_months = {
        event["id"]: set(month_span(event["start"], event["end"])) for event in spec["events"]
    }
    labels = np.asarray(
        [
            1.0 if any(month in windows for windows in event_months.values()) else 0.0
            for month in months
        ]
    )
    conventional = np.column_stack((volatility, correlation, residual))
    logistic, logistic_reason = regularized_logistic_scores(
        conventional,
        labels,
        estimation_scores,
        float(analysis["logistic_C"]),
    )
    scored_months = months[origin:]
    scored_split = split[origin:]

    def trimmed(array: np.ndarray) -> np.ndarray:
        return np.asarray(array[origin:], dtype=float)

    scores = {
        "rolling_volatility": trimmed(volatility),
        "rolling_correlation": trimmed(correlation),
        "var_residual": trimmed(residual),
        "drr_spectral": trimmed(spectral),
        "drr_rooting": trimmed(rooting_score),
        "drr_full": trimmed(combined),
    }
    if logistic is not None:
        scores["regularized_logistic"] = trimmed(logistic)
    withheld = {}
    if logistic is None:
        withheld["regularized_logistic"] = logistic_reason
    events = {
        event_id: [month in windows for month in scored_months]
        for event_id, windows in event_months.items()
    }
    return {
        "months": scored_months,
        "split": scored_split,
        "events": events,
        "scores": scores,
        "withheld": withheld,
        "anchors": {
            "correlation_estimation_median": float(correlation_level),
            "rooting_lag_estimation_median": float(lag_level),
        },
    }


def run_structural_change_benchmark() -> Dict:
    """Execute the frozen QBO plan and return the artifact."""
    spec = load_preregistration()
    series = load_qbo_series(spec)
    frame = compute_feature_frame(series["values"], series["months"], spec)
    evaluated = evaluate_panel(frame, spec)
    artifact = {
        "schema_version": "external-evidence-artifact-v1",
        "study_id": spec["id"],
        "preregistration_sha256": preregistration_sha256(),
        "dataset": spec["dataset"],
        "analysis": spec["analysis"],
        "events": spec["events"],
        "window_rule": spec["window_rule"],
        "estimation_rule": spec["estimation_rule"],
        "information_sets": spec["information_sets"],
        "decision_rule": spec["decision_rule"],
        "uncertainty_rule": spec["uncertainty_rule"],
        "anchors": frame["anchors"],
        "models": evaluated["models"],
        "unique_event_ids": evaluated["unique_event_ids"],
        "uncertainty": evaluated["uncertainty"],
        "claim": evaluated["claim"],
        "series": _series_table(frame),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    return json_ready(artifact)


def write_benchmark_artifact(artifact: Dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qbo_structural_change_benchmark.json"
    markdown_path = output_dir / "qbo_structural_change_benchmark.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_report(artifact), encoding="utf-8")
    return json_path, markdown_path


def _series_table(frame: Dict) -> List[Dict]:
    rows = []
    names = list(frame["scores"])
    for index, month in enumerate(frame["months"]):
        rows.append(
            {
                "month": month,
                "split": frame["split"][index],
                "events": [event_id for event_id, mask in frame["events"].items() if mask[index]],
                "scores": {name: _number(frame["scores"][name][index]) for name in names},
            }
        )
    return rows


def _number(value: float):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _standardize(values: np.ndarray, estimation: np.ndarray) -> np.ndarray:
    sample = values[estimation]
    sample = sample[np.isfinite(sample)]
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if sample.size < 2:
        return out
    center = float(np.mean(sample))
    spread = float(np.std(sample, ddof=1))
    if spread == 0.0:
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - center) / spread
    return out


def _expanding_var_rms(series: np.ndarray, min_history: int) -> np.ndarray:
    n_rows = len(series)
    scores = np.full(n_rows, np.nan)
    for index in range(min_history, n_rows):
        past = series[:index]
        design = np.column_stack((np.ones(len(past) - 1), past[:-1]))
        coefficient, _, _, _ = np.linalg.lstsq(design, past[1:], rcond=None)
        prediction = np.concatenate(([1.0], series[index - 1])) @ coefficient
        residual = series[index] - prediction
        scores[index] = float(np.sqrt(np.mean(residual**2)))
    return scores


def _month_index(year: int, month: int) -> int:
    return year * 12 + month - 1
