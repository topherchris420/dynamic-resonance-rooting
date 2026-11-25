"""One call that runs DRR's experimental mathematical features and records the evidence.

:func:`analyze_mathematical_features` is the opt-in entry point behind
``DynamicResonanceRooting.analyze_system(..., mathematical_features=True)``. It

1. regularizes missing values or irregular timestamps (recorded, never silent);
2. runs the data-quality checks of :mod:`drr_framework.data_quality`;
3. tests every channel pair for ``n:m`` phase synchronization with
   :class:`drr_framework.phase_sync.PhaseSynchronyAnalyzer`;
4. runs the same test on phase-randomized copies of the data, which keep each
   channel's spectrum and destroy every relationship, as negative controls;
5. describes each channel with recurrence quantification and a second-order
   linear recurrence, and the trajectory with oriented areas;
6. assembles a :class:`drr_framework.math_evidence.MathematicalEvidenceRecord`.

Nothing here changes DRR's existing outputs. The features are experimental:
their measured behavior is in ``results/expected/resonance_math_benchmark.json``,
and a result on real data needs its own validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .data_quality import assess_series_quality, regularize_series
from .math_evidence import array_sha256, binomial_upper_tail, build_evidence_record
from .modules import phase_randomized_surrogate
from .phase_sync import PhaseSynchronyAnalyzer
from .recurrence import recurrence_quantification, second_order_recurrence
from .trajectory_geometry import cyclic_order, oriented_area_matrix

MATH_FEATURES_VERSION = "drr_math_features_v1"


def _negative_controls(
    values: np.ndarray,
    analyzer: PhaseSynchronyAnalyzer,
    names: Sequence[str],
    n_controls: int,
    random_state: int,
) -> Dict[str, Any]:
    """Phase-randomized copies of the data, each tested like the original.

    Under a calibrated test each copy has at most an ``alpha`` chance of any
    synchronized pair. The controls are flagged when the count of such copies
    has a binomial upper-tail probability below 0.01.
    """
    if n_controls <= 0:
        return {"n_controls": 0, "flagged": False, "method": "phase_randomization"}
    rng = np.random.default_rng(random_state)
    declared = 0
    for _ in range(n_controls):
        surrogate = phase_randomized_surrogate(values, rng)
        result = analyzer.analyze(surrogate, names)
        declared += any(pair["status"] == "synchronized" for pair in result["pairs"])
    tail = binomial_upper_tail(declared, n_controls, analyzer.alpha)
    return {
        "method": "phase_randomization",
        "n_controls": int(n_controls),
        "random_state": int(random_state),
        "controls_with_any_synchronized_pair": int(declared),
        "expected_at_most": float(analyzer.alpha * n_controls),
        "binomial_upper_tail": tail,
        "flagged": bool(tail < 0.01),
    }


def _characterize(
    values: np.ndarray,
    sampling_rate: float,
    names: Sequence[str],
    quality: Mapping[str, Any],
    recurrence_points: int,
) -> Dict[str, Any]:
    """Per-channel recurrence and linear-recurrence descriptions."""
    out: Dict[str, Any] = {}
    for name, column in zip(names, values.T):
        record: Dict[str, Any] = {
            "second_order_recurrence": second_order_recurrence(column, sampling_rate)
        }
        frequency = quality["channels"].get(name, {}).get("dominant_frequency_hz")
        window = column[-recurrence_points:] if recurrence_points > 0 else column
        try:
            record["recurrence"] = recurrence_quantification(
                window,
                sampling_rate=sampling_rate,
                frequency=frequency if frequency and frequency > 0 else None,
                max_points=max(recurrence_points, 64),
            )
            record["recurrence"]["analysed_samples"] = int(window.size)
        except ValueError as exc:
            record["recurrence"] = {"error": str(exc)}
        out[name] = record
    return out


def analyze_mathematical_features(
    data: Any,
    sampling_rate: float,
    *,
    column_names: Optional[Sequence[str]] = None,
    timestamps: Optional[Any] = None,
    phase_options: Optional[Mapping[str, Any]] = None,
    quality_thresholds: Optional[Mapping[str, float]] = None,
    negative_controls: int = 19,
    recurrence_points: int = 1000,
    random_state: int = 0,
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the experimental features on ``data`` (``(time,)`` or ``(time, channels)``).

    Missing values may be ``NaN``; with ``timestamps`` (seconds) the rows may be
    irregularly spaced. Either way the series is interpolated onto a regular
    grid at ``sampling_rate`` and the interpolation is recorded and checked.
    ``phase_options`` are passed to :class:`PhaseSynchronyAnalyzer`.
    ``recurrence_points`` bounds the trailing window used for recurrence
    quantification, whose memory grows quadratically. The result is
    deterministic for a fixed ``random_state``.
    """
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    raw = np.asarray(data, dtype=float)
    values = raw[:, None] if raw.ndim == 1 else raw
    if values.ndim != 2:
        raise ValueError("data must have shape (time,) or (time, channels)")
    names: List[str] = (
        [str(name) for name in column_names]
        if column_names is not None
        else [f"dim_{index}" for index in range(values.shape[1])]
    )
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("column_names must be unique and match the number of channels")
    input_sha256 = array_sha256(
        values if timestamps is None else np.column_stack([timestamps, values])
    )

    regularization = None
    if timestamps is not None or not np.all(np.isfinite(values)):
        values, regularization = regularize_series(values, timestamps, sampling_rate)
        values = np.asarray(values, dtype=float).reshape(-1, len(names))

    quality = assess_series_quality(
        values,
        sampling_rate,
        column_names=names,
        regularization=regularization,
        thresholds=dict(quality_thresholds) if quality_thresholds else None,
    )
    options = dict(phase_options or {})
    analyzer = PhaseSynchronyAnalyzer(sampling_rate, **options)
    phase = None
    geometry = None
    controls: Dict[str, Any] = {"n_controls": 0, "flagged": False, "method": "phase_randomization"}
    if values.shape[1] >= 2:
        phase = analyzer.analyze(values, names)
        area = oriented_area_matrix(values)
        geometry = {
            "oriented_area": area.tolist(),
            "cyclic_order": cyclic_order(area),
            "note": (
                "Oriented areas are a function of lag-one cross-covariance and add no information "
                "beyond second-order statistics; see drr_framework.trajectory_geometry."
            ),
        }
        controls = _negative_controls(values, analyzer, names, negative_controls, random_state)

    characterization = _characterize(values, sampling_rate, names, quality, recurrence_points)
    configuration = {
        "version": MATH_FEATURES_VERSION,
        "sampling_rate": float(sampling_rate),
        "phase": analyzer.parameters(),
        "quality_thresholds": quality["thresholds"],
        "negative_controls": int(negative_controls),
        "recurrence_points": int(recurrence_points),
        "random_state": int(random_state),
    }
    measurements: Dict[str, Any] = {
        "pairs": phase["pairs"] if phase is not None else [],
        "dfa_alpha": {
            name: (record.get("dfa") or {}).get("alpha")
            for name, record in quality["channels"].items()
        },
        "dominant_frequency_hz": {
            name: record.get("dominant_frequency_hz")
            for name, record in quality["channels"].items()
        },
    }
    record = build_evidence_record(
        analysed_data=values,
        sampling_rate=sampling_rate,
        column_names=names,
        configuration=configuration,
        quality=quality,
        phase=phase,
        measurements=measurements,
        negative_controls=controls,
        input_sha256=input_sha256,
        data_source=data_source,
    )
    return {
        "method": MATH_FEATURES_VERSION,
        "experimental": True,
        "data_quality": quality,
        "phase_synchrony": phase,
        "negative_controls": controls,
        "trajectory_geometry": geometry,
        "characterization": characterization,
        "evidence_record": record.to_dict(),
    }


__all__ = ["MATH_FEATURES_VERSION", "analyze_mathematical_features"]
