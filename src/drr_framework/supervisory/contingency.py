"""Supplementary binary metrology; separate from lead-window event detection.

All numerical metrics use the public NIST Contingent API. Labels are independently
ascertained institution-quarter outcomes, never inferred from an event inventory.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Optional, Tuple

import numpy as np

from .challenge_models import BinaryEventLabel, _integer, _labels_as_of, _quarter, _text
from .common import canonical_json, day, instant, stable_id


PACKAGE_NAME = "contingency-tools"
REVIEWED_VERSION = "0.2.3"
DESCRIPTION = "NIST Contingency-based supplementary detection metrology"
EVIDENCE_CLASS = "BINARY_CLASSIFICATION_METROLOGY"
_MODES = {"pre_specified", "tuning_derived", "retrospective_sensitivity"}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the finite floating-point range") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; use None for an unavailable score")
    return float(value)


@dataclass(frozen=True)
class ContingencyScore:
    """One score for an explicitly identified target quarter, in its original units.

    Higher scores predict the positive class. ``source_id`` binds to the supplied
    score artifact; its generation and absence of model leakage need separate review.
    """

    institution_id: str
    target_period: str
    as_of: str
    score: Optional[float]
    source_id: str
    score_family: str
    withholding_reason: Optional[str] = None

    def __post_init__(self):
        for name in ("institution_id", "source_id", "score_family"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "target_period", day(self.target_period))
        _quarter(self.target_period)
        object.__setattr__(self, "as_of", instant(self.as_of).isoformat())
        if self.score is not None:
            object.__setattr__(self, "score", _finite(self.score, "score"))
            if self.withholding_reason is not None:
                raise ValueError("An available score cannot have a withholding reason")
        elif self.withholding_reason is None:
            object.__setattr__(self, "withholding_reason", "Score unavailable in supplied record")
        if self.withholding_reason is not None:
            _text(self.withholding_reason, "withholding_reason")

    @classmethod
    def from_challenge(cls, result, *, target_period):
        """Bind an unchanged Phase 1 score to a caller-chosen classification target.

        No lead-window labels are derived. Choosing the fitted/current quarter does
        not turn an after-quarter score into a prospective forecast of that quarter.
        """
        from .challenge_models import ConventionalChallengeResult

        if not isinstance(result, ConventionalChallengeResult):
            raise ValueError("result must be a ConventionalChallengeResult")
        if result.status not in {"available", "withheld"} or (
            (result.status == "available") != (result.score is not None)
        ):
            raise ValueError("Inconsistent challenge score availability")
        return cls(
            result.institution_id,
            target_period,
            result.review.as_of,
            result.score,
            result.result_id,
            stable_id({"method": result.method, "config": result.config}),
            result.withholding_reason,
        )


@dataclass(frozen=True)
class ContingencyDesign:
    """Declared population, label semantics, and threshold protocol.

    An event inventory alone does not establish binary classification semantics.
    Explicit quarter labels also require a documented negative ascertainment rule.
    Tuning maximizes the declared metric, breaking exact ties at the highest threshold.
    """

    observation_population: str
    event_definition: str
    score_family: str
    registered_at: str
    thresholds: Tuple[float, ...]
    threshold_selection_mode: str = "pre_specified"
    label_semantics: str = "unspecified"
    negative_label_definition: Optional[str] = None
    tuning_as_of: Optional[str] = None
    tuning_metric: str = "mcc"
    maximum_prediction_cells: int = 2_000_000

    def __post_init__(self):
        for name in ("observation_population", "event_definition", "score_family"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "registered_at", instant(self.registered_at).isoformat())
        if isinstance(self.thresholds, (str, bytes)):
            raise ValueError("thresholds must be a nonempty sequence of distinct finite numbers")
        try:
            thresholds = tuple(_finite(t, "threshold") for t in self.thresholds)
        except TypeError as exc:
            raise ValueError("thresholds must be a sequence") from exc
        if not thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError("thresholds must be nonempty and distinct")
        object.__setattr__(self, "thresholds", tuple(sorted(thresholds)))
        for name in ("threshold_selection_mode", "label_semantics", "tuning_metric"):
            _text(getattr(self, name), name)
        if self.threshold_selection_mode not in _MODES:
            raise ValueError("Unknown threshold_selection_mode")
        if self.label_semantics not in {
            "explicit_institution_quarter",
            "event_inventory_only",
            "unspecified",
        }:
            raise ValueError("Unknown label_semantics")
        if self.negative_label_definition is not None:
            _text(self.negative_label_definition, "negative_label_definition")
        if self.threshold_selection_mode == "pre_specified" and len(thresholds) != 1:
            raise ValueError(
                "Pre-specified evaluation requires one threshold; use sensitivity mode for a family"
            )
        if self.tuning_metric not in {"mcc", "f1", "f2"}:
            raise ValueError("tuning_metric must be mcc, f1, or f2")
        if self.threshold_selection_mode == "tuning_derived":
            if self.tuning_as_of is None:
                raise ValueError("Tuning-derived evaluation requires tuning_as_of")
            object.__setattr__(self, "tuning_as_of", instant(self.tuning_as_of).isoformat())
            if instant(self.registered_at) >= instant(self.tuning_as_of):
                raise ValueError("The tuning protocol must be registered before selection")
        elif self.tuning_as_of is not None:
            raise ValueError("tuning_as_of applies only to tuning-derived evaluation")
        _integer(self.maximum_prediction_cells, "maximum_prediction_cells")

    @property
    def pre_specified_threshold(self):
        return self.thresholds[0] if self.threshold_selection_mode == "pre_specified" else None


@dataclass(frozen=True)
class ContingencyThresholdResult:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    mcc: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    f2: Optional[float]
    withholding_reasons: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ContingencyResult:
    design: ContingencyDesign
    evaluation_as_of: str
    status: str
    withholding_reason: Optional[str]
    package_version: Optional[str]
    observation_count: int
    positive_label_count: int
    negative_label_count: Optional[int]
    unlabeled_observation_count: int
    unscored_observation_count: int
    evaluated_observation_count: int
    evaluated_positive_count: int
    evaluated_negative_count: int
    selected_threshold: Optional[float]
    threshold_results: Tuple[ContingencyThresholdResult, ...]
    average_precision: Optional[float]
    average_precision_withholding_reason: Optional[str]
    prospective: bool
    information_set_json: str
    limitations: Tuple[str, ...]
    package_name: str = PACKAGE_NAME
    evidence_class: str = EVIDENCE_CLASS
    description: str = DESCRIPTION

    @property
    def information_set(self):
        return json.loads(self.information_set_json)

    @property
    def evaluated_thresholds(self):
        return tuple(row.threshold for row in self.threshold_results)

    @property
    def threshold_selection_mode(self):
        return self.design.threshold_selection_mode

    @property
    def pre_specified_threshold(self):
        return self.design.pre_specified_threshold

    @property
    def result_id(self):
        return stable_id(self)


def _version(name):
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def _load_contingent():
    # The 0.2.3 wheel declares >=3.11 but uses Python 3.12 type-alias syntax.
    if sys.version_info < (3, 12):
        raise ImportError("contingency-tools 0.2.3 requires Python 3.12 syntax")
    if _version(PACKAGE_NAME) != REVIEWED_VERSION:
        raise ImportError("Install the reviewed contingency-tools==0.2.3 with .[contingency]")
    try:
        return import_module("contingency").Contingent
    except SyntaxError as exc:
        raise ImportError(
            "The installed Contingency package has incompatible Python syntax"
        ) from exc


def _scores(rows, design):
    rows = tuple(rows)
    if any(not isinstance(r, ContingencyScore) for r in rows):
        raise ValueError("scores must contain ContingencyScore records")
    if any(r.score_family != design.score_family for r in rows):
        raise ValueError("All scores must belong to the declared score family")
    keys = [(r.institution_id, r.target_period) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Only one score per institution/target quarter is permitted")
    return tuple(sorted(rows, key=lambda r: (r.institution_id, r.target_period)))


def _population(scores, labels, design, cutoff):
    keys = {(r.institution_id, r.target_period) for r in scores}
    selected = _labels_as_of(
        (label for label in labels if (label.institution_id, label.reporting_period) in keys),
        design.event_definition,
        cutoff,
    )
    selected = tuple(selected[k] for k in sorted(selected))
    lookup = {(label.institution_id, label.reporting_period): label for label in selected}
    pairs = tuple(
        (row, lookup[(row.institution_id, row.target_period)])
        for row in scores
        if row.score is not None
        and (row.institution_id, row.target_period) in lookup
        and lookup[(row.institution_id, row.target_period)].value is not None
    )
    positives = sum(label.value == 1 for label in selected)
    negatives = sum(label.value == 0 for label in selected)
    audit = dict(
        scores=scores,
        selected_labels=selected,
        label_cutoff=cutoff,
        complete_case_keys=tuple((r.institution_id, r.target_period) for r, _ in pairs),
        observation_count=len(scores),
        positive_label_count=positives,
        negative_label_count=negatives,
        unlabeled_observation_count=len(scores) - positives - negatives,
        unscored_observation_count=sum(r.score is None for r in scores),
        complete_case_count=len(pairs),
        population_hash=stable_id({"scores": scores, "labels": selected}),
    )
    return pairs, audit


def _metric(value, undefined):
    if undefined:
        return None, undefined
    if np.ma.is_masked(value) or not np.isfinite(value):
        return None, "NIST returned a masked or non-finite value"
    return float(value), None


def _threshold_metrics(backend, pairs, thresholds, maximum_cells):
    if len(pairs) * len(thresholds) > maximum_cells:
        return (), "Exact threshold family exceeds maximum_prediction_cells; no subsampling applied"
    actual = np.asarray([label.value for _, label in pairs], dtype=bool)
    scores = np.asarray([row.score for row, _ in pairs], dtype=float)
    # Public binary constructor preserves raw score units and inclusive equality.
    # from_scalar rescales scores to [0,1], so its weights are not raw thresholds.
    model = backend(y_true=actual, y_pred=np.less_equal.outer(thresholds, scores))
    values = dict(
        mcc=model.mcc, precision=model.precision, recall=model.recall, f1=model.F, f2=model.F2
    )
    results = []
    for index, threshold in enumerate(thresholds):
        tp, fp, fn, tn = (int(getattr(model, name)[index]) for name in ("TP", "FP", "FN", "TN"))
        undefined = dict(
            mcc=(
                "MCC has a zero marginal denominator"
                if 0 in (tp + fn, tn + fp, tp + fp, tn + fn)
                else None
            ),
            precision="No predicted positives" if tp + fp == 0 else None,
            recall="No observed positives" if tp + fn == 0 else None,
            f1="No observed or predicted positives" if 2 * tp + fp + fn == 0 else None,
            f2="No observed or predicted positives" if 5 * tp + fp + 4 * fn == 0 else None,
        )
        metrics, reasons = {}, []
        for name, array in values.items():
            metrics[name], reason = _metric(array[index], undefined[name])
            if reason:
                reasons.append((name, reason))
        results.append(
            ContingencyThresholdResult(
                threshold, tp, fp, fn, tn, **metrics, withholding_reasons=tuple(reasons)
            )
        )
    return tuple(results), None


def _average_precision(backend, pairs, maximum_cells):
    if not any(label.value == 1 for _, label in pairs):
        return None, "Average precision requires observed positives", ()
    scores = np.asarray([row.score for row, _ in pairs], dtype=float)
    thresholds = np.unique(scores)
    if len(scores) * (len(thresholds) + 1) > maximum_cells:
        return (
            None,
            "Exact average-precision family exceeds maximum_prediction_cells; no approximation applied",
            (),
        )
    actual = np.asarray([label.value for _, label in pairs], dtype=bool)
    predictions = np.vstack(
        (np.less_equal.outer(thresholds, scores), np.zeros(len(scores), dtype=bool))
    )
    # Include every distinct raw score (ties stay together) and an all-negative
    # endpoint. This also works for constant/extreme scores without rescaling.
    value = backend(y_true=actual, y_pred=predictions).expected("aps")
    value, reason = _metric(value, None)
    return value, reason, tuple(float(t) for t in thresholds)


def evaluate_contingency(
    scores, labels, *, design, evaluation_as_of, tuning_scores=(), tuning_labels=()
):
    """Evaluate explicit binary labels without calling or changing event backtesting.

    Metrics concern the scored, explicitly labeled subset; missingness and the
    entire supplied population remain recorded. Tuning uses a separate, earlier
    target-period block and labels available strictly before ``tuning_as_of``.
    No evaluation outcome chooses a threshold. Sensitivity mode selects none.
    """
    if not isinstance(design, ContingencyDesign):
        raise ValueError("design must be a ContingencyDesign")
    scores, tuning_scores = _scores(scores, design), _scores(tuning_scores, design)
    labels, tuning_labels = tuple(labels), tuple(tuning_labels)
    if any(not isinstance(label, BinaryEventLabel) for label in labels + tuning_labels):
        raise ValueError("labels must contain independently supplied BinaryEventLabel records")
    if not scores:
        raise ValueError("The observation population must contain at least one score record")
    evaluation_as_of = instant(evaluation_as_of).isoformat()
    first = min(instant(row.as_of) for row in scores)
    if instant(evaluation_as_of) < max(instant(row.as_of) for row in scores):
        raise ValueError("Evaluation cutoff precedes score availability")
    if instant(design.registered_at) > instant(evaluation_as_of):
        raise ValueError("Evaluation cutoff precedes the declared protocol")
    mode = design.threshold_selection_mode
    if mode != "retrospective_sensitivity" and instant(design.registered_at) >= first:
        raise ValueError("Threshold protocol must be registered before evaluation scores")
    if mode != "tuning_derived" and (tuning_scores or tuning_labels):
        raise ValueError("Tuning inputs require tuning-derived mode")
    pairs, population = _population(scores, labels, design, evaluation_as_of)
    valid_semantics = design.label_semantics == "explicit_institution_quarter" and bool(
        design.negative_label_definition
    )
    reason = None
    if not valid_semantics:
        reason = "Binary classification semantics require explicit quarter labels and a documented negative ascertainment rule"
    info = dict(
        evaluation_population=population,
        threshold_comparison="score >= threshold in original score units",
        metrics_population="scored observations with explicit binary labels at the evaluation cutoff",
        threshold_selection_mode=mode,
        tuning=None,
        average_precision_raw_thresholds=(),
        average_precision_all_negative_endpoint=False,
        numerical_library_versions={
            name: _version(name)
            for name in (PACKAGE_NAME, "beartype", "jaxtyping", "numpy", "scipy")
        },
    )
    tuning_pairs = ()
    if mode == "tuning_derived":
        cutoff = instant(design.tuning_as_of)
        if cutoff >= first:
            raise ValueError("Tuning cutoff must precede every evaluation score")
        if any(
            instant(row.as_of) >= cutoff or instant(row.target_period) >= cutoff
            for row in tuning_scores
        ):
            raise ValueError("Tuning scores and target quarters must precede the tuning cutoff")
        if tuning_scores and max(row.target_period for row in tuning_scores) >= min(
            row.target_period for row in scores
        ):
            raise ValueError(
                "Tuning and evaluation target-period blocks must be strictly separated"
            )
        if {row.source_id for row in scores} & {row.source_id for row in tuning_scores}:
            raise ValueError("Tuning cannot reuse evaluation score artifacts")
        label_cutoff = (cutoff - timedelta(microseconds=1)).isoformat()
        tuning_pairs, tuning_population = _population(
            tuning_scores, tuning_labels, design, label_cutoff
        )
        info["tuning"] = dict(
            population=tuning_population,
            candidate_results=(),
            selection_metric=design.tuning_metric,
            tie_rule="highest threshold among exact metric ties",
            selected_threshold=None,
        )
        if not tuning_pairs:
            reason = (
                reason
                or "No permitted scored and labeled tuning history; no threshold fallback applied"
            )
    selected = design.pre_specified_threshold
    thresholds = design.thresholds
    threshold_results, ap, ap_reason = (), None, None
    package_version = _version(PACKAGE_NAME)
    if reason is None:
        try:
            backend = _load_contingent()
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    if mode == "tuning_derived":
                        candidates, reason = _threshold_metrics(
                            backend, tuning_pairs, thresholds, design.maximum_prediction_cells
                        )
                        info["tuning"]["candidate_results"] = candidates
                        available = [
                            r for r in candidates if getattr(r, design.tuning_metric) is not None
                        ]
                        if not available:
                            reason = (
                                reason
                                or "Tuning objective is undefined at every declared threshold"
                            )
                        else:
                            chosen = max(
                                available,
                                key=lambda r: (getattr(r, design.tuning_metric), r.threshold),
                            )
                            selected = chosen.threshold
                            thresholds = (selected,)
                            info["tuning"]["selected_threshold"] = selected
                    if reason is None and not pairs:
                        reason = "No scored observations with explicitly known binary labels"
                    if reason is None:
                        threshold_results, reason = _threshold_metrics(
                            backend, pairs, thresholds, design.maximum_prediction_cells
                        )
                    if reason is None:
                        ap, ap_reason, ap_thresholds = _average_precision(
                            backend, pairs, design.maximum_prediction_cells
                        )
                        info.update(
                            average_precision_raw_thresholds=ap_thresholds,
                            average_precision_all_negative_endpoint=bool(ap_thresholds),
                        )
        except ImportError as exc:
            reason = f"Optional Contingency dependency unavailable: {exc}"
        except (FloatingPointError, OverflowError, RuntimeWarning) as exc:
            reason = f"NIST numerical evaluation withheld: {type(exc).__name__}: {exc}"
        if reason:
            threshold_results, ap = (), None
    if reason:
        ap_reason = reason
    info["threshold_sensitivity"] = (
        {
            name: {
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "defined_threshold_count": len(values),
            }
            for name in ("mcc", "precision", "recall", "f1", "f2")
            for values in [
                [getattr(r, name) for r in threshold_results if getattr(r, name) is not None]
            ]
        }
        if mode == "retrospective_sensitivity"
        else None
    )
    prospective = mode != "retrospective_sensitivity" and all(
        instant(row.as_of) < instant(_quarter(row.target_period).start_time.to_pydatetime())
        for row in scores
    )
    info["prospective_timing_satisfied"] = prospective
    info["prospective_timing_basis"] = (
        "Every score predates the start of its target quarter; this checks declared timestamps, not model provenance"
    )
    limitations = (
        "Binary classification metrology is distinct from event-detection counts, lead times, and right censoring; no composite is formed.",
        "Metrics describe the scored, explicitly labeled subset; missing scores and labels can create selection bias.",
        "Institution-quarter observations can share episodes and serial or cross-institution dependence; no independence or uncertainty claim is made.",
        "Registration, label ascertainment, and score provenance are caller declarations requiring independent review.",
        "NIST software does not establish DRR validity, approval, or appropriateness for supervisory use.",
    )
    if not prospective:
        limitations += (
            "This result does not establish prospective classification performance; consult threshold mode and target timing.",
        )
    if mode == "retrospective_sensitivity":
        limitations += (
            "Threshold sensitivity uses evaluation outcomes retrospectively and selects no threshold.",
        )
    status = "available"
    if reason:
        status = "withheld"
    elif ap_reason or any(r.withholding_reasons for r in threshold_results):
        status = "partially_withheld"
    elif len(pairs) < len(scores):
        status = "available_with_exclusions"
    return ContingencyResult(
        design,
        evaluation_as_of,
        status,
        reason,
        package_version,
        len(scores),
        population["positive_label_count"],
        population["negative_label_count"] if valid_semantics else None,
        population["unlabeled_observation_count"],
        population["unscored_observation_count"],
        len(pairs) if threshold_results else 0,
        sum(label.value == 1 for _, label in pairs) if threshold_results else 0,
        sum(label.value == 0 for _, label in pairs) if threshold_results else 0,
        selected,
        threshold_results,
        ap,
        ap_reason,
        prospective,
        canonical_json(info),
        limitations,
    )
