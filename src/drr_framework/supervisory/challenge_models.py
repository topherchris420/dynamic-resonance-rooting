"""Conventional econometric challenge models with explicit historical information sets.

These research outputs are separate from scalar ``BaselineResult`` records and
analyst judgment. No coefficient inference, causal interpretation, automatic
specification search, regularization fallback, or outcome imputation is performed.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..datasets import RegulatoryAnalysisDataset, SupervisoryPanelDataset
from .common import canonical, canonical_json, day, instant, stable_id


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _names(values, name, minimum=1):
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of names")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of names") from exc
    for value in values:
        _text(value, name)
    if len(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain >= {minimum} unique names")
    return values


def _quarter(value):
    value = day(value)
    period = pd.Period(value, freq="Q")
    if value != period.end_time.date().isoformat():
        raise ValueError("reporting_period must be a calendar quarter end")
    return period


def _periods(end, count):
    return tuple(p.end_time.date().isoformat() for p in pd.period_range(end=end, periods=count))


@dataclass(frozen=True)
class ChallengeReview:
    as_of: str
    reporting_period: str

    def __post_init__(self):
        object.__setattr__(self, "as_of", instant(self.as_of).isoformat())
        object.__setattr__(self, "reporting_period", day(self.reporting_period))
        _quarter(self.reporting_period)
        if self.reporting_period > day(self.as_of):
            raise ValueError("Review cannot precede the current reporting period")


@dataclass(frozen=True)
class LaggedFeature:
    metric: str
    lag: int = 1

    def __post_init__(self):
        _text(self.metric, "metric")
        _integer(self.lag, "lag")

    @property
    def name(self):
        return f"L{self.lag}.{self.metric}"


@dataclass(frozen=True)
class BinaryEventLabel:
    """Explicit event/non-event *in this quarter*, separately sourced from features.

    ``None`` means unknown. A zero must be an explicitly ascertained non-event
    under ``definition``; absence from an event list is never converted to zero.
    Multiple vintages are selected by availability, with conflicts rejected.
    """

    institution_id: str
    reporting_period: str
    value: Optional[int]
    available_as_of: str
    definition: str
    source: str
    event_at: Optional[str] = None

    def __post_init__(self):
        for name in ("institution_id", "definition", "source"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "reporting_period", day(self.reporting_period))
        _quarter(self.reporting_period)
        object.__setattr__(self, "available_as_of", instant(self.available_as_of).isoformat())
        if self.value is not None:
            if not isinstance(self.value, (bool, int)) or self.value not in (0, 1):
                raise ValueError("Label value must be explicit 0, 1 or None (unknown)")
            object.__setattr__(self, "value", int(self.value))
        if instant(self.available_as_of) < instant(self.reporting_period):
            raise ValueError("A quarterly outcome cannot be known before the quarter ends")
        if self.event_at is not None:
            object.__setattr__(self, "event_at", instant(self.event_at).isoformat())
            if (
                self.value != 1
                or pd.Period(day(self.event_at), freq="Q") != _quarter(self.reporting_period)
                or instant(self.event_at) > instant(self.available_as_of)
            ):
                raise ValueError("event_at requires a positive label in the same reporting quarter")

    @property
    def label_id(self):
        return stable_id(self)


@dataclass(frozen=True)
class VARChallengeConfig:
    institution_id: str
    form: str
    variables: Tuple[str, ...]
    registered_at: str
    training_periods: int = 24
    lag_order: int = 1
    minimum_residual_df: int = 8
    max_condition_number: float = 1e8
    warning_threshold: float = 3.0

    def __post_init__(self):
        _text(self.institution_id, "institution_id")
        _text(self.form, "form")
        object.__setattr__(self, "variables", _names(self.variables, "variables", 2))
        object.__setattr__(self, "registered_at", instant(self.registered_at).isoformat())
        for name in ("training_periods", "lag_order", "minimum_residual_df"):
            _integer(getattr(self, name), name)
        for name in ("max_condition_number", "warning_threshold"):
            _positive(getattr(self, name), name)
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must exceed 1")


@dataclass(frozen=True)
class PanelLogitConfig:
    institutions: Tuple[str, ...]
    form: str
    features: Tuple[LaggedFeature, ...]
    event_definition: str
    registered_at: str
    training_periods: int = 24
    minimum_observations: int = 40
    minimum_institutions: int = 2
    minimum_events: int = 5
    minimum_nonevents: int = 5
    max_condition_number: float = 1e8
    max_iterations: int = 100
    convergence_tolerance: float = 1e-8
    warning_threshold: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "institutions", _names(self.institutions, "institutions"))
        try:
            object.__setattr__(self, "features", tuple(self.features))
        except TypeError as exc:
            raise ValueError("features must contain LaggedFeature records") from exc
        if not self.features or any(not isinstance(f, LaggedFeature) for f in self.features):
            raise ValueError("features must contain LaggedFeature records")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")
        if any(f.metric in {"institution_id", "reporting_period"} for f in self.features):
            raise ValueError("Feature metrics cannot use panel identity-column names")
        _text(self.form, "form")
        _text(self.event_definition, "event_definition")
        object.__setattr__(self, "registered_at", instant(self.registered_at).isoformat())
        for name in (
            "training_periods",
            "minimum_observations",
            "minimum_institutions",
            "minimum_events",
            "minimum_nonevents",
            "max_iterations",
        ):
            _integer(getattr(self, name), name)
        for name in ("max_condition_number", "convergence_tolerance", "warning_threshold"):
            _positive(getattr(self, name), name)
        if self.max_condition_number <= 1 or self.warning_threshold > 1:
            raise ValueError("Invalid conditioning tolerance or probability threshold")
        if self.minimum_institutions > len(self.institutions):
            raise ValueError("minimum_institutions exceeds the configured institution scope")


@dataclass(frozen=True)
class ConventionalChallengeResult:
    """Immutable envelope; method-specific fit and audit data return fresh JSON copies.

    VAR scores are residual-standardized forecast errors. Logit scores are fitted
    conditional probabilities. Neither is a scalar trailing ``BaselineResult``.
    """

    method: str
    institution_id: str
    review: ChallengeReview
    config: Union[VARChallengeConfig, PanelLogitConfig]
    status: str
    score: Optional[float]
    withholding_reason: Optional[str]
    information_set_json: str
    fit_json: str = "{}"
    limitations: Tuple[str, ...] = ()
    evidence_class: str = "CONVENTIONAL_ECONOMETRIC_EVIDENCE"

    @property
    def information_set(self):
        return json.loads(self.information_set_json)

    @property
    def fit(self):
        return json.loads(self.fit_json)

    @property
    def flagged(self):
        return None if self.score is None else self.score >= self.config.warning_threshold

    @property
    def result_id(self):
        return stable_id(self)


def _reviews(reviews, config):
    reviews = tuple(reviews)
    if not reviews or any(not isinstance(r, ChallengeReview) for r in reviews):
        raise ValueError("reviews must contain ChallengeReview records")
    if any(
        instant(b.as_of) <= instant(a.as_of) or b.reporting_period <= a.reporting_period
        for a, b in zip(reviews, reviews[1:])
    ):
        raise ValueError("Review times and reporting periods must be unique and increasing")
    if instant(config.registered_at) >= instant(reviews[0].as_of):
        raise ValueError("The specification must be registered before the first review")
    return reviews


def _labels_as_of(labels, definition, cutoff):
    selected = {}
    for label in labels:
        if not isinstance(label, BinaryEventLabel):
            raise ValueError("labels must contain BinaryEventLabel records")
        if instant(label.available_as_of) > instant(cutoff):
            continue
        if label.definition != definition:
            raise ValueError("Event label definition differs from the registered design")
        key = (label.institution_id, label.reporting_period)
        old = selected.get(key)
        if old is None or instant(old.available_as_of) < instant(label.available_as_of):
            selected[key] = label
        elif old.available_as_of == label.available_as_of and old != label:
            raise ValueError(f"Conflicting event labels for {key}")
    return selected


def _load_statsmodels():
    # Import lazily: the base installation can still run all existing baselines.
    package = import_module("statsmodels")
    api = import_module("statsmodels.api")
    return api, package.__version__


def _package_version(name):
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class _WindowedStore:
    """Restrict dataset validation while retaining the original vintage/lineage rules."""

    store: object
    dates: Tuple[str, ...]

    def as_of(self, cutoff):
        return tuple(o for o in self.store.as_of(cutoff) if o.reporting_period in self.dates)


def _snapshot(store, registry, institution, form, variables, dates, cutoff, allow_synthetic):
    """Reconstruct immutable records, never use a caller-mutated display DataFrame."""
    dataset = RegulatoryAnalysisDataset.from_vintage_store(
        _WindowedStore(store, tuple(dates)),
        registry,
        institution_id=institution,
        form=form,
        as_of=cutoff,
        metrics=variables,
        allow_synthetic=allow_synthetic,
    )
    records = tuple(o for o in dataset.observations if o.reporting_period in dates)
    frame = pd.DataFrame(index=dates, columns=variables, dtype=float)
    for record in records:
        frame.loc[record.reporting_period, record.metric] = record.value
    definitions = tuple(
        registry.resolve(
            form,
            o.metric,
            o.reporting_period,
            as_of=cutoff,
            version=o.definition_version,
            allow_synthetic=allow_synthetic,
        )
        for o in records
    )
    definitions = tuple({stable_id(d): d for d in definitions}.values())
    return frame, records, definitions


def _comparability(records, variables, policy, cutoff, form, dates, institution_class):
    reasons, events = [], {}
    for variable in variables:
        metric_records = [o for o in records if o.metric == variable]
        if len({(o.unit, o.definition_version, o.perimeter_version) for o in metric_records}) > 1:
            reasons.append(f"Definition/unit/perimeter break: {variable}")
        if any(o.perimeter_version == "unspecified" for o in metric_records):
            reasons.append(f"Unspecified reporting perimeter: {variable}")
        if policy is not None:
            for event in policy.relevant(
                as_of=cutoff, form=form, metric=variable, institution_class=institution_class
            ):
                if event.comparability_break:
                    effective = (
                        pd.Period(day(event.effective_date), freq="Q").end_time.date().isoformat()
                    )
                    if dates[0] <= effective <= dates[-1]:
                        events[event.event_id] = event
                        reasons.append(f"Known comparability breakpoint: {event.identifier}")
    return tuple(sorted(set(reasons))), tuple(events[k] for k in sorted(events))


def _information(review, config, cutoff, dates, records, definitions, *, labels=(), events=()):
    records = tuple(sorted(records, key=lambda o: o.key))
    return dict(
        review=canonical(review),
        configuration_id=stable_id(config),
        observation_cutoff_exclusive=review.as_of,
        reconstruction_cutoff=cutoff,
        training_periods=dates,
        training_start=dates[0],
        training_end=dates[-1],
        source_observations=canonical(records),
        source_ids=tuple(o.observation_id for o in records),
        source_vintages=tuple(sorted({o.source_vintage for o in records})),
        input_hash=stable_id(records),
        semantic_definitions=canonical(definitions),
        training_labels=canonical(labels),
        training_label_ids=tuple(label.label_id for label in labels),
        policy_events=canonical(events),
        threshold_mode="pre_specified",
        threshold=config.warning_threshold,
        registration_basis="caller declaration; timestamp is not proof of external registration",
        preprocessing="raw levels; training-only centering/scaling; no peers or interpolation",
        numerical_library_versions={
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": _package_version("scipy"),
            "drr-framework": _package_version("drr-framework"),
        },
    )


def _result(method, institution, review, config, info, fit, reason=None, score=None):
    limitations = (
        "Descriptive fitted parameters only; no coefficient p-values or causal claims.",
        "Short histories and sparse events limit estimation and outcomes analysis.",
        "Ingestion-constrained reconstruction cannot establish what another system knew.",
        "Independent conceptual-soundness and intended-use review remain pending.",
    )
    if method == "panel_logit":
        limitations += (
            "Pooled common slopes; institution effects, serial/cross-firm dependence and rare-event bias are not modeled.",
            "Class counts refer to labeled institution-quarters, not necessarily independent economic episodes.",
            "Unknown labels remain excluded and counted; label selection can bias the observed sample.",
        )
    else:
        limitations += (
            "Fixed-lag linear VAR in levels assumes a stable specification; stationarity is not established.",
            "Forecast-error scores describe unusual observations, not event probabilities.",
        )
    return ConventionalChallengeResult(
        method,
        institution,
        review,
        config,
        "withheld" if reason else "available",
        score,
        reason,
        canonical_json(info),
        canonical_json(fit),
        limitations,
    )


def _design_diagnostics(design, max_condition):
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    diagnostics = dict(
        design_rank=rank, design_columns=design.shape[1], design_condition_number=condition
    )
    reason = None
    if rank < design.shape[1]:
        reason = "Numerically unidentified design"
    elif not np.isfinite(condition) or condition > max_condition:
        reason = "Design exceeds configured condition-number tolerance"
    return diagnostics, reason


def walk_forward_var(
    store,
    registry,
    *,
    reviews,
    config: VARChallengeConfig,
    policy_context=None,
    institution_class=None,
    allow_synthetic=False,
):
    """Refit a fixed VAR on exactly ``training_periods`` lagged target quarters.

    Each fit uses that many target rows plus ``lag_order`` initial observations.
    The current quarter is scored only after fitting; it never calibrates the
    model, residual variance, normalization, lag order, or alert threshold.
    """
    reviews = _reviews(reviews, config)
    results = []
    for review in reviews:
        cutoff = (instant(review.as_of) - timedelta(microseconds=1)).isoformat()
        target = _quarter(review.reporting_period)
        training_dates = _periods(target - 1, config.training_periods)
        dates = _periods(target, config.training_periods + config.lag_order + 1)
        records, definitions, events = (), (), ()
        reason = None
        values = np.full((len(dates), len(config.variables)), np.nan)
        try:
            frame, records, definitions = _snapshot(
                store,
                registry,
                config.institution_id,
                config.form,
                config.variables,
                dates,
                cutoff,
                allow_synthetic,
            )
            values = frame.to_numpy(dtype=float)
            breaks, events = _comparability(
                records,
                config.variables,
                policy_context,
                cutoff,
                config.form,
                dates,
                institution_class,
            )
            reason = "; ".join(breaks) or None
        except ValueError as exc:
            reason = f"Data unavailable: {exc}"
        info = _information(
            review, config, cutoff, training_dates, records, definitions, events=events
        )
        info.update(
            lag_history_start=dates[0],
            variables=config.variables,
            allow_synthetic=allow_synthetic,
            institution_class=institution_class,
            training_source_ids=tuple(
                o.observation_id for o in records if o.reporting_period < review.reporting_period
            ),
            scoring_source_ids=tuple(
                o.observation_id for o in records if o.reporting_period == review.reporting_period
            ),
        )
        fit = dict(
            selected_lag_order=config.lag_order,
            lag_selection="pre_specified_fixed",
            training_count=config.training_periods,
            finite_history_periods=int(np.isfinite(values[:-1]).all(axis=1).sum()),
            missing_cells=int((~np.isfinite(values)).sum()),
            coefficient_inference="not_reported",
            package="statsmodels",
            package_version=None,
            score_definition="sqrt(mean((current - forecast)**2 / training_residual_variance))",
        )
        if reason is None and not np.isfinite(values).all():
            reason = "Insufficient history or missing/non-finite observations in the exact configured window"
        df = config.training_periods - (1 + config.lag_order * len(config.variables))
        fit["residual_df"] = df
        if reason is None and df < config.minimum_residual_df:
            reason = "Insufficient residual degrees of freedom for the configured VAR"
        if reason:
            results.append(_result("var", config.institution_id, review, config, info, fit, reason))
            continue
        score = None
        try:
            api, version = _load_statsmodels()
            fit["package_version"] = version
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                training = values[:-1]
                center, scale = training.mean(axis=0), training.std(axis=0, ddof=0)
                if np.any(scale <= 0):
                    raise ValueError("Constant configured VAR variable; no variable was removed")
                z = (training - center) / scale
                p = config.lag_order
                design = np.column_stack(
                    [np.ones(config.training_periods)]
                    + [z[p - lag : len(z) - lag] for lag in range(1, p + 1)]
                )
                diagnostics, reason = _design_diagnostics(design, config.max_condition_number)
                fit.update(diagnostics, normalization_mean=center, normalization_scale=scale)
                if reason is None:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model = api.tsa.VAR(z, missing="raise").fit(maxlags=p, ic=None, trend="c")
                    fit["warnings"] = tuple(f"{w.category.__name__}: {w.message}" for w in caught)
                    covariance = np.asarray(model.sigma_u)
                    condition = float(np.linalg.cond(covariance))
                    fit.update(
                        coefficients=model.params,
                        coefficient_rows=("intercept",)
                        + tuple(f"L{lag}.{v}" for lag in range(1, p + 1) for v in config.variables),
                        coefficient_columns=config.variables,
                        coefficient_units="training-standardized variables",
                        residuals=model.resid,
                        residual_covariance=covariance,
                        residual_condition_number=condition,
                        residual_mean=np.mean(model.resid, axis=0),
                        residual_rmse=np.sqrt(np.mean(model.resid**2, axis=0)),
                        stable_linear_dynamics=bool(model.is_stable()),
                    )
                    if (
                        not np.isfinite(covariance).all()
                        or not np.isfinite(model.params).all()
                        or np.linalg.matrix_rank(covariance) < len(config.variables)
                        or np.linalg.eigvalsh(covariance).min() <= np.finfo(float).eps
                        or not np.isfinite(condition)
                        or condition > config.max_condition_number
                    ):
                        reason = "Singular or materially ill-conditioned VAR residual covariance"
                    elif any(issubclass(w.category, RuntimeWarning) for w in caught):
                        reason = "VAR numerical warning: " + "; ".join(fit["warnings"])
                    else:
                        forecast = model.forecast(z[-p:], steps=1)[0] * scale + center
                        error = values[-1] - forecast
                        residual_sd = np.sqrt(np.diag(covariance)) * scale
                        score = float(np.sqrt(np.mean((error / residual_sd) ** 2)))
                        if not np.isfinite(score) or not np.isfinite(forecast).all():
                            raise ValueError("Non-finite VAR forecast or score")
                        fit.update(
                            forecast=forecast,
                            observed=values[-1],
                            forecast_error=error,
                            training_residual_sd=residual_sd,
                            forecast_lag_values=training[-p:],
                        )
        except ImportError as exc:
            reason = f"Optional econometrics dependency unavailable: {exc}"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            reason = f"VAR estimation withheld: {type(exc).__name__}: {exc}"
        results.append(
            _result(
                "var",
                config.institution_id,
                review,
                config,
                info,
                fit,
                reason,
                None if reason else score,
            )
        )
    return tuple(results)


def _panel_predictors(panel: SupervisoryPanelDataset, keys, features):
    """Exact calendar lags on a raw as-of panel; never shift over missing quarters."""
    frame = panel.frame.set_index([panel.institution_column, panel.date_column])
    rows = []
    for institution, date in keys:
        rows.append(
            [
                frame.loc[
                    (institution, (_quarter(date) - feature.lag).end_time.date().isoformat()),
                    feature.metric,
                ]
                for feature in features
            ]
        )
    return np.asarray(rows, dtype=float).reshape(len(rows), len(features))


def _fit_logit(panel, keys, outcomes, score_keys, config, fit):
    """Pooled binary logit fitted by statsmodels; all scale parameters are training-only."""
    api, version = _load_statsmodels()
    fit["package_version"] = version
    x = _panel_predictors(panel, keys, config.features)
    current = _panel_predictors(panel, score_keys, config.features)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        center, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
        fit.update(normalization_mean=center, normalization_scale=scale)
        if np.any(scale <= 0):
            return None, "Constant configured logit feature; no feature was removed"
        design = np.column_stack((np.ones(len(x)), (x - center) / scale))
        score_design = np.column_stack((np.ones(len(current)), (current - center) / scale))
        diagnostics, reason = _design_diagnostics(design, config.max_condition_number)
        fit.update(diagnostics)
        if reason:
            return None, reason
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit["fit_attempted"] = True
            model = api.Logit(
                np.asarray(outcomes, dtype=float), design, missing="raise", check_rank=True
            )
            result = model.fit(
                method="newton",
                maxiter=config.max_iterations,
                tol=config.convergence_tolerance,
                disp=False,
                full_output=True,
            )
        fit["warnings"] = tuple(f"{w.category.__name__}: {w.message}" for w in caught)
        fit["converged"] = bool(result.mle_retvals.get("converged", False))
        fit["iterations"] = result.mle_retvals.get("iterations")
        if not fit["converged"]:
            return None, "Logit failed to converge under the registered iteration budget"
        if any(
            issubclass(w.category, RuntimeWarning)
            or w.category.__name__
            in {"PerfectSeparationWarning", "ConvergenceWarning", "HessianInversionWarning"}
            for w in caught
        ):
            return None, "Logit numerical/convergence warning: " + "; ".join(fit["warnings"])
        information = -np.asarray(model.hessian(result.params))
        condition = float(np.linalg.cond(information))
        eigenvalues = np.linalg.eigvalsh(information)
        fit.update(
            information_condition_number=condition,
            information_minimum_eigenvalue=float(eigenvalues.min()),
        )
        if (
            not np.isfinite(result.params).all()
            or not np.isfinite(information).all()
            or not np.isfinite(condition)
            or condition > config.max_condition_number
            or eigenvalues.min() <= np.finfo(float).eps * max(1.0, eigenvalues.max())
        ):
            return None, "Logit information matrix is unidentified or materially ill-conditioned"
        scores = np.asarray(result.predict(score_design))
        if not np.isfinite(scores).all():
            return None, "Non-finite logit probabilities"
        fit.update(
            coefficients=result.params,
            coefficient_rows=("intercept",) + tuple(f.name for f in config.features),
            coefficient_units="training-standardized predictors; log-odds",
            scoring_predictors=current,
            log_likelihood=float(result.llf),
        )
        return scores, None


def walk_forward_panel_logit(
    store,
    registry,
    *,
    reviews,
    labels,
    config: PanelLogitConfig,
    policy_context=None,
    institution_classes=None,
    allow_synthetic=False,
):
    """Build raw ``SupervisoryPanelDataset`` snapshots and refit at every review.

    The fixed institution scope and calendar window are never shortened to make
    fitting succeed. Only explicitly labeled earlier target quarters whose labels
    were available *before* this review can enter the likelihood. Missing labels
    are counted, not inferred. No current-quarter outcomes enter training.
    """
    reviews = _reviews(reviews, config)
    labels = tuple(labels)
    results = []
    variables = tuple(dict.fromkeys(f.metric for f in config.features))
    for review in reviews:
        cutoff = (instant(review.as_of) - timedelta(microseconds=1)).isoformat()
        target = _quarter(review.reporting_period)
        training_dates = _periods(target - 1, config.training_periods)
        source_dates = _periods(
            target - 1, config.training_periods + max(f.lag for f in config.features)
        )
        label_map = _labels_as_of(labels, config.event_definition, cutoff)
        keys = tuple((i, d) for i in config.institutions for d in training_dates)
        used_labels = tuple(
            label_map[k] for k in keys if k in label_map and label_map[k].value is not None
        )
        used_keys = tuple((label.institution_id, label.reporting_period) for label in used_labels)
        outcomes = tuple(label.value for label in used_labels)
        records, definitions, events, frames, reasons = [], [], [], [], []
        for institution in config.institutions:
            try:
                frame, facts, meanings = _snapshot(
                    store,
                    registry,
                    institution,
                    config.form,
                    variables,
                    source_dates + (review.reporting_period,),
                    cutoff,
                    allow_synthetic,
                )
                records.extend(facts)
                definitions.extend(meanings)
                breaks, relevant = _comparability(
                    facts,
                    variables,
                    policy_context,
                    cutoff,
                    config.form,
                    source_dates + (review.reporting_period,),
                    (institution_classes or {}).get(institution),
                )
                reasons.extend(f"{institution}: {r}" for r in breaks)
                events.extend(relevant)
            except ValueError as exc:
                reasons.append(f"{institution}: data unavailable: {exc}")
                frame = pd.DataFrame(index=source_dates, columns=variables, dtype=float)
            frame = frame.copy()
            frame = frame.reindex(source_dates)
            frame["institution_id"] = institution
            frame["reporting_period"] = source_dates
            frames.append(frame)
        # Direct construction avoids the legacy adapter's retrospective cleaning.
        panel = SupervisoryPanelDataset(
            frame=pd.concat(frames, ignore_index=True),
            institution_column="institution_id",
            date_column="reporting_period",
            metric_columns=variables,
            peer_group_column=None,
            sampling_rate=4.0,
            metadata={
                "as_of": cutoff,
                "standardize": False,
                "interpolate": False,
                "transform": "level",
            },
        )
        definitions = tuple({stable_id(d): d for d in definitions}.values())
        events = tuple({e.event_id: e for e in events}.values())
        pooling_breaks, _ = _comparability(
            records, variables, None, cutoff, config.form, source_dates, None
        )
        reasons.extend(f"Panel pooling: {reason}" for reason in pooling_breaks)
        info = _information(
            review,
            config,
            cutoff,
            training_dates,
            records,
            definitions,
            labels=used_labels,
            events=events,
        )
        info.update(
            institutions=config.institutions,
            features=canonical(config.features),
            lag_history_start=source_dates[0],
            training_row_keys=used_keys,
            allow_synthetic=allow_synthetic,
            institution_classes=tuple(
                (i, (institution_classes or {}).get(i)) for i in config.institutions
            ),
        )
        missing = ~np.isfinite(panel.frame[list(variables)].to_numpy(dtype=float))
        # A rectangular panel includes padding for features whose lags differ.
        # Only cells referenced by the fixed training/scoring designs are required.
        required_dates = {
            metric: {
                (_quarter(date) - feature.lag).end_time.date().isoformat()
                for feature in config.features
                if feature.metric == metric
                for date in training_dates + (review.reporting_period,)
            }
            for metric in variables
        }
        required = np.asarray(
            [
                [date in required_dates[metric] for metric in variables]
                for date in panel.frame[panel.date_column]
            ],
            dtype=bool,
        )
        missing_required = missing & required
        n, positives = len(outcomes), int(sum(outcomes))
        fit = dict(
            package="statsmodels",
            package_version=None,
            link="logit",
            optimizer="newton",
            coefficient_inference="not_reported",
            configured_training_rows=len(keys),
            labeled_observation_count=n,
            event_count=positives,
            nonevent_count=n - positives,
            event_fraction=positives / n if n else None,
            unlabeled_observation_count=len(keys) - n,
            usable_observation_count=0,
            usable_institution_count=0,
            missing_cells=int(missing_required.sum()),
            missing_cells_by_metric=dict(zip(variables, missing_required.sum(axis=0).tolist())),
            unused_missing_cells=int((missing & ~required).sum()),
            converged=None,
            iterations=None,
            fit_attempted=False,
        )
        x = _panel_predictors(panel, used_keys, config.features)
        complete = np.isfinite(x).all(axis=1)
        fit["usable_observation_count"] = int(complete.sum())
        fit["usable_institution_count"] = len({k[0] for k, ok in zip(used_keys, complete) if ok})
        if missing_required.any():
            reasons.append(
                "Insufficient history or missing/non-finite required features in the exact configured panel window"
            )
        if n < max(config.minimum_observations, len(config.features) + 2):
            reasons.append("Too few usable labeled observations")
        if positives < config.minimum_events or n - positives < config.minimum_nonevents:
            reasons.append("Too few explicitly labeled events or non-events")
        if fit["usable_institution_count"] < config.minimum_institutions:
            reasons.append("Too few usable institutions")
        reason = "; ".join(reasons) or None
        scores = None
        score_keys = tuple((i, review.reporting_period) for i in config.institutions)
        if reason is None:
            try:
                scores, reason = _fit_logit(panel, used_keys, outcomes, score_keys, config, fit)
            except ImportError as exc:
                reason = f"Optional econometrics dependency unavailable: {exc}"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
                if fit["fit_attempted"]:
                    fit["converged"] = False
                reason = f"Logit estimation withheld: {type(exc).__name__}: {exc}"
            # PerfectSeparationError inherits Exception rather than ValueError.
            except Exception as exc:
                if type(exc).__name__ != "PerfectSeparationError":
                    raise
                fit["converged"] = False
                reason = f"Logit separation: {exc}"
        for index, institution in enumerate(config.institutions):
            results.append(
                _result(
                    "panel_logit",
                    institution,
                    review,
                    config,
                    info,
                    fit,
                    reason,
                    None if reason else float(scores[index]),
                )
            )
    return tuple(results)
