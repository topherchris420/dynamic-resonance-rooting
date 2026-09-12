"""Repeatable specification challenges. Failed runs never count as supporting evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

from .common import canonical, stable_id
from .analyzer import DRRConfig, LFBORegimeAnalyzer
from .change_detection import robust_z


@dataclass(frozen=True)
class SignalSpecification:
    name: str
    lookback: int = 24
    transformation: str = "level"
    standardize: bool = True
    rooting_lag: int = 2
    rooting_method: str = "lagged_correlation"
    correction: str = "max_statistic"
    omit_variable: Optional[str] = None
    omit_period: Optional[str] = None
    peer_definition: Optional[str] = None
    vintage: Optional[str] = None


@dataclass(frozen=True)
class SpecificationResult:
    specification: SignalSpecification
    supported: bool
    direction: Optional[int]
    magnitude: Optional[float]
    status: str = "evaluated"
    failure: Optional[str] = None


@dataclass(frozen=True)
class SignalRobustnessReport:
    signal_key: str
    specifications_tested: Tuple[SpecificationResult, ...]
    percentage_surviving: Optional[float]
    direction_stability: Optional[float]
    magnitude_stability: Optional[float]
    peer_sensitivity: str
    transformation_sensitivity: str
    data_vintage_sensitivity: str
    contradictory_evidence: Tuple[str, ...]
    failure_cases: Tuple[str, ...]
    classification: str
    evaluated_count: int
    planned_count: int
    interpretation: str = (
        "Specification survival is a sensitivity summary, not a probability that a claim is true."
    )


def run_falsification(signal_key, specifications, evaluator):
    """`evaluator(spec)` reruns the same target claim under each specification."""
    specifications = tuple(specifications)
    if not specifications or len({s.name for s in specifications}) != len(specifications):
        raise ValueError("Specifications need unique names and at least one planned run")
    results = []
    for spec in specifications:
        try:
            result = evaluator(spec)
            if result.specification != spec or result.status not in {
                "evaluated",
                "not_applicable",
                "failed",
            }:
                raise ValueError("Evaluator returned an invalid specification result")
            if result.direction is not None and result.direction not in (-1, 0, 1):
                raise ValueError("Direction must be -1, 0 or 1")
            if result.magnitude is not None and not np.isfinite(result.magnitude):
                raise ValueError("Magnitude must be finite or null")
            results.append(result)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            results.append(SpecificationResult(spec, False, None, None, "failed", str(exc)))
    valid = [r for r in results if r.status == "evaluated"]
    supported = [r for r in valid if r.supported]
    # Denominator includes failed/non-applicable specifications, transparently reported.
    survival = 100 * len(supported) / len(results) if valid else None
    directions = [r.direction for r in valid if r.direction is not None]
    direction_stability = (
        max(directions.count(d) for d in set(directions)) / len(directions) if directions else None
    )
    magnitudes = [abs(r.magnitude) for r in valid if r.magnitude is not None]
    stability = min(magnitudes) / max(magnitudes) if magnitudes and max(magnitudes) > 0 else None

    def sensitivity(field):
        relevant = [r for r in valid if getattr(r.specification, field) is not None]
        if len({getattr(r.specification, field) for r in relevant}) < 2:
            return "not tested across alternatives"
        return (
            "sensitive"
            if len({r.supported for r in relevant}) > 1
            else "stable in tested alternatives"
        )

    classification = (
        "not supported"
        if not supported
        else (
            "robust"
            if survival >= 80 and len(valid) == len(results)
            else "mixed" if survival >= 50 else "fragile"
        )
    )
    return SignalRobustnessReport(
        signal_key,
        tuple(results),
        survival,
        direction_stability,
        stability,
        sensitivity("peer_definition"),
        sensitivity("transformation"),
        sensitivity("vintage"),
        tuple(r.specification.name + ": claim did not survive" for r in valid if not r.supported),
        tuple(
            r.specification.name + ": " + str(r.failure or r.status)
            for r in results
            if r.status != "evaluated"
        ),
        classification,
        len(valid),
        len(results),
    )


def falsify_material_change(dataset, metric, *, percent_threshold=5.0, z_threshold=3.0):
    """Challenge a specific current-period deviation from recent history.

    Historical omissions affect the reference distribution, never compress the
    time axis of a lagged model. Current/prior quarter observations stay fixed.
    """
    values = dataset.frame[metric].to_numpy(dtype=float)
    specifications = [
        SignalSpecification(f"lookback-{w}", lookback=w, standardize=False) for w in (8, 12, 24)
    ]
    specifications += [
        SignalSpecification("differences", transformation="diff"),
        SignalSpecification("percent-changes", transformation="pct_change"),
    ]
    specifications += [
        SignalSpecification("omit-historical-" + p, omit_period=p) for p in dataset.dates[-5:-2]
    ]

    def evaluate(spec):
        v = values[-spec.lookback - 1 :].copy()
        dates = dataset.dates[-spec.lookback - 1 :]
        if spec.omit_period in dates:
            v[list(dates).index(spec.omit_period)] = np.nan
        if len(v) < 5 or not np.isfinite(v[-2:]).all():
            raise ValueError("Insufficient current/prior observations")
        raw_pct = 100 * (v[-1] - v[-2]) / abs(v[-2]) if v[-2] != 0 else None
        if spec.transformation == "diff":
            v = np.diff(v)
        elif spec.transformation == "pct_change":
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.diff(v) / np.abs(v[:-1]) * 100
        z = robust_z(v[-1], v[:-1])
        # Percent threshold applies to the original QoQ claim, not a percent-of-percent.
        support = (z is not None and abs(z) >= z_threshold) or (
            spec.transformation == "level"
            and raw_pct is not None
            and abs(raw_pct) >= percent_threshold
        )
        return SpecificationResult(spec, support, int(np.sign(v[-1] - np.nanmedian(v[:-1]))), z)

    return run_falsification(
        f"{dataset.institution_id}:{metric}:material_change", specifications, evaluate
    )


def falsify_drr_signal(dataset, config=None):
    """Challenge the structural-surprise claim with the real DRR pipeline."""
    config = config or DRRConfig()
    specs = [
        SignalSpecification(
            "reference",
            lookback=config.lookback,
            rooting_lag=config.rooting_max_lag,
            rooting_method=config.rooting_method,
            correction=config.rooting_correction,
        ),
        SignalSpecification(
            "short-window", lookback=max(config.minimum_observations, config.lookback - 4)
        ),
        SignalSpecification("differences", transformation="diff", lookback=config.lookback),
        SignalSpecification(
            "percent-changes", transformation="pct_change", lookback=config.lookback
        ),
        SignalSpecification("unstandardized", standardize=False, lookback=config.lookback),
        SignalSpecification("lag-one", rooting_lag=1, lookback=config.lookback),
        SignalSpecification(
            "alternate-rooting", rooting_method="transfer_entropy", lookback=config.lookback
        ),
        SignalSpecification("uncorrected-inference", correction="none", lookback=config.lookback),
    ]
    specs += [
        SignalSpecification("omit-" + name, omit_variable=name, lookback=config.lookback)
        for name in dataset.variable_names
        if len(dataset.variable_names) > 1
    ]

    def evaluate(spec):
        names = tuple(n for n in dataset.variable_names if n != spec.omit_variable)
        subset = type(dataset).from_vintage_store(
            dataset._store,
            dataset._registry,
            institution_id=dataset.institution_id,
            form=dataset.filing_type,
            as_of=dataset.available_as_of,
            metrics=names,
            allow_synthetic=dataset.metadata.get("allow_synthetic", False),
        )
        cfg = replace(
            config,
            lookback=spec.lookback,
            transformation=spec.transformation,
            standardize=spec.standardize,
            rooting_max_lag=spec.rooting_lag,
            rooting_method=spec.rooting_method,
            rooting_correction=spec.correction,
        )
        result = LFBORegimeAnalyzer(cfg).analyze(subset)
        if result["status"] == "unavailable":
            raise ValueError(result["limitation"])
        surprise = result["structural_surprise"]
        return SpecificationResult(
            spec, surprise["flagged"], 1 if surprise["flagged"] else 0, surprise["score"]
        )

    return run_falsification(f"{dataset.institution_id}:structural_surprise", specs, evaluate)
