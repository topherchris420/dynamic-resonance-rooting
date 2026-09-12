"""Semantic adapter over the existing DRR facade; no replacement of core methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .common import canonical, DECISION_BOUNDARY


@dataclass(frozen=True)
class DRRConfig:
    lookback: int = 24
    minimum_observations: int = 16
    window_size: int = 12
    rooting_method: str = "lagged_correlation"
    rooting_max_lag: int = 2
    rooting_n_surrogates: int = 99
    rooting_correction: str = "max_statistic"
    seed: int = 42
    standardize: bool = True
    transformation: str = "level"
    topology_threshold: float = 0.25

    def __post_init__(self):
        if (
            self.minimum_observations < 8
            or self.lookback < self.minimum_observations
            or self.window_size < 4
            or self.rooting_max_lag < 1
            or self.rooting_n_surrogates < 0
        ):
            raise ValueError("Invalid DRR history/inference configuration")
        if (
            self.transformation not in {"level", "diff", "pct_change"}
            or self.rooting_method not in {"lagged_correlation", "transfer_entropy"}
            or self.rooting_correction not in {"max_statistic", "none"}
        ):
            raise ValueError("Unsupported DRR specification")
        if not 0 <= self.topology_threshold <= 1:
            raise ValueError("Topology threshold must be in [0,1]")


class LFBORegimeAnalyzer:
    def __init__(self, config=None):
        self.config = config or DRRConfig()

    def analyze(self, dataset, *, previous=None):
        config = self.config
        base = dict(
            status="unavailable",
            variables=list(dataset.variable_names),
            as_of=dataset.available_as_of,
            config=canonical(config),
            decision_boundary=DECISION_BOUNDARY,
            structural_alert=False,
        )
        for metric in dataset.variable_names:
            records = [o for o in dataset.observations if o.metric == metric][-config.lookback :]
            if len({(o.unit, o.definition_version, o.perimeter_version) for o in records}) > 1:
                return dict(
                    base,
                    limitation="Comparability break in the DRR window; segment the series before analysis",
                )
        try:
            values = dataset.to_drr_input()[-config.lookback :]
        except ValueError as exc:
            return dict(base, limitation=str(exc))
        if config.transformation == "diff":
            values = np.diff(values, axis=0)
        elif config.transformation == "pct_change":
            if np.any(values[:-1] == 0):
                return dict(base, limitation="Percent changes undefined for zero denominator")
            values = np.diff(values, axis=0) / np.abs(values[:-1]) * 100
        if len(values) < config.minimum_observations:
            return dict(
                base,
                limitation=f"{len(values)} complete quarters available; {config.minimum_observations} required",
            )
        if config.standardize:
            # Fit on prior rows: the current observation does not calibrate its own scale.
            mean, std = values[:-1].mean(axis=0), values[:-1].std(axis=0)
            values = (values - mean) / np.where(std > 0, std, 1.0)
        from ..analysis import DynamicResonanceRooting
        from ..structural_surprise import structural_surprise
        from ..topology_dynamics import topology_drift, root_migration, summarize_topology
        from ..reporting import serialize_analysis_results

        drr = DynamicResonanceRooting(sampling_rate=dataset.sampling_rate)
        result = drr.analyze_system(
            values,
            multivariate=len(dataset.variable_names) > 1,
            window_size=config.window_size,
            rooting_method=config.rooting_method,
            rooting_max_lag=config.rooting_max_lag,
            rooting_n_surrogates=config.rooting_n_surrogates,
            rooting_random_state=config.seed,
            rooting_correction=config.rooting_correction,
        )
        names = {f"dim_{i}": name for i, name in enumerate(dataset.variable_names)}

        def rename(value):
            if isinstance(value, dict):
                return {names.get(k, k): rename(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [rename(v) for v in value]
            return names.get(value, value) if isinstance(value, str) else value

        result = rename(serialize_analysis_results(result))
        rooting = result.get("rooting_analysis", {})
        surprise = structural_surprise(
            values, min_history=max(8, len(values) // 2), feature_names=dataset.variable_names
        )
        finite = surprise.score[:-1][np.isfinite(surprise.score[:-1])]
        threshold = float(np.quantile(finite, 0.975)) if len(finite) >= 4 else None
        surprise_flag = bool(
            threshold is not None
            and float(surprise.score[-1]) > threshold
            and float(surprise.score[-1]) > 3.0
        )
        matrix = rooting.get("score_matrix")
        drift = migration = None
        topology = None
        if matrix is not None:
            topology = canonical(summarize_topology(np.asarray(matrix)))
            if (
                previous
                and previous.get("variables") == base["variables"]
                and previous.get("config") == base["config"]
            ):
                prior = previous.get("rooting", {}).get("score_matrix")
                if prior is not None:
                    drift = topology_drift(np.asarray(prior), np.asarray(matrix))
                    migration = root_migration(np.asarray(prior), np.asarray(matrix))
        inference_error = rooting.get("error")
        return dict(
            base,
            status="partial" if inference_error else "available",
            limitation=inference_error,
            observation_count=len(values),
            frequency_unit="cycles per year (quarterly sampling)",
            resonance_depth=result.get("resonance_depths", {}),
            resonance_persistence=result.get("resonance_depth_details", {}),
            frequencies={
                k: v.get("dominant_freq") for k, v in result.get("resonances", {}).items()
            },
            rooting=rooting,
            rooting_backend=rooting.get("method"),
            state_space=result.get("state_space_analysis", {}).get("diagnostics", {}),
            structural_surprise={
                "score": float(surprise.score[-1]),
                "prior_threshold": threshold,
                "threshold_history": len(finite),
                "flagged": surprise_flag,
            },
            topology=topology,
            topology_drift=drift,
            root_migration=migration,
            structural_alert=surprise_flag
            or bool(drift is not None and drift >= config.topology_threshold),
            limitations=[
                "Quarterly histories are short; spectral and dependence estimates can be unstable.",
                "State-space fit/smoothing describe this as-of window; smoothed historical states are retrospective.",
                "Structural surprise and topology thresholds are exploratory heuristics, not calibrated distress probabilities.",
            ],
        )
