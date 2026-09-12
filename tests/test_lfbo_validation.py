from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_regulatory_foundation import observation, definition, dataset
from drr_framework.supervisory.baselines import run_baselines, matched_evaluation
from drr_framework.supervisory.analyzer import LFBORegimeAnalyzer, DRRConfig
from drr_framework.supervisory.falsification import (
    falsify_material_change,
    run_falsification,
    SignalSpecification,
    SpecificationResult,
)
from drr_framework.supervisory.backtesting import walk_forward_validate
from drr_framework.supervisory.vintage import VintageStore
from drr_framework.supervisory.semantics import SemanticRegistry


def history(n=28):
    rows = []
    for i, p in enumerate(pd.period_range("2017Q1", periods=n, freq="Q")):
        filing = (p.end_time.normalize() + pd.Timedelta(days=35)).date().isoformat()
        rows.append(
            observation(
                p.end_time.date().isoformat(),
                100 + 0.3 * i + np.sin(i),
                original_filing_date=filing,
                ingestion_date=filing,
                available_as_of=filing,
            )
        )
    return tuple(rows)


def test_baselines_exclude_current_from_reference_and_do_not_invent_event_quality():
    records = list(history())
    records[-1] = replace(records[-1], value=200)
    result = run_baselines(dataset(records))
    assert any(r.method == "robust_z" and r.flagged for r in result)
    matched = matched_evaluation(
        [{"as_of": "2020-01-01", "baseline_alert": False, "drr_alert": True}]
    )
    assert matched["incremental_alert_count"] == 1
    assert matched["incremental_detection_rate"] is None
    assert matched["outcomes_status"].startswith("not_estimable")


def test_semantic_drr_mapping_and_determinism():
    d = dataset(history())
    result = LFBORegimeAnalyzer(DRRConfig(rooting_n_surrogates=9)).analyze(d)
    assert result["status"] == "available"
    assert set(result["resonance_depth"]) == {"SYN_ASSETS"}
    assert "dim_0" not in str(result)
    assert result == LFBORegimeAnalyzer(DRRConfig(rooting_n_surrogates=9)).analyze(d)


def test_falsification_reports_failures_and_is_deterministic():
    d = dataset(history())
    assert falsify_material_change(d, "SYN_ASSETS") == falsify_material_change(d, "SYN_ASSETS")

    def evaluate(spec):
        if spec.name == "failed":
            raise ValueError("deliberate unsupported specification")
        return SpecificationResult(spec, True, 1, 4)

    report = run_falsification(
        "test", [SignalSpecification("ok"), SignalSpecification("failed")], evaluate
    )
    assert report.percentage_surviving == 50 and report.classification == "mixed"
    assert report.failure_cases and report.evaluated_count == 1


def test_walk_forward_remains_unchanged_when_future_revision_is_appended():
    records = history()
    dates = [o.available_as_of for o in records[-3:]]
    args = dict(
        institution="A", form="FR Y-9C", review_dates=dates, allow_synthetic=True, enable_drr=False
    )
    initial = walk_forward_validate(
        VintageStore(records), SemanticRegistry((definition(),)), **args
    )
    old = records[-3]
    revision = replace(
        old,
        value=900,
        amendment_date="2025-10-01",
        ingestion_date="2025-10-01",
        available_as_of="2025-10-01",
        source_vintage="later",
        supersedes=old.observation_id,
    )
    future = walk_forward_validate(
        VintageStore(records + (revision,)), SemanticRegistry((definition(),)), **args
    )
    assert initial == future
