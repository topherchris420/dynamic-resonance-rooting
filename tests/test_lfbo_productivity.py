from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_regulatory_foundation import definition, observation, dataset
from drr_framework.supervisory.change_detection import detect_material_changes, percentile
from drr_framework.supervisory.peer_analysis import (
    PeerGroupDefinition,
    analyze_peers,
    SignalContext,
)
from drr_framework.supervisory.semantics import SemanticRegistry
from drr_framework.supervisory.vintage import VintageStore
from drr_framework.supervisory.monitoring import (
    MonitoringSignal,
    ReviewState,
    compare_review_states,
    AttentionBudget,
)


def series_records(values, institution="A"):
    return tuple(
        observation(
            p.end_time.date().isoformat(),
            v,
            institution_id=institution,
            original_filing_date="2025-08-01",
        )
        for p, v in zip(pd.period_range("2023Q1", periods=len(values), freq="Q"), values)
    )


def test_quarter_changes_yoy_percentile_and_confidence():
    values = [100, 101, 102, 103, 104, 105, 106, 130]
    d = dataset(series_records(values))
    change = detect_material_changes(d)[0]
    assert change.raw_change == 24
    assert change.yoy_change == 27
    assert change.normalized_change == pytest.approx(24 / 106 * 100)
    assert change.historical_percentile == 100
    assert change.persistence == 7
    assert change.data_confidence == 1
    assert change.flagged
    assert percentile(2, [1, 2, 2, 3]) == 50
    missing = values.copy()
    missing[-2] = None
    c = detect_material_changes(dataset(series_records(missing)))[0]
    assert c.raw_change is None
    assert c.data_confidence < 1


def test_definition_and_perimeter_breaks_prevent_spurious_changes():
    records = list(series_records([100, 101, 102, 103, 104, 105, 106, 500]))
    records[-1] = replace(records[-1], perimeter_version="new perimeter")
    change = detect_material_changes(dataset(records))[0]
    assert not change.flagged and change.raw_change is None
    assert any("breakpoint" in s for s in change.comparison_limitations)


def test_peer_cohort_is_reproducible_and_paired():
    records = tuple(
        o
        for i, change in (("A", 50), ("B", 1), ("C", 2), ("D", 3))
        for o in series_records([100, 100 + change], i)
    )
    cohort = PeerGroupDefinition(
        "test", ("D", "B", "A", "C"), "Synthetic matched peers", "2023-01-01", "2023-01-01"
    )
    args = dict(
        institution="A",
        form="FR Y-9C",
        metric="SYN_ASSETS",
        period="2023-06-30",
        as_of="2025-09-01",
        allow_synthetic=True,
    )
    p = analyze_peers(VintageStore(records), SemanticRegistry((definition(),)), cohort, **args)
    assert p.context == SignalContext.FIRM_SPECIFIC
    assert p.percentile == 100 and p.median == 102
    assert p.included == ("B", "C", "D")
    assert p == analyze_peers(
        VintageStore(records[::-1]), SemanticRegistry((definition(),)), cohort, **args
    )
    with pytest.raises(ValueError, match="not available"):
        analyze_peers(
            VintageStore(records),
            SemanticRegistry((definition(),)),
            replace(cohort, as_of="2026-01-01"),
            **args,
        )


def test_monitoring_suppresses_unchanged_and_deprioritizes_weakened():
    signal = MonitoringSignal(
        "A:assets", "A", "assets", "2025-03-31", "Assets changed", "e1", 70, 1
    )
    prev = ReviewState("2025-08-01", signals=(signal,))
    same = ReviewState("2025-09-01", signals=(signal,))
    assert AttentionBudget().select(compare_review_states(prev, same)).review_first == ()
    lower = replace(same, signals=(replace(signal, materiality=40, evidence_id="e2"),))
    selected = AttentionBudget().select(compare_review_states(prev, lower))
    assert selected.review_first == () and len(selected.deprioritized) == 1
    new = replace(same, signals=(replace(signal, key="B:assets", institution="B"), signal))
    assert len(AttentionBudget(top_n=1).select(compare_review_states(prev, new)).review_first) == 1
    removed = replace(same, signals=())
    assert len(compare_review_states(prev, removed).disappeared) == 1
