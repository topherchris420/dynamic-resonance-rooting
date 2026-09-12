"""Historical monitoring rebuilt from each review date's available records."""

from __future__ import annotations

from .analyzer import LFBORegimeAnalyzer
from .baselines import run_baselines, matched_evaluation
from .common import instant, stable_id
from .change_detection import detect_material_changes


def walk_forward_validate(
    store,
    registry,
    *,
    institution,
    form,
    review_dates,
    metrics=None,
    baseline_config=None,
    drr_config=None,
    allow_synthetic=False,
    enable_drr=True,
    events=None,
):
    from ..datasets import RegulatoryAnalysisDataset

    dates = tuple(review_dates)
    if any(instant(b) <= instant(a) for a, b in zip(dates, dates[1:])):
        raise ValueError("Review dates must be unique and increasing")
    rows = []
    previous = None
    for date in dates:
        dataset = RegulatoryAnalysisDataset.from_vintage_store(
            store,
            registry,
            institution_id=institution,
            form=form,
            as_of=date,
            metrics=metrics,
            allow_synthetic=allow_synthetic,
        )
        baselines = run_baselines(dataset, baseline_config)
        drr = (
            LFBORegimeAnalyzer(drr_config).analyze(dataset, previous=previous)
            if enable_drr
            else {"status": "disabled", "structural_alert": False}
        )
        baseline_alert = any(b.flagged for b in baselines) or any(
            c.flagged for c in detect_material_changes(dataset)
        )
        rows.append(
            dict(
                as_of=instant(date).isoformat(),
                reporting_period=dataset.reporting_period,
                baseline_alert=baseline_alert,
                drr_alert=bool(drr.get("structural_alert")),
                drr_status=drr["status"],
                source_ids=tuple(o.observation_id for o in dataset.observations),
                source_vintages=dataset.filing_vintage,
                input_hash=stable_id(dataset.observations),
                later_revision_count=None,
            )
        )
        previous = drr
    result = matched_evaluation(
        rows, events=None if events is None else tuple(instant(d).isoformat() for d in events)
    )
    # Post-run revision audit is evaluation-only, never an input to historical alerts.
    if dates:
        for row in rows:
            row["later_revision_count"] = len(store.compare_vintages(row["as_of"], dates[-1]))
    return {
        "mode": "point_in_time_reconstruction",
        "rows": rows,
        "matched_evaluation": result,
        "boundary": "Uses ingestion-constrained availability. A later download cannot prove what the local system knew historically.",
    }
