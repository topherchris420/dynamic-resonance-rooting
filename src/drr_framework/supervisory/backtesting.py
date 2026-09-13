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
    policy_context=None,
    institution_class=None,
):
    from ..datasets import RegulatoryAnalysisDataset

    dates = tuple(review_dates)
    if any(instant(b) <= instant(a) for a, b in zip(dates, dates[1:])):
        raise ValueError("Review dates must be unique and increasing")
    rows = []
    previous = None
    drr_cfg = drr_config
    if drr_cfg is None:
        from .analyzer import DRRConfig

        drr_cfg = DRRConfig()
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
        breakpoints = (
            {
                metric: policy_context.breakpoints(
                    as_of=date,
                    form=form,
                    metric=metric,
                    dates=dataset.dates,
                    institution_class=institution_class,
                )
                for metric in dataset.variable_names
            }
            if policy_context is not None
            else {}
        )
        baselines = run_baselines(dataset, baseline_config, breakpoints=breakpoints)
        active_break = any(
            point in dataset.dates[-drr_cfg.lookback :]
            for points in breakpoints.values()
            for point in points
        )
        if not enable_drr:
            drr = {"status": "disabled", "structural_alert": False}
        elif active_break:
            drr = {
                "status": "unavailable",
                "structural_alert": False,
                "limitation": "Reporting comparability break inside the DRR analysis window",
            }
        else:
            drr = LFBORegimeAnalyzer(drr_cfg).analyze(dataset, previous=previous)
        baseline_alert = any(b.flagged for b in baselines) or any(
            c.flagged for c in detect_material_changes(dataset, breakpoints=breakpoints)
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
                breakpoints=breakpoints,
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
            revisions = store.compare_vintages(row["as_of"], dates[-1])
            row["later_revision_count"] = sum(
                revision.current.institution_id == institution
                and revision.current.form == form
                and (metrics is None or revision.current.metric in metrics)
                for revision in revisions
            )
    return {
        "mode": "point_in_time_reconstruction",
        "rows": rows,
        "matched_evaluation": result,
        "boundary": "Uses ingestion-constrained availability. A later download cannot prove what the local system knew historically.",
    }
