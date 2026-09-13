"""Complete analyst workflow: source observations through briefs and a passport.

Run: python examples/lfbo_monitoring_workbench.py
Interactive review: drr-monitor --demo --serve
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drr_framework.supervisory.demo import (
    synthetic_monitoring_lab,
    synthetic_perspectives,
    PREVIOUS_REVIEW,
    CURRENT_REVIEW,
)
from drr_framework.supervisory.evidence_ledger import EvidenceLedger, AnalystReview
from drr_framework.supervisory.feedback import evaluate_signal_usefulness
from drr_framework.supervisory.workbench import MonitoringWorkbench, WorkbenchConfig
from drr_framework.supervisory.cli import export_run
from drr_framework.supervisory.backtesting import walk_forward_validate
from drr_framework.supervisory.common import canonical_json, write_immutable
from drr_framework.supervisory.issues import MonitoringIssue, IssueTracker


def main(output_dir="results/lfbo-monitoring-lab"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Data, semantic definitions and cohort are generated independently of the models.
    store, registry, cohort, policy, entities = synthetic_monitoring_lab()
    perspectives = synthetic_perspectives(store)
    store.export_jsonl(output / "synthetic-observations.jsonl")
    write_immutable(output / "synthetic-registry.json", canonical_json(registry.definitions) + "\n")
    ledger = EvidenceLedger(output / "ledger.sqlite")
    workbench = MonitoringWorkbench(
        store,
        registry,
        ledger,
        cohort=cohort,
        policy=policy,
        entities=entities,
        perspectives=perspectives,
        config=WorkbenchConfig(allow_synthetic=True),
    )
    before, previous, _ = workbench.run(PREVIOUS_REVIEW)
    current, state, passport = workbench.run(CURRENT_REVIEW, previous=previous)
    selected = current["attention"]["review_first"]
    if selected:
        oid = selected[0]["signal"]["evidence_id"]
        ledger.review(
            AnalystReview(
                oid,
                "investigate",
                "Synthetic analyst",
                "The funding movement survives baseline checks; inspect the synthetic filing inputs and peer exclusions.",
                CURRENT_REVIEW,
                3,
            )
        )
        IssueTracker(output / "issues.sqlite").append(
            MonitoringIssue(
                "demo-funding-review",
                "DEMO-A",
                "Reconcile funding movement",
                "Check filing details and selected peers",
                "synthetic_test",
                (oid,),
                "normal",
                "investigating",
                "Synthetic analyst",
                CURRENT_REVIEW,
                CURRENT_REVIEW,
                unresolved_questions=(
                    "Does the filing explain the change in funding composition?",
                ),
            ),
            ledger,
        )
    # Re-run at the same cutoff to include the recorded analyst disposition.
    current, state, passport = workbench.run(CURRENT_REVIEW, previous=previous)
    destination = export_run(current, state, passport, ledger, output)
    dates = ("2025-08-10", "2025-11-10", PREVIOUS_REVIEW, CURRENT_REVIEW)
    validation = walk_forward_validate(
        store,
        registry,
        institution="DEMO-A",
        form="FR Y-9C",
        review_dates=dates,
        allow_synthetic=True,
    )
    write_immutable(destination / "walk-forward-validation.json", canonical_json(validation) + "\n")
    write_immutable(
        destination / "usefulness.json",
        canonical_json(evaluate_signal_usefulness(ledger)) + "\n",
    )
    print(f"Filing revisions: {len(store.compare_vintages(PREVIOUS_REVIEW,CURRENT_REVIEW))}")
    print(
        f"Review first: {len(current['attention']['review_first'])}; deprioritized: {len(current['attention']['deprioritized'])}"
    )
    print(f"Open {destination/'index.html'}")
    return destination


if __name__ == "__main__":
    main()
