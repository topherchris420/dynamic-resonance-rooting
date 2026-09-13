"""Deterministic smoke benchmark for the heaviest LFBO workbench path."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from drr_framework.supervisory import EvidenceLedger, MonitoringWorkbench, WorkbenchConfig
from drr_framework.supervisory.demo import CURRENT_REVIEW, synthetic_monitoring_lab


def main() -> int:
    store, registry, cohort, policy, entities = synthetic_monitoring_lab()
    with tempfile.TemporaryDirectory() as directory:
        ledger = EvidenceLedger(Path(directory) / "ledger.sqlite")
        workbench = MonitoringWorkbench(
            store,
            registry,
            ledger,
            cohort=cohort,
            policy=policy,
            entities=entities,
            config=WorkbenchConfig(allow_synthetic=True),
        )
        started = time.perf_counter()
        cold, _, _ = workbench.run(CURRENT_REVIEW)
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        warm, _, _ = workbench.run(CURRENT_REVIEW)
        warm_seconds = time.perf_counter() - started
    if cold["state_id"] != warm["state_id"]:
        raise RuntimeError("Cached and uncached monitoring states differ")
    print(
        "LFBO benchmark: "
        f"{len(cold['analyses'])} institutions, "
        f"{sum(len(a['observations']) for a in cold['analyses'])} observations, "
        f"cold={cold_seconds:.3f}s, warm={warm_seconds:.3f}s, "
        f"DRR cache entries={len(workbench._cache)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
