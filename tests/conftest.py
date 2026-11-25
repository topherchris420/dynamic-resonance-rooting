import pytest

from drr_framework.supervisory.demo import synthetic_monitoring_lab
from drr_framework.supervisory.evidence_ledger import EvidenceLedger
from drr_framework.supervisory.workbench import MonitoringWorkbench, WorkbenchConfig


@pytest.fixture
def lab(tmp_path):
    store, registry, cohort, policy, entities = synthetic_monitoring_lab()
    ledger = EvidenceLedger(tmp_path / "ledger.sqlite")
    return MonitoringWorkbench(
        store,
        registry,
        ledger,
        cohort=cohort,
        policy=policy,
        entities=entities,
        config=WorkbenchConfig(allow_synthetic=True, enable_drr=False),
    )
