"""The README's performance language stays tied to the external artifact."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "expected" / "qbo_structural_change_benchmark.json"


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_architecture_names_core_substrate_and_adapters():
    readme = _read("README.md")
    architecture = _read("docs/architecture.md")
    assert "DRR core → validation substrate → domain adapters" in architecture
    for layer in ("validation substrate", "Domain adapters"):
        assert layer.lower() in readme.lower()
        assert layer.lower() in architecture.lower()
    assert "A domain claim stays in its adapter." in architecture


def test_narrative_benchmarks_point_at_the_external_artifact():
    benchmarks = _read("DRR_BENCHMARKS.md")
    report = _read("DRR_VALIDATION_REPORT.md")
    assert "narrative summaries" in benchmarks
    assert "qbo_structural_change_benchmark.json" in benchmarks
    assert "narrative validation memo" in report
    assert "qbo_structural_change_benchmark.json" in report


def test_reviewed_claim_status_is_the_status_named_in_the_docs():
    status = json.loads(ARTIFACT.read_text(encoding="utf-8"))["claim"]["status"]
    assert status == "not_supported"
    for name in (
        "README.md",
        "docs/external-evidence.md",
        "docs/faq.md",
        "docs/architecture.md",
        "DRR_BENCHMARKS.md",
        "DRR_VALIDATION_REPORT.md",
        "DRR_LIMITATIONS.md",
    ):
        assert status in _read(name)
