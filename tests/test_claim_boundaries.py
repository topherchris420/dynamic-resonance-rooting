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


def test_benchmark_documents_point_at_both_artifacts():
    for name in ("DRR_BENCHMARKS.md", "DRR_VALIDATION_REPORT.md"):
        text = _read(name)
        assert "qbo_structural_change_benchmark.json" in text, name
        assert "rooting_calibration_study.json" in text, name


def test_withdrawn_figures_do_not_return():
    """These figures had no generating code or data in the repository."""
    withdrawn = (
        "**0.842**",
        "**0.891**",
        "**18.5**",
        "+185 bps",
        "+6 to +10 days",
        "12 to 18 days",
        "D_R < 0.25$) with 0",
    )
    for name in ("DRR_BENCHMARKS.md", "README.md", "docs/faq.md"):
        text = _read(name)
        for figure in withdrawn:
            assert figure not in text, (name, figure)
    assert "Empirical Answer" not in _read("DRR_BENCHMARKS.md")


def test_benchmark_calibration_rows_are_the_artifact_rows():
    from drr_framework.calibration import render_calibration_report

    calibration = json.loads(
        (ROOT / "results" / "expected" / "rooting_calibration_study.json").read_text(
            encoding="utf-8"
        )
    )
    rendered = set(render_calibration_report(calibration).splitlines())
    benchmarks = _read("DRR_BENCHMARKS.md")
    rows = [
        line
        for line in benchmarks.splitlines()
        if line.startswith("| 0.")
        or line.startswith("| 1.")
        or line.startswith("| ar1_")
        or line.startswith("| tone_")
    ]
    calibration_rows = [row for row in rows if "conventional" not in row and "ablation" not in row]
    assert len(calibration_rows) >= 20
    for row in calibration_rows:
        assert row in rendered, row


def test_reviewed_claim_status_is_the_status_named_in_the_docs():
    status = json.loads(ARTIFACT.read_text(encoding="utf-8"))["claim"]["status"]
    assert status == "not_supported"
    assert "not supported" in _read("README.md").lower()
    for name in (
        "docs/external-evidence.md",
        "docs/faq.md",
        "docs/architecture.md",
        "DRR_BENCHMARKS.md",
        "DRR_VALIDATION_REPORT.md",
        "DRR_LIMITATIONS.md",
    ):
        assert status in _read(name)
