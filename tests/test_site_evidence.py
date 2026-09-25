"""The landing page shows computed numbers, and its evidence matches the artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "index.html"
SCRIPT = ROOT / "assets" / "web-demo" / "script.js"

sys.path.insert(0, str(ROOT / "scripts"))
import sync_site_evidence  # noqa: E402


def _embedded() -> dict:
    match = sync_site_evidence.BLOCK.search(PAGE.read_text(encoding="utf-8"))
    assert match, "index.html lost its evidence block"
    return json.loads(match.group(2))


def test_page_evidence_matches_the_committed_artifacts():
    assert _embedded() == sync_site_evidence.build_evidence()
    assert sync_site_evidence.main(["--check"]) == 0


def test_page_reports_the_reviewed_external_claim_status():
    qbo = _embedded()["qbo"]
    artifact = json.loads(sync_site_evidence.QBO.read_text(encoding="utf-8"))
    assert qbo["status"] == artifact["claim"]["status"] == "not_supported"
    assert "not supported" in PAGE.read_text(encoding="utf-8").lower()


def test_page_runs_the_tested_engine_and_no_hardcoded_readings():
    page = PAGE.read_text(encoding="utf-8")
    assert '<script src="assets/web-demo/drr-engine.js"></script>' in page
    assert page.index("drr-engine.js") < page.index("assets/web-demo/script.js")
    # The v4.3 page printed readings such as 0.847 and 84.7% that no computation produced.
    visible = re.sub(
        r'<script type="application/json" id="evidence-data">.*?</script>', "", page, flags=re.S
    )
    # Slider <output> elements echo their input's value; everything else must be computed.
    visible = re.sub(r"<output[^>]*>.*?</output>", "", visible, flags=re.S)
    assert not re.search(r">\s*\d+\.\d{2,}\s*%?\s*<", visible)
    script = SCRIPT.read_text(encoding="utf-8")
    assert "DRR.rooting(" in script and "DRR.depth(" in script
