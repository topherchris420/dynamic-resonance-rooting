# Expected Outputs

This directory contains compact expected outputs for examples and smoke-test
workflows. They are intentionally small so reviewers can compare a fresh run
against committed reference artifacts without downloading external data.

`qbo_structural_change_benchmark.json` is the preregistered external comparison
on the vendored NOAA CPC QBO snapshots. The Markdown file beside it is rendered
from that JSON. Regenerate both with
`python scripts/run_structural_change_benchmark.py --output-dir results/expected`.
