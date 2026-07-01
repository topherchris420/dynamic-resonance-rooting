# Research Quality Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Dynamic Resonance Rooting into a research-quality open-source project without changing the scientific claims or core algorithmic behavior.

**Architecture:** Keep DRR's research code intact, move the package into a modern `src/` layout, document the public APIs and workflows, and add deterministic examples and reviewer-facing artifacts. Preserve historical/prototype material by moving it into labeled archival locations instead of deleting it.

**Tech Stack:** Python, NumPy/SciPy, pandas, scikit-learn, NetworkX, matplotlib, pytest, GitHub Actions, Mermaid Markdown diagrams.

---

## Behavior Lock

- Baseline command: `py -m pytest`
- Baseline result before edits: `41 passed`
- Behavior to preserve: public imports from `drr_framework`, deterministic reproduction summary, physics/policy/supervisory examples, analysis/reporting APIs, and existing tests.

## Audit Findings

- Root contains a broken historical Python fragment named `dynamic-resonance-rooting`; it duplicates current package concepts and should be archived as provenance, not executable code.
- Root contains browser demo support files (`style.css`, `script.js`, workers) that obscure package/docs structure; support assets should move under `assets/web-demo/` while `index.html` remains root-compatible for GitHub Pages.
- `data/raw` and `data/processed` are empty files, not usable dataset directories; replace them with real example dataset folders and README guidance.
- `drr_framework/` at the repository root works, but a `src/drr_framework/` layout improves packaging discipline and avoids accidental local-import masking.
- `setup.py` and `pyproject.toml` duplicate package metadata; modernize around `pyproject.toml` and keep `setup.py` minimal only if needed.
- `docs/conf.py` and `docs/index.rst` are placeholders; docs need a Markdown-first research navigation structure.
- Public API docs, developer guide, user guide, FAQ, architecture diagrams, reproducibility instructions, and expected outputs are missing or scattered.
- Community files and issue/PR templates are missing.
- Tooling is minimal: CI runs tests, but lint/format/pre-commit/type-check configuration is absent.
- Some public functions/classes have thin docstrings, direct `print` calls, or mojibake in user-facing labels; fix only presentation issues, not research logic.

## Cleanup Plan

- **Pass 1: Structure and provenance**
  - Move `drr_framework/` to `src/drr_framework/`.
  - Move browser support files to `assets/web-demo/` and update `index.html`.
  - Move the broken root prototype fragment to `paper/legacy-prototype-fragment.md`.
  - Replace empty `data/raw` and `data/processed` placeholder files with dataset folders and README files.

- **Pass 2: Documentation and scientific communication**
  - Rewrite `README.md` as a research landing page.
  - Add `docs/architecture.md` with Mermaid diagrams for workflow, data pipeline, resonance detection, rooting, phase transition detection, and package architecture.
  - Add `docs/api.md`, `docs/user-guide.md`, `docs/developer-guide.md`, `docs/reproducibility.md`, `docs/examples.md`, and `docs/faq.md`.
  - Add `paper/README.md` with paper/supplement guidance that does not invent results.

- **Pass 3: Reproducible examples**
  - Add a deterministic example dataset under `data/raw/`.
  - Add expected reproduction outputs under `data/processed/` or `results/expected/`.
  - Add a practical quickstart example that loads data, detects resonance, computes metrics, visualizes, and exports results.
  - Add tests that execute the new example and validate key outputs.

- **Pass 4: Engineering quality**
  - Modernize packaging metadata in `pyproject.toml`.
  - Add lint/format/pre-commit configuration and improve CI commands.
  - Add targeted docstrings/type hints/logging fixes for presentation-only code paths.
  - Keep dependency additions restricted to optional development tooling.

- **Pass 5: Community and discoverability**
  - Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue templates, and PR template.
  - Add repository keywords/classifiers in package metadata and README guidance for GitHub topics.

## Verification Plan

- Run `py -m pytest`.
- Run `py -m pytest tests/test_examples.py` after adding example coverage.
- Run `py -m compileall src examples scripts tests`.
- Run configured lint/type commands where tools are available; if optional tools are unavailable locally, report that explicitly.
- Inspect `git diff --stat` and changed files before final reporting.
