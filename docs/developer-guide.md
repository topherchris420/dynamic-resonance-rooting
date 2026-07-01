# Developer Guide

## Repository Layout

```text
src/drr_framework/     Python package
examples/              Runnable research workflows
tests/                 Unit and workflow tests
docs/                  Research, API, and reviewer documentation
data/                  Small checked-in example data
results/expected/      Compact expected outputs
assets/                README and web-demo assets
paper/                 Paper and supplement scaffolding
```

## Local Setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Quality Checks

```bash
python -m pytest
python -m compileall src examples scripts tests
python -m ruff check .
python -m black --check .
```

`ruff`, `black`, `mypy`, and `pre-commit` are optional development tools exposed
through the `dev` extra. The core package dependencies remain focused on the
research runtime.

## Contribution Principles

- Preserve the scientific intent and caveats.
- Add tests before changing behavior.
- Keep examples deterministic with documented random seeds.
- Prefer small, reviewable changes.
- Do not claim validation, performance, or empirical results that are not
  represented by checked-in evidence.

## Release Checklist

- Run tests and syntax checks.
- Regenerate expected reproduction artifacts if the benchmark intentionally
  changes.
- Update README, docs, and citation metadata when public APIs or research
  framing changes.
- Confirm community templates and security guidance remain accurate.
