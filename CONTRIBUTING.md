# Contributing

Thank you for considering a contribution to Dynamic Resonance Rooting (DRR).
This project welcomes research, documentation, examples, tests, and careful
engineering improvements.

## Ground Rules

- Preserve the research voice and caveats.
- Do not invent results, benchmarks, validation claims, or citations.
- Add or update tests before changing behavior.
- Keep examples deterministic where possible.
- Prefer small pull requests with a clear research or engineering rationale.

## Development Setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
python -m pytest
python -m compileall src examples scripts tests
python -m ruff check .
python -m black --check .
```

## Pull Request Checklist

- Explain the motivation and scientific or engineering scope.
- State whether behavior changed.
- Include tests or explain why the change is documentation-only.
- Update docs or examples for public-facing changes.
- Keep validation and deployment claims conservative.

## Research Contributions

For new methods, include:

- Problem statement and assumptions.
- API surface and expected inputs.
- Synthetic or empirical validation plan.
- Limitations and non-goals.
- Reproducible scripts or expected outputs when possible.
