# User Guide

## Installation

```bash
git clone https://github.com/topherchris420/dynamic-resonance-rooting.git
cd dynamic-resonance-rooting
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Quick Start

```python
from drr_framework import DynamicResonanceRooting, generate_coupled_oscillator

sampling_rate = 200.0
_, data = generate_coupled_oscillator(
    sampling_rate=sampling_rate,
    target_frequency_hz=12.5,
    random_state=42,
)

drr = DynamicResonanceRooting(embedding_dim=3, tau=2, sampling_rate=sampling_rate)
results = drr.analyze_system(data, multivariate=True, window_size=256)

print(results["resonance_depths"])
print(results["rooting_analysis"]["significant_edges"])
```

## Choosing A Workflow

- Use `examples/physics_lab.py` for coupled oscillator and physical-system
  diagnostics.
- Use `examples/policy_lab.py` for tabular policy observables.
- Use `examples/supervisory_policy_lab.py` for institution panels, supervisory
  caveats, validation-readiness packets, and Tableau exports.
- Use `examples/quickstart_resonance_export.py` for a compact load-detect-export
  demonstration.

## Interpreting Outputs

- `resonances`: dominant frequencies and spectral evidence.
- `resonance_depths`: scalar persistence/stability scores by dimension.
- `resonance_depth_details`: component-level scores and confidence intervals.
- `rooting_analysis`: directed lagged relationships and edge metadata. Its
  `surrogate_method` names the null the edges survived; the default circular
  shift assumes stationarity, and the [FAQ](faq.md) says when to use another.
- `state_space_analysis`: transition, measurement, likelihood, stability, and
  impulse-response diagnostics.

DRR outputs are research diagnostics. Domain conclusions require separate
validation, calibration, and review.

## Analyze a CSV

After installing the package, run `drr-analyze --help` (or
`python -m drr_framework.csv_analysis --help`). The command works locally without
an account or external data service. Use explicit signal columns so a numeric
identifier or timestamp is not accidentally treated as a signal.

```bash
drr-analyze data/raw/coupled_oscillator_sample.csv \
  --columns dim_0 dim_1 dim_2 \
  --sampling-rate 200 --time-column time_seconds \
  --window-size 256 --max-lag 4 --n-surrogates 99 \
  --random-state 42 --output-dir results/csv-run-001
```

The destination must not already exist. A successful run writes:

| File | Contents |
| --- | --- |
| `report.md` | Channel frequencies and depth, tested edges, interpretation boundaries, and reproduction settings |
| `analysis.json` | Full spectral and depth results, candidate/significant edges, raw and adjusted p-value matrices, settings, source checksum, and dependency versions |

Matrix rows are sources and columns are targets; `rooting.matrix_columns` records
the exact channel order. Lags are exported in samples and seconds. Undefined
numerical entries are JSON `null`, never nonstandard `NaN`. The run ID hashes the
input checksum, row counts, parameters, and recorded versions. Identical inputs
and settings in the same environment produce identical artifacts; this identifier
is not a signature or proof that a scientific claim is valid.

### Input requirements

- UTF-8 CSV with unique, nonempty headers and at least two distinct signal columns.
- At least 32 rows; the depth window (default 256) must fit after preprocessing.
- Every selected value must be numeric and finite. Constant channels are rejected.
- Rows must contain the header's number of fields, including unselected columns.
- Optional timestamps must be numeric **seconds**, strictly increasing and evenly
  spaced, matching `1 / sampling_rate` within relative tolerance `1e-4` and absolute
  tolerance `1e-9` seconds. Without timestamps, regular sampling is an unchecked
  assumption stated in the report. Date strings require explicit conversion first.

The command never sorts, fills gaps, drops rows, or resamples data. Resolve those
choices upstream and retain the preprocessing record. `--difference` explicitly
analyzes first differences and records the loss of one observation. Differencing
can help with a random walk, but does not establish stationarity for arbitrary data.

### Choose the null deliberately

| `--surrogate-method` | Assumption |
| --- | --- |
| `circular_shift` (default) | Stationarity; preserves within-channel dependence apart from a seam |
| `phase_randomization` | Stationary linear Gaussian processes; preserves each channel's periodogram |
| `permutation` | Serial independence; unsuitable for autocorrelated signals |

These assumptions are **not automatically verified**. Consult the
[null-model study](MATH_RESEARCH.md) before drawing conclusions. Spectra use FFT,
rooting uses lagged correlation, and edge selection uses max-statistic correction.
The CLI deliberately keeps those choices fixed; use the Python API for other operators.

`--n-surrogates 0` runs exploration only. Positive surrogate counts still impose a
minimum p-value of `1 / (n_surrogates + 1)`; a report warns when that cannot reach
`--alpha`. No significant edge is not proof of independence. Resonance depth is
descriptive, and statistical direction does not establish causation.

### Common fixes

| Error | Next step |
| --- | --- |
| Column not found | Use exact headers; quote names containing spaces |
| Non-finite or nonnumeric value | Inspect the reported column and CSV line; resolve missing data explicitly |
| Timestamps do not match | Check time units, sample rate, duplicates, ordering, and gaps |
| Window exceeds observations | Choose a smaller `--window-size` (minimum 8) or provide more observations |
| Constant channel | Select a varying observable; inspect preprocessing |
| Output directory exists | Choose a new run directory to preserve earlier evidence |

Invalid data or configuration exits with status 2 and no result bundle. The source
file is never modified. Larger channel counts, lag searches, and surrogate budgets
increase computation; start with a small, justified set of observables.
