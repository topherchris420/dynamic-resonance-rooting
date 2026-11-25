# Amendments

Changes to `protocol.json` after registration (2026-11-25), each with its
reason and whether any study result had been seen. None so far.

## 1. Clarifications recorded before the full run (2026-11-25)

The previous session's operator and runner code was lost before it was
committed (the session stopped mid-work). The code was rewritten from
`protocol.json` in `src/drr_framework/v3_reconstruction.py` and
`scripts/run_v3_reconstruction.py`. Writing it required fixing three details
that the protocol does not state. None changes a registered choice.

- **Random-number streams inside a resonance family.** One
  `numpy.random.default_rng(seed + 1000*j)` per family is consumed in a fixed
  order within each trial: the series, then the 99 AR(1) null draws, the 99
  isotonic null draws, then the 99 Lambda surrogates.
- **Random-number streams in the QBO spectral checks.** One generator seeded
  20760926 is consumed for u30 (AR(1) D*, isotonic D*, Lambda), then u50 in the
  same order.
- **QBO-band ratio.** Taken as the largest I_k / S-hat_k over Fourier bins
  18-24; its period is 572 / k months.

A smoke run at reduced trial and surrogate counts (`--smoke`) was made to check
the wiring before the full run. Its outputs are not study results, and no
implementation choice was changed after it.

## 2. Run record and one exploratory observation (2026-11-25)

The full run (`scripts/run_v3_reconstruction.py --workers 4`) finished in
692 s under Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1. Its artifact is
`results/expected/v3_reconstruction/v3_reconstruction.json`. No registered
choice was changed after the run, and no study was rerun.

While writing unit tests (before the full run), a Lambda argmax was seen on the
last interior Fourier bin of a line-in-red-noise series. A separate check
after the run (seed 20261125, 1,000 series each, no surrogates) found the
Lambda argmax on bin 255 of 255 in 16 white-noise and 28 unit-variance AR(0.9)
series, against about 3.9 expected if the argmax fell on every bin equally
often. The 30 hPa QBO argmax (2.0 months) is also the last interior bin (285 of
285). This is an exploratory observation about where the isotonic fit leaves
the largest ratios. It is not a registered finding and it changes no result.
