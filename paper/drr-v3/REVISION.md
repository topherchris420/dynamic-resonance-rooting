# Revision notes: November 2026

`main.tex` revises *Dynamic Resonance Rooting: Calibrated Resonance and Lead–Lag
Diagnostics for Complex Adaptive Systems*. The September 2026 version (SSRN
5285993, drr-framework 4.3.0 at commit `862c4f7`) is kept here as
`ssrn-5285993-sep2026.pdf`. `main.pdf` is the compiled revision.

Build: `latexmk -pdf main.tex`, or run `pdflatex main.tex` twice. You need a
standard TeX Live install with `pgfplots`. `main.tex` uses the vector figures in
`figures/` when they are present and falls back to cropping the September PDF
otherwise. Figure 4 is drawn inline with pgfplots.

`figures/figure_{1,2,3,5}.pdf` were cut from the September PDF with
`pdftocairo -pdf -r 72 -f PAGE -l PAGE -x X -y Y -W W -H H -paperw W -paperh H
-nocenter -noshrink`. Unlike an `\includegraphics` crop, this drops the rest of
the page, so the compiled PDF carries no hidden September text. To use the
original figure sources instead, overwrite those files.

## What changed

**New evidence from the repository after commit `862c4f7`.** Every number is read from a committed
artifact under `results/expected/`:

- **New Section VIII, "Where the Implemented Test Stays Calibrated."**
  - Dependent-noise null study: fGn up to H = 0.95, GARCH(1,1) and random
    walks, for three nulls. Includes a new pgfplots figure.
  - Phase and harmonic benchmark: rooting is inflated on coherent oscillators
    (0.145), shared trends (0.150), random walks (0.188), shared sweeps (0.098)
    and irregular timestamps (0.072). The seam ablation (0.040 → 0.170) ties the
    oscillator failure to Remark 1.
  - Paired robustness benchmark against raw and AR(1)-residual correlation.
    At H = 0.95 this benchmark and the dependent-noise study disagree, but their
    intervals overlap, so the revision calls H = 0.95 borderline.
- **Section IX now covers three preregistered external tests.** QBO is
  unchanged. Added: the daily crypto-return financial holdout and the lead–lag
  breakdown study, both *not supported*. A summary table lists all three.
- **Evidence ledger:** a new claim and status table that includes the phase
  test, the DRR-Mellin benchmark and the withdrawn financial-stress figures.
- **Abstract, introduction, contributions** (new C4 and C5), **discussion,
  limitations, next steps, conclusion and appendix** are updated to match.

**Independent reconstruction of the v3 studies (new Section VII).** The
original v3 scripts were lost, so the studies were rebuilt from a protocol
committed before any run (`experiments/v3_reconstruction/protocol.json`). The
code is `src/drr_framework/v3_reconstruction.py` and
`scripts/run_v3_reconstruction.py`; the artifact and its rendered report are in
`results/expected/v3_reconstruction/`.

- All 12 registered findings (F1–F12) hold. All 123 compared Monte Carlo rates
  are consistent with the published values under Fisher tests with Holm
  correction. All 20 QBO quantities agree. The eight committed v2 rows the
  protocol names reproduce trial for trial.
- One comparison failed its rule: the long-run raw reverse correlation at
  φ = 0.9 is 0.218, against 0.228 in Table 1. Prop. 2 predicts 0.219, so the
  published value is the one off. Table 1 keeps 0.228, as the author chose, with
  a footnote giving the rebuilt 0.218 and the predicted 0.219.
- New exploratory observation, not registered: Λ's maximum lands on the last
  interior Fourier bin more often than chance. The unexplained 2.0-month line
  in the 30 hPa QBO series sits on that bin, and the QBO text now says it may
  be an estimator artifact.
- New Table 5 compares published and rebuilt values for the cells the claims
  rest on. The ledger marks the v3 rows "reconstructed" and adds a row for
  F1–F12. Contribution C7, next step (1) and the conclusion are updated.

**Corrections and clarifications to the September text**

- The PDF repeated the abstract heading ("Abstract / Abstract—"). It now
  appears once.
- Eq. (1): the time average in C and the mean envelope in A are now written out.
- §IV-C: the source process is stated explicitly,
  x_t = φx_{t−1} + √(1−φ²) ε_t. Prop. 2(ii) relies on this form.
- Welch [25] was cited for log-parabolic peak interpolation, which that paper
  does not cover. It is now cited only for the spectrum.
- Wilson intervals were used but never cited. Wilson (1927) is now cited.
- Remark 1 now points forward to the measured seam failure.
- The §IX-B QBO reverse-edge discussion notes that the monthly QBO is close
  to the coherent-oscillator regime, where the v2 null is anti-conservative.
- The abstract, C6, Limitations, data availability and the appendix now say
  that the original v3 scripts and artifacts were lost, and point to the
  reconstruction that replaces them.
- Section cross-references now use the same Roman numerals as the headings.
- Theorem 1 is described everywhere as a statement about the distribution of an
  invariant score, not the score itself.
- The text separates the exact family-wise error guarantee of Prop. 1 (circular
  stationarity, mutual independence) from the measured calibration of the
  implemented test.
- Monotone spectra are presented as the chosen null class for "memory without
  rhythm", not as a property of all processes with memory; AR(2) with complex
  roots is given as a counterexample.

**Unchanged:** Thm. 1, Props. 1–2 and their proofs, and every value in the
original Tables 1–5 and Figures 1–4. The September tables and figures appear
in the same order. The figure numbers referenced by the appendix shifted
because the new figure is placed earlier.

## Open items for the author

- The original v3 study scripts (`run_study.py`, `run_spectral_excess.py`,
  `make_figures.py`) and their JSON artifacts were lost. The tables and figures
  still show the original values; the reconstruction checks them statistically
  but not number for number. Figures 2, 3 and 5 were not redrawn from the
  reconstruction.
- The September appendix reported a test count (398 passed, 18 skipped) at
  commit `862c4f7`. The revision drops it because the suite has changed and was
  not rerun for this revision.
- `figures/` holds crops of the September figures. If the original figure
  sources are available, overwrite those files with them; `main.tex` will pick
  them up without further changes.
