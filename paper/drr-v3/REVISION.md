# Revision notes: November 2026

`main.tex` revises *Dynamic Resonance Rooting: Calibrated Resonance and Lead–Lag
Diagnostics for Complex Adaptive Systems*. The September 2026 version (SSRN
5285993, drr-framework 4.3.0 at commit `862c4f7`) is kept here as
`ssrn-5285993-sep2026.pdf`. `main.pdf` is the compiled revision.

Build: `latexmk -pdf main.tex`, or run `pdflatex main.tex` twice. You need a
standard TeX Live install with `pgfplots`. Figures 1–3 and the QBO figure are
cropped from the September PDF, so keep that file next to `main.tex`.

## What changed

**New evidence from the repository after commit `862c4f7`.** Every number is read from a committed
artifact under `results/expected/`:

- **New Section VII, "Where the Surrogate Null Holds."**
  - Dependent-noise null study: fGn up to H = 0.95, GARCH(1,1) and random
    walks, for three nulls. Includes a new pgfplots figure.
  - Phase and harmonic benchmark: rooting is inflated on coherent oscillators
    (0.145), shared trends (0.150), random walks (0.188), shared sweeps (0.098)
    and irregular timestamps (0.072). The seam ablation (0.040 → 0.170) ties the
    oscillator failure to Remark 1.
  - Paired robustness benchmark against raw and AR(1)-residual correlation.
    At H = 0.95 this benchmark and the dependent-noise study disagree, but their
    intervals overlap, so the revision calls H = 0.95 borderline.
- **Section VIII now covers three preregistered external tests.** QBO is
  unchanged. Added: the daily crypto-return financial holdout and the lead–lag
  breakdown study, both *not supported*. A summary table lists all three.
- **Evidence ledger:** a new claim and status table that includes the phase
  test, the DRR-Mellin benchmark and the withdrawn financial-stress figures.
- **Abstract, introduction, contributions** (new C4 and C5), **discussion,
  limitations, next steps, conclusion and appendix** are updated to match.

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
- The §VIII-B QBO reverse-edge discussion notes that the monthly QBO is close
  to the coherent-oscillator regime, where the v2 null is anti-conservative.
- Limitations now state that the v3 operators and their study scripts are not
  yet in the repository.

**Unchanged:** Thm. 1, Props. 1–2 and their proofs, and every value in the
original Tables 1–5 and Figures 1–4. The September tables and figures appear
in the same order. The figure numbers referenced by the appendix shifted
because the new figure is placed earlier.

## Open items for the author

- The v3 study scripts (`run_study.py`, `run_spectral_excess.py`,
  `make_figures.py`) and their JSON artifacts are cited as accompanying the
  manuscript, but they are not in this repository. Tables 1–4 (v3 columns), the
  Λ column and Figures 2–4 cannot be checked from the repository until they
  are added.
- The September appendix reported a test count (398 passed, 18 skipped) at
  commit `862c4f7`. The revision drops it because the suite has changed and was
  not rerun for this revision.
- The figure crops come from the September PDF. If the figure sources are
  available, replacing the crops with the original vector files would be
  cleaner.
