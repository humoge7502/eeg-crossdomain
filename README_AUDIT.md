# Audit module — read this before the LODO transfer result means anything

This folder is designed to sit inside your existing `eeg-crossdomain/`
repo as-is (`src/` and `scripts/` merge directly; nothing here overwrites
Stage 1/2/4/5/6 of the pipeline described in the project reference doc).

## Why this exists

Both of your submitted papers share one move: before reporting a model's
headline number, build the control that would let the number collapse,
and report what happens.

- NeuroClick: the dwell-plus-propensity logistic baseline showed the
  first-visit gain came from propensity, not the sequence architecture.
- INS-HDGS-CMT: the density-matched static/random graph nulls showed the
  "dynamic functional graph" pathway worked through node-feature
  processing, not the measured connectivity it was built to capture.

This project's core claim — that harmonization + domain alignment lets a
model trained on two neuromarketing paradigms transfer to a third — needs
the same treatment, and needs it *before* the main LODO experiment
(`scripts/04_run_lodo_experiment.py`) is run and written up. Two things
are added:

### 1. `src/construct_floor.py` — the construct-floor check

Answers: how much of each dataset's own label can a plain linear probe,
using only that dataset's own generic band-power features, recover with
*no* cross-dataset information at all? This is the ceiling any transfer
result has to clear. It is the direct analogue of INS-HDGS-CMT's linear
probe on the label's own defining terms (0.92 ROC-AUC from gaze alone),
which made every gaze-touching model's headline score uninterpretable
without that number sitting next to it.

### 2. `src/null_alignment.py` — the null-alignment control

Answers: does CORAL/Riemannian alignment (Stage 3) beat alignment
procedures that are shape-matched but structurally empty? Two nulls:
- **shuffled-identity**: same alignment math, fake domain labels
- **random-target**: same alignment math, a random covariance matched
  only on scale to the real target

This is the direct analogue of the density-matched static/random graph
nulls in INS-HDGS-CMT Section 3.4.

### `scripts/05_run_audit.py`

Runs both checks, writes `results/audit_report.md`, and states the
decision rule explicitly: the main transfer result is only reportable as
evidence of real shared structure if it (1) exceeds the construct floor
on the held-out dataset, AND (2) significantly beats all three nulls
after Holm correction, within its own declared family.

## Current status

Everything here runs on **synthetic, clearly-labelled placeholder data**
(see the `# VERIFY` markers in `construct_floor.py` and
`null_alignment.py`) — the same convention your existing smoke tests
already use, since this sandbox has no access to the real Figshare/
OpenNeuro downloads either. On Brev:

1. Confirm `scripts/02_preprocess_all.py`'s Stage-2 cache format.
2. Point `load_harmonized_features()` in `src/construct_floor.py` at
   that real cache instead of `_synthetic_harmonized_dataset()`.
3. Re-run `scripts/05_run_audit.py` — nothing else changes.
4. Only then run `scripts/04_run_lodo_experiment.py` and interpret its
   output against `results/audit_report.md`.

## Direct test run on this synthetic placeholder data

`real_alignment` did **not** significantly beat any of the three nulls
after Holm correction on the synthetic data used for smoke-testing here
— which is the correct behavior of the audit, since the synthetic
generator has no real cross-dataset structure built in by design. This
is not a result about your actual datasets; it's confirmation the
decision rule fires correctly on data where it should.
