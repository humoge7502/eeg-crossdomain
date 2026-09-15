# Cross-domain EEG project — status and priority order

GPU window on Brev closes 25th of this month. This file is the single
source of truth for what's done, what's blocking, and what order to run
things in. Read patches/README.md for exact commands.

## Priority order (do these in sequence)

1. **`patches/`** — DONE (2026-09-15). Patches applied (`diff -q patches/train.py
   src/train.py` -> identical), one LODO pass run, predictions saved
   (`results/lodo_predictions_*.npz`), permutation+equivalence audit complete:
   all three folds p_holm = 1.000 AND equivalent to null within +/-0.10 kappa
   (`results/lodo_permutation_equivalence_report.md`). See EXPERIMENT_LOG
   Section 17 for interpretation.

2. **Restaurant-Logo motor confound** — DONE (2026-09-15). Built and ran
   `scripts/25_rl_motor_confound_control.py`: epochs truncated to [0, median RT
   = 0.844 s) — strictly before the median keypress, one global cutoff so no
   per-trial RT leaks into the features — same 36-d features, same LOSOCV
   protocol, family-level within-subject permutation control. Report:
   `results/rl_motor_confound_report.md` (+ EXPERIMENT_LOG Section 30).

3. **ds007406-harmonization sensitivity check** — DONE (2026-09-15). LODO
   re-run with ds007406 dropped (`configs/default_no_ds007406.yaml --tag nods`):
   both remaining folds null AND equivalent to null (NeuMa kappa 0.0011,
   RL kappa 0.0000, p_holm = 1.000, CIs within +/-0.10). The null-transfer
   claim does not depend on ds007406's harmonization.
   Report: `results/lodo_permutation_equivalence_nods.md` (+ Section 29).

## Already done (do not redo)

- Epoch-wise subject-leakage bug found and fixed (Section 12 of your
  EXPERIMENT_LOG.md) — within-dataset baselines already re-validated
  subject-wise.
- Six-condition null-transfer sweep (Section 15) — solid, keep as is.
- construct_floor.py / null_alignment.py audit modules (see src/) — these
  test the domain-ALIGNMENT step specifically (Stage 3), a different
  question from the permutation test above (which tests the LODO
  transfer RESULT itself). Run both; they check different things.

## Known open risk

The synthetic smoke-test of scripts/06 (pure noise, no real signal)
showed that ds007406 (n=10) and possibly restaurant_logo (n=15) may not
have enough subjects to support an EQUIVALENCE claim even when the true
effect is exactly zero. If the real run shows this too, report those
folds as underpowered/inconclusive rather than null — don't force a
"no transfer" reading onto a result that's actually "couldn't tell."
