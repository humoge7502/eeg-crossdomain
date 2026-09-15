# How to apply these patches — read this first, deadline-critical

GPU window closes 25th. Here is the minimum path to a submittable core
statistical result, in order.

## Step 1 — apply two patches (no GPU needed, do this first, ~2 minutes)

```bash
cd eeg-crossdomain-workspace
cp patches/train.py src/train.py
cp patches/04_run_lodo_experiment.py scripts/04_run_lodo_experiment.py
```

Both are complete drop-in replacements (not diffs) — every line of your
original logic is preserved; the only additions are marked `# PATCH` in
each file. Training logic, architecture, loss, optimizer, and
hyperparameters are byte-for-byte unchanged.

## Step 2 — one LODO run (GPU, same cost as your existing lodo_log2.txt run)

```bash
nohup python3 scripts/04_run_lodo_experiment.py > lodo_log3.txt 2>&1 &
```

This now additionally saves, per held-out fold:
`results/lodo_predictions_<dataset>.npz` (y_true, y_prob, subject_ids).

**Check the console output for one thing specifically**, printed by the
patched loader at the top of the log:
```
[lodo] Loaded neuma: ..., subject_ids=present (42 subjects)
[lodo] Loaded restaurant_logo: ..., subject_ids=MISSING
```
If any dataset says `subject_ids=MISSING`, the permutation test for that
dataset's fold will run in an anti-conservative, trial-level fallback
mode (script 06 will say so loudly in its own output and in the report).
If you see this, the fix is upstream — check whether that dataset's
`_epochs_common.npz` was built by a version of
`scripts/02b_resample_truncate_epochs.py` before subject_id tracking was
added, and re-run 02b for that dataset specifically before proceeding
(no need to redo Stage 1 preprocessing, just the resample/truncate step).

## Step 3 — permutation + equivalence audit (CPU, seconds, run as many times as you like)

```bash
pip install scikit-learn numpy --quiet  # already installed if scripts 02-04 run
python3 scripts/06_lodo_permutation_and_equivalence.py \
    --n-permutations 2000 \
    --equivalence-margin 0.10 \
    --out results/lodo_permutation_equivalence_report.md
```

This does NOT retrain anything — it reuses the fixed predictions from
Step 2. You can safely re-run it with different `--equivalence-margin`
values (e.g. 0.05 for a stricter bound, if a reviewer pushes back on 0.10
as too generous) without touching the GPU again.

## What "done" looks like

Open `results/lodo_permutation_equivalence_report.md`. For the null-transfer
claim to be submittable as currently framed, you want, per held-out
dataset:
- `p_holm` NOT significant on kappa and AUC (supports "no detected
  transfer"), AND
- `equivalent to null? = YES` (supports "and we had the power to say so
  within +/-0.10 kappa")

If a fold shows neither — not significant AND not equivalent — that
fold is underpowered, not null. Say so explicitly in the manuscript
rather than reporting it as a clean negative. Based on the synthetic
smoke-test included with this patch set, this is a real risk for
`ds007406` (n=10) and possibly `restaurant_logo` (n=15) even under pure
noise — plan the writeup to handle that outcome rather than being
surprised by it.

## If you're tight on time and can only do one thing

Step 1 + Step 2 + Step 3 in that order, once, on the real data. That
alone converts the paper's current "we observed near-zero kappa" into
"we observed near-zero kappa, formally indistinguishable from chance
(p=X, Holm-corrected), and equivalent to no meaningful transfer within
+/-0.10 kappa (or: underpowered to make that claim, for dataset Y)" —
which is the single highest-value fix available before submission.
