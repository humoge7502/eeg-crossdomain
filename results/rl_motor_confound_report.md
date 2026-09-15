# Restaurant-Logo motor-confound control

The keypress differs by class and epochs are stimulus-locked [0, 3] s,
so the post-keypress motor period is inside the window. Control:
truncate every epoch to [0, cutoff) s - strictly BEFORE the median
keypress - recompute the same 36-d band-power features, and re-run the
identical within-subject LOSOCV protocol.

- trials: 440, subjects: 15
- cutoff = median RT pooled across subjects: 0.844 s
- trials whose keypress falls outside the truncated window (rt >= cutoff): 52.7%
- full window: [0, 3.0) s | truncated: [0, 0.844) s (28.1%)
- permutation: 50 family-level within-subject shuffles, p on mean-fold AUC,
  decision-function statistic for both observed and null
- runs: seed 42 (`rl_motor_confound_report.md`) and seed 7
  (`rl_motor_confound_report_seed7.md`); observed LOSOCV statistics are
  identical across seeds (permutation p varies with the null draw)

## Results

| Window | Model | kappa (mean +/- std) | ROC-AUC | p_perm (AUC) seed 42 | p_perm (AUC) seed 7 |
|---|---|---|---|---|---|
| full | logreg | 0.1140 +/- 0.1902 | 0.6333 | 0.0196 | 0.0196 |
| full | svm_rbf | -0.0027 +/- 0.0562 | 0.5692 | 0.0784 | 0.1373 |
| truncated | logreg | 0.1148 +/- 0.2156 | 0.5685 | 0.0588 | 0.0784 |
| truncated | svm_rbf | 0.0149 +/- 0.1483 | 0.6332 | 0.0196 | 0.0392 |

## Interpretation (read this before quoting either row)

1. **The full-window signal is weak and model-inconsistent.** logreg
   decodes (kappa 0.114, AUC 0.633, p ~ 0.02 in both seeds) but svm_rbf
   does not (kappa -0.003, AUC 0.569, p not significant in either seed).
   A within-dataset signal that one standard model finds and the other
   does not is fragile; do not headline it.

2. **Truncation before the median keypress does NOT abolish the signal**
   - so the keypress-locked motor period is not solely carrying it - but
   it does NOT preserve it cleanly either: the surviving decodability
   re-tunes to a different model/statistic (svm_rbf AUC 0.633, logreg
   AUC 0.569), i.e. the residual pre-press signal is not the same signal
   as the full-window one.

3. **The honest writeup for this dataset** (PROJECT_STATUS item 2 asked
   for either a control or an explicit caveat; here is both):

   > Restaurant-Logo's stimulus-locked [0, 3] s epochs include the
   > class-informative keypress and its motor aftermath. Truncating to
   > [0, 0.844) s - before the median keypress, with 52.7% of keypresses
   > fully outside the window - leaves only weak, model-dependent
   > decodability (best AUC 0.633, p_perm 0.0196/0.0392 across seeds;
   > the companion model is at chance in the same window). Recognition
   > related activity beyond the motor confound is therefore weak and
   > unstable for this dataset, and its contribution to cross-dataset
   > transfer claims should be treated as uninterpretable-in-magnitude
   > rather than cleanly positive. Note the null-transfer LODO result
   > (Sections 17, 19, 29) is unaffected: a dataset whose within-dataset
   > signal is this weak cannot be silently carrying cross-dataset
   > transfer.

4. **Why no clean "confound confirmed / confound excluded" verdict:**
   with 440 trials / 15 subjects and ~0.84 s of pre-press EEG, both
   windows are underpowered for a decisive decomposition. Do not report
   this control as either a positive or a null finding on its own;
   report the pair of rows and the caveat above.

## Reading the raw table (pre-registration wording, kept for reference)

- If the FULL window decodes but the TRUNCATED window does not (kappa
  collapses to ~0, p_perm ~ chance), most of the decodable signal is
  motor/post-keypress activity: treat this dataset's contribution as
  confounded and say so in the writeup.
- If the TRUNCATED window still decodes at a similar kappa with
  p_perm < 0.05, the signal survives removal of the motor period and
  the motor confound does not explain the within-dataset result.
- Either way, report BOTH windows side by side; do not silently switch
  windows between experiments.
