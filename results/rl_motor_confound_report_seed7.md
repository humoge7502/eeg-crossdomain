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
- permutation: 50 family-level within-subject shuffles, p on mean-fold AUC

| Window | Model | kappa (mean +/- std) | ROC-AUC | p_perm (AUC) |
|---|---|---|---|---|
| full | logreg | 0.1140 +/- 0.1902 | 0.6333 | 0.0196 |
| full | svm_rbf | -0.0027 +/- 0.0562 | 0.5692 | 0.1373 |
| truncated | logreg | 0.1148 +/- 0.2156 | 0.5685 | 0.0784 |
| truncated | svm_rbf | 0.0149 +/- 0.1483 | 0.6332 | 0.0392 |

## Reading this table

- If the FULL window decodes but the TRUNCATED window does not (kappa
  collapses to ~0, p_perm ~ chance), most of the decodable signal is
  motor/post-keypress activity: treat this dataset's contribution as
  confounded and say so in the writeup.
- If the TRUNCATED window still decodes at a similar kappa with
  p_perm < 0.05, the signal survives removal of the motor period and
  the motor confound does not explain the within-dataset result.
- Either way, report BOTH windows side by side; do not silently switch
  windows between experiments.
