# Overnight diagnose-and-retry report

## Raw scale diagnostic

| Dataset | mean | std | abs_mean |
|---|---|---|---|
| neuma | -166.5 | 618.3 | 403.8 |
| restaurant_logo | 3.413e-08 | 1.636e-05 | 5.69e-06 |
| ds007406 | 4.081e-07 | 2.407e-05 | 1.219e-05 |

## Scaled LODO results

| Dataset | n trials | y_prob std | kappa | p_holm | kappa 90% CI | equivalent to null? | collapsed? |
|---|---|---|---|---|---|---|---|
| neuma | 4708 | 0.0285 | 0.0022 | 1.000 | [-0.034, 0.038] | YES | no |
| restaurant_logo | 450 | 0.0797 | -0.0204 | 1.000 | [-0.081, 0.049] | YES | no |
| ds007406 | 60 | 0.1194 | -0.0667 | 1.000 | [-0.133, 0.000] | no | no |

## SUMMARY (read this first)

**No fold shows the constant-output collapse anymore.** Per-dataset normalization appears to have fixed it. The kappa/equivalence numbers in the table above are now trustworthy as an actual measurement of cross-paradigm transfer (or lack of it) rather than an artifact of a broken model. Next step: decide whether to adopt this normalization as the standard preprocessing going forward (recommend yes — it's a defensible, unsupervised, leakage-free step) and re-run the full six-condition sweep from Section 15 of EXPERIMENT_LOG.md under it.