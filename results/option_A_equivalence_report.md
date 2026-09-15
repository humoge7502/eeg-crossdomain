# Option A equivalence audit

n_boot=5000, equivalence_margin=+/-0.1 kappa, subject-level bootstrap (90% CI)

| Condition | Dataset | n_trials | n_subjects | Observed kappa | 90% CI | Equivalent to null? |
|---|---|---|---|---|---|---|
| noalign | neuma | 4708 | 42 | -0.0080 | [-0.0302, 0.0135] | YES |
| noalign | restaurant_logo | 440 | 15 | -0.0858 | [-0.1636, -0.0003] | no |
| noalign | ds007406 | 60 | 10 | -0.0667 | [-0.3667, 0.2333] | no |
| coral | neuma | 4708 | 42 | -0.0141 | [-0.0344, 0.0048] | YES |
| coral | restaurant_logo | 440 | 15 | -0.0345 | [-0.0823, 0.0137] | YES |
| coral | ds007406 | 60 | 10 | -0.0333 | [-0.2667, 0.2000] | no |
