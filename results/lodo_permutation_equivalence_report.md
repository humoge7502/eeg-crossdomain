# LODO permutation test + equivalence audit

n_permutations=2000, equivalence_margin=+/-0.1 kappa


| Dataset | n trials | n subj | conservative? | kappa | p_holm | AUC | p_holm | kappa 90% CI | equivalent to null? |
|---|---|---|---|---|---|---|---|---|---|
| neuma | 4708 | 42 | yes | -0.0008 | 1.000 | 0.487 | 1.000 | [-0.002, 0.000] | YES |
| restaurant_logo | 450 | 15 | yes | 0.0000 | 1.000 | 0.476 | 1.000 | [0.000, 0.000] | YES |
| ds007406 | 60 | 10 | yes | 0.0000 | 1.000 | 0.483 | 1.000 | [0.000, 0.000] | YES |

## Reading this table

- `p_holm` significant (< 0.05) on kappa or AUC would mean the fixed predictions ARE distinguishable from chance for that held-out dataset — i.e., evidence FOR some transfer, contradicting the null claim.
- `equivalent to null? = YES` means the 90% bootstrap CI on kappa falls entirely within +/-0.1, supporting a positive equivalence claim ("no transfer exceeding this margin") rather than a mere failure to reject chance.
- A fold with neither a significant p AND equivalence = YES is underpowered: report it as inconclusive, not as a null finding.
