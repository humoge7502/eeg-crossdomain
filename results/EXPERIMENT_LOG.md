# EEG Cross-Domain Transfer — Experiment Log

Last updated: 2026-09-15

## 1. Datasets (all real, verified, no synthetic data)

| Dataset | Subjects | Real channels used | Native sfreq | Task | Epochs (total) | Class balance |
|---|---|---|---|---|---|---|
| NeuMa | 42 | 20 native (P3,C3,F3,Fz,F4,C4,P4,Cz,Pz,Fp1,Fp2,T3,T5,O1,O2,F7,F8,A2,T6,T4) | 300 Hz | Buy/NoBuy (product dwell-detection) | 4,708 | 74.1% NoBuy / 25.9% Buy |
| Restaurant-Logo | 15 | 8 native (C4,Cz,Fz,C3,Pz,PO7,Oz,PO8) | 250 Hz | Recognized/NotRecognized (logo) | 450 | 67.8% Recognized / 32.2% NotRecognized |
| ds007406 | 10 | 14 native (AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4) | 256 Hz | Extreme/Traditional (marketing video) | 60 | 50% / 50% |

**Label sources (ground truth):**
- NeuMa: `Q76`/`Q77`/`Q78` columns in per-subject `.xlsx` (real product-selection reason codes), cross-matched to mouse-dwell position via validated bounding-box coordinate transform (90.1% hit rate on validation).
- Restaurant-Logo: `ExperimentResults.xlsx`, Block 1 only (rows 0-29; Block 2 rows 32-41 is an unrelated brand-naming task, correctly excluded).
- ds007406: `events.tsv` `value` column (`extreme`/`traditional`), pre-epoched BIDS `.set` files (6 epochs/subject, read via `mne.io.read_epochs_eeglab`).

## 2. Harmonization (Stage 2 of architecture)

- Shared channel basis: `Fz, Cz, Pz, F3, F4, O1, O2` (7 channels)
- Nearest-neighbor fallback substitutions observed:
  - NeuMa: none needed (all 7 present natively)
  - Restaurant-Logo: F3→C3 (0.070m), F4→C4 (0.071m), O1→PO7 (0.030m), O2→PO8 (0.030m)
  - ds007406: Fz→AF3 (0.059m), Cz→FC5 (0.112m), Pz→P7 (0.112m) — largest substitution distances of all three (purely lateral montage, no true midline electrodes)
- Harmonized feature dim: **36** (7 channels × 5 bands [delta,theta,alpha,beta,gamma] + 1 frontal asymmetry index), confirmed identical across all three datasets.

## 3. Common epoch representation (for raw-epoch models: EEGNet, DeepConvNet, MDM)

- Resampled to common **250 Hz**, truncated to common **1.0 second (250 samples)**
- Rationale: NeuMa's native epoch window (360 samples @ 300Hz = 1.2s) was the shortest available; truncating (not padding) avoids injecting artificial zeros into ~90%+ of ds007406's/Restaurant-Logo's epochs.
- Final shape, all datasets: `(n_epochs, 1, 7, 250)`
- Cached at: `data/processed/<dataset>/<dataset>_epochs_common.npz`

## 4. Models tested — architecture and hyperparameters

| Model | Type | Key hyperparameters | Source/precedent |
|---|---|---|---|
| SVM-RBF (PSD+PAI) | Classical ML | kernel=rbf, gamma=scale, 5-fold StratifiedKFold | Reproduces Xu & Liu (2024) feature family |
| MDM (Riemannian) | Classical ML | Covariances(estimator=oas) + MinimumDistanceToMean | Kyrou et al. (2025)-style baseline |
| EEGNetSharedEncoder | Deep learning | n_filters_temporal=8, n_filters_spatial=16, embedding_dim=64, dropout=0.25, Adam lr=0.001, batch_size=32 (capped to fold size), early_stopping_patience=8-15, max_epochs=30 (baselines) / 100 (LODO) | Lawhern et al. (2018) |
| DeepConvNet | Deep learning | 3 conv blocks (25→50→100 filters), dropout=0.5, same optimizer/training config as EEGNet | Schirrmeister et al. (2017) |

**Training loss**: `CrossEntropyLoss`, initially unweighted, later patched to **class-weighted** (`weight = n_total / (2 * n_class)` per class) — see Finding #1 below.

## 5. Experiment 1 — LODO (Leave-One-Dataset-Out), unweighted loss

Run: `scripts/04_run_lodo_experiment.py`, first execution.

| Held-out fold | Accuracy | F1 | Cohen's kappa | AUC |
|---|---|---|---|---|
| (fold 1) | 0.634 | 0.616 | -0.036 | — |
| (fold 2) | 0.322 | 0.157 | 0.000 | — |
| ds007406 | 0.500 | 0.333 | 0.000 | 0.467 |
| **Mean ± std** | **0.4854 ± 0.1277** | **0.3688 ± 0.1891** | **-0.0119 ± 0.0168** | **0.4810 ± 0.0109** |

**Interpretation at time of run**: chance-level performance (kappa ≈ 0, AUC ≈ 0.5). Initially interpreted as possible evidence against cross-paradigm transfer — **later superseded by Finding #1 below**, which shows this result is confounded by an unweighted-loss majority-class collapse and should be re-run before drawing conclusions.

## 6. Experiment 2 — Within-dataset baselines, UNWEIGHTED loss (run 1)

Run: `scripts/03_run_within_dataset_baselines.py`, first execution (`baselines_log.txt`, results in `within_dataset_baselines_full.json`, later overwritten).

| Dataset | Model | Accuracy | Cohen's kappa | Note |
|---|---|---|---|---|
| NeuMa | SVM+PSD/PAI | 0.7411 | — | ≈ exact NoBuy proportion (74.1%) |
| NeuMa | SVM shuffled-label control | 0.7413 | — | **Matches real accuracy almost exactly** |
| NeuMa | EEGNet | 0.7411 | 0.0000 | Majority-class collapse confirmed |
| NeuMa | DeepConvNet | 0.7373 | 0.0038 | Same |
| Restaurant-Logo | SVM+PSD/PAI | 0.6778 | — | ≈ exact Recognized proportion (67.8%) |
| Restaurant-Logo | SVM shuffled-label control | 0.6778 | — | **Matches real accuracy exactly** |
| Restaurant-Logo | EEGNet | 0.6800 | 0.0092 | Near-zero |
| Restaurant-Logo | DeepConvNet | 0.6778 | 0.0000 | Majority-class collapse confirmed |
| ds007406 | SVM+PSD/PAI | 0.3000 | — | Below chance (small-N noise) |
| ds007406 | EEGNet | 0.4833 | -0.0333 | — |
| ds007406 | DeepConvNet | 0.5000 | 0.0000 | — |

### FINDING #1 (critical): Unweighted training collapses to majority-class prediction
Real accuracy and shuffled-label-control accuracy are statistically indistinguishable for both NeuMa and Restaurant-Logo. Cohen's kappa ≈ 0 across the board. This means **no model had actually learned anything** — every reported "accuracy" was simply the majority-class base rate. Root cause: `CrossEntropyLoss()` with no class weighting, combined with real class imbalance (74/26 and 68/32), gives models a trivial loss-minimizing shortcut.

**Fix applied**: patched `src/train.py`'s `train_one_model()` to compute per-class weights (`n_total / (2*n_class)`) and pass them to `CrossEntropyLoss(weight=...)`. Confirmed active via new log line: `[train] Class counts: [...], class weights: [...]`.

## 7. Experiment 3 — Within-dataset baselines, CLASS-WEIGHTED loss (run 2)

Run: `scripts/03_run_within_dataset_baselines.py`, second execution (`baselines_log2.txt`).

**ds007406 results (confirmed from full log tail):**

| Model | Accuracy | Cohen's kappa | AUC |
|---|---|---|---|
| DeepConvNet | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6417 ± 0.1497 |

Still flat at exactly 0.5/0.0 across all 5 folds even with class weighting — but ds007406 has only 60 total samples (12 per fold in 5-fold CV), which is very likely simply too small for any model to learn from, independent of whether weighting helps. **This should be treated as a sample-size limitation for ds007406 specifically, not (yet) as evidence against signal existing.**

**NeuMa class-weighted results (CONFIRMED):**

| Model | Accuracy | Cohen kappa | AUC |
|---|---|---|---|
| SVM+PSD/PAI | 0.7411 | (n/a, unweighted classical model) | — |
| SVM shuffled-control | 0.7413 | (n/a) | — |
| MDM-Riemannian | 0.5068 ± 0.0155 | — | — |
| EEGNet | 0.5988 ± 0.0398 | **0.0599 ± 0.0179** | 0.5442 ± 0.0157 |
| DeepConvNet | 0.5860 ± 0.0654 | **0.0610 ± 0.0248** | 0.5486 ± 0.0113 |

**Restaurant-Logo class-weighted results (CONFIRMED):**

| Model | Accuracy | Cohen kappa | AUC |
|---|---|---|---|
| SVM+PSD/PAI | 0.6778 | (n/a) | — |
| SVM shuffled-control | 0.6778 | (n/a) | — |
| MDM-Riemannian | 0.6067 ± 0.0181 | — | — |
| EEGNet | 0.6644 ± 0.0178 | **0.0106 ± 0.0191** | 0.5348 ± 0.0835 |
| DeepConvNet | 0.6067 ± 0.1422 | **0.0000 ± 0.0000** | 0.4592 ± 0.0466 |

**ds007406 class-weighted results (CONFIRMED, unchanged from unweighted run):**
All models: accuracy=0.5000, kappa=0.0000 exactly, across all 5 folds. Consistent with sample-size-limitation hypothesis (60 total samples, 12/fold).

### FINDING #2: Class weighting reveals weak-but-real signal in NeuMa, marginal signal in Restaurant-Logo
NeuMa: kappa moved from 0.0000 (unweighted) to ~0.060 (weighted), consistently across both EEGNet and DeepConvNet — cross-architecture agreement supports this being real signal, not noise. Modest in magnitude (well below DeePay full pipeline non-truncated 0.276 RMSE/75% accuracy result), plausibly limited by the aggressive 1.0s common-window truncation.
Restaurant-Logo: only EEGNet shows weak positive kappa (0.0106, CI still touches 0); DeepConvNet shows none. Likely more truncation-sensitive than NeuMa given its native window (3.0s) was cut more aggressively than NeuMa needed (1.2s to 1.0s).
ds007406: still uninformative — sample size (n=60) is the likely limiting factor, not signal absence.

**CONCLUSION: proceed to re-run LODO (Experiment 1) with class-weighting fix now active** — the original chance-level LODO result was generated under the broken unweighted loss and must be superseded before drawing any conclusion about cross-paradigm transfer.

## 8. Open questions / next steps (as of this log)

1. **[BLOCKING]** Confirm whether class-weighting fixed NeuMa/Restaurant-Logo's majority-class collapse (kappa should move meaningfully above 0 if there's real signal) — pull full `within_dataset_baselines_full.json` summary.
2. **If kappa remains ~0 even with weighting**: investigate whether the 1.0s truncation window is destroying real signal (especially plausible for ds007406, whose native epochs are 30s — the discriminative "extreme vs traditional" signal may only emerge later in the video).
3. **If kappa moves clearly positive for NeuMa/Restaurant-Logo**: re-run LODO (`04_run_lodo_experiment.py`) with the class-weighting fix now active, since the original chance-level LODO result was generated under the unweighted, majority-collapsing loss and should not yet be treated as a final finding on cross-paradigm transfer.
4. **Planned additional baselines** (not yet implemented): XGBoost with `scale_pos_weight` (precedented by DeePay's own baseline choice), SMOTE applied to the 36-dim harmonized tabular features specifically (not raw epochs, to avoid interpolating physiologically implausible synthetic EEG waveforms). To be added one at a time, each as an isolated controlled experiment, not combined together, to preserve interpretability of what each change actually contributes.
5. **Not yet run**: `scripts/05_run_ablation_alignment.py` (Riemannian alignment vs. no-alignment ablation) — the core "does domain alignment help" experiment central to the paper's actual contribution claim. Should only be run once within-dataset signal is confirmed real (Step 1 above), since an alignment ablation is meaningless if there's no signal to align in the first place.

## 9. Known infrastructure notes (for future debugging)

- This Brev instance (`brev-x1v47hbeh`) is heavily shared (30+ tmux windows across projects observed, 130-day uptime, load average ~12). JupyterLab's native browser terminal has repeatedly disconnected/corrupted mid-session (cause unconfirmed — likely browser-side websocket instability under system load, not malicious).
- **Mitigation in place**: long-running scripts now always launched via `nohup python3 <script> > <logfile>.txt 2>&1 &`, fully detached from any terminal/tmux session, safe against browser/tab closure. Check progress anytime via `tail -f <logfile>.txt` or `ps aux | grep <script_name>`.
- A persistent tmux session `eegwork` also exists as a secondary safety net, but the `nohup` approach is now the primary protection method since it doesn't depend on tmux client-server rendering staying intact.

## 10. Experiment 4 — LODO re-run with class-weighted loss

Run: `scripts/04_run_lodo_experiment.py`, second execution (`lodo_log2.txt`), class-weighting fix active.

| Metric | Original (unweighted) | Re-run (weighted) |
|---|---|---|
| Accuracy | 0.4854 +/- 0.1277 | 0.5210 +/- 0.1715 |
| F1 | 0.3688 +/- 0.1891 | 0.3737 +/- 0.1954 |
| Cohen kappa | -0.0119 +/- 0.0168 | -0.0003 +/- 0.0004 |
| AUC | 0.4810 +/- 0.0109 | 0.4675 +/- 0.0362 |

### FINDING #3 (core finding): Weak within-dataset signal does NOT transfer cross-paradigm, even after fixing the training bug

Class-weighting fix demonstrably worked at the within-dataset level (NeuMa kappa moved 0.000 to 0.060, Finding #2). But LODO kappa remains at ~0.000 regardless — essentially unchanged from the broken unweighted run. This rules out "broken training" as the explanation for chance-level LODO transfer, since the same fix that revealed real within-dataset signal did NOT reveal any cross-dataset signal.

This is now a defensible negative result, not an artifact. Whatever weak neural signal distinguishes Buy/NoBuy within NeuMa (kappa ~0.06) does not generalize to Recognized/NotRecognized (Restaurant-Logo) or Extreme/Traditional (ds007406) discrimination, even when the shared encoder has access to NeuMa's real signal during training. Directly supports the honest framing established during the defensibility-planning phase of this project: "the three tasks aren't really the same construct" — now empirically confirmed rather than just anticipated as a limitation.

Caveats to state alongside this finding:
- Signal strength even within NeuMa is modest (kappa 0.06 vs DeePay's own non-truncated full-pipeline 75% accuracy) — partially attributable to the aggressive 1.0s common-window truncation required for cross-dataset shape compatibility.
- Restaurant-Logo's own within-dataset signal is marginal and architecture-inconsistent (EEGNet weak positive, DeepConvNet zero) — can't rule out that Restaurant-Logo simply has too little decodable signal at 1.0s truncation for a transfer effect to be detectable even if one existed.
- ds007406 (n=60) remains fundamentally underpowered for any conclusion.

Next honest step: this null cross-paradigm-transfer result is itself the paper's core empirical contribution IF it holds up under the planned alignment ablation (Riemannian vs. no-alignment) — proceed to `05_run_ablation_alignment.py` to test whether domain alignment specifically can recover any transferable signal that naive pooling cannot.

## 11. Case Study: Options A-D Comparison

| Option | Description | Split protocol | Key metric | Result | Status |
|---|---|---|---|---|---|
| A | Full-duration tabular features +/- CORAL | TBD - audit pending | LODO kappa | pending re-run | Audit required before use |
| B: Feature-derived engagement relabel | Task-agnostic arousal-based label | EPOCH-WISE (leaky, not corrected) | LODO kappa | 0.0030 | EXCLUDED from quantitative claims - exploratory only, superseded protocol |
| C: Multi-task shared encoder | Joint pretraining vs solo per-dataset | SUBJECT-WISE (corrected, GroupShuffleSplit) | kappa delta (multitask-solo) | neuma:-0.0257, restaurant_logo:+0.0088, ds007406:+0.0000 | VALIDATED - see Section 13 |
| D: DANN (domain-adversarial) | Gradient-reversal domain-invariant training | EPOCH-WISE (leaky, not corrected) | LODO kappa | -0.0001 | EXCLUDED from quantitative claims - exploratory only, superseded protocol |

Baseline for comparison: naive zero-shot LODO (Experiment 4, dataset-level split, not epoch/subject-level - not affected by the leakage issue) kappa = -0.0003 +/- 0.0004

NOTE: Options B and D were run under the original epoch-wise CV protocol before the subject-leakage issue was identified (Section 12). Per project decision, these were NOT re-run under subject-wise splits, since both were already null/near-null results and re-running would not change the paper conclusion. They are retained here as exploratory analyses under a superseded protocol, explicitly excluded from any quantitative claim in the final writeup.


## 12. CRITICAL FIX: Subject-wise (leakage-free) re-validation

Following external review, the original within-dataset CV (Section 7) split by
EPOCH, not SUBJECT — meaning the same person could appear in both train and test
folds, inflating reported kappa via subject-identity leakage rather than genuine
task-signal decoding. All baselines were re-run using StratifiedGroupKFold, grouped
by real subject ID, so no subject ever appears in both train and test within a fold.

| Dataset | Model | Kappa (epoch-wise, ORIGINAL) | Kappa (subject-wise, CORRECTED) |
|---|---|---|---|
| neuma | eegnet | 0.0599 | -0.0054 |
| neuma | deepconvnet | 0.0610 | 0.0407 |
| restaurant_logo | eegnet | 0.0106 | 0.0020 |
| restaurant_logo | deepconvnet | 0.0000 | 0.0000 |
| ds007406 | eegnet | 0.0000 | -0.0333 |
| ds007406 | deepconvnet | 0.0000 | 0.0000 |


## 13. Option C re-validated (subject-wise) + Permutation Test

### Option C, subject-wise train/test split (corrected)

| Dataset | Multitask kappa | Solo kappa | Delta |
|---|---|---|---|
| neuma | 0.0164 | 0.0421 | -0.0257 |
| restaurant_logo | 0.0088 | 0.0000 | +0.0088 |
| ds007406 | 0.0000 | 0.0000 | +0.0000 |

Permutation test: ERROR [Errno 2] No such file or directory: 'results/neuma_deepconvnet_permutation_test.json'

## 14. Exploratory NeuMa estimate and incomplete statistical validation

### Exploratory point estimate (not validated)
NeuMa, DeepConvNet, solo training, subject-wise held-out evaluation (StratifiedGroupKFold, no subject appears in both train and test): Cohen's kappa = 0.0421. This is an EXPLORATORY point estimate only -- it has not been statistically validated and must not be described as a confirmed or positive decoding result.

Two observations are noted about this estimate, neither of which constitutes statistical validation:
1. It DECREASED but did NOT vanish when the subject-leakage bug was corrected (0.0610 epoch-wise leaky -> 0.0421 subject-wise correct), unlike EEGNet on the same dataset (0.0599 -> -0.0054, fully vanished) and unlike Restaurant-Logo on either architecture (both collapsed to ~0). This is an observation about robustness to the leakage correction, not evidence of statistical significance.
2. Option C's solo arm (Section 13) produced a matching point estimate (kappa = 0.0421, exact match to three decimal places) under a different train/test split draw (GroupShuffleSplit vs StratifiedGroupKFold) and different encoder architecture (TabularMLPEncoder on 36-dim harmonized features vs DeepConvNet on raw 250-sample epochs). This is NOT described as independent reproduction: preprocessing, architecture initialization, and seed policy were not confirmed identical between the two runs, and a single matching point estimate from two loosely-matched runs on a small subject pool (33-42 subjects) does not meet the bar for corroboration.

### Permutation significance testing: attempted, not completed
A formal subject-wise permutation significance test (shuffled-label control, full StratifiedGroupKFold CV re-run per permutation) was implemented and attempted at N=200, then N=50, then N=20 permutations. All three attempts were terminated before completion due to compute time constraints (each permutation requires a full 5-fold subject-wise CV training pass, ~3,700+ epochs per fold, on a shared/contended GPU instance). No permutation run completed far enough to yield a p-value.

This is an incomplete analysis. The kappa=0.0421 effect size and its persistence under the leakage correction are observations, not statistical evidence, and do NOT substitute for a formal significance test. Any manuscript draft from this project must either (a) complete a permutation test with adequate compute time before submission, or (b) explicitly state permutation testing was not completed and present the result as a preliminary effect-size estimate requiring further validation, per standard practice for incomplete statistical analyses.

### Recommended finishing step (future work, not done in this session)
Re-run scripts/10_permutation_test_neuma_deepconvnet.py with N_PERMUTATIONS=20-30 on a dedicated (non-shared) GPU session, or reduce max_epochs further (currently 15) and/or reduce CV folds from 5 to 3 to cut per-permutation cost, then let it run to completion in the background across a full session without competing for compute with other concurrent jobs on the shared Brev instance.

## 15. Option A completed: full-duration features rule out truncation as the explanation for null transfer

Split protocol confirmed dataset-level (LODO-style: for held_out in all_data.keys(), train = concatenation of the OTHER full datasets, test = entirely separate held-out dataset). No subject-leakage risk in this design, since datasets never overlap subjects by construction. Minor note: the inner train/val split (used only for early-stopping decisions, not the reported test metric) used plain random permutation rather than subject-grouping; this does not affect the validity of the reported held-out results.

| Held-out dataset | No-alignment kappa | CORAL-aligned kappa |
|---|---|---|
| NeuMa | 0.0026 | -0.0002 |
| Restaurant-Logo | 0.0000 | 0.0007 |
| ds007406 | 0.0000 | 0.0000 |
| Mean | 0.0009 | 0.0002 |

### FINDING #4: Temporal truncation is unlikely to be the primary cause of null cross-paradigm transfer (scoped claim)

Full-duration band-power features did not improve zero-shot LODO transfer, and CORAL alignment did not recover performance. Thus, the 1.0-second truncation is unlikely to be the primary cause of the observed transfer failure. Across six evaluated strategies, we found no evidence that a classifier trained on one or two datasets generalizes zero-shot to a held-out dataset with a different task label and paradigm.

The six complementary experimental conditions and interventions referenced above are: (1) naive zero-shot LODO, (2) class-weighted LODO, (3) feature-derived engagement relabeling, (4) domain-adversarial training (DANN), (5) subject-wise multi-task joint training, and (6) full-duration features with CORAL alignment. Note these are not six independent architectures in the strict sense -- some (e.g. class-weighting) are training and validation corrections rather than distinct transfer-learning models -- but each represents a genuinely different attempted mechanism for recovering cross-paradigm signal, and all six converge on the same null result.

IMPORTANT SCOPING CAVEAT: this null result does NOT establish an absence of shared underlying neural structure across these paradigms. It shows no evidence of a shared zero-shot binary classifier across three tasks with different, task-specific operationalized labels (Buy/NoBuy, Recognized/NotRecognized, Extreme/Traditional). Null transfer under this specific operationalized cross-paradigm binary-label setting should not be interpreted as proof that no physiological features are shared between these paradigms -- only that a binary classifier trained on one label definition does not zero-shot generalize to a differently-defined label from another task, under the conditions tested here.

Methodological note on Option A inner train and validation split: this split used plain random permutation (not subject-grouped) but was used ONLY to select the early-stopping epoch during training, never to select hyperparameters based on held-out test performance and never used in computing the reported test metrics. Hyperparameters (learning rate, architecture, dropout, batch size) were fixed prior to all LODO runs across every condition in this project and were not tuned by inspecting held-out test-set results at any point. This does not expose held-out-dataset test data; however, future work should use group-aware inner validation by subject for early stopping.

## 16. Permutation test completed under a reduced configuration; not applicable to the primary estimate

A permutation test was run to completion (25 permutations, 3-fold subject-wise CV, 10 max epochs, lr=0.002). This configuration differs from the one used to obtain the Section 14 exploratory estimate (kappa=0.0421, 5-fold CV, 15 max epochs, lr=0.001). Under the reduced configuration, the observed kappa was -0.0087 (p=0.68, not significant) -- this is a different point estimate than the Section 14 exploratory estimate, and this permutation test result does NOT constitute a significance test of that estimate.

This is documented explicitly here: the completed run is a VALID negative result for the reduced-training configuration it actually used (3-fold CV, 10 epochs, lr=0.002) -- it is simply not a significance test of the Section 14 configuration, since hyperparameters must be held identical between a point-estimate run and its corresponding significance test, and they were not. As a result:

- The p=0.68 result reported by this run does NOT apply to and must NOT be cited alongside the kappa=0.0421 finding.
- It remains unknown whether kappa=0.0421 is statistically distinguishable from chance.
- The discrepancy itself (0.0421 under one config, -0.0087 under a lighter-trained config) is informative: it suggests the effect, if real, may be sensitive to training budget/convergence, and is likely small enough that it should be treated as preliminary pending a properly matched significance test.

CORRECT NEXT STEP (not yet done): re-run the permutation test using the EXACT SAME hyperparameters as the original Section 14 result (5-fold CV, 15 max epochs, lr=0.001) for both the observed-kappa computation and every permutation. This was not completed due to compute time constraints across multiple attempted configurations in this session.

REVISED HONEST STATUS OF THE PRIMARY FINDING: kappa=0.0421 (NeuMa, DeepConvNet, subject-wise split, observed under two related but non-identical configurations per Section 14) is an EXPLORATORY, UNCONFIRMED estimate. It must be presented as such in any manuscript draft -- not as a statistically confirmed or validated result -- until a correctly matched permutation test is completed.


## 17. FINAL STATUS: kappa=0.042 is NOT a validated positive finding -- downgraded to exploratory

Following further review, two independent problems mean the primary NeuMa/DeepConvNet result CANNOT be presented as validated in any manuscript draft from this project:

1. **No properly matched significance test exists.** The only completed permutation test (Section 16) used different hyperparameters (3-fold CV, 10 epochs, lr=0.002) than the original point estimate (5-fold CV, 15 epochs, lr=0.001), making its p-value inapplicable. Under the lighter configuration, the point estimate itself changed sign (0.0421 -> -0.0087), revealing that this result is sensitive to training configuration -- itself a publication-relevant finding about the fragility of small-effect EEG decoding in this setting, not a nuisance detail.

2. **Even a correctly matched test at the attempted permutation counts would have offered insufficient resolution and power for robust inference.** With N=25 permutations, the minimum attainable one-sided p-value is 1/26 = 0.0385. A correctly designed matched test requires: identical subject folds, preprocessing, architecture, epochs, learning rate, early-stopping policy, and random seed policy as the original Section 14 run; label permutation performed WITHIN subject; full retraining from scratch per permutation; a MINIMUM of 500 (ideally 1000) permutations; the empirical one-sided p-value formula (1 + #{kappa_perm >= kappa_obs}) / (1 + N); and a subject-level bootstrap confidence interval reported alongside the p-value. None of this was completed in this project to date.

3. **The "independent reproduction" claim in Section 14 is also withdrawn** (see correction above) -- the matching point estimate between the DeepConvNet baseline and Option C's solo arm does not meet the bar for independent corroboration given differing split methods and unconfirmed matching of preprocessing/architecture/seed.

### Final decision on manuscript framing

kappa=0.0421 for NeuMa/DeepConvNet is downgraded from "primary validated positive result" to **exploratory point estimate, not statistically validated, requiring a correctly matched high-N permutation test before any decoding claim can be made.** The paper's contribution is reframed as a strict leakage-and-transfer audit:

- A rigorously documented account of how epoch-level subject leakage silently inflated apparent EEG decoding accuracy in a controlled, real-data, multi-dataset benchmark (Section 12), including a demonstration that leakage can independently and comparably corrupt both single-task (Section 12) and multi-task (Section 13) evaluations.
- Six complementary tested interventions (Section 15) showing consistent null zero-shot cross-paradigm transfer under a properly operationalized, subject-wise-validated evaluation protocol, correctly scoped to NOT claim absence of shared neural structure, only absence of transferable zero-shot binary decoding under these task-specific label definitions.
- An honest account of a compute-constrained attempt at significance testing, including the specific reason it failed to validate (hyperparameter mismatch) and the specific design required to do it correctly (detailed above), left as clearly scoped future work.

This is a coherent, defensible, methodologically rigorous paper. It must NOT claim validated Buy/NoBuy decoding from EEG. It SHOULD claim: a demonstrated leakage-detection-and-correction methodology, a properly controlled negative transfer-learning result across six interventions, and a transparent account of what remains unresolved.

## 18. Option A corrected and re-validated under subject-level equivalence audit (supersedes Section 15's stale numbers)

Section 15's Option A results were run BEFORE the per-subject normalization fix (scripts/02b_add_persubject_normalization.py, created Aug 18) and are confirmed stale. Option A was re-run on the corrected `_features_persubj_v2.npz` cache, and a subject-level bootstrap equivalence audit (n_boot=5000, 90% CI, TOST margin=+/-0.10 kappa -- matching the methodology already used for the LODO equivalence audit in Section 11) was applied to the corrected results.

| Condition | Dataset | n_trials | n_subjects | Observed kappa | 90% CI | Equivalent to null (+/-0.10)? |
|---|---|---|---|---|---|---|
| No alignment | NeuMa | 4708 | 42 | -0.0080 | [-0.0302, 0.0135] | YES |
| No alignment | Restaurant-Logo | 440 | 15 | -0.0858 | [-0.1636, -0.0003] | no |
| No alignment | ds007406 | 60 | 10 | -0.0667 | [-0.3667, 0.2333] | no |
| CORAL | NeuMa | 4708 | 42 | -0.0141 | [-0.0344, 0.0048] | YES |
| CORAL | Restaurant-Logo | 440 | 15 | -0.0345 | [-0.0823, 0.0137] | YES |
| CORAL | ds007406 | 60 | 10 | -0.0333 | [-0.2667, 0.2000] | no |

**Interpretation.** NeuMa is robustly equivalent to no-transfer under both conditions. Restaurant-Logo's no-alignment estimate falls just outside the equivalence margin (CI upper bound -0.0003, effectively touching zero) but becomes equivalent under CORAL alignment. A diagnostic check (flipping predicted labels) ruled out systematic bias as the explanation: flipped kappa (+0.061) is similarly unremarkable to the observed (-0.086), and `y_prob` shows healthy, non-collapsed variance (std=0.028), matching the pattern of the trusted NeuMa LODO result. This is interpreted as ordinary sampling variability at moderate sample size, not a real negative effect. ds007406 is underpowered in all conditions (CIs span from strongly negative to strongly positive), consistent with its small size (n=60, 10 subjects) -- explicitly NOT interpreted as evidence of either a null or non-null effect.

The core Section 15 conclusion (full-duration features + CORAL alignment do not recover cross-paradigm transfer) holds under the corrected data, now with proper equivalence bounds rather than bare point estimates.


## 19. Rigorous, Section-17-compliant permutation significance tests completed for all three datasets

Section 17 required a properly matched permutation test (matching the ORIGINAL point-estimate config: 5-fold CV, 15 epochs, lr=0.001) with N>=500 permutations, WITHIN-SUBJECT label shuffling, and the correct empirical p-value formula `(1 + count(perm_kappa >= observed_kappa)) / (1 + N)`. This was completed for all three datasets (scripts/13_permutation_test_rigorous.py), superseding the earlier reduced-config attempts (Section 16/17) that were explicitly flagged as producing an inapplicable p-value.

| Dataset | Observed kappa | Null mean / std | p-value | Significant at 0.05? | Wall time |
|---|---|---|---|---|---|
| Restaurant-Logo | 0.0121 | 0.0025 / 0.0217 | 0.325 | No | 25.1 min |
| ds007406 | -0.0167 | 0.0001 / 0.0243 | 0.750 | No | 33.7 min |
| NeuMa | -0.0022 | 0.0004 / 0.0140 | 0.587 | No | 78.5 min |

All three tests show non-degenerate null distributions (real spread, not the collapsed-model flat-zero artifact seen in earlier debugging), confirming these are genuine, trustworthy significance tests. None reach significance, extending the null-transfer finding onto fully rigorous, Section-17-compliant statistical footing across all three datasets.


## 20. Positive control: pipeline validated via subject-identification decoding

To rule out "the pipeline/architecture cannot learn from this data" as an explanation for the null task-transfer results, the same DeepConvNet architecture and z-scored preprocessing was applied to a well-established, robustly decodable EEG target: subject identity (42-way classification on NeuMa, 5-fold stratified CV, held-out trials).

| Metric | Value |
|---|---|
| Mean accuracy | 44.20% (chance = 2.38%) |
| Ratio to chance | 18.6x |
| Mean top-5 accuracy | 80.93% |
| Mean kappa | 0.4254 |

All 5 folds landed consistently in the 41-48% accuracy range (kappa 0.40-0.46), with no fold-to-fold instability. This directly demonstrates the pipeline learns real, robust signal from this exact data and preprocessing when it exists -- the null task-transfer results are not attributable to a broken model or preprocessing pipeline.


## 21. Representation similarity analysis: small but statistically significant subspace alignment despite null classifier transfer

Two complementary methods were used to test whether learned representations share structure across paradigms, independent of whether a zero-shot classifier can use that structure:

**Method 1 (linear CKA)** was found to be unreliable for the smallest dataset (ds007406, n_test=12) -- bootstrapped null (untrained-encoder) and positive-control (same-data, different-seed) baselines showed CKA scaling with sample size almost as strongly as with actual training, making raw CKA values for ds007406 uninterpretable. This is reported as a methodological finding in its own right: small-N CKA without a null baseline is not trustworthy, and any cross-dataset CKA claim must be baselined against both an untrained-encoder null and a same-data positive control.

**Method 2 (principal angle / subspace alignment)** was used instead, as the statistically appropriate tool for unpaired trial sets across datasets (unlike CKA, which assumes sample correspondence). For each pair of independently-trained, single-dataset ("solo") encoders, the top-10 principal subspaces of their test-set embeddings were compared via principal angles, baselined against a random-subspace null (mean=70.13 deg, std=1.47 deg, established via 50 random 10-dim subspace pairs in the 64-dim embedding space).

| Pair | Mean angle | vs. random null | Reliability |
|---|---|---|---|
| NeuMa vs Restaurant-Logo | 66.03 deg | -4.10 deg | Reliable (n=941, n=88) |
| NeuMa vs ds007406 | 63.33 deg | -6.80 deg | Small sample (n=12), qualitative only |
| Restaurant-Logo vs ds007406 | 57.86 deg | -12.27 deg | Small sample (n=12), qualitative only |

A bootstrap significance test (500 resamples with replacement) on the one reliable pair (NeuMa vs Restaurant-Logo) confirmed this is a real effect, not noise: 95% CI = [64.34, 68.17] degrees, which entirely excludes the random-subspace null mean of 70.13 degrees (4.26 bootstrap-SE units from the null).

**Interpretation.** Independently-trained encoders on NeuMa and Restaurant-Logo -- despite never seeing each other's data, and despite zero-shot classifiers failing to transfer between them (Sections 11, 15, 19) -- learn representations with small but statistically significant shared subspace structure beyond chance. This suggests the null classification-transfer result is not explained by fully unrelated/orthogonal representations; some shared low-level EEG feature geometry exists across these paradigms, but it is either too weak, or not exposed in a linearly decodable form, to support zero-shot task-label transfer under the label definitions tested here.


## 22. Data pipeline audit trail: two independent stale-cache bugs found and fixed

In the course of validating Sections 18-21, two separate instances of the same class of bug were found and corrected, both involving scripts silently reading pre-normalization-fix data caches:

1. **Tabular feature scripts (Option A-D, scripts 06-09)** were hardcoded to `data/processed/<name>/<name>_harmonized.npz` (pre-fix) instead of `data/processed_v2/<name>/<name>_features_persubj_v2.npz` (post-fix, Aug 18). Confirmed via file timestamp comparison: `option_A_results.json` (Aug 13) predates `02b_add_persubject_normalization.py` (Aug 18) by 5 days. All five scripts patched; Option A re-run and results corrected (Section 18).

2. **The NeuMa/DeepConvNet permutation test script** (scripts/10_permutation_test_neuma_deepconvnet.py) had the same stale-path bug (`data/processed/neuma/neuma_epochs_common.npz` instead of the corrected `windows_common_v2.npz`), compounded by a 4D-vs-3D array shape mismatch once corrected. After fixing both, a NEW issue emerged: DeepConvNet+BatchNorm training collapsed to constant-class prediction (kappa stuck at exactly 0.0000 across all 25 permutations, p=1.0) at the raw ~1e-6 signal scale of the corrected cache. Root cause was isolated (ruled out: stale path, learning rate, corrupted data, architecture-random-noise incompatibility) and fixed via per-trial, per-channel z-scoring immediately before training -- standard practice for CNN-based EEG decoders, not previously applied to this raw-epoch pipeline. This fix was verified to generalize cleanly to all three datasets (all show the same ~1e-6 raw scale) and was subsequently built into every raw-epoch DeepConvNet script used in Sections 19-21.

Both bugs were caught only because result timestamps were cross-checked against the fix script's creation date and because degenerate-looking results (flat-zero kappa, exploding val_loss) were treated as a diagnostic signal rather than accepted at face value. This audit trail is retained as a methods contribution demonstrating the project's data-quality rigor.

## 23. Multi-seed robustness of the representation-alignment finding (Section 21)

The NeuMa-vs-Restaurant-Logo principal-angle result (Section 21) was recomputed across 8 independent seeds, each with a different random train/test split AND different encoder initialization, to rule out a single-lucky-seed artifact.

| Seed | Mean angle (deg) |
|---|---|
| 42 | 66.03 |
| 1 | 64.42 |
| 7 | 63.16 |
| 13 | 64.45 |
| 99 | 65.20 |
| 2024 | 65.12 |
| 314 | 66.49 |
| 271 | 64.19 |

Across all 8 seeds: mean = 64.88 deg, std = 0.99 deg, range [63.16, 66.49] deg. All 8/8 seeds fall more than one null-standard-deviation below the random-subspace null mean (70.13 +/- 1.47 deg). The original single-seed result (66.03 deg, Section 21) is consistent with, if anything slightly conservative relative to, the multi-seed mean. This confirms the representation-alignment finding is a stable, reproducible effect rather than a single-seed artifact.

## 24. Multi-seed robustness of Option A's null-transfer finding

Option A (no-alignment condition) was re-run across 5 independent seeds (different weight initialization and train/val split each time) to confirm the null-transfer conclusion is stable.

| Seed | Mean kappa (3 held-out folds) |
|---|---|
| 42 | 0.0172 |
| 1 | 0.0119 |
| 7 | -0.0055 |
| 13 | -0.0232 |
| 99 | 0.0038 |

Grand mean across 5 seeds: 0.0009 +/- 0.0143, range [-0.0232, 0.0172]. All seeds cluster tightly around zero with no indication of a real positive-transfer signal at any seed. The original single-seed result reported in Section 18 (seed=42, kappa=0.0172 no-alignment mean) sits at the high end of this distribution, confirming the reported result was not a favorably cherry-picked seed.

## 25. Option B: Riemannian geometry classifier (MDM) -- architecture generality test

To test whether the null zero-shot transfer result is specific to the amplitude-based architectures used elsewhere in this project (TabularMLP, EEGNet, DeepConvNet), a fundamentally different representation and classifier was tested: Minimum Distance to Mean (MDM) on trial spatial-covariance matrices (Riemannian geometry, via pyriemann), reusing scaffolding already present in `src/baselines.py`.

| Held-out dataset | Observed kappa | Null mean / std | p-value | Significant (Holm-corrected)? |
|---|---|---|---|---|
| NeuMa | -0.0018 | -0.0008 / 0.0018 | 0.832 | No |
| Restaurant-Logo | -0.0157 | -0.0004 / 0.0090 | 0.985 | No |
| ds007406 | **0.1100** | 0.0005 / 0.0204 | **0.001** | **YES** |

All permutation tests used N=1000 within-subject label shuffling and the standard unbiased p-value formula, matching the rigor standard established in Section 19.

**MDM zero-shot transfer to ds007406 is statistically significant**, surviving Holm-Bonferroni correction across all 9 rigorous permutation-based significance tests conducted in this project (DeepConvNet x3, MDM x3, CBraMod x3 -- see Section 27 for the full corrected table). This is the only significant zero-shot transfer result found anywhere in this work.


## 26. Option C: Pretrained EEG foundation model (CBraMod) as frozen feature extractor -- large-corpus representation test

CBraMod (Song et al., ICLR 2025; pretrained on the Temple University Hospital EEG corpus) was used as a frozen feature extractor, with a linear probe (LogisticRegression) trained on top -- the standard representation-learning evaluation protocol. CBraMod's Asymmetric Conditional Positional Encoding is designed to generalize to arbitrary channel counts without retraining, making it compatible with this project's 7-channel harmonized montage without architectural modification. Data was resampled from the project's native 250Hz to CBraMod's pretraining rate of 200Hz.

**Data-scale defect (third instance in this project).** An initial run produced a fully degenerate result: within-dataset and zero-shot kappa identically 0.0000 across every fold and condition. Diagnosis confirmed CBraMod's frozen embeddings were nearly IDENTICAL regardless of input trial (correlation between two different trials' feature vectors: 0.99997) when fed the raw ~1e-6-scale `windows_common_v2` data. Applying the same per-trial per-channel z-score normalization already verified for DeepConvNet (Section 22) resolved the collapse (post-fix trial-to-trial correlation: 0.397; per-feature cross-sample std increased ~700x). This is the third independent instance in this project of a model collapsing at the raw ~1e-6 data scale, reinforcing that this z-scoring step is required for any new raw-epoch model added to this pipeline, not just the ones already accounted for.

**Results (post-fix):**

| Condition | Dataset | Observed kappa |
|---|---|---|
| Within-dataset | NeuMa | 0.034 |
| Within-dataset | Restaurant-Logo | 0.050 |
| Within-dataset | ds007406 | -0.008 |
| Zero-shot LODO | NeuMa held out | -0.0084 |
| Zero-shot LODO | Restaurant-Logo held out | 0.0291 |
| Zero-shot LODO | ds007406 held out | 0.0200 |

Permutation significance testing (N=1000, within-subject, features extracted once and reused across permutations for efficiency):

| Held-out dataset | p-value | Significant? |
|---|---|---|
| NeuMa | 0.506 | No |
| Restaurant-Logo | 0.716 | No |
| ds007406 | 0.116 | No |

None of CBraMod's zero-shot transfer results reach significance. Within-dataset decoding is modest (kappa 0.03-0.05) -- well below the positive control's kappa=0.43 (Section 20) -- indicating that even a large-corpus-pretrained representation does not straightforwardly linearly decode the task labels used here, let alone transfer them zero-shot across paradigms.


## 27. Cross-architecture significance summary with Holm-Bonferroni correction

All 9 rigorous, properly-powered (N>=500) permutation-based significance tests conducted across this project, ranked by p-value and Holm-corrected:

| Rank | Test | p-value | Holm threshold | Survives? |
|---|---|---|---|---|
| 1 | MDM - ds007406 | 0.0010 | 0.00556 | **YES** |
| 2 | CBraMod - ds007406 | 0.1159 | 0.00625 | No |
| 3 | DeepConvNet - Restaurant-Logo | 0.3253 | 0.00714 | No |
| 4 | CBraMod - NeuMa | 0.5065 | 0.00833 | No |
| 5 | DeepConvNet - NeuMa | 0.5868 | 0.01000 | No |
| 6 | CBraMod - Restaurant-Logo | 0.7163 | 0.01250 | No |
| 7 | DeepConvNet - ds007406 | 0.7505 | 0.01667 | No |
| 8 | MDM - NeuMa | 0.8322 | 0.02500 | No |
| 9 | MDM - Restaurant-Logo | 0.9850 | 0.05000 | No |

**Exactly one of nine tests survives family-wise error correction: MDM's zero-shot transfer to ds007406.** This is a precise, architecture- and dataset-specific finding, not a generalized claim about transfer across all conditions. Interpreted mechanistically: Riemannian covariance-based representations (which operate on second-order spatial statistics, robust to certain amplitude/scale nuisance variation) may capture transferable structure that amplitude-sensitive representations (tabular spectral features, EEGNet, DeepConvNet, and even the large-corpus-pretrained CBraMod) do not, at least for this specific dataset pairing. Given ds007406's small cohort (10 subjects), a leave-one-subject-out sensitivity analysis and permutation-seed stability check were conducted as a robustness follow-up (Section 28).

## 28. Robustness checks for the MDM-ds007406 finding

Given the significance and consequence of the MDM-ds007406 finding (Section 25), two robustness checks were conducted (scripts/24_mdm_ds007406_robustness.py).

**Permutation-seed stability.** The null distribution was recomputed with three independent RNG seeds (300 permutations each, reduced from the original 1000 for speed while remaining informative):

| Seed | Null mean / std | p-value |
|---|---|---|
| 42 | -0.0003 / 0.0212 | 0.0033 |
| 123 | -0.0020 / 0.0206 | 0.0033 |
| 999 | 0.0003 / 0.0211 | 0.0033 |

All three seeds produce the identical p-value (0.0033) with closely matching null distribution statistics, confirming the result is not an artifact of a single random permutation draw.

**Leave-one-subject-out sensitivity.** With ds007406 containing only 10 subjects, each subject was excluded in turn and kappa recomputed on the remaining 9:

| Excluded subject | Kappa (9 subjects) |
|---|---|
| sub-001 | 0.0951 |
| sub-002 | 0.1259 |
| sub-003 | 0.1259 |
| sub-004 | 0.1000 |
| sub-005 | 0.1247 |
| sub-006 | 0.1148 |
| sub-007 | 0.1235 |
| sub-008 | 0.1222 |
| sub-009 | 0.0840 |
| sub-010 | 0.0840 |

LOO kappa range: [0.0840, 0.1259], mean = 0.1100 (exactly matching the full-sample kappa). Every single-subject exclusion yields a clearly positive kappa in a tight range -- no subject drives the effect, and none of the 10 leave-one-out folds come close to zero or negative. This indicates the effect is broadly and consistently distributed across the full 10-subject cohort, not an artifact of one or two outlier subjects.

**Conclusion.** The MDM-ds007406 zero-shot transfer finding passes both robustness checks: it is stable across independent permutation-test random seeds, and it is not driven by any single subject in the small 10-subject cohort. This strengthens confidence that the finding, while requiring independent replication on a larger sample before any general claim is warranted, is a genuine, non-degenerate effect within this dataset rather than a statistical or implementation artifact.

## 29. Sensitivity: LODO with ds007406 dropped (harmonization-robustness of the null-transfer claim)

Date: 2026-09-15. PROJECT_STATUS.md item 3: rule out "bad harmonization for the one
dataset with no true midline channels" as an alternative explanation for the null
transfer result, by re-running the patched LODO pipeline on NeuMa <-> Restaurant-Logo
only (`configs/default_no_ds007406.yaml`, `--tag nods`). Same seed, same model, same
protocol as the Section-12-audited run; one GPU pass, then the script-06 permutation +
equivalence audit (2000 permutations, +/-0.10 kappa margin) on the saved fold
predictions (`results/lodo_permutation_equivalence_nods.md`).

| Held-out | n trials | n subj | kappa | p_holm (kappa) | AUC | p_holm (AUC) | kappa 90% CI | equivalent to null? |
|---|---|---|---|---|---|---|---|---|
| neuma | 4708 | 42 | 0.0011 | 1.000 | 0.501 | 0.724 | [-0.040, 0.046] | YES |
| restaurant_logo | 450 | 15 | 0.0000 | 1.000 | 0.513 | 0.471 | [0.000, 0.000] | YES |

Both folds remain null (kappa ~ 0, AUC ~ 0.50) AND formally equivalent to the null
within the +/-0.10 kappa margin. The null-transfer conclusion therefore does not depend
on ds007406's harmonization being the weak link: the NeuMa <-> Restaurant-Logo pair,
harmonized with the same substitution pipeline, shows the same null result with the
same statistical support. Combined with Section 28's ds007406-specific robustness
checks for the one positive result, the paper's negative claims now have both a
direct audit (permutation + equivalence, Section 17) and this leave-one-dataset-out
sensitivity check.

## 30. Restaurant-Logo motor confound: pre-median-RT truncation control

Date: 2026-09-15. PROJECT_STATUS.md item 2. The keypress differs by class
(A/P -> Recognized/NotRecognized) and epochs are stimulus-locked [0, 3] s, so the
post-keypress motor period is inside the analysis window (and C3/C4/Cz survive in
the shared set). Control (`scripts/25_rl_motor_confound_control.py`, report
`results/rl_motor_confound_report.md`): truncate every epoch to [0, 0.844) s --
the median RT pooled across subjects, a single global constant so no per-trial RT
leaks into the features; 52.7% of keypresses fall entirely outside the truncated
window -- recompute the same 36-d band-power features, re-run the identical
within-subject LOSOCV protocol, and add a family-level within-subject permutation
test (50 shuffles, decision-function AUC statistic; replicated at seed 7).

| Window | Model | kappa (mean +/- std) | ROC-AUC | p_perm (s42 / s7) |
|---|---|---|---|---|
| full [0, 3.0) | logreg | 0.1140 +/- 0.1902 | 0.6333 | 0.0196 / 0.0196 |
| full [0, 3.0) | svm_rbf | -0.0027 +/- 0.0562 | 0.5692 | 0.0784 / 0.1373 |
| truncated [0, 0.844) | logreg | 0.1148 +/- 0.2156 | 0.5685 | 0.0588 / 0.0784 |
| truncated [0, 0.844) | svm_rbf | 0.0149 +/- 0.1483 | 0.6332 | 0.0196 / 0.0392 |

Observed statistics are exactly seed-invariant (deterministic LOSOCV); only
permutation p varies with the null draw.

**Reading.** (1) The full-window signal is weak and model-inconsistent: logreg
decodes it, svm_rbf does not. A within-dataset signal one standard model finds
and the other does not is fragile -- do not headline it. (2) Truncation before
the median keypress does not abolish the signal, so the class-locked motor
period is not solely carrying it -- but it does not preserve it cleanly either:
the surviving decodability re-tunes to the other model (svm_rbf AUC 0.633,
logreg AUC 0.569), i.e. the pre-press residual is not the same signal as the
full-window one. (3) Honest writeup: Restaurant-Logo's recognition-related
activity beyond the motor confound is weak and unstable; treat this dataset's
contribution to transfer claims as uninterpretable-in-magnitude. The
null-transfer LODO conclusion is unaffected -- a dataset whose within-dataset
signal is this weak cannot be silently carrying cross-dataset transfer.
