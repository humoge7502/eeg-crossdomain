"""
construct_floor.py

The construct-floor check.

Rationale (this is the audit's answer to "the three tasks aren't really the
same construct"): before any cross-dataset transfer claim means anything,
we need to know how much of EACH dataset's OWN label is trivially
recoverable from generic, task-agnostic features, WITHIN that dataset,
using nothing that resembles domain adaptation at all.

This is the direct structural analogue of two checks already run in the
prior papers:
  - NeuroClick's dwell-plus-propensity baseline, which showed the
    "first1 gain" was explained by propensity rather than sequence modelling
  - INS-HDGS-CMT's linear probe on the label's own defining EEG/gaze terms
    (Supplementary Table S15), which showed 0.92 ROC-AUC was recoverable
    from gaze statistics alone, making every gaze-touching model's headline
    score uninterpretable without that floor

Here the same move is applied one level up: per dataset, per label, what
ROC-AUC does a plain logistic regression on generic band-power features
reach with NO cross-dataset information at all? Any subsequent transfer
result has to be read against this floor, not against chance (0.5).

If a model trained on two harmonized datasets and evaluated on the third
scores BELOW what an in-dataset-only linear probe reaches, harmonization
is doing worse than doing nothing. If it scores AT the floor, transfer
has not been demonstrated to add anything the receiving dataset's own
generic statistics didn't already carry. Only a score ABOVE the floor,
using held-out-dataset labels that a linear probe on that dataset's own
features cannot reach, is evidence of real shared structure.

# VERIFY: harmonized feature loading currently reads the Stage-2 output
# format described in the project reference doc (per-channel band power +
# frontal alpha asymmetry over the shared 7-channel basis: Fz, Cz, Pz, F3,
# F4, O1, O2). Swap `load_harmonized_features()` for the real loader once
# scripts/02_preprocess_all.py's cache format is confirmed on Brev.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stats_utils import bootstrap_ci

SHARED_CHANNELS = ["Fz", "Cz", "Pz", "F3", "F4", "O1", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class DatasetSpec:
    name: str
    n_subjects: int
    label_name: str
    native_channels: int
    # class balance used only for synthetic generation / smoke testing
    positive_rate: float = 0.4


# The three datasets as described in the project reference document.
DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec("NeuMa", n_subjects=42, label_name="Buy/NoBuy", native_channels=19,
                positive_rate=0.13),  # matches NeuMa's observed Buy prevalence
    DatasetSpec("RestaurantLogo", n_subjects=15, label_name="Recognized/NotRecognized",
                native_channels=8, positive_rate=0.5),
    DatasetSpec("ds007406", n_subjects=10, label_name="Extreme/Traditional",
                native_channels=14, positive_rate=0.5),
]


def _synthetic_harmonized_dataset(spec: DatasetSpec, seed: int) -> Dict[str, np.ndarray]:
    """
    Generates a harmonized-feature-shaped synthetic dataset: one row per
    (subject, trial), n_features = len(SHARED_CHANNELS) * len(BANDS) + 1
    (the +1 is the frontal alpha asymmetry index), matching Stage 2's
    documented output dimensionality exactly.

    This is a smoke-testing stand-in only, in the same spirit as the
    existing repo's synthetic smoke tests (Section 9.2 of the reference
    doc) — it is NOT a claim about what the real data will show. Replace
    with load_harmonized_features(spec.name) once real Stage-2 caches
    exist on Brev.
    """
    rng = np.random.default_rng(seed)
    n_features = len(SHARED_CHANNELS) * len(BANDS) + 1
    trials_per_subject = 12
    n_total = spec.n_subjects * trials_per_subject

    subject_ids = np.repeat(np.arange(spec.n_subjects), trials_per_subject)
    # weak, subject-specific generic signal (so the floor is neither 0.5 nor 1.0)
    subject_bias = rng.normal(0, 0.4, size=spec.n_subjects)
    X = rng.normal(0, 1.0, size=(n_total, n_features))
    signal_weight = rng.normal(0, 1.0, size=n_features)
    latent = X @ signal_weight * 0.15 + subject_bias[subject_ids]
    prob = 1.0 / (1.0 + np.exp(-latent))
    # shift to match the declared positive rate
    threshold = np.quantile(prob, 1.0 - spec.positive_rate)
    y = (prob >= threshold).astype(int)

    return {"X": X, "y": y, "groups": subject_ids}


def load_harmonized_features(dataset_name: str, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    # VERIFY: replace this dispatcher with a real loader reading
    # scripts/02_preprocess_all.py's cached Stage-2 output once it exists.
    Falls back to synthetic data (clearly labelled in every downstream
    report) so the audit pipeline is runnable end-to-end before real
    data is available.
    """
    for spec in DATASET_SPECS:
        if spec.name == dataset_name:
            return _synthetic_harmonized_dataset(spec, seed=seed)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def construct_floor_for_dataset(dataset_name: str, seed: int = 42, n_splits: int = 5):
    """
    Fits a plain, generic-feature logistic regression WITHIN one dataset,
    using grouped (subject-disjoint) k-fold cross-validation so no
    subject's own trials leak into their own held-out fold -- the same
    leakage discipline used throughout both prior papers.

    Returns per-fold ROC-AUC plus a bootstrap CI on the mean. This is the
    floor: the ceiling a "successful transfer" claim into this dataset
    must clear to mean anything.
    """
    data = load_harmonized_features(dataset_name, seed=seed)
    X, y, groups = data["X"], data["y"], data["groups"]

    n_groups = len(np.unique(groups))
    n_splits_eff = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits_eff)

    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue  # fold has only one class present; skip, as in INS-HDGS-CMT's
                       # exclusion of single-class subjects from test folds
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y[train_idx])
        prob = clf.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], prob))

    mean_auc, lo, hi = bootstrap_ci(fold_aucs)
    return {
        "dataset": dataset_name,
        "n_folds_evaluable": len(fold_aucs),
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def run_all_construct_floors(seed: int = 42):
    """
    Runs the construct-floor check for every dataset and prints a summary
    table. This should run BEFORE any LODO transfer experiment; the
    transfer results in scripts/04_run_lodo_experiment.py should always be
    reported alongside these numbers, never in isolation.
    """
    results = []
    print(f"{'Dataset':<18}{'Label':<28}{'Floor AUC':<12}{'95% CI':<20}{'Folds'}")
    print("-" * 90)
    for spec in DATASET_SPECS:
        r = construct_floor_for_dataset(spec.name, seed=seed)
        results.append(r)
        ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"{spec.name:<18}{spec.label_name:<28}{r['mean_auc']:.3f}       {ci_str:<20}{r['n_folds_evaluable']}")
    print("-" * 90)
    print(
        "NOTE: figures above are from SYNTHETIC data (see # VERIFY in "
        "load_harmonized_features). Re-run after wiring the real Stage-2 "
        "cache on Brev before these numbers are used in any report."
    )
    return results


if __name__ == "__main__":
    run_all_construct_floors()
