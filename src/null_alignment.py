"""
null_alignment.py

The null-alignment control.

Rationale: this is the direct structural analogue of the density-matched
static/random graph nulls in INS-HDGS-CMT (Section 3.4), which showed the
"dynamic functional graph" pathway worked through its node-feature
processing, NOT through the measured time-varying connectivity -- because
replacing the real adjacency with a random one, matched only on density,
cost nothing.

Applied here: domain alignment (Stage 3, CORAL or Riemannian) is only
evidence of real cross-dataset structure if it beats alignment procedures
that are shape-matched but structurally uninformative. Two nulls:

  1. SHUFFLED-IDENTITY null: CORAL is computed as usual, but the
     dataset-identity labels used to decide "which distribution aligns to
     which" are randomly permuted before alignment. This preserves the
     alignment procedure's statistical machinery (same covariance
     re-centering, same matrix operations) while breaking the
     correspondence between the alignment and the TRUE domain gap. If
     real alignment doesn't beat this, the gain isn't coming from
     correctly identifying and closing the actual distribution shift.

  2. RANDOM-TARGET null: instead of aligning each source dataset's
     covariance to the real target dataset's covariance, align it to a
     synthetic covariance matrix with the same trace (overall scale) but
     a randomly rotated eigenbasis. This preserves "alignment changes the
     feature scale/rotation" while removing "alignment moves toward the
     TARGET's specific geometry."

Both nulls are deliberately cheap and fast (no GPU needed) so they run on
every LODO fold as a matter of course, not as an afterthought.

# VERIFY: this module implements CORAL directly (no pyriemann dependency)
# so it runs standalone on CPU here. The real Stage-3 pipeline on Brev
# should also expose a Riemannian-alignment path (via pyriemann); add a
# parallel `riemannian_align()` + null pair there using the same interface
# once pyriemann is confirmed working on the GPU image.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from construct_floor import DATASET_SPECS, load_harmonized_features
from stats_utils import summarize_comparison, holm_correction


def _coral_align(X_source: np.ndarray, X_target: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Standard CORAL (Sun & Saenko, 2016), as named and cited in the project
    reference doc: re-color the source covariance to match the target's.
    """
    Xs = X_source - X_source.mean(axis=0, keepdims=True)
    Xt = X_target - X_target.mean(axis=0, keepdims=True)

    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])

    # whiten by source covariance, then color by target covariance
    Cs_sqrt_inv = _matrix_inv_sqrt(Cs)
    Ct_sqrt = _matrix_sqrt(Ct)

    Xs_aligned = Xs @ Cs_sqrt_inv @ Ct_sqrt
    return Xs_aligned + X_target.mean(axis=0, keepdims=True)


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-8, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_inv_sqrt(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-8, None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _random_rotated_covariance(reference_cov: np.ndarray, seed: int) -> np.ndarray:
    """Same eigenvalue spectrum (so same 'scale') as reference_cov, random eigenbasis."""
    rng = np.random.default_rng(seed)
    eigvals, _ = np.linalg.eigh(reference_cov)
    d = reference_cov.shape[0]
    random_mat = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(random_mat)
    return q @ np.diag(np.clip(eigvals, 1e-8, None)) @ q.T


def real_alignment(X_source, X_target):
    return _coral_align(X_source, X_target)


def shuffled_identity_null(X_source, X_target, seed: int):
    """
    Pools source+target, randomly re-splits into two groups of the same
    sizes, then aligns using those fake group identities. The alignment
    machinery runs identically; only which-points-belong-to-which-domain
    is scrambled.
    """
    rng = np.random.default_rng(seed)
    pooled = np.vstack([X_source, X_target])
    idx = rng.permutation(len(pooled))
    fake_source = pooled[idx[: len(X_source)]]
    fake_target = pooled[idx[len(X_source):]]
    aligned_fake_source = _coral_align(fake_source, fake_target)
    # return only the rows corresponding to the ORIGINAL source points,
    # re-identified via the permutation, so shapes match downstream use
    source_positions = idx[: len(X_source)]
    order = np.argsort(source_positions)
    return aligned_fake_source[order]


def random_target_null(X_source, X_target, seed: int):
    """Aligns source to a random covariance matched only on eigenvalue scale to the real target."""
    Xs = X_source - X_source.mean(axis=0, keepdims=True)
    Ct_real = np.cov(X_target - X_target.mean(axis=0, keepdims=True), rowvar=False)
    Ct_random = _random_rotated_covariance(Ct_real, seed=seed)

    Cs = np.cov(Xs, rowvar=False) + 1e-4 * np.eye(Xs.shape[1])
    Cs_sqrt_inv = _matrix_inv_sqrt(Cs)
    Ct_random_sqrt = _matrix_sqrt(Ct_random + 1e-4 * np.eye(Ct_random.shape[0]))
    aligned = Xs @ Cs_sqrt_inv @ Ct_random_sqrt
    return aligned + X_source.mean(axis=0, keepdims=True)


def _train_eval(X_train, y_train, X_test, y_test) -> float:
    if len(np.unique(y_test)) < 2:
        return float("nan")
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(X_train), y_train)
    prob = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return roc_auc_score(y_test, prob)


def run_lodo_with_alignment_variants(seed: int = 42) -> List[Dict]:
    """
    For each held-out ("target") dataset in turn, trains on the pooled
    (aligned) remaining two ("source") datasets and evaluates on the held
    out one, under four conditions:
        no_adaptation      -- Stage-3 skipped entirely (the ablation
                               control already specified in Section 5.2
                               of the project reference doc)
        real_alignment      -- CORAL as designed
        shuffled_identity_null
        random_target_null

    Returns one row per (target_dataset, condition, fold-equivalent
    subject-group) so the results can be paired and Holm-corrected
    downstream by run_null_alignment_audit().
    """
    rows = []
    dataset_names = [s.name for s in DATASET_SPECS]

    for held_out in dataset_names:
        target_data = load_harmonized_features(held_out, seed=seed)
        source_names = [n for n in dataset_names if n != held_out]

        source_X, source_y = [], []
        for sn in source_names:
            d = load_harmonized_features(sn, seed=seed)
            source_X.append(d["X"])
            source_y.append(d["y"])
        X_source_raw = np.vstack(source_X)
        y_source = np.concatenate(source_y)
        X_target = target_data["X"]
        y_target = target_data["y"]

        # subject groups in the target set define the paired units for
        # the audit's statistics (mirrors "participant-paired" testing)
        groups_target = target_data["groups"]

        conditions = {
            "no_adaptation": lambda: X_source_raw,
            "real_alignment": lambda: real_alignment(X_source_raw, X_target),
            "shuffled_identity_null": lambda: shuffled_identity_null(X_source_raw, X_target, seed=seed),
            "random_target_null": lambda: random_target_null(X_source_raw, X_target, seed=seed),
        }

        for cond_name, cond_fn in conditions.items():
            X_source_cond = cond_fn()
            # per-subject AUC in the target set, so comparisons across
            # conditions can be paired subject-by-subject
            for subj in np.unique(groups_target):
                subj_mask = groups_target == subj
                auc = _train_eval(X_source_cond, y_source, X_target[subj_mask], y_target[subj_mask])
                rows.append({
                    "held_out_dataset": held_out,
                    "subject": int(subj),
                    "condition": cond_name,
                    "auc": auc,
                })
    return rows


def run_null_alignment_audit(seed: int = 42):
    """
    Runs the full LODO x condition sweep, then reports three paired
    families (each Holm-corrected within itself, matching the family
    structure used in both prior papers):
        real_alignment vs no_adaptation        (does alignment help at all)
        real_alignment vs shuffled_identity_null  (does it use the real domain gap)
        real_alignment vs random_target_null      (does it use the real target geometry)
    """
    rows = run_lodo_with_alignment_variants(seed=seed)

    import collections
    by_key = collections.defaultdict(dict)
    for r in rows:
        key = (r["held_out_dataset"], r["subject"])
        by_key[key][r["condition"]] = r["auc"]

    families = {
        "real_vs_no_adaptation": ("real_alignment", "no_adaptation"),
        "real_vs_shuffled_identity": ("real_alignment", "shuffled_identity_null"),
        "real_vs_random_target": ("real_alignment", "random_target_null"),
    }

    print("Null-alignment audit (per held-out dataset, subject-paired)\n" + "=" * 70)
    all_summaries = []
    for fam_name, (cond_a, cond_b) in families.items():
        per_dataset_summaries = []
        for spec in DATASET_SPECS:
            a_vals, b_vals = [], []
            for (held_out, subj), conds in by_key.items():
                if held_out != spec.name:
                    continue
                if cond_a in conds and cond_b in conds:
                    a_vals.append(conds[cond_a])
                    b_vals.append(conds[cond_b])
            if len(a_vals) == 0:
                continue
            summary = summarize_comparison(a_vals, b_vals, label=f"{fam_name} | held_out={spec.name}")
            per_dataset_summaries.append(summary)
        all_summaries.extend(per_dataset_summaries)

    # Holm-correct within the whole declared family (all held-out datasets
    # x this comparison type), consistent with how both prior papers
    # correct within a declared family rather than globally.
    pvals = [s["p_raw"] for s in all_summaries]
    adjusted = holm_correction(pvals) if pvals else []
    for s, p_adj in zip(all_summaries, adjusted):
        s["p_holm"] = p_adj

    print(f"{'Comparison':<45}{'Mean Δ AUC':<12}{'95% CI':<20}{'p_raw':<10}{'p_holm':<10}{'n'}")
    print("-" * 105)
    for s in all_summaries:
        ci_str = f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]"
        print(f"{s['label']:<45}{s['mean_diff']:<12.4f}{ci_str:<20}{s['p_raw']:<10.3f}{s['p_holm']:<10.3f}{s['n_pairs']}")

    print("-" * 105)
    print(
        "Reading: real_alignment must significantly exceed ALL THREE nulls "
        "for domain alignment to be credited with using genuine cross-"
        "dataset structure, not just 'doing something' to the features.\n"
        "NOTE: figures above are from SYNTHETIC data. Re-run on Brev with "
        "real Stage-2 caches before these numbers appear in any manuscript."
    )
    return all_summaries


if __name__ == "__main__":
    run_null_alignment_audit()
