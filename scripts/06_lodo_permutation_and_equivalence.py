"""
scripts/06_lodo_permutation_and_equivalence.py

The script standing between the current draft and a submittable core
result (per the strict-second-reviewer read): a formal significance test
on the actual LODO transfer metric, plus an equivalence bound so the null
claim means "no transfer detectable above X" rather than just "p > 0.05".

DESIGN: tests are run against the FIXED predictions saved by the patched
scripts/04_run_lodo_experiment.py (y_true, y_prob, subject_ids per
held-out fold) — NOT by retraining per permutation. This is the same
design already used for INS-HDGS-CMT's within-subject label-permutation
test (Section 3.1 of that paper: "5000 shuffles of the labels within each
held-out subject... shuffled against fixed predictions"), extended here
from a single dataset to the cross-dataset LODO setting. It answers "is
the association between these fixed predictions and the true labels
distinguishable from chance", which is the right question for a trained,
already-evaluated model — it does NOT re-test whether the training
procedure itself could fit random labels.

WHY THIS MATTERS FOR THE DEADLINE: no GPU retraining is required. This
runs entirely on saved .npz files, in seconds, on CPU. The only GPU cost
is the ONE additional LODO run needed to produce those .npz files with
the patches/04_run_lodo_experiment.py and patches/train.py changes
applied — the same cost as your existing lodo_log2.txt run.

TWO THINGS THIS SCRIPT REPORTS, PER HELD-OUT DATASET AND POOLED:
  1. A one-sided permutation p-value for kappa and for AUC (Holm-corrected
     across the 3 held-out folds, within each metric family separately —
     matching the family-correction convention from both your prior
     papers).
  2. A TOST-style equivalence check: whether the observed effect's
     bootstrap CI falls entirely inside a pre-specified "no meaningful
     transfer" band (default +/- 0.10 kappa, a conventional small-effect
     threshold in EEG decoding — override with --equivalence-margin if
     you have a better-justified value for this literature).

If subject_ids are present for a given held-out dataset, permutation and
bootstrap are BOTH done within-subject (the conservative, correct choice
for non-independent trials). If subject_ids are missing for a dataset,
the script runs anyway but prints a loud warning and tags that fold's
results as "ANTI-CONSERVATIVE (trial-level, no subject grouping)" in the
report — do not treat that fold's p-value as final until the upstream
subject_id caching gap is closed for it.

Usage:
    python scripts/06_lodo_permutation_and_equivalence.py \\
        --n-permutations 2000 --equivalence-margin 0.10 \\
        --out results/lodo_permutation_equivalence_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stats_utils import holm_correction  # noqa: E402

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]


def load_predictions(results_dir: Path, name: str):
    path = results_dir / f"lodo_predictions_{name}.npz"
    if not path.exists():
        print(f"[audit] WARNING: {path} not found — skipping {name}. "
              f"Re-run the patched scripts/04_run_lodo_experiment.py first.")
        return None
    data = np.load(path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    subject_ids = data["subject_ids"] if "subject_ids" in data else None
    return {"y_true": y_true, "y_prob": y_prob, "subject_ids": subject_ids}


def observed_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # single-class held-out set
    return kappa, auc


def permutation_null(y_true, y_prob, subject_ids, n_perm: int, seed: int):
    """
    Builds the null distribution of (kappa, auc) by shuffling y_true
    against the FIXED y_prob/y_pred, within-subject when subject_ids is
    available (else across the whole held-out set, with a caveat flagged
    by the caller).
    """
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)

    null_kappas = np.empty(n_perm)
    null_aucs = np.empty(n_perm)

    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        idx_by_subject = {s: np.where(subject_ids == s)[0] for s in unique_subjects}
    else:
        idx_by_subject = None

    for p in range(n_perm):
        if idx_by_subject is not None:
            shuffled = y_true.copy()
            for s, idxs in idx_by_subject.items():
                shuffled[idxs] = rng.permutation(y_true[idxs])
        else:
            shuffled = rng.permutation(y_true)

        null_kappas[p] = cohen_kappa_score(shuffled, y_pred)
        try:
            null_aucs[p] = roc_auc_score(shuffled, y_prob)
        except ValueError:
            null_aucs[p] = np.nan

    return null_kappas, null_aucs


def one_sided_p(observed: float, null_dist: np.ndarray) -> float:
    """(1 + #{null >= observed}) / (1 + N) — the exact formula your own
    INS-HDGS-CMT paper used for its label-permutation test."""
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.isnan(observed):
        return float("nan")
    return float((1 + np.sum(null_dist >= observed)) / (1 + len(null_dist)))


def subject_bootstrap_ci(y_true, y_prob, subject_ids, n_boot: int, seed: int, ci: float = 0.90):
    """
    Subject-level bootstrap (resample subjects with replacement, keep all
    their trials) — the correct unit of resampling given within-subject
    trial correlation. Falls back to trial-level bootstrap (flagged by
    the caller) if subject_ids is unavailable.
    """
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)

    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        idx_by_subject = {s: np.where(subject_ids == s)[0] for s in unique_subjects}
        n_subj = len(unique_subjects)
        boot_kappas, boot_aucs = [], []
        for _ in range(n_boot):
            chosen = rng.choice(unique_subjects, size=n_subj, replace=True)
            idxs = np.concatenate([idx_by_subject[s] for s in chosen])
            if len(np.unique(y_true[idxs])) < 2:
                continue
            boot_kappas.append(cohen_kappa_score(y_true[idxs], y_pred[idxs]))
            try:
                boot_aucs.append(roc_auc_score(y_true[idxs], y_prob[idxs]))
            except ValueError:
                pass
    else:
        n = len(y_true)
        boot_kappas, boot_aucs = [], []
        for _ in range(n_boot):
            idxs = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idxs])) < 2:
                continue
            boot_kappas.append(cohen_kappa_score(y_true[idxs], y_pred[idxs]))
            try:
                boot_aucs.append(roc_auc_score(y_true[idxs], y_prob[idxs]))
            except ValueError:
                pass

    alpha = (1.0 - ci) / 2.0

    def _ci(vals):
        if len(vals) == 0:
            return float("nan"), float("nan")
        lo, hi = np.quantile(vals, [alpha, 1 - alpha])
        return float(lo), float(hi)

    kappa_lo, kappa_hi = _ci(boot_kappas)
    auc_lo, auc_hi = _ci(boot_aucs)
    return {"kappa_ci": (kappa_lo, kappa_hi), "auc_ci": (auc_lo, auc_hi)}


def tost_equivalence(ci_lo: float, ci_hi: float, margin: float) -> bool:
    """True if the CI falls entirely within [-margin, +margin] — the
    two-one-sided-tests (TOST) equivalence criterion."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return (ci_lo >= -margin) and (ci_hi <= margin)


def run_audit(results_dir: Path, n_perm: int, equivalence_margin: float, seed: int):
    fold_reports = []

    for name in DATASET_NAMES:
        preds = load_predictions(results_dir, name)
        if preds is None:
            continue

        y_true, y_prob, subject_ids = preds["y_true"], preds["y_prob"], preds["subject_ids"]
        conservative = subject_ids is not None

        obs_kappa, obs_auc = observed_metrics(y_true, y_prob)
        null_kappas, null_aucs = permutation_null(y_true, y_prob, subject_ids, n_perm, seed)
        p_kappa = one_sided_p(obs_kappa, null_kappas)
        p_auc = one_sided_p(obs_auc, null_aucs) if not np.isnan(obs_auc) else float("nan")

        ci = subject_bootstrap_ci(y_true, y_prob, subject_ids, n_boot=5000, seed=seed, ci=0.90)
        kappa_equiv = tost_equivalence(*ci["kappa_ci"], margin=equivalence_margin)

        fold_reports.append({
            "dataset": name,
            "n_trials": int(len(y_true)),
            "n_subjects": int(len(np.unique(subject_ids))) if subject_ids is not None else None,
            "conservative_within_subject": conservative,
            "observed_kappa": float(obs_kappa),
            "observed_auc": float(obs_auc) if not np.isnan(obs_auc) else None,
            "p_kappa_raw": p_kappa,
            "p_auc_raw": p_auc,
            "kappa_ci_90": ci["kappa_ci"],
            "auc_ci_90": ci["auc_ci"],
            "equivalent_to_no_transfer": kappa_equiv,
            "equivalence_margin": equivalence_margin,
        })

    # Holm-correct within each metric family across the (up to 3) folds
    kappa_pvals = [r["p_kappa_raw"] for r in fold_reports]
    auc_pvals = [r["p_auc_raw"] for r in fold_reports if not np.isnan(r["p_auc_raw"])]
    if kappa_pvals:
        kappa_adj = holm_correction(kappa_pvals)
        for r, p in zip(fold_reports, kappa_adj):
            r["p_kappa_holm"] = float(p)
    auc_idx = 0
    if auc_pvals:
        auc_adj = holm_correction(auc_pvals)
        for r in fold_reports:
            if not np.isnan(r["p_auc_raw"]):
                r["p_auc_holm"] = float(auc_adj[auc_idx])
                auc_idx += 1
            else:
                r["p_auc_holm"] = float("nan")

    return fold_reports


def write_report(fold_reports, out_path: Path, n_perm: int, margin: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# LODO permutation test + equivalence audit\n")
    lines.append(f"n_permutations={n_perm}, equivalence_margin=+/-{margin} kappa\n")

    lines.append("\n| Dataset | n trials | n subj | conservative? | kappa | p_holm | AUC | p_holm | kappa 90% CI | equivalent to null? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in fold_reports:
        cons = "yes" if r["conservative_within_subject"] else "**NO — trial-level fallback**"
        auc_str = f"{r['observed_auc']:.3f}" if r["observed_auc"] is not None else "n/a"
        p_auc_str = f"{r['p_auc_holm']:.3f}" if not np.isnan(r.get("p_auc_holm", float('nan'))) else "n/a"
        ci = r["kappa_ci_90"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        equiv = "YES" if r["equivalent_to_no_transfer"] else "no"
        lines.append(
            f"| {r['dataset']} | {r['n_trials']} | {r['n_subjects']} | {cons} | "
            f"{r['observed_kappa']:.4f} | {r['p_kappa_holm']:.3f} | {auc_str} | {p_auc_str} | "
            f"{ci_str} | {equiv} |"
        )

    any_nonconservative = any(not r["conservative_within_subject"] for r in fold_reports)
    lines.append("\n## Reading this table\n")
    lines.append(
        "- `p_holm` significant (< 0.05) on kappa or AUC would mean the fixed "
        "predictions ARE distinguishable from chance for that held-out dataset "
        "— i.e., evidence FOR some transfer, contradicting the null claim.\n"
        "- `equivalent to null? = YES` means the 90% bootstrap CI on kappa falls "
        f"entirely within +/-{margin}, supporting a positive equivalence claim "
        "(\"no transfer exceeding this margin\") rather than a mere failure to "
        "reject chance.\n"
        "- A fold with neither a significant p AND equivalence = YES is "
        "underpowered: report it as inconclusive, not as a null finding.\n"
    )
    if any_nonconservative:
        lines.append(
            "\n**WARNING: at least one fold ran without subject_ids** "
            "(trial-level permutation/bootstrap, anti-conservative — treat its "
            "p-value and CI as provisional until the upstream subject_id "
            "caching gap is closed for that dataset; see patches/README.md).\n"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(fold_reports, f, indent=2)
    print(f"\n[audit] Report written -> {out_path}")
    print(f"[audit] JSON written -> {out_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--equivalence-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/lodo_permutation_equivalence_report.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fold_reports = run_audit(results_dir, args.n_permutations, args.equivalence_margin, args.seed)

    if not fold_reports:
        print("[audit] No lodo_predictions_*.npz files found. Run the patched "
              "scripts/04_run_lodo_experiment.py first (see patches/README.md).")
        return

    write_report(fold_reports, Path(args.out), args.n_permutations, args.equivalence_margin)


if __name__ == "__main__":
    main()
