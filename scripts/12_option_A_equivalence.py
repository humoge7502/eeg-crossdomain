"""
Applies the same subject-level bootstrap + TOST equivalence methodology
used in scripts/06_lodo_permutation_and_equivalence.py to Option A's
(full-duration tabular features, CORAL) results, so reporting is
consistent across the whole paper (margin=+/-0.10 kappa, 90% CI,
subject-level bootstrap resampling, n_boot=5000).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from src.utils import project_root, ensure_dir

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
CONDITIONS = ["noalign", "coral"]
EQUIVALENCE_MARGIN = 0.10
N_BOOT = 5000
SEED = 42


def subject_bootstrap_ci(y_true, y_prob, subject_ids, n_boot, seed, ci=0.90):
    """Identical logic to scripts/06_lodo_permutation_and_equivalence.py's
    subject_bootstrap_ci, duplicated here to keep this script standalone."""
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)

    if subject_ids is not None and subject_ids.size > 0:
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
        conservative = True
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
        conservative = False

    alpha = (1.0 - ci) / 2.0

    def _ci(vals):
        if len(vals) == 0:
            return float("nan"), float("nan")
        lo, hi = np.quantile(vals, [alpha, 1 - alpha])
        return float(lo), float(hi)

    kappa_lo, kappa_hi = _ci(boot_kappas)
    auc_lo, auc_hi = _ci(boot_aucs)
    return {"kappa_ci": (kappa_lo, kappa_hi), "auc_ci": (auc_lo, auc_hi), "conservative_within_subject": conservative}


def tost_equivalence(ci_lo, ci_hi, margin):
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return (ci_lo >= -margin) and (ci_hi <= margin)


def main():
    results_dir = ensure_dir(project_root() / "results")
    all_reports = {}

    for condition in CONDITIONS:
        for name in DATASET_NAMES:
            fpath = results_dir / f"optionA_predictions_{condition}_{name}.npz"
            if not fpath.exists():
                print(f"[skip] {fpath} not found")
                continue
            d = np.load(fpath)
            y_true, y_prob, subject_ids = d["y_true"], d["y_prob"], d["subject_ids"]
            y_pred = (y_prob >= 0.5).astype(int)
            observed_kappa = float(cohen_kappa_score(y_true, y_pred))
            try:
                observed_auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                observed_auc = None

            ci = subject_bootstrap_ci(y_true, y_prob, subject_ids, n_boot=N_BOOT, seed=SEED, ci=0.90)
            kappa_equiv = tost_equivalence(*ci["kappa_ci"], margin=EQUIVALENCE_MARGIN)

            key = f"{condition}_{name}"
            report = {
                "dataset": name,
                "condition": condition,
                "n_trials": int(len(y_true)),
                "n_subjects": int(len(np.unique(subject_ids))) if subject_ids.size else None,
                "conservative_within_subject": ci["conservative_within_subject"],
                "observed_kappa": observed_kappa,
                "observed_auc": observed_auc,
                "kappa_ci_90": ci["kappa_ci"],
                "auc_ci_90": ci["auc_ci"],
                "equivalent_to_no_transfer": kappa_equiv,
                "equivalence_margin": EQUIVALENCE_MARGIN,
            }
            all_reports[key] = report
            print(f"[{key}] kappa={observed_kappa:.4f}, 90% CI=[{ci['kappa_ci'][0]:.4f}, {ci['kappa_ci'][1]:.4f}], "
                  f"equivalent_to_null={kappa_equiv}")

    out_path = results_dir / "option_A_equivalence_report.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Markdown summary
    md_lines = ["# Option A equivalence audit\n",
                f"n_boot={N_BOOT}, equivalence_margin=+/-{EQUIVALENCE_MARGIN} kappa, subject-level bootstrap (90% CI)\n",
                "| Condition | Dataset | n_trials | n_subjects | Observed kappa | 90% CI | Equivalent to null? |",
                "|---|---|---|---|---|---|---|"]
    for key, r in all_reports.items():
        ci_str = f"[{r['kappa_ci_90'][0]:.4f}, {r['kappa_ci_90'][1]:.4f}]"
        md_lines.append(f"| {r['condition']} | {r['dataset']} | {r['n_trials']} | {r['n_subjects']} | "
                         f"{r['observed_kappa']:.4f} | {ci_str} | {'YES' if r['equivalent_to_no_transfer'] else 'no'} |")
    md_path = results_dir / "option_A_equivalence_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
