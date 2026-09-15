"""
scripts/07_diagnose_and_retry_lodo.py

Designed to run unattended overnight via nohup. Does four things in
order, logging clearly at each step, and writes one skimmable SUMMARY
block at the very end so the first thing you read in the morning is the
answer, not a wall of epoch logs.

  1. DIAGNOSE: prints raw per-dataset X scale (mean/std/min/max) from the
     cached *_epochs_common.npz files. This directly tests the leading
     hypothesis for last night's collapse: configs/experiment_v2.yaml
     declares unit_scale_to_volts as 1e-6 for neuma vs 1.0 for the other
     two datasets. If that scaling was applied to the Stage-2 tabular
     features but NOT to the raw epochs cached for EEGNet/DeepConvNet,
     a batchnorm layer trained on one dataset's scale could saturate to
     a near-constant output the moment it sees a wildly different-scale
     domain — which is exactly what last night's std=0.0000 findings
     look like.

  2. FIX (if warranted): applies per-dataset z-score normalization to the
     raw X arrays IN MEMORY before training — each dataset normalized
     using only its own mean/std, unsupervised, no cross-dataset or
     cross-label information used. This is the same per-dataset
     unsupervised-normalization logic your own experiment_v2.yaml already
     declares for the Stage-2 tabular pipeline (per_participant_normalization),
     just applied one level up, at the dataset level, for the raw-epoch
     models. Original cached files on disk are NOT modified.

  3. RETRAIN: re-runs the exact same 3-fold LODO training as
     scripts/04_run_lodo_experiment.py (same model, same hyperparameters,
     same class-weighting, same early stopping) on the normalized data,
     saving fixed predictions to results/lodo_predictions_scaled_<name>.npz
     (a NEW path — your original unscaled results/lodo_predictions_*.npz
     from last night are untouched).

  4. RE-AUDIT: immediately re-runs the permutation + equivalence test
     (same logic as scripts/06) against the new scaled predictions,
     writing results/lodo_scaled_permutation_equivalence_report.md.

Usage (run this, then go to sleep):
    nohup python3 scripts/07_diagnose_and_retry_lodo.py > diagnose_retry_log.txt 2>&1 &

In the morning:
    tail -100 diagnose_retry_log.txt
    cat results/lodo_scaled_permutation_equivalence_report.md
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.train import EEGTensorDataset, train_one_model, evaluate_model
from src.models import EEGNetSharedEncoder, CrossDomainDecoder
from torch.utils.data import DataLoader
from stats_utils import holm_correction  # from the audit module added earlier

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------
# STEP 1: DIAGNOSE
# ---------------------------------------------------------------------

def load_raw(cfg, name):
    path = project_root() / cfg["paths"]["processed_dir"] / name / f"{name}_epochs_common.npz"
    if not path.exists():
        return None
    data = np.load(path)
    subject_ids = data["subject_ids"] if "subject_ids" in data else None
    return {"X": data["X"], "y": data["y"], "subject_ids": subject_ids}


def diagnose_scale(all_raw):
    log("\n" + "=" * 70)
    log("STEP 1/4 -- DIAGNOSING RAW SCALE PER DATASET")
    log("=" * 70)
    stats = {}
    for name, d in all_raw.items():
        X = d["X"]
        s = {
            "mean": float(X.mean()), "std": float(X.std()),
            "min": float(X.min()), "max": float(X.max()),
            "abs_mean": float(np.abs(X).mean()),
        }
        stats[name] = s
        log(f"  {name:<18} mean={s['mean']:.6g} std={s['std']:.6g} "
            f"min={s['min']:.6g} max={s['max']:.6g} abs_mean={s['abs_mean']:.6g}")

    abs_means = {k: v["abs_mean"] for k, v in stats.items() if v["abs_mean"] > 0}
    if len(abs_means) >= 2:
        ratio = max(abs_means.values()) / min(abs_means.values())
        log(f"\n  Largest/smallest abs_mean ratio across datasets: {ratio:,.1f}x")
        if ratio > 100:
            log("  >>> LARGE SCALE MISMATCH CONFIRMED. This is very likely contributing "
                "to (or fully explaining) last night's constant-output collapse on "
                "restaurant_logo and ds007406. Proceeding to per-dataset normalization + retrain.")
        else:
            log("  >>> Scale mismatch is not dramatic. The collapse may have a different "
                "cause (e.g. early-stopping-on-epoch-1, a dead/saturated layer from "
                "initialization). Proceeding with normalization + retrain anyway, since "
                "it is a safe, standard step regardless, but flag this in the morning "
                "for a second look if the collapse persists after this run.")
    return stats


# ---------------------------------------------------------------------
# STEP 2: FIX -- per-dataset z-score normalization, in memory only
# ---------------------------------------------------------------------

def normalize_per_dataset(all_raw):
    log("\n" + "=" * 70)
    log("STEP 2/4 -- APPLYING PER-DATASET Z-SCORE NORMALIZATION (in memory)")
    log("=" * 70)
    normalized = {}
    for name, d in all_raw.items():
        X = d["X"]
        mu, sigma = X.mean(), X.std()
        sigma = sigma if sigma > 1e-12 else 1.0
        X_norm = (X - mu) / sigma
        normalized[name] = {"X": X_norm, "y": d["y"], "subject_ids": d["subject_ids"]}
        log(f"  {name:<18} normalized using its OWN mean={mu:.6g}, std={sigma:.6g} "
            f"(unsupervised, no cross-dataset or label information used)")
    return normalized


# ---------------------------------------------------------------------
# STEP 3: RETRAIN -- same LODO loop as scripts/04, on normalized data
# ---------------------------------------------------------------------

def run_lodo_fold_normalized(held_out, all_data, cfg, device):
    train_names = [k for k in all_data.keys() if k != held_out]
    X_train = np.concatenate([all_data[k]["X"] for k in train_names], axis=0)
    y_train = np.concatenate([all_data[k]["y"] for k in train_names], axis=0)
    X_test, y_test = all_data[held_out]["X"], all_data[held_out]["y"]
    test_subject_ids = all_data[held_out]["subject_ids"]

    n_channels, n_timepoints = X_train.shape[2], X_train.shape[3]
    encoder = EEGNetSharedEncoder(
        n_channels=n_channels, n_timepoints=n_timepoints,
        n_filters_temporal=cfg["model"]["n_filters_temporal"],
        n_filters_spatial=cfg["model"]["n_filters_spatial"],
        embedding_dim=cfg["model"]["embedding_dim"], dropout=cfg["model"]["dropout"],
    )
    model = CrossDomainDecoder(encoder, embedding_dim=cfg["model"]["embedding_dim"],
                                n_classes=2, dropout=cfg["model"]["dropout"])

    n_val = max(1, int(0.15 * len(y_train)))
    perm = np.random.default_rng(cfg["training"]["seed"]).permutation(len(y_train))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(EEGTensorDataset(X_train[train_idx], y_train[train_idx]),
                               batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(EEGTensorDataset(X_train[val_idx], y_train[val_idx]),
                             batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(EEGTensorDataset(X_test, y_test),
                              batch_size=cfg["training"]["batch_size"], shuffle=False)

    trained = train_one_model(model, train_loader, val_loader, device,
                               epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"],
                               early_stopping_patience=cfg["training"]["early_stopping_patience"])
    test_results = evaluate_model(trained["model"], test_loader, device)

    return {
        "held_out": held_out, "trained_on": train_names,
        "metrics": test_results["metrics"],
        "y_true": test_results["y_true"], "y_prob": test_results["y_prob"],
        "test_subject_ids": test_subject_ids,
    }


def retrain_lodo(normalized_data, cfg, device, results_dir):
    log("\n" + "=" * 70)
    log("STEP 3/4 -- RETRAINING LODO ON NORMALIZED DATA")
    log("=" * 70)
    fold_results = []
    for held_out in normalized_data.keys():
        log(f"\n{'-'*60}\nLODO fold (scaled): held out = {held_out}\n{'-'*60}")
        fold_result = run_lodo_fold_normalized(held_out, normalized_data, cfg, device)
        fold_results.append(fold_result)
        log(f"[lodo-scaled] held_out={held_out} -> {fold_result['metrics']}")

        y_prob = fold_result["y_prob"]
        log(f"[lodo-scaled] {held_out} y_prob stats: mean={y_prob.mean():.4f} "
            f"std={y_prob.std():.4f} min={y_prob.min():.4f} max={y_prob.max():.4f} "
            f"{'<<< STILL COLLAPSED' if y_prob.std() < 1e-6 else '(varies across trials -- good)'}")

        pred_path = results_dir / f"lodo_predictions_scaled_{held_out}.npz"
        save_kwargs = {"y_true": fold_result["y_true"], "y_prob": fold_result["y_prob"]}
        if fold_result.get("test_subject_ids") is not None:
            save_kwargs["subject_ids"] = fold_result["test_subject_ids"]
        np.savez(pred_path, **save_kwargs)
        log(f"[lodo-scaled] Saved -> {pred_path}")
    return fold_results


# ---------------------------------------------------------------------
# STEP 4: RE-AUDIT -- permutation test + equivalence bound, on the new predictions
# ---------------------------------------------------------------------

def observed_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return kappa, auc


def permutation_null(y_true, y_prob, subject_ids, n_perm, seed):
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)
    null_kappas, null_aucs = np.empty(n_perm), np.empty(n_perm)
    idx_by_subject = None
    if subject_ids is not None:
        idx_by_subject = {s: np.where(subject_ids == s)[0] for s in np.unique(subject_ids)}
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


def one_sided_p(observed, null_dist):
    null_dist = null_dist[~np.isnan(null_dist)]
    if len(null_dist) == 0 or np.isnan(observed):
        return float("nan")
    return float((1 + np.sum(null_dist >= observed)) / (1 + len(null_dist)))


def bootstrap_ci_subject(y_true, y_prob, subject_ids, n_boot, seed, ci=0.90):
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)
    boot_kappas = []
    if subject_ids is not None:
        unique_s = np.unique(subject_ids)
        idx_by_s = {s: np.where(subject_ids == s)[0] for s in unique_s}
        for _ in range(n_boot):
            chosen = rng.choice(unique_s, size=len(unique_s), replace=True)
            idxs = np.concatenate([idx_by_s[s] for s in chosen])
            if len(np.unique(y_true[idxs])) < 2:
                continue
            boot_kappas.append(cohen_kappa_score(y_true[idxs], y_pred[idxs]))
    else:
        n = len(y_true)
        for _ in range(n_boot):
            idxs = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idxs])) < 2:
                continue
            boot_kappas.append(cohen_kappa_score(y_true[idxs], y_pred[idxs]))
    if not boot_kappas:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_kappas, [alpha, 1 - alpha])
    return float(lo), float(hi)


def reaudit(fold_results, results_dir, n_perm=2000, margin=0.10, seed=42):
    log("\n" + "=" * 70)
    log("STEP 4/4 -- RE-AUDITING SCALED RESULTS (permutation + equivalence)")
    log("=" * 70)
    reports = []
    for fr in fold_results:
        y_true, y_prob, subj = fr["y_true"], fr["y_prob"], fr["test_subject_ids"]
        obs_kappa, obs_auc = observed_metrics(y_true, y_prob)
        null_k, null_a = permutation_null(y_true, y_prob, subj, n_perm, seed)
        p_k = one_sided_p(obs_kappa, null_k)
        p_a = one_sided_p(obs_auc, null_a) if not np.isnan(obs_auc) else float("nan")
        ci_lo, ci_hi = bootstrap_ci_subject(y_true, y_prob, subj, 5000, seed)
        equiv = (not np.isnan(ci_lo)) and (ci_lo >= -margin) and (ci_hi <= margin)
        reports.append({
            "dataset": fr["held_out"], "n_trials": int(len(y_true)),
            "y_prob_std": float(y_prob.std()),
            "observed_kappa": float(obs_kappa), "observed_auc": float(obs_auc) if not np.isnan(obs_auc) else None,
            "p_kappa_raw": p_k, "p_auc_raw": p_a,
            "kappa_ci_90": (ci_lo, ci_hi), "equivalent_to_no_transfer": equiv,
        })
    kp = [r["p_kappa_raw"] for r in reports]
    if kp:
        adj = holm_correction(kp)
        for r, p in zip(reports, adj):
            r["p_kappa_holm"] = float(p)
    return reports


def write_final_report(scale_stats, reports, out_path: Path, margin: float):
    lines = ["# Overnight diagnose-and-retry report\n"]
    lines.append("## Raw scale diagnostic\n")
    lines.append("| Dataset | mean | std | abs_mean |\n|---|---|---|---|")
    for name, s in scale_stats.items():
        lines.append(f"| {name} | {s['mean']:.4g} | {s['std']:.4g} | {s['abs_mean']:.4g} |")

    lines.append("\n## Scaled LODO results\n")
    lines.append("| Dataset | n trials | y_prob std | kappa | p_holm | kappa 90% CI | equivalent to null? | collapsed? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    still_collapsed = []
    for r in reports:
        collapsed = r["y_prob_std"] < 1e-6
        if collapsed:
            still_collapsed.append(r["dataset"])
        ci = r["kappa_ci_90"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "n/a"
        lines.append(
            f"| {r['dataset']} | {r['n_trials']} | {r['y_prob_std']:.4f} | "
            f"{r['observed_kappa']:.4f} | {r['p_kappa_holm']:.3f} | {ci_str} | "
            f"{'YES' if r['equivalent_to_no_transfer'] else 'no'} | "
            f"{'STILL COLLAPSED' if collapsed else 'no'} |"
        )

    lines.append("\n## SUMMARY (read this first)\n")
    if still_collapsed:
        lines.append(
            f"**Still collapsed after normalization: {', '.join(still_collapsed)}.** "
            "Scale mismatch was not the (sole) cause for these — needs the "
            "epoch-by-epoch training curve inspected next (val_loss stuck flat "
            "from epoch 1 would point to a dead/saturated layer from "
            "initialization rather than input scale; check `diagnose_retry_log.txt` "
            "training curves for these folds specifically)."
        )
    else:
        lines.append(
            "**No fold shows the constant-output collapse anymore.** Per-dataset "
            "normalization appears to have fixed it. The kappa/equivalence numbers "
            "in the table above are now trustworthy as an actual measurement of "
            "cross-paradigm transfer (or lack of it) rather than an artifact of a "
            "broken model. Next step: decide whether to adopt this normalization "
            "as the standard preprocessing going forward (recommend yes — it's a "
            "defensible, unsupervised, leakage-free step) and re-run the full "
            "six-condition sweep from Section 15 of EXPERIMENT_LOG.md under it."
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log(f"\n[report] Written -> {out_path}")


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    all_raw = {}
    for name in DATASET_NAMES:
        d = load_raw(cfg, name)
        if d is not None:
            all_raw[name] = d
    if len(all_raw) < 3:
        log(f"[ERROR] Only {len(all_raw)}/3 datasets found in cache — aborting.")
        return

    scale_stats = diagnose_scale(all_raw)
    normalized = normalize_per_dataset(all_raw)
    fold_results = retrain_lodo(normalized, cfg, device, results_dir)
    reports = reaudit(fold_results, results_dir)
    write_final_report(scale_stats, reports,
                        results_dir / "lodo_scaled_permutation_equivalence_report.md",
                        margin=0.10)

    log("\n" + "=" * 70)
    log("DONE. Read results/lodo_scaled_permutation_equivalence_report.md first.")
    log("=" * 70)


if __name__ == "__main__":
    main()
