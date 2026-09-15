import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.baselines import DeepConvNet
from src.train import EEGTensorDataset, train_one_model, evaluate_model

N_CV_FOLDS = 5
MAX_EPOCHS = 15
LEARNING_RATE = 0.001


def load_dataset(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    X = d["X"]
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mu) / sd
    return X[:, None, :, :], d["y"], d["subject_ids"]


def permute_labels_within_subject(y, groups, rng):
    y_perm = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def run_subjectwise_cv_once(X, y, groups, cfg, device, n_splits=N_CV_FOLDS, max_epochs=MAX_EPOCHS, lr=LEARNING_RATE):
    n_channels, n_timepoints = X.shape[2], X.shape[3]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["training"]["seed"])
    fold_kappas = []
    for train_idx, test_idx in sgkf.split(X.reshape(len(X), -1), y, groups=groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_train = groups[train_idx]
        train_subjects = np.unique(groups_train)
        rng = np.random.default_rng(cfg["training"]["seed"])
        n_val = max(1, int(0.2 * len(train_subjects)))
        val_subjects = set(rng.choice(train_subjects, size=n_val, replace=False))
        val_mask = np.isin(groups_train, list(val_subjects))
        tr_idx = np.where(~val_mask)[0]; val_idx = np.where(val_mask)[0]

        train_loader = DataLoader(EEGTensorDataset(X_train[tr_idx], y_train[tr_idx]), batch_size=64, shuffle=True)
        val_loader = DataLoader(EEGTensorDataset(X_train[val_idx], y_train[val_idx]), batch_size=64, shuffle=False)
        test_loader = DataLoader(EEGTensorDataset(X_test, y_test), batch_size=64, shuffle=False)

        model = DeepConvNet(n_channels=n_channels, n_timepoints=n_timepoints)
        trained = train_one_model(model, train_loader, val_loader, device, epochs=max_epochs, lr=lr, early_stopping_patience=5)
        result = evaluate_model(trained["model"], test_loader, device)
        fold_kappas.append(result["metrics"]["cohen_kappa"])
    return float(np.mean(fold_kappas)), fold_kappas


def subject_bootstrap_kappa_ci(fold_kappas, _unused, n_boot=5000, seed=42, ci=0.90):
    rng = np.random.default_rng(seed)
    vals = np.array(fold_kappas)
    boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neuma", "restaurant_logo", "ds007406"])
    parser.add_argument("--n-permutations", type=int, default=500)
    args = parser.parse_args()
    name = args.dataset
    N_PERMUTATIONS = args.n_permutations

    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    X, y, groups = load_dataset(name)
    print(f"[{name}] Loaded X={X.shape}, subjects={len(np.unique(groups))}")
    print(f"[{name}] Config: {N_CV_FOLDS}-fold CV, {MAX_EPOCHS} epochs, lr={LEARNING_RATE}, "
          f"{N_PERMUTATIONS} permutations (within-subject shuffle)")

    t_start = time.time()

    print(f"\n[{name}] Computing OBSERVED kappa (real labels)...")
    observed_kappa, observed_fold_kappas = run_subjectwise_cv_once(X, y, groups, cfg, device)
    print(f"[{name}] Observed kappa: {observed_kappa:.4f} (per-fold: {[round(k,4) for k in observed_fold_kappas]})")

    ci_low_boot, ci_high_boot = subject_bootstrap_kappa_ci(observed_fold_kappas, None, seed=cfg["training"]["seed"])
    print(f"[{name}] Bootstrap 90% CI (over CV folds): [{ci_low_boot:.4f}, {ci_high_boot:.4f}]")

    print(f"\n[{name}] Running {N_PERMUTATIONS} WITHIN-SUBJECT label-shuffled permutations...")
    rng = np.random.default_rng(cfg["training"]["seed"])
    perm_kappas = []
    for i in range(N_PERMUTATIONS):
        y_shuffled = permute_labels_within_subject(y, groups, rng)
        k, _ = run_subjectwise_cv_once(X=X, y=y_shuffled, groups=groups, cfg=cfg, device=device)
        perm_kappas.append(k)
        if (i + 1) % 25 == 0 or i == 0:
            elapsed_min = (time.time() - t_start) / 60
            print(f"  [{name}] permutation {i+1}/{N_PERMUTATIONS}: kappa={k:.4f} (elapsed: {elapsed_min:.1f} min)")

    perm_kappas = np.array(perm_kappas)
    p_value = float((1 + np.sum(perm_kappas >= observed_kappa)) / (1 + N_PERMUTATIONS))
    ci_low_null, ci_high_null = np.percentile(perm_kappas, [2.5, 97.5])

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*60}\nRIGOROUS PERMUTATION TEST RESULT: {name}\n{'='*60}")
    print(f"Observed kappa: {observed_kappa:.4f}")
    print(f"Bootstrap 90% CI (over CV folds): [{ci_low_boot:.4f}, {ci_high_boot:.4f}]")
    print(f"Permutation null (within-subject): mean={perm_kappas.mean():.4f}, std={perm_kappas.std():.4f}")
    print(f"Null 95% range: [{ci_low_null:.4f}, {ci_high_null:.4f}]")
    print(f"p-value (1+count(perm>=obs))/(1+N): {p_value:.4f}")
    print(f"Significant at 0.05: {p_value < 0.05}")
    print(f"Total wall time: {total_min:.1f} minutes")

    result = {
        "dataset": name,
        "config": {"n_cv_folds": N_CV_FOLDS, "max_epochs": MAX_EPOCHS, "lr": LEARNING_RATE,
                    "n_permutations": N_PERMUTATIONS, "permutation_scheme": "within_subject"},
        "observed_kappa": observed_kappa,
        "observed_fold_kappas": observed_fold_kappas,
        "bootstrap_ci_90": [ci_low_boot, ci_high_boot],
        "null_mean": float(perm_kappas.mean()),
        "null_std": float(perm_kappas.std()),
        "null_95ci": [float(ci_low_null), float(ci_high_null)],
        "p_value": p_value,
        "p_value_formula": "(1 + count(perm_kappa >= observed_kappa)) / (1 + N_PERMUTATIONS)",
        "significant_at_0.05": p_value < 0.05,
        "wall_time_minutes": total_min,
    }
    out_path = results_dir / f"{name}_deepconvnet_permutation_test_RIGOROUS.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
