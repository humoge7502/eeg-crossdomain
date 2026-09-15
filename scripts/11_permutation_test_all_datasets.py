import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.baselines import DeepConvNet
from src.train import EEGTensorDataset, train_one_model, evaluate_model

N_PERMUTATIONS = 25
N_CV_FOLDS = 3
MAX_EPOCHS = 10

def load_dataset(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    X = d["X"]
    # Per-trial, per-channel z-scoring: raw values are ~1e-6 scale, which
    # caused DeepConvNet+BatchNorm to collapse to constant-class prediction.
    # Verified on neuma: standardizing each trial's time axis to unit
    # variance stabilizes training. Applying the same fix here for
    # restaurant_logo and ds007406, which sit at the same raw scale.
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mu) / sd
    return X[:, None, :, :], d["y"], d["subject_ids"]

def run_subjectwise_cv_once(X, y, groups, cfg, device, n_splits=N_CV_FOLDS, max_epochs=MAX_EPOCHS):
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
        trained = train_one_model(model, train_loader, val_loader, device, epochs=max_epochs, lr=0.002, early_stopping_patience=5)
        result = evaluate_model(trained["model"], test_loader, device)
        fold_kappas.append(result["metrics"]["cohen_kappa"])
    return float(np.mean(fold_kappas)), fold_kappas

def run_for_dataset(name, cfg, device):
    X, y, groups = load_dataset(name)
    print(f"\n{'='*60}\nDATASET: {name}\n{'='*60}")
    print(f"Loaded {name}: X={X.shape}, subjects={len(np.unique(groups))}")
    print(f"Config: {N_CV_FOLDS}-fold CV, {MAX_EPOCHS} max epochs, {N_PERMUTATIONS} permutations")

    print("\nComputing OBSERVED kappa (subject-wise CV, real labels)...")
    observed_kappa, observed_fold_kappas = run_subjectwise_cv_once(X, y, groups, cfg, device)
    print(f"Observed kappa: {observed_kappa:.4f} (per-fold: {[round(k,4) for k in observed_fold_kappas]})")

    rng_ci = np.random.default_rng(cfg["training"]["seed"])
    boot_means = []
    for _ in range(2000):
        sample = rng_ci.choice(observed_fold_kappas, size=len(observed_fold_kappas), replace=True)
        boot_means.append(np.mean(sample))
    ci_low_boot, ci_high_boot = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI (over CV folds, subject-disjoint): [{ci_low_boot:.4f}, {ci_high_boot:.4f}]")

    print(f"\nRunning {N_PERMUTATIONS} label-shuffled permutations (subject-wise CV each time)...")
    rng = np.random.default_rng(cfg["training"]["seed"])
    perm_kappas = []
    for i in range(N_PERMUTATIONS):
        y_shuffled = rng.permutation(y)
        k, _ = run_subjectwise_cv_once(X, y_shuffled, groups, cfg, device)
        perm_kappas.append(k)
        print(f"  [{name}] permutation {i+1}/{N_PERMUTATIONS}: kappa={k:.4f}")

    perm_kappas = np.array(perm_kappas)
    p_value = float(np.mean(perm_kappas >= observed_kappa))
    ci_low_null, ci_high_null = np.percentile(perm_kappas, [2.5, 97.5])

    print(f"\n{'='*60}\nPERMUTATION TEST RESULT: {name}\n{'='*60}")
    print(f"Observed kappa: {observed_kappa:.4f}")
    print(f"Bootstrap CI on observed: [{ci_low_boot:.4f}, {ci_high_boot:.4f}]")
    print(f"Permutation null: mean={perm_kappas.mean():.4f}, std={perm_kappas.std():.4f}")
    print(f"Null 95% range: [{ci_low_null:.4f}, {ci_high_null:.4f}]")
    print(f"p-value (observed >= null): {p_value:.4f}")
    print(f"Significant at 0.05: {p_value < 0.05}")

    result = {
        "dataset": name,
        "observed_kappa": observed_kappa,
        "observed_fold_kappas": observed_fold_kappas,
        "bootstrap_ci_95": [float(ci_low_boot), float(ci_high_boot)],
        "n_permutations": N_PERMUTATIONS,
        "n_cv_folds": N_CV_FOLDS,
        "max_epochs": MAX_EPOCHS,
        "null_mean": float(perm_kappas.mean()),
        "null_std": float(perm_kappas.std()),
        "null_95ci": [float(ci_low_null), float(ci_high_null)],
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }
    return result

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    all_results = {}
    for name in ["restaurant_logo", "ds007406"]:
        result = run_for_dataset(name, cfg, device)
        all_results[name] = result
        with open(results_dir / f"{name}_deepconvnet_permutation_test.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> results/{name}_deepconvnet_permutation_test.json")

    with open(results_dir / "all_datasets_permutation_test_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved combined summary -> results/all_datasets_permutation_test_summary.json")

if __name__ == "__main__":
    main()
