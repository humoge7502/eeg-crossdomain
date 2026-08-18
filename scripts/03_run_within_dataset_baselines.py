import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.baselines import svm_psd_pai_baseline, mdm_riemannian_baseline, DeepConvNet
from src.models import EEGNetSharedEncoder, CrossDomainDecoder
from src.train import EEGTensorDataset, train_one_model, evaluate_model
from src.evaluate import compute_metrics


def load_harmonized(name, cfg):
    path = project_root() / cfg["paths"]["processed_dir"] / name / f"{name}_harmonized.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    subject_ids = d["subject_ids"] if "subject_ids" in d else None
    return d["X"], d["y"], subject_ids


def load_epochs_common(name, cfg):
    path = project_root() / cfg["paths"]["processed_dir"] / name / f"{name}_epochs_common.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    subject_ids = d["subject_ids"] if "subject_ids" in d else None
    return d["X"], d["y"], subject_ids


def run_deep_model_cv(model_name, model_fn, X, y, groups, cfg, device, n_splits=5, max_epochs=30):
    n_channels, n_timepoints = X.shape[2], X.shape[3]
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return {"model": model_name, "error": "too few subjects for group CV"}

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["training"]["seed"])
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X.reshape(len(X), -1), y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Inner validation split, ALSO subject-wise (not just random), for early stopping
        train_subjects = np.unique(groups_train)
        rng = np.random.default_rng(cfg["training"]["seed"] + fold_idx)
        n_val_subjects = max(1, int(0.2 * len(train_subjects)))
        val_subjects = set(rng.choice(train_subjects, size=n_val_subjects, replace=False))
        val_mask = np.isin(groups_train, list(val_subjects))
        tr_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]

        train_loader = DataLoader(EEGTensorDataset(X_train[tr_idx], y_train[tr_idx]),
                                    batch_size=min(cfg["training"]["batch_size"], max(len(tr_idx),1)), shuffle=True)
        val_loader = DataLoader(EEGTensorDataset(X_train[val_idx], y_train[val_idx]),
                                  batch_size=min(cfg["training"]["batch_size"], max(len(val_idx),1)), shuffle=False)
        test_loader = DataLoader(EEGTensorDataset(X_test, y_test),
                                   batch_size=min(cfg["training"]["batch_size"], len(y_test)), shuffle=False)

        model = model_fn(n_channels, n_timepoints)
        trained = train_one_model(model, train_loader, val_loader, device,
                                    epochs=max_epochs, lr=cfg["training"]["learning_rate"],
                                    early_stopping_patience=8)
        result = evaluate_model(trained["model"], test_loader, device)
        fold_metrics.append(result["metrics"])
        print(f"    fold {fold_idx+1}/{n_splits} (test subjects: {sorted(set(groups[test_idx]))}): {result['metrics']}")

    mean_metrics = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    std_metrics = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    return {"model": model_name, "mean": mean_metrics, "std": std_metrics, "n_splits": n_splits,
            "split_type": "subject-wise (StratifiedGroupKFold)"}


def make_eegnet(n_channels, n_timepoints):
    encoder = EEGNetSharedEncoder(n_channels=n_channels, n_timepoints=n_timepoints, embedding_dim=64)
    return CrossDomainDecoder(encoder, embedding_dim=64, n_classes=2)


def make_deepconvnet(n_channels, n_timepoints):
    return DeepConvNet(n_channels=n_channels, n_timepoints=n_timepoints)


def run_all_baselines_for_dataset(name, cfg, device):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    results = {}

    harmonized = load_harmonized(name, cfg)
    if harmonized is not None:
        X_h, y_h, groups_h = harmonized
        if groups_h is None:
            print(f"  WARNING: no subject_ids in harmonized features — skipping group-wise SVM")
        else:
            print(f"  Harmonized features: X={X_h.shape}, subjects={len(np.unique(groups_h))}, class balance: {np.bincount(y_h)}")
            svm_result = svm_psd_pai_baseline(X_h, y_h, groups_h, cv_folds=5)
            print(f"  SVM+PSD/PAI (subject-wise): {svm_result}")
            results["svm_psd_pai"] = svm_result

            y_shuffled = np.random.default_rng(cfg["training"]["seed"]).permutation(y_h)
            shuffled_result = svm_psd_pai_baseline(X_h, y_shuffled, groups_h, cv_folds=5)
            print(f"  SVM (shuffled labels control, subject-wise): {shuffled_result}")
            results["svm_shuffled_control"] = shuffled_result

    epochs_data = load_epochs_common(name, cfg)
    if epochs_data is not None:
        X_e, y_e, groups_e = epochs_data
        if groups_e is None:
            print(f"  WARNING: no subject_ids in epochs_common — SKIPPING MDM/EEGNet/DeepConvNet "
                  f"(re-run scripts/02_preprocess_all.py + 02b to regenerate with subject_ids)")
        else:
            print(f"  Raw epochs (common): X={X_e.shape}, subjects={len(np.unique(groups_e))}, class balance: {np.bincount(y_e)}")

            try:
                mdm_result = mdm_riemannian_baseline(X_e.squeeze(1), y_e, groups_e, cv_folds=5)
                print(f"  MDM-Riemannian (subject-wise): {mdm_result}")
                results["mdm_riemannian"] = mdm_result
            except Exception as e:
                print(f"  MDM-Riemannian FAILED: {e}")
                results["mdm_riemannian"] = {"error": str(e)}

            print(f"  Training EEGNet (subject-wise CV)...")
            eegnet_result = run_deep_model_cv("EEGNet", make_eegnet, X_e, y_e, groups_e, cfg, device)
            print(f"  EEGNet CV result: {eegnet_result}")
            results["eegnet"] = eegnet_result

            print(f"  Training DeepConvNet (subject-wise CV)...")
            deepconv_result = run_deep_model_cv("DeepConvNet", make_deepconvnet, X_e, y_e, groups_e, cfg, device)
            print(f"  DeepConvNet CV result: {deepconv_result}")
            results["deepconvnet"] = deepconv_result

    return results


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    all_results = {}
    for name in ["neuma", "restaurant_logo", "ds007406"]:
        all_results[name] = run_all_baselines_for_dataset(name, cfg, device)

    out_path = results_dir / "within_dataset_baselines_subjectwise.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}\nSaved -> {out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
