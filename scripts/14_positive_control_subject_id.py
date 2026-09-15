"""
Positive control: can this exact pipeline (DeepConvNet, same z-scored
windows_common_v2 preprocessing) decode something well-established and
robustly decodable from EEG -- subject identity -- on the same data
where task-label (Buy/NoBuy etc.) decoding came back null?

If subject-ID decoding succeeds well above chance, that rules out
"the pipeline/architecture just doesn't work on this data" as an
explanation for the null task-transfer results elsewhere in this project.

Uses a stratified (not subject-wise) trial-level split, since decoding
subject identity is inherently a within-subject-identity task -- every
subject must appear in both train and test by construction.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.baselines import DeepConvNet
from src.train import EEGTensorDataset, train_one_model

N_CV_FOLDS = 5
MAX_EPOCHS = 20


def load_dataset_for_subject_id(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    X = d["X"]
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mu) / sd
    X = X[:, None, :, :]

    subject_ids_raw = d["subject_ids"]
    unique_subjects = np.unique(subject_ids_raw)
    subj_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    y_subject = np.array([subj_to_idx[s] for s in subject_ids_raw])
    return X, y_subject, len(unique_subjects)


class DeepConvNetMultiClass(nn.Module):
    """Same conv backbone as DeepConvNet, swapped to an n_classes-way head."""
    def __init__(self, n_channels, n_timepoints, n_classes, dropout=0.5):
        super().__init__()
        base = DeepConvNet(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2, dropout=dropout)
        self.block1 = base.block1
        self.block2 = base.block2
        self.block3 = base.block3
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        feats = self._forward_features(x)
        return self.classifier(feats), feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["neuma", "restaurant_logo", "ds007406"])
    args = parser.parse_args()
    name = args.dataset

    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    X, y_subject, n_subjects = load_dataset_for_subject_id(name)
    n_channels, n_timepoints = X.shape[2], X.shape[3]
    print(f"[{name}] Positive control: subject-ID decoding, {n_subjects}-way classification")
    print(f"[{name}] X={X.shape}, y_subject unique={n_subjects}")

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=cfg["training"]["seed"])
    fold_accs, fold_top5, fold_kappas = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X.reshape(len(X), -1), y_subject)):
        X_train, y_train = X[train_idx], y_subject[train_idx]
        X_test, y_test = X[test_idx], y_subject[test_idx]

        n_val = max(1, int(0.15 * len(y_train)))
        rng = np.random.default_rng(cfg["training"]["seed"])
        perm = rng.permutation(len(y_train))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        train_loader = DataLoader(EEGTensorDataset(X_train[tr_idx], y_train[tr_idx]), batch_size=64, shuffle=True)
        val_loader = DataLoader(EEGTensorDataset(X_train[val_idx], y_train[val_idx]), batch_size=64, shuffle=False)
        test_loader = DataLoader(EEGTensorDataset(X_test, y_test), batch_size=64, shuffle=False)

        model = DeepConvNetMultiClass(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_subjects)
        trained = train_one_model(model, train_loader, val_loader, device, epochs=MAX_EPOCHS, lr=0.001, early_stopping_patience=5)

        trained["model"].eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = trained["model"](xb)
                logits = out[0] if isinstance(out, tuple) else out
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        preds = all_logits.argmax(axis=1)

        acc = accuracy_score(all_labels, preds)
        try:
            top5 = top_k_accuracy_score(all_labels, all_logits, k=min(5, n_subjects), labels=np.arange(n_subjects))
        except ValueError:
            top5 = None
        kappa = cohen_kappa_score(all_labels, preds)

        fold_accs.append(acc); fold_kappas.append(kappa)
        if top5 is not None: fold_top5.append(top5)
        chance = 1.0 / n_subjects
        print(f"[{name}] fold {fold_i+1}/{N_CV_FOLDS}: acc={acc:.4f} (chance={chance:.4f}), "
              f"top5={top5}, kappa={kappa:.4f}")

    result = {
        "dataset": name,
        "task": "subject_identification_positive_control",
        "n_subjects": n_subjects,
        "chance_accuracy": 1.0 / n_subjects,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_top5_accuracy": float(np.mean(fold_top5)) if fold_top5 else None,
        "mean_kappa": float(np.mean(fold_kappas)),
        "fold_accuracies": fold_accs,
        "fold_kappas": fold_kappas,
    }
    out_path = results_dir / f"{name}_positive_control_subject_id.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}\nPOSITIVE CONTROL RESULT: {name}\n{'='*60}")
    print(f"Mean accuracy: {result['mean_accuracy']:.4f} (chance: {result['chance_accuracy']:.4f})")
    print(f"Mean kappa: {result['mean_kappa']:.4f}")
    print(f"Ratio to chance: {result['mean_accuracy'] / result['chance_accuracy']:.1f}x")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
