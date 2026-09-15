import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
ORIG_SFREQ = 250
TARGET_SFREQ = 200


def load_and_resample(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype(np.float32)
    n_times_new = int(X.shape[-1] * TARGET_SFREQ / ORIG_SFREQ)
    X_resampled = resample(X, n_times_new, axis=-1)
    # CRITICAL FIX: raw windows_common_v2 data is at ~1e-6 scale, which
    # collapses CBraMod's frozen embeddings to near-identical output
    # regardless of input (verified: trial-to-trial correlation 0.9999
    # without this fix, 0.40 with it). Per-trial per-channel z-scoring,
    # same recipe verified for DeepConvNet earlier in this project.
    mu = X_resampled.mean(axis=-1, keepdims=True)
    sd = X_resampled.std(axis=-1, keepdims=True) + 1e-8
    X_resampled = (X_resampled - mu) / sd
    return X_resampled, d["y"].astype(np.int64), d["subject_ids"]


def extract_features(model, X, device, batch_size=64):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).to(device)
            out = model(xb, return_features=True)
            f = out["features"] if isinstance(out, dict) else out
            f = f.reshape(f.shape[0], -1).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0)


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")

    from braindecode.models import CBraMod
    print("Loading pretrained CBraMod (braindecode/cbramod-pretrained)...")
    model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_chans=7,
                                      sfreq=TARGET_SFREQ, n_times=200, return_encoder_output=True).to(device)
    for p in model.parameters():
        p.requires_grad = False
    print("Loaded. All parameters frozen.")

    print("\nLoading and resampling data (250Hz -> 200Hz)...")
    all_data = {}
    for name in DATASET_NAMES:
        X, y, groups = load_and_resample(name)
        all_data[name] = (X, y, groups)
        print(f"[{name}] resampled X shape: {X.shape}")

    print("\nExtracting frozen CBraMod features for all datasets...")
    all_features = {}
    for name in DATASET_NAMES:
        X, y, groups = all_data[name]
        feats = extract_features(model, X, device)
        all_features[name] = feats
        print(f"[{name}] feature shape: {feats.shape}")

    print("\n" + "=" * 60 + "\nWITHIN-DATASET LINEAR PROBE (subject-wise CV)\n" + "=" * 60)
    within_results = {}
    for name in DATASET_NAMES:
        feats, y, groups = all_features[name], all_data[name][1], all_data[name][2]
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            print(f"[{name}] too few subjects, skipping")
            continue
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["training"]["seed"])
        fold_kappas = []
        for train_idx, test_idx in sgkf.split(feats, y, groups=groups):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(feats[train_idx], y[train_idx])
            preds = clf.predict(feats[test_idx])
            fold_kappas.append(cohen_kappa_score(y[test_idx], preds))
        within_results[name] = {"fold_kappas": fold_kappas, "mean_kappa": float(np.mean(fold_kappas))}
        print(f"[{name}] within-dataset kappa: {np.mean(fold_kappas):.4f} (per-fold: {[round(k,4) for k in fold_kappas]})")

    print("\n" + "=" * 60 + "\nZERO-SHOT LODO TRANSFER (CBraMod frozen features)\n" + "=" * 60)
    lodo_results = {}
    for held_out in DATASET_NAMES:
        train_names = [k for k in DATASET_NAMES if k != held_out]
        X_train = np.concatenate([all_features[k] for k in train_names], axis=0)
        y_train = np.concatenate([all_data[k][1] for k in train_names], axis=0)
        X_test, y_test = all_features[held_out], all_data[held_out][1]

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        kappa = cohen_kappa_score(y_test, preds)
        lodo_results[held_out] = float(kappa)
        print(f"[LODO] held_out={held_out}: kappa={kappa:.4f}")

    results = {
        "model": "CBraMod (braindecode/cbramod-pretrained), frozen, linear probe",
        "resampled_to_hz": TARGET_SFREQ,
        "within_dataset": within_results,
        "zero_shot_lodo": lodo_results,
    }
    with open(results_dir / "cbramod_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> results/cbramod_results.json")


if __name__ == "__main__":
    main()
