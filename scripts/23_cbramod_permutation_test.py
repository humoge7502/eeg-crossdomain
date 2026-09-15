import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
ORIG_SFREQ = 250
TARGET_SFREQ = 200
N_PERMUTATIONS = 1000


def load_and_resample(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype(np.float32)
    n_times_new = int(X.shape[-1] * TARGET_SFREQ / ORIG_SFREQ)
    X_resampled = resample(X, n_times_new, axis=-1)
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
            f = out["features"].reshape(out["features"].shape[0], -1).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0)


def permute_labels_within_subject(y, groups, rng):
    y_perm = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")

    from braindecode.models import CBraMod
    print("Loading CBraMod and extracting features (once)...")
    model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_chans=7,
                                      sfreq=TARGET_SFREQ, n_times=200, return_encoder_output=True).to(device)
    for p in model.parameters():
        p.requires_grad = False

    all_data, all_features = {}, {}
    for name in DATASET_NAMES:
        X, y, groups = load_and_resample(name)
        all_data[name] = (X, y, groups)
        all_features[name] = extract_features(model, X, device)
        print(f"[{name}] features: {all_features[name].shape}")

    print("\n" + "=" * 60 + "\nPERMUTATION TESTS: ZERO-SHOT LODO TRANSFER\n" + "=" * 60)
    results = {}
    for held_out in DATASET_NAMES:
        train_names = [k for k in DATASET_NAMES if k != held_out]
        X_train = np.concatenate([all_features[k] for k in train_names], axis=0)
        y_train = np.concatenate([all_data[k][1] for k in train_names], axis=0)
        X_test, y_test, test_groups = all_features[held_out], all_data[held_out][1], all_data[held_out][2]

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        observed_kappa = cohen_kappa_score(y_test, clf.predict(X_test))
        print(f"[{held_out}] Observed kappa: {observed_kappa:.4f}")

        rng = np.random.default_rng(cfg["training"]["seed"])
        perm_kappas = []
        for i in range(N_PERMUTATIONS):
            y_test_shuffled = permute_labels_within_subject(y_test, test_groups, rng)
            clf_p = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf_p.fit(X_train, y_train)
            perm_kappas.append(cohen_kappa_score(y_test_shuffled, clf_p.predict(X_test)))
            if (i + 1) % 200 == 0:
                print(f"  [{held_out}] permutation {i+1}/{N_PERMUTATIONS}")

        perm_kappas = np.array(perm_kappas)
        p_value = float((1 + np.sum(perm_kappas >= observed_kappa)) / (1 + N_PERMUTATIONS))
        results[held_out] = {
            "observed_kappa": float(observed_kappa),
            "null_mean": float(perm_kappas.mean()),
            "null_std": float(perm_kappas.std()),
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        }
        print(f"[{held_out}] p={p_value:.4f}, null mean/std={perm_kappas.mean():.4f}/{perm_kappas.std():.4f}")

    with open(results_dir / "cbramod_permutation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> results/cbramod_permutation_results.json")


if __name__ == "__main__":
    main()
