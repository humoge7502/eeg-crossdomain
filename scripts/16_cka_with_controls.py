import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
N_BOOTSTRAP = 200


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def cka_bootstrap(emb_a, emb_b, n_boot=N_BOOTSTRAP, seed=42):
    n = min(len(emb_a), len(emb_b))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx_a = rng.choice(len(emb_a), size=n, replace=(len(emb_a) < n * 2))
        idx_b = rng.choice(len(emb_b), size=n, replace=(len(emb_b) < n * 2))
        vals.append(linear_cka(emb_a[idx_a], emb_b[idx_b]))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_features_and_labels(cfg):
    data, labels = {}, {}
    for name in DATASET_NAMES:
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        data[name] = d["X"].astype(np.float32)
        labels[name] = d["y"].astype(np.int64)
    return data, labels


def extract_embeddings(encoder, X, device):
    encoder.eval()
    with torch.no_grad():
        return encoder(torch.tensor(X).to(device)).cpu().numpy()


def train_solo_encoder(X, y, input_dim, device, seed, epochs=40):
    set_seed(seed)
    encoder = TabularMLPEncoder(input_dim=input_dim, embedding_dim=64).to(device)
    head = nn.Linear(64, 2).to(device)
    model = nn.Sequential(encoder, head)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    class_counts = np.bincount(y, minlength=2)
    weights = torch.tensor([len(y) / (2.0 * max(c, 1)) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    Xt = torch.tensor(X).to(device); yt = torch.tensor(y).to(device)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = criterion(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
    return encoder


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")
    all_features, all_labels = load_features_and_labels(cfg)

    results = {}

    print("=" * 60 + "\nNULL BASELINE: untrained random encoders\n" + "=" * 60)
    for name in DATASET_NAMES:
        X = all_features[name]
        input_dim = X.shape[1]
        set_seed(1)
        enc_a = TabularMLPEncoder(input_dim=input_dim, embedding_dim=64).to(device)
        set_seed(2)
        enc_b = TabularMLPEncoder(input_dim=input_dim, embedding_dim=64).to(device)
        emb_a = extract_embeddings(enc_a, X, device)
        emb_b = extract_embeddings(enc_b, X, device)
        mean, std, lo, hi = cka_bootstrap(emb_a, emb_b)
        results[f"null_untrained_{name}"] = {"mean": mean, "std": std, "ci95": [lo, hi], "n": len(X)}
        print(f"[null_untrained_{name}] CKA = {mean:.4f} +/- {std:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}] (n={len(X)})")

    print("\n" + "=" * 60 + "\nPOSITIVE CONTROL: same-dataset, different-seed encoders\n" + "=" * 60)
    rng = np.random.default_rng(cfg["training"]["seed"])
    for name in DATASET_NAMES:
        X, y = all_features[name], all_labels[name]
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(0.2 * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        enc_seed10 = train_solo_encoder(X[train_idx], y[train_idx], X.shape[1], device, seed=10)
        enc_seed20 = train_solo_encoder(X[train_idx], y[train_idx], X.shape[1], device, seed=20)
        emb_a = extract_embeddings(enc_seed10, X[test_idx], device)
        emb_b = extract_embeddings(enc_seed20, X[test_idx], device)
        mean, std, lo, hi = cka_bootstrap(emb_a, emb_b)
        results[f"positive_control_{name}"] = {"mean": mean, "std": std, "ci95": [lo, hi], "n_test": len(test_idx)}
        print(f"[positive_control_{name}] CKA = {mean:.4f} +/- {std:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}] (n_test={len(test_idx)})")

    print("\n" + "=" * 60 + "\nCROSS-DATASET CKA (solo encoders, bootstrap-stabilized)\n" + "=" * 60)
    solo_encoders, solo_test_X = {}, {}
    for name in DATASET_NAMES:
        X, y = all_features[name], all_labels[name]
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(0.2 * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        solo_encoders[name] = train_solo_encoder(X[train_idx], y[train_idx], X.shape[1], device, seed=cfg["training"]["seed"])
        solo_test_X[name] = X[test_idx]
        print(f"[{name}] solo encoder trained, n_test={len(test_idx)}")

    for i, k1 in enumerate(DATASET_NAMES):
        for k2 in DATASET_NAMES[i + 1:]:
            emb_a = extract_embeddings(solo_encoders[k1], solo_test_X[k1], device)
            emb_b = extract_embeddings(solo_encoders[k2], solo_test_X[k2], device)
            mean, std, lo, hi = cka_bootstrap(emb_a, emb_b)
            key = f"cross_dataset_{k1}_vs_{k2}"
            results[key] = {"mean": mean, "std": std, "ci95": [lo, hi],
                              "n_test_a": len(emb_a), "n_test_b": len(emb_b)}
            print(f"[{key}] CKA = {mean:.4f} +/- {std:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}] "
                  f"(n_a={len(emb_a)}, n_b={len(emb_b)})")

    out_path = results_dir / "cka_with_controls_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
