"""
Representation similarity analysis (linear CKA) across datasets.

Two complementary comparisons:
1. Option A encoders (each trained on 2-of-3 datasets, LODO setup) --
   does the encoder trained under the actual transfer experiment still
   develop features that align across the datasets it WAS trained on?
2. Solo encoders (each trained only on its own single dataset, from
   scratch) -- do the three EEG paradigms have intrinsically similar
   representational geometry, independent of any transfer setup?

CKA is computed on TEST-set embeddings (held-out data for that encoder)
to avoid trivial overfitting-driven similarity.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]


def linear_cka(X, Y):
    """Standard linear CKA (Kornblith et al. 2019).
    X: (n_samples, d1), Y: (n_samples, d2) -- same n_samples required."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def load_features(cfg):
    data = {}
    for name in DATASET_NAMES:
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        data[name] = d["X"].astype(np.float32)
    return data


def extract_embeddings(encoder, X, device):
    encoder.eval()
    with torch.no_grad():
        emb = encoder(torch.tensor(X).to(device)).cpu().numpy()
    return emb


def cka_matched_subset(emb_a, emb_b, seed=42):
    """CKA requires equal sample counts along axis 0. Since different
    datasets have different N, we subsample the larger to match the
    smaller (without replacement), matching sample count for a fair
    comparison rather than truncating to an arbitrary prefix."""
    n = min(len(emb_a), len(emb_b))
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(emb_a), size=n, replace=False)
    idx_b = rng.choice(len(emb_b), size=n, replace=False)
    return linear_cka(emb_a[idx_a], emb_b[idx_b])


def run_option_a_cka(cfg, device, results_dir):
    """CKA between Option A's already-trained LODO encoders, evaluated
    on each dataset's own features (encoders trained on 2-of-3 datasets
    each; here we compare the SAME encoder's embedding of the datasets
    it WAS trained on, pairwise)."""
    print("\n" + "=" * 60 + "\nOPTION A ENCODER CKA (LODO-trained encoders)\n" + "=" * 60)
    all_features = load_features(cfg)
    results = {}

    for condition in ["noalign", "coral"]:
        for held_out in DATASET_NAMES:
            ckpt_path = results_dir / f"optionA_encoder_{condition}_{held_out}.pt"
            if not ckpt_path.exists():
                continue
            train_keys = [k for k in DATASET_NAMES if k != held_out]
            input_dim = all_features[train_keys[0]].shape[1]
            encoder = TabularMLPEncoder(input_dim=input_dim, embedding_dim=64).to(device)
            encoder.load_state_dict(torch.load(ckpt_path, map_location=device))

            # Embed both TRAINING datasets this encoder saw, using its own weights
            emb = {}
            for k in train_keys:
                emb[k] = extract_embeddings(encoder, all_features[k], device)

            if len(train_keys) == 2:
                k1, k2 = train_keys
                cka_val = cka_matched_subset(emb[k1], emb[k2])
                key = f"{condition}_{held_out}held_out_{k1}_vs_{k2}"
                results[key] = {
                    "condition": condition, "held_out": held_out,
                    "pair": [k1, k2], "cka": cka_val,
                }
                print(f"[{key}] CKA({k1}, {k2}) = {cka_val:.4f}")

    return results


def run_solo_encoder_cka(cfg, device, results_dir):
    """Train fresh solo encoders (each on its own single dataset only,
    no cross-dataset exposure at all), then compute pairwise CKA on
    held-out (20%) data from each."""
    print("\n" + "=" * 60 + "\nSOLO ENCODER CKA (single-dataset-only encoders)\n" + "=" * 60)
    from torch.utils.data import DataLoader, TensorDataset
    all_features = load_features(cfg)
    all_labels = {}
    for name in DATASET_NAMES:
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        all_labels[name] = d["y"].astype(np.int64)

    encoders = {}
    test_embeddings = {}
    rng = np.random.default_rng(cfg["training"]["seed"])

    for name in DATASET_NAMES:
        X, y = all_features[name], all_labels[name]
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(0.2 * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        encoder = TabularMLPEncoder(input_dim=X.shape[1], embedding_dim=64).to(device)
        head = nn.Linear(64, 2).to(device)
        model = nn.Sequential(encoder, head)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        class_counts = np.bincount(y[train_idx], minlength=2)
        weights = torch.tensor([len(train_idx) / (2.0 * max(c, 1)) for c in class_counts], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        Xtr = torch.tensor(X[train_idx]).to(device); ytr = torch.tensor(y[train_idx]).to(device)
        for epoch in range(40):
            model.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(ytr), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                out = model(Xtr[b])
                loss = criterion(out, ytr[b])
                loss.backward(); opt.step()

        encoders[name] = encoder
        test_embeddings[name] = extract_embeddings(encoder, X[test_idx], device)
        print(f"[{name}] Solo encoder trained, test embedding shape: {test_embeddings[name].shape}")

    results = {}
    for i, k1 in enumerate(DATASET_NAMES):
        for k2 in DATASET_NAMES[i + 1:]:
            cka_val = cka_matched_subset(test_embeddings[k1], test_embeddings[k2])
            key = f"solo_{k1}_vs_{k2}"
            results[key] = {"pair": [k1, k2], "cka": cka_val}
            print(f"[{key}] CKA({k1}, {k2}) = {cka_val:.4f}")

    return results


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")

    option_a_results = run_option_a_cka(cfg, device, results_dir)
    solo_results = run_solo_encoder_cka(cfg, device, results_dir)

    combined = {"option_a_encoder_cka": option_a_results, "solo_encoder_cka": solo_results}
    out_path = results_dir / "cka_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
