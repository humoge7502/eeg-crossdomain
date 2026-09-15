"""
Principal angle / subspace alignment analysis -- the correct tool for
comparing learned representations from UNPAIRED trial sets across
different datasets (unlike CKA, which assumes sample correspondence).

For each pair of datasets, computes the k-dimensional principal
subspaces of their solo-encoder test embeddings (via SVD) and measures
the principal angles between subspaces. Small angles => well-aligned
subspaces (shared representational geometry). Angles near 90 degrees
=> orthogonal/unrelated subspaces.

ds007406 is included but explicitly flagged given its tiny n_test=12 --
interpret qualitatively only, not as a headline number.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
K_COMPONENTS = 10  # top-k subspace dimensions to compare (of 64 total embedding dims)


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


def principal_angles(emb_a, emb_b, k):
    """Returns k principal angles (radians) between the top-k left
    singular subspaces of emb_a and emb_b. Uses SVD-based subspace
    extraction then the standard principal-angle formula via SVD of
    Qa^T @ Qb (Qa, Qb orthonormal bases for each subspace)."""
    k_eff = min(k, emb_a.shape[1], emb_a.shape[0] - 1, emb_b.shape[0] - 1)
    Ua, _, _ = np.linalg.svd(emb_a - emb_a.mean(axis=0), full_matrices=False)
    Ub, _, _ = np.linalg.svd(emb_b - emb_b.mean(axis=0), full_matrices=False)
    Qa = Ua[:, :k_eff] if Ua.shape[1] >= k_eff else Ua
    Qb = Ub[:, :k_eff] if Ub.shape[1] >= k_eff else Ub
    # This SVD gives left singular vectors in sample space; for subspace
    # comparison we actually want the embedding-space (column) subspace.
    # Redo properly: subspace of the 64-dim embedding space spanned by
    # the top-k directions of variation (right singular vectors of X).
    _, _, Va = np.linalg.svd(emb_a - emb_a.mean(axis=0), full_matrices=False)
    _, _, Vb = np.linalg.svd(emb_b - emb_b.mean(axis=0), full_matrices=False)
    Qa = Va[:k_eff, :].T  # (embedding_dim, k_eff), orthonormal columns
    Qb = Vb[:k_eff, :].T
    M = Qa.T @ Qb
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles_rad = np.arccos(singular_values)
    return angles_rad, k_eff


def summarize_angles(angles_rad):
    angles_deg = np.degrees(angles_rad)
    return {
        "mean_angle_deg": float(angles_deg.mean()),
        "min_angle_deg": float(angles_deg.min()),
        "max_angle_deg": float(angles_deg.max()),
        "angles_deg": [float(a) for a in angles_deg],
    }


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")
    all_features, all_labels = load_features_and_labels(cfg)

    rng = np.random.default_rng(cfg["training"]["seed"])
    solo_encoders, solo_test_X, test_sizes = {}, {}, {}

    print("Training solo encoders (one per dataset)...")
    for name in DATASET_NAMES:
        X, y = all_features[name], all_labels[name]
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(0.2 * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        solo_encoders[name] = train_solo_encoder(X[train_idx], y[train_idx], X.shape[1], device, seed=cfg["training"]["seed"])
        solo_test_X[name] = X[test_idx]
        test_sizes[name] = len(test_idx)
        print(f"[{name}] n_test={len(test_idx)}")

    # Random-subspace null baseline: angles between two RANDOM k-dim
    # subspaces of a 64-dim space (analytically, random subspaces in
    # high dim tend to be near-orthogonal; this gives us the chance floor)
    print("\nComputing random-subspace null baseline...")
    null_angles_all = []
    for _ in range(50):
        A = rng.standard_normal((64, K_COMPONENTS))
        B = rng.standard_normal((64, K_COMPONENTS))
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        M = Qa.T @ Qb
        sv = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
        null_angles_all.append(np.degrees(np.arccos(sv)).mean())
    null_mean_angle = float(np.mean(null_angles_all))
    null_std_angle = float(np.std(null_angles_all))
    print(f"Random-subspace null: mean angle = {null_mean_angle:.2f} +/- {null_std_angle:.2f} degrees "
          f"(90 deg = fully orthogonal/unrelated)")

    results = {"k_components": K_COMPONENTS,
               "random_subspace_null_deg": {"mean": null_mean_angle, "std": null_std_angle},
               "pairs": {}}

    print(f"\nPairwise principal angles (top-{K_COMPONENTS} subspaces):")
    for i, k1 in enumerate(DATASET_NAMES):
        for k2 in DATASET_NAMES[i + 1:]:
            emb_a = extract_embeddings(solo_encoders[k1], solo_test_X[k1], device)
            emb_b = extract_embeddings(solo_encoders[k2], solo_test_X[k2], device)
            angles_rad, k_eff = principal_angles(emb_a, emb_b, K_COMPONENTS)
            summary = summarize_angles(angles_rad)
            summary["k_effective"] = k_eff
            summary["n_test_a"] = test_sizes[k1]
            summary["n_test_b"] = test_sizes[k2]
            summary["small_sample_warning"] = min(test_sizes[k1], test_sizes[k2]) < 20
            key = f"{k1}_vs_{k2}"
            results["pairs"][key] = summary
            flag = " *** SMALL SAMPLE (n<20), INTERPRET WITH CAUTION ***" if summary["small_sample_warning"] else ""
            print(f"[{key}] mean angle = {summary['mean_angle_deg']:.2f} deg "
                  f"(range {summary['min_angle_deg']:.1f}-{summary['max_angle_deg']:.1f}), "
                  f"vs random-null {null_mean_angle:.2f} deg{flag}")

    out_path = results_dir / "principal_angle_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
