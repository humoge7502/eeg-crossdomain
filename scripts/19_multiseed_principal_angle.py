"""
Multi-seed robustness check for the principal-angle representation
alignment finding (Section 21). Repeats the NeuMa-vs-Restaurant-Logo
solo-encoder training and principal-angle computation across multiple
independent seeds (different train/test splits AND different encoder
initializations each time), to confirm the -4.1 deg deviation from the
random-subspace null is not an artifact of one lucky seed.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder

K_COMPONENTS = 10
SEEDS = [42, 1, 7, 13, 99, 2024, 314, 271]
RANDOM_NULL_MEAN = 70.13
RANDOM_NULL_STD = 1.47


def load_features_and_labels(cfg, name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
    d = np.load(p)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


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


def mean_principal_angle_deg(emb_a, emb_b, k):
    k_eff = min(k, emb_a.shape[1])
    _, _, Va = np.linalg.svd(emb_a - emb_a.mean(axis=0), full_matrices=False)
    _, _, Vb = np.linalg.svd(emb_b - emb_b.mean(axis=0), full_matrices=False)
    Qa = Va[:k_eff, :].T
    Qb = Vb[:k_eff, :].T
    M = Qa.T @ Qb
    sv = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    return float(np.degrees(np.arccos(sv)).mean())


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")

    X1, y1 = load_features_and_labels(cfg, "neuma")
    X2, y2 = load_features_and_labels(cfg, "restaurant_logo")

    print(f"Running principal-angle analysis across {len(SEEDS)} independent seeds "
          f"(different train/test splits AND encoder inits each time)...")
    angles = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx1 = rng.permutation(len(y1)); n_test1 = max(1, int(0.2 * len(y1)))
        test_idx1, train_idx1 = idx1[:n_test1], idx1[n_test1:]
        idx2 = rng.permutation(len(y2)); n_test2 = max(1, int(0.2 * len(y2)))
        test_idx2, train_idx2 = idx2[:n_test2], idx2[n_test2:]

        enc1 = train_solo_encoder(X1[train_idx1], y1[train_idx1], X1.shape[1], device, seed=seed)
        enc2 = train_solo_encoder(X2[train_idx2], y2[train_idx2], X2.shape[1], device, seed=seed)

        emb1 = extract_embeddings(enc1, X1[test_idx1], device)
        emb2 = extract_embeddings(enc2, X2[test_idx2], device)

        angle = mean_principal_angle_deg(emb1, emb2, K_COMPONENTS)
        angles.append(angle)
        print(f"  seed={seed}: mean angle = {angle:.2f} deg")

    angles = np.array(angles)
    print(f"\n{'='*60}\nMULTI-SEED ROBUSTNESS RESULT\n{'='*60}")
    print(f"Angles across {len(SEEDS)} seeds: {[round(a,2) for a in angles]}")
    print(f"Mean: {angles.mean():.2f} deg, Std: {angles.std():.2f} deg")
    print(f"Min: {angles.min():.2f} deg, Max: {angles.max():.2f} deg")
    print(f"Random-subspace null: {RANDOM_NULL_MEAN} +/- {RANDOM_NULL_STD} deg")
    n_below_null = int(np.sum(angles < RANDOM_NULL_MEAN - RANDOM_NULL_STD))
    print(f"Seeds with angle > 1 null-std below the null mean: {n_below_null}/{len(SEEDS)}")

    result = {
        "seeds": SEEDS,
        "angles_deg": [float(a) for a in angles],
        "mean_deg": float(angles.mean()),
        "std_deg": float(angles.std()),
        "min_deg": float(angles.min()),
        "max_deg": float(angles.max()),
        "random_null_mean_deg": RANDOM_NULL_MEAN,
        "random_null_std_deg": RANDOM_NULL_STD,
        "n_seeds_below_null_minus_1std": n_below_null,
        "n_seeds_total": len(SEEDS),
    }
    with open(results_dir / "multiseed_principal_angle_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> results/multiseed_principal_angle_results.json")


if __name__ == "__main__":
    main()
