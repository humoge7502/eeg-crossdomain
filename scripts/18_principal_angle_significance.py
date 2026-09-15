"""
Bootstrap significance test for the one reliable principal-angle result
(NeuMa vs Restaurant-Logo, n=941/88 -- large enough for trustworthy
resampling). Resamples trials WITH replacement from each dataset's test
embeddings, recomputes the mean principal angle each time, and reports
a 95% CI -- so we can compare against the random-subspace null's spread
rather than eyeballing a single point estimate.
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
N_BOOTSTRAP = 500


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
    rng = np.random.default_rng(cfg["training"]["seed"])

    X1, y1 = load_features_and_labels(cfg, "neuma")
    X2, y2 = load_features_and_labels(cfg, "restaurant_logo")

    idx1 = rng.permutation(len(y1)); n_test1 = max(1, int(0.2 * len(y1)))
    test_idx1, train_idx1 = idx1[:n_test1], idx1[n_test1:]
    idx2 = rng.permutation(len(y2)); n_test2 = max(1, int(0.2 * len(y2)))
    test_idx2, train_idx2 = idx2[:n_test2], idx2[n_test2:]

    print("Training solo encoders...")
    enc1 = train_solo_encoder(X1[train_idx1], y1[train_idx1], X1.shape[1], device, seed=cfg["training"]["seed"])
    enc2 = train_solo_encoder(X2[train_idx2], y2[train_idx2], X2.shape[1], device, seed=cfg["training"]["seed"])

    emb1 = extract_embeddings(enc1, X1[test_idx1], device)
    emb2 = extract_embeddings(enc2, X2[test_idx2], device)

    observed_angle = mean_principal_angle_deg(emb1, emb2, K_COMPONENTS)
    print(f"Observed mean angle: {observed_angle:.2f} deg")

    print(f"Bootstrapping ({N_BOOTSTRAP} resamples)...")
    boot_angles = []
    for _ in range(N_BOOTSTRAP):
        idx_a = rng.choice(len(emb1), size=len(emb1), replace=True)
        idx_b = rng.choice(len(emb2), size=len(emb2), replace=True)
        boot_angles.append(mean_principal_angle_deg(emb1[idx_a], emb2[idx_b], K_COMPONENTS))
    boot_angles = np.array(boot_angles)
    ci_lo, ci_hi = np.percentile(boot_angles, [2.5, 97.5])

    print("\nRandom-subspace null (from previous run): mean=70.13, std=1.47 deg")
    print(f"Observed NeuMa-vs-RestaurantLogo angle: {observed_angle:.2f} deg")
    print(f"Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] deg")
    null_z = (70.13 - boot_angles.mean()) / boot_angles.std()
    print(f"Distance from null mean in bootstrap-SE units: {null_z:.2f}")
    print(f"Does 95% CI exclude the null mean (70.13)? {'YES -- significant' if ci_hi < 70.13 else 'NO -- not significant'}")

    result = {
        "observed_angle_deg": observed_angle,
        "bootstrap_mean": float(boot_angles.mean()),
        "bootstrap_std": float(boot_angles.std()),
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "random_null_mean_deg": 70.13,
        "random_null_std_deg": 1.47,
        "excludes_null": bool(ci_hi < 70.13),
    }
    with open(results_dir / "principal_angle_significance_neuma_restlogo.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> results/principal_angle_significance_neuma_restlogo.json")


if __name__ == "__main__":
    main()
