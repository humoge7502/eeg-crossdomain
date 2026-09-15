"""
Multi-seed robustness check for Option A (full-duration tabular
features, LODO). Repeats the no-alignment condition across multiple
seeds (different weight init AND different train/val split each time)
to confirm the null-transfer conclusion (kappa ~0) is stable, not a
single-seed artifact.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.evaluate import compute_metrics

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]
SEEDS = [42, 1, 7, 13, 99]


def load_all(cfg):
    data = {}
    for name in DATASET_NAMES:
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        data[name] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
    return data


def train_eval(X_train, y_train, X_test, y_test, device, seed):
    set_seed(seed)
    class_counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([len(y_train) / (2.0 * max(c, 1)) for c in class_counts], dtype=torch.float32).to(device)
    encoder = TabularMLPEncoder(input_dim=X_train.shape[1], embedding_dim=64)
    head = ClassificationHead(embedding_dim=64, n_classes=2)
    model = nn.Sequential(encoder, head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_val = max(1, int(0.15 * len(y_train)))
    perm = np.random.default_rng(seed).permutation(len(y_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.tensor(X_train[tr_idx]); ytr = torch.tensor(y_train[tr_idx])
    Xv = torch.tensor(X_train[val_idx]); yv = torch.tensor(y_train[val_idx])
    Xte = torch.tensor(X_test); yte = torch.tensor(y_test)

    best_val = float("inf"); best_state = None; patience = 0
    for epoch in range(60):
        model.train()
        perm2 = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 32):
            idx = perm2[i:i + 32]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = criterion(model(Xv.to(device)), yv.to(device)).item()
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(yte.numpy(), preds, probs)
    return metrics["cohen_kappa"]


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / "results")
    all_data = load_all(cfg)

    print(f"Multi-seed Option A (no-alignment) across {len(SEEDS)} seeds...")
    per_seed_mean = []
    all_results = {}
    for seed in SEEDS:
        fold_kappas = []
        for held_out in DATASET_NAMES:
            train_keys = [k for k in DATASET_NAMES if k != held_out]
            X_train = np.concatenate([all_data[k][0] for k in train_keys], axis=0)
            y_train = np.concatenate([all_data[k][1] for k in train_keys], axis=0)
            X_test, y_test = all_data[held_out]
            kappa = train_eval(X_train, y_train, X_test, y_test, device, seed)
            fold_kappas.append(kappa)
        mean_kappa = float(np.mean(fold_kappas))
        per_seed_mean.append(mean_kappa)
        all_results[f"seed_{seed}"] = {"fold_kappas": fold_kappas, "mean_kappa": mean_kappa}
        print(f"  seed={seed}: mean kappa across 3 held-out folds = {mean_kappa:.4f} "
              f"(per-fold: {[round(k,4) for k in fold_kappas]})")

    per_seed_mean = np.array(per_seed_mean)
    print(f"\n{'='*60}\nMULTI-SEED OPTION A ROBUSTNESS RESULT\n{'='*60}")
    print(f"Mean kappa across {len(SEEDS)} seeds: {per_seed_mean.mean():.4f} +/- {per_seed_mean.std():.4f}")
    print(f"Range: [{per_seed_mean.min():.4f}, {per_seed_mean.max():.4f}]")

    result = {
        "seeds": SEEDS,
        "per_seed_results": all_results,
        "grand_mean_kappa": float(per_seed_mean.mean()),
        "grand_std_kappa": float(per_seed_mean.std()),
        "min_kappa": float(per_seed_mean.min()),
        "max_kappa": float(per_seed_mean.max()),
    }
    with open(results_dir / "multiseed_option_A_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> results/multiseed_option_A_results.json")


if __name__ == "__main__":
    main()
