import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.evaluate import compute_metrics, summarize_lodo_results

def load_all_with_engagement(cfg):
    # Harmonized feature layout: 7 channels x 5 bands (delta,theta,alpha,beta,gamma) + 1 asymmetry = 36
    # Arousal/engagement proxy: mean beta+gamma power across channels (bands index 3,4 per channel)
    # minus mean alpha power (channels index 2 per channel), then median-split within each dataset.
    data = {}
    for name in cfg["datasets"].keys():
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        X = d["X"].astype(np.float32)
        n_bands = 5
        n_ch = 7
        X_reshaped = X[:, :n_ch*n_bands].reshape(-1, n_ch, n_bands)
        beta = X_reshaped[:, :, 3].mean(axis=1)
        gamma = X_reshaped[:, :, 4].mean(axis=1)
        alpha = X_reshaped[:, :, 2].mean(axis=1)
        engagement_score = (beta + gamma) - alpha
        median = np.median(engagement_score)
        engagement_label = (engagement_score > median).astype(np.int64)
        data[name] = (X, engagement_label)
        print(f"[{name}] engagement label balance: {np.bincount(engagement_label)}")
    return data

def train_eval(X_train, y_train, X_test, y_test, cfg, device, tag):
    class_counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor([len(y_train)/(2.0*max(c,1)) for c in class_counts], dtype=torch.float32).to(device)
    encoder = TabularMLPEncoder(input_dim=X_train.shape[1], embedding_dim=64)
    head = ClassificationHead(embedding_dim=64, n_classes=2)
    model = nn.Sequential(encoder, head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_val = max(1, int(0.15*len(y_train)))
    perm = np.random.default_rng(cfg["training"]["seed"]).permutation(len(y_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr = torch.tensor(X_train[tr_idx]); ytr = torch.tensor(y_train[tr_idx])
    Xv = torch.tensor(X_train[val_idx]); yv = torch.tensor(y_train[val_idx])
    Xte = torch.tensor(X_test); yte = torch.tensor(y_test)

    best_val = float("inf"); best_state = None; patience = 0
    for epoch in range(60):
        model.train()
        perm2 = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 32):
            idx = perm2[i:i+32]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = criterion(model(Xv.to(device)), yv.to(device)).item()
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k,v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(yte.numpy(), preds, probs)
    print(f"  [{tag}] {metrics}")
    return metrics

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    print("="*60 + "\nOPTION B: Feature-derived engagement relabeling (task-agnostic)\n" + "="*60)
    all_data = load_all_with_engagement(cfg)

    fold_results = []
    for held_out in all_data.keys():
        train_keys = [k for k in all_data if k != held_out]
        X_train = np.concatenate([all_data[k][0] for k in train_keys], axis=0)
        y_train = np.concatenate([all_data[k][1] for k in train_keys], axis=0)
        X_test, y_test = all_data[held_out]
        print(f"\n--- held_out={held_out} ---")
        m = train_eval(X_train, y_train, X_test, y_test, cfg, device, held_out)
        fold_results.append({"held_out": held_out, "metrics": m})

    summary = summarize_lodo_results(fold_results)
    print("\nOPTION B SUMMARY:", {k: round(v["mean"],4) for k,v in summary.items()})

    with open(results_dir / "option_B_results.json", "w") as f:
        json.dump({"summary": summary, "per_fold": fold_results}, f, indent=2)
    print("Saved -> results/option_B_results.json")

if __name__ == "__main__":
    main()
