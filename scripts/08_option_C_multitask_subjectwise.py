import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.evaluate import compute_metrics

def load_all(cfg):
    data = {}
    for name in cfg["datasets"].keys():
        p = project_root() / "data" / "processed" / name / f"{name}_harmonized.npz"
        d = np.load(p, allow_pickle=True)
        subject_ids = d["subject_ids"] if "subject_ids" in d else None
        if subject_ids is None:
            print(f"[{name}] WARNING: no subject_ids in harmonized features — cannot do subject-wise split")
        data[name] = (d["X"].astype(np.float32), d["y"].astype(np.int64), subject_ids)
    return data

def subject_wise_split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], groups[train_idx], groups[test_idx]

def train_model(model, Xtr, ytr, Xv, yv, device, epochs=60, lr=0.001, patience=10):
    counts = np.bincount(ytr, minlength=2)
    w = torch.tensor([len(ytr)/(2.0*max(c,1)) for c in counts], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr).to(device); ytr_t = torch.tensor(ytr).to(device)
    Xv_t = torch.tensor(Xv).to(device); yv_t = torch.tensor(yv).to(device)

    best_val = float("inf"); best_state = None; bad = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(ytr_t), 32):
            idx = perm[i:i+32]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv_t), yv_t).item()
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    all_data = load_all(cfg)

    print("="*60 + "\nOPTION C (SUBJECT-WISE): Multi-task shared encoder vs solo\n" + "="*60)

    splits = {}
    for name, (X, y, groups) in all_data.items():
        if groups is None:
            print(f"[{name}] SKIPPED entirely — no subject_ids available")
            continue
        Xtr, ytr, Xte, yte, gtr, gte = subject_wise_split(X, y, groups, test_size=0.2, seed=cfg["training"]["seed"])
        splits[name] = (Xtr, ytr, Xte, yte, gtr)
        print(f"[{name}] train={len(ytr)} epochs / {len(np.unique(gtr))} subjects, "
              f"test={len(yte)} epochs / {len(np.unique(gte))} subjects")

    if len(splits) < 2:
        print("Not enough datasets with subject_ids to run multi-task comparison.")
        return

    shared_encoder = TabularMLPEncoder(input_dim=36, embedding_dim=64).to(device)
    heads = {name: ClassificationHead(embedding_dim=64, n_classes=2).to(device) for name in splits.keys()}
    params = list(shared_encoder.parameters())
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.Adam(params, lr=0.001)

    criteria = {}
    inner_splits = {}
    for name, (Xtr, ytr, Xte, yte, gtr) in splits.items():
        counts = np.bincount(ytr, minlength=2)
        w = torch.tensor([len(ytr)/(2.0*max(c,1)) for c in counts], dtype=torch.float32).to(device)
        criteria[name] = nn.CrossEntropyLoss(weight=w)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg["training"]["seed"])
        inner_tr, inner_val = next(gss.split(Xtr, ytr, groups=gtr))
        inner_splits[name] = (inner_tr, inner_val)

    print("\nTraining joint multi-task model (subject-disjoint train/val per dataset)...")
    for epoch in range(60):
        shared_encoder.train()
        for h in heads.values(): h.train()
        total_loss = 0.0
        for name, (Xtr, ytr, Xte, yte, gtr) in splits.items():
            inner_tr, _ = inner_splits[name]
            Xb = torch.tensor(Xtr[inner_tr]).to(device)
            yb = torch.tensor(ytr[inner_tr]).to(device)
            emb = shared_encoder(Xb)
            logits = heads[name](emb)
            loss = criteria[name](logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 15 == 0:
            print(f"  epoch {epoch}: total_loss={total_loss:.4f}")

    print("\nEvaluating multi-task shared encoder on held-out SUBJECTS per dataset:")
    multitask_results = {}
    shared_encoder.eval()
    for name, (Xtr, ytr, Xte, yte, gtr) in splits.items():
        heads[name].eval()
        with torch.no_grad():
            emb = shared_encoder(torch.tensor(Xte).to(device))
            logits = heads[name](emb)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        m = compute_metrics(yte, preds, probs)
        multitask_results[name] = m
        print(f"  [{name}] multitask (held-out subjects): {m}")

    print("\nTraining solo (single-dataset) encoders, same subject-wise split, for comparison...")
    solo_results = {}
    for name, (Xtr, ytr, Xte, yte, gtr) in splits.items():
        enc = TabularMLPEncoder(input_dim=36, embedding_dim=64).to(device)
        head = ClassificationHead(embedding_dim=64, n_classes=2).to(device)
        model = nn.Sequential(enc, head)
        inner_tr, inner_val = inner_splits[name]
        model = train_model(model, Xtr[inner_tr], ytr[inner_tr], Xtr[inner_val], ytr[inner_val], device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte).to(device))
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        m = compute_metrics(yte, preds, probs)
        solo_results[name] = m
        print(f"  [{name}] solo (held-out subjects): {m}")

    print("\nOPTION C (SUBJECT-WISE) COMPARISON:")
    for name in splits.keys():
        mt_k = multitask_results[name]["cohen_kappa"]
        solo_k = solo_results[name]["cohen_kappa"]
        print(f"  {name}: multitask={mt_k:.4f}  solo={solo_k:.4f}  delta={mt_k-solo_k:+.4f}")

    with open(results_dir / "option_C_subjectwise_results.json", "w") as f:
        json.dump({"multitask": multitask_results, "solo": solo_results}, f, indent=2)
    print("Saved -> results/option_C_subjectwise_results.json")

if __name__ == "__main__":
    main()
