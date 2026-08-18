import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.evaluate import compute_metrics

def load_all(cfg):
    data = {}
    for name in cfg["datasets"].keys():
        p = project_root() / "data" / "processed" / name / f"{name}_harmonized.npz"
        d = np.load(p)
        data[name] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
    return data

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    all_data = load_all(cfg)

    print("="*60 + "\nOPTION C: Multi-task shared encoder vs solo per-dataset\n" + "="*60)

    splits = {}
    for name, (X, y) in all_data.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        splits[name] = (Xtr, ytr, Xte, yte)

    shared_encoder = TabularMLPEncoder(input_dim=36, embedding_dim=64).to(device)
    heads = {name: ClassificationHead(embedding_dim=64, n_classes=2).to(device) for name in all_data.keys()}
    params = list(shared_encoder.parameters())
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.Adam(params, lr=0.001)

    criteria = {}
    for name, (Xtr, ytr, _, _) in splits.items():
        counts = np.bincount(ytr, minlength=2)
        w = torch.tensor([len(ytr)/(2.0*max(c,1)) for c in counts], dtype=torch.float32).to(device)
        criteria[name] = nn.CrossEntropyLoss(weight=w)

    print("Training joint multi-task model...")
    for epoch in range(60):
        shared_encoder.train()
        for h in heads.values(): h.train()
        total_loss = 0.0
        for name, (Xtr, ytr, _, _) in splits.items():
            Xb = torch.tensor(Xtr).to(device)
            yb = torch.tensor(ytr).to(device)
            emb = shared_encoder(Xb)
            logits = heads[name](emb)
            loss = criteria[name](logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  epoch {epoch}: total_loss={total_loss:.4f}")

    print("\nEvaluating multi-task shared encoder per dataset:")
    multitask_results = {}
    shared_encoder.eval()
    for name, (_, _, Xte, yte) in splits.items():
        heads[name].eval()
        with torch.no_grad():
            emb = shared_encoder(torch.tensor(Xte).to(device))
            logits = heads[name](emb)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        m = compute_metrics(yte, preds, probs)
        multitask_results[name] = m
        print(f"  [{name}] multitask: {m}")

    print("\nTraining solo (single-dataset) encoders for comparison...")
    solo_results = {}
    for name, (Xtr, ytr, Xte, yte) in splits.items():
        enc = TabularMLPEncoder(input_dim=36, embedding_dim=64).to(device)
        head = ClassificationHead(embedding_dim=64, n_classes=2).to(device)
        model = nn.Sequential(enc, head)
        opt2 = torch.optim.Adam(model.parameters(), lr=0.001)
        counts = np.bincount(ytr, minlength=2)
        w = torch.tensor([len(ytr)/(2.0*max(c,1)) for c in counts], dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        Xb = torch.tensor(Xtr).to(device); yb = torch.tensor(ytr).to(device)
        for epoch in range(60):
            model.train()
            opt2.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward(); opt2.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte).to(device))
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        m = compute_metrics(yte, preds, probs)
        solo_results[name] = m
        print(f"  [{name}] solo: {m}")

    print("\nOPTION C COMPARISON (multitask vs solo, kappa):")
    for name in all_data.keys():
        mt_k = multitask_results[name]["cohen_kappa"]
        solo_k = solo_results[name]["cohen_kappa"]
        print(f"  {name}: multitask={mt_k:.4f}  solo={solo_k:.4f}  delta={mt_k-solo_k:+.4f}")

    with open(results_dir / "option_C_results.json", "w") as f:
        json.dump({"multitask": multitask_results, "solo": solo_results}, f, indent=2)
    print("Saved -> results/option_C_results.json")

if __name__ == "__main__":
    main()
