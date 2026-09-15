import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.dann import DomainDiscriminator
from src.evaluate import compute_metrics, summarize_lodo_results

def load_all(cfg):
    data = {}
    for name in cfg["datasets"].keys():
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        data[name] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
    return data

def run_dann_fold(held_out, all_data, cfg, device):
    train_keys = [k for k in all_data if k != held_out]
    dataset_names = list(all_data.keys())
    domain_id = {name: i for i, name in enumerate(dataset_names)}

    X_train = np.concatenate([all_data[k][0] for k in train_keys], axis=0)
    y_train = np.concatenate([all_data[k][1] for k in train_keys], axis=0)
    domain_train = np.concatenate([np.full(len(all_data[k][1]), domain_id[k]) for k in train_keys], axis=0)
    X_test, y_test = all_data[held_out]
    domain_test = np.full(len(y_test), domain_id[held_out])

    X_all_domain = np.concatenate([X_train, X_test], axis=0)
    domain_all = np.concatenate([domain_train, domain_test], axis=0)

    encoder = TabularMLPEncoder(input_dim=36, embedding_dim=64).to(device)
    task_head = ClassificationHead(embedding_dim=64, n_classes=2).to(device)
    domain_head = DomainDiscriminator(embedding_dim=64, n_domains=len(dataset_names), lambda_=0.5).to(device)

    params = list(encoder.parameters()) + list(task_head.parameters()) + list(domain_head.parameters())
    opt = torch.optim.Adam(params, lr=0.001)

    counts = np.bincount(y_train, minlength=2)
    task_weights = torch.tensor([len(y_train)/(2.0*max(c,1)) for c in counts], dtype=torch.float32).to(device)
    task_criterion = nn.CrossEntropyLoss(weight=task_weights)
    domain_criterion = nn.CrossEntropyLoss()

    Xtr_task = torch.tensor(X_train).to(device); ytr_task = torch.tensor(y_train).to(device)
    Xall_dom = torch.tensor(X_all_domain).to(device); yall_dom = torch.tensor(domain_all).to(device)

    for epoch in range(60):
        encoder.train(); task_head.train(); domain_head.train()

        emb_task = encoder(Xtr_task)
        task_logits = task_head(emb_task)
        task_loss = task_criterion(task_logits, ytr_task)

        emb_dom = encoder(Xall_dom)
        domain_logits = domain_head(emb_dom)
        domain_loss = domain_criterion(domain_logits, yall_dom)

        total_loss = task_loss + domain_loss
        opt.zero_grad(); total_loss.backward(); opt.step()

    encoder.eval(); task_head.eval()
    with torch.no_grad():
        emb_test = encoder(torch.tensor(X_test).to(device))
        logits = task_head(emb_test)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(y_test, preds, probs)
    print(f"  held_out={held_out}: {metrics}")
    return {"held_out": held_out, "metrics": metrics}

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    all_data = load_all(cfg)

    print("="*60 + "\nOPTION D: Domain-Adversarial Neural Network (DANN)\n" + "="*60)

    fold_results = []
    for held_out in all_data.keys():
        print(f"\n--- held_out={held_out} ---")
        r = run_dann_fold(held_out, all_data, cfg, device)
        fold_results.append(r)

    summary = summarize_lodo_results(fold_results)
    print("\nOPTION D SUMMARY:", {k: round(v["mean"],4) for k,v in summary.items()})

    with open(results_dir / "option_D_results.json", "w") as f:
        json.dump({"summary": summary, "per_fold": fold_results}, f, indent=2)
    print("Saved -> results/option_D_results.json")

if __name__ == "__main__":
    main()
