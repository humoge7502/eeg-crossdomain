import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.models import TabularMLPEncoder, ClassificationHead
from src.evaluate import compute_metrics, summarize_lodo_results
from src.alignment import coral_align

def load_all(cfg):
    data = {}
    for name in cfg["datasets"].keys():
        p = project_root() / "data" / "processed_v2" / name / f"{name}_features_persubj_v2.npz"
        d = np.load(p)
        subject_ids = d["subject_ids"] if "subject_ids" in d else None
        data[name] = (d["X"].astype(np.float32), d["y"].astype(np.int64), subject_ids)
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
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(Xv.to(device))
            vloss = criterion(vout, yv.to(device)).item()
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k,v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(yte.numpy(), preds, probs)
    print(f"  [{tag}] {metrics}")
    return metrics, yte.numpy(), probs, encoder.state_dict()

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    all_data = load_all(cfg)

    print("="*60 + "\nOPTION A: Full-duration tabular features, no truncation\n" + "="*60)

    fold_results_noalign = []
    fold_results_coral = []
    for held_out in all_data.keys():
        train_keys = [k for k in all_data if k != held_out]
        X_train = np.concatenate([all_data[k][0] for k in train_keys], axis=0)
        y_train = np.concatenate([all_data[k][1] for k in train_keys], axis=0)
        X_test, y_test, test_subject_ids = all_data[held_out]

        print(f"\n--- held_out={held_out} (no alignment) ---")
        m1, yt1, yp1, enc_state1 = train_eval(X_train, y_train, X_test, y_test, cfg, device, f"noalign_{held_out}")
        fold_results_noalign.append({"held_out": held_out, "metrics": m1})
        np.savez(results_dir / f"optionA_predictions_noalign_{held_out}.npz",
                 y_true=yt1, y_prob=yp1,
                 subject_ids=test_subject_ids if test_subject_ids is not None else np.array([]))
        torch.save(enc_state1, results_dir / f"optionA_encoder_noalign_{held_out}.pt")

        print(f"--- held_out={held_out} (CORAL aligned to target stats) ---")
        X_train_aligned = coral_align(X_train, X_test).astype(np.float32)
        m2, yt2, yp2, enc_state2 = train_eval(X_train_aligned, y_train, X_test, y_test, cfg, device, f"coral_{held_out}")
        fold_results_coral.append({"held_out": held_out, "metrics": m2})
        np.savez(results_dir / f"optionA_predictions_coral_{held_out}.npz",
                 y_true=yt2, y_prob=yp2,
                 subject_ids=test_subject_ids if test_subject_ids is not None else np.array([]))
        torch.save(enc_state2, results_dir / f"optionA_encoder_coral_{held_out}.pt")

    summary_noalign = summarize_lodo_results(fold_results_noalign)
    summary_coral = summarize_lodo_results(fold_results_coral)

    print("\n" + "="*60 + "\nOPTION A SUMMARY\n" + "="*60)
    print("No alignment:", {k: round(v["mean"],4) for k,v in summary_noalign.items()})
    print("CORAL aligned:", {k: round(v["mean"],4) for k,v in summary_coral.items()})

    with open(results_dir / "option_A_results.json", "w") as f:
        json.dump({"no_alignment": {"summary": summary_noalign, "per_fold": fold_results_noalign},
                    "coral_alignment": {"summary": summary_coral, "per_fold": fold_results_coral}}, f, indent=2)
    print("Saved -> results/option_A_results.json")

if __name__ == "__main__":
    main()
