"""Training utilities: standardization fit on train only, class-weighted CE, early stopping on val loss, window->trial aggregation."""
import numpy as np, torch, torch.nn as nn, random

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class ZScore:
    """Per-channel z-score for raw windows (stats from TRAIN windows only)."""
    def fit(self, X): self.mu = X.mean(axis=(0, 2), keepdims=True); self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-12; return self
    def transform(self, X): return ((X - self.mu) / self.sd).astype(np.float32)

def train_torch(model, Xtr, ytr, Xva, yva, device, max_epochs=100, patience=15, lr=1e-3, bs=64, seed=0, wd=1e-4):
    seed_all(seed); model = model.to(device); cnt = np.bincount(ytr, minlength=2); w = torch.tensor(len(ytr) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.tensor(Xtr).unsqueeze(1); ytr_t = torch.tensor(ytr, dtype=torch.long); Xva_t = torch.tensor(Xva).unsqueeze(1).to(device); yva_t = torch.tensor(yva, dtype=torch.long).to(device)
    best, best_state, bad, g = np.inf, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(ytr_t), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device); opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(crit(model(Xva_t), yva_t)) if len(yva) else 0.0
        if vl < best - 1e-5: best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, {"epochs_run": ep + 1, "best_val_loss": float(best)}

@torch.no_grad()
def predict_proba(model, X, device, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs): out.append(torch.softmax(model(torch.tensor(X[i:i + bs]).unsqueeze(1).to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def aggregate_windows_to_trials(p_win, trial_ids):
    """Mean window probability per trial; returns (unique_trial_ids, p_trial) in first-appearance order."""
    order = {}; sums = {}; cnts = {}
    for p, t in zip(p_win, trial_ids):
        if t not in order: order[t] = len(order); sums[t] = 0.0; cnts[t] = 0
        sums[t] += float(p); cnts[t] += 1
    tids = sorted(order, key=order.get); return np.array(tids), np.array([sums[t] / cnts[t] for t in tids])
