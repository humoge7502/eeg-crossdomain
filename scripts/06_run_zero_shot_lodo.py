#!/usr/bin/env python
"""Experiment 3 - direct zero-shot LODO boundary test.
Target = one dataset; sources = the other two. No target labels used for training.
Variants: naive, classweight, coral, dann. Test set = frozen held-out target participants.
Labels are NOT semantically merged; this is explicitly a boundary-condition experiment."""
import argparse, json, sys, logging, datetime, time
from pathlib import Path
import numpy as np, yaml, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.tabular_v2 import TabularEncoder, Head, EncHead
from src.models.coral_dann_v2 import coral_fit, coral_apply, DomainDiscriminator
from src.models.train_v2 import seed_all
from src.evaluation.metrics_v2 import compute_all, choose_threshold

DATASETS = ["neuma", "restaurant_logo", "ds007406"]


def load_features(root, name, feat):
    f = np.load(root / name / f"{name}_features_{feat}.npz", allow_pickle=True)
    return {"X": f["X"].astype(np.float32), "y": f["y"].astype(int),
            "sid": f["subject_ids"].astype(str), "tid": f["trial_ids"].astype(str)}


def train_dann(Xs, ys, Xt, Xva, yva, device, seed, epochs, patience, lam=0.5):
    seed_all(seed)
    enc = TabularEncoder(Xs.shape[1]).to(device)
    head = Head(enc.emb_dim).to(device)
    disc = DomainDiscriminator(enc.emb_dim, 2, lam).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()) + list(disc.parameters()), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(ys, minlength=2)
    w = torch.tensor(len(ys) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    task_crit = nn.CrossEntropyLoss(weight=w)
    dom_crit = nn.CrossEntropyLoss()
    Xs_t = torch.tensor(Xs); ys_t = torch.tensor(ys, dtype=torch.long); Xt_t = torch.tensor(Xt)
    Xv = torch.tensor(Xva).to(device); yv = torch.tensor(yva, dtype=torch.long).to(device)
    best, state, bad, g = np.inf, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        enc.train(); head.train(); disc.train()
        perm = torch.randperm(len(ys_t), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            xs = Xs_t[idx].to(device); yb = ys_t[idx].to(device)
            n = min(len(idx), len(Xt_t))
            xt = Xt_t[torch.randperm(len(Xt_t), generator=g)[:n]].to(device)
            zs, zt = enc(xs), enc(xt)
            loss = task_crit(head(zs), yb)
            loss = loss + 0.5 * (dom_crit(disc(zs), torch.zeros(len(zs), dtype=torch.long, device=device))
                                 + dom_crit(disc(zt), torch.ones(len(zt), dtype=torch.long, device=device)))
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            vl = float(task_crit(head(enc(Xv)), yv))
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                     {k: v.detach().clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        enc.load_state_dict(state[0]); head.load_state_dict(state[1])
    enc.eval(); head.eval()
    return EncHead(enc, head)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32).to(device)), 1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--features", default="v2")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    root = ROOT / (cfg["output_root"] + ("_smoke" if a.smoke else ""))
    splits_dir = root / "splits"
    suffix = "" if a.features == "v2" else "_" + a.features.replace("_v2", "")
    out = ROOT / a.output_dir
    pred_dir = out / "predictions" / (("exp3_lodo_smoke" if a.smoke else "exp3_lodo") + suffix)
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / f"06_lodo_{ts}.log"), logging.StreamHandler()])
    log = logging.getLogger("e3")
    device = torch.device(f"cuda:{a.gpu if a.gpu < torch.cuda.device_count() else 0}" if torch.cuda.is_available() else "cpu")
    epochs = 3 if a.smoke else a.epochs
    patience = 2 if a.smoke else a.patience
    seeds = a.seeds[:1] if a.smoke else a.seeds
    variants = ["naive", "coral"] if a.smoke else ["naive", "classweight", "coral", "dann"]
    data = {n: load_features(root, n, a.features) for n in DATASETS}
    log.info(f"device={device} features={a.features} seeds={seeds}")
    quick = []
    for target in DATASETS:
        sources = [s for s in DATASETS if s != target]
        sp_all = json.load(open(splits_dir / f"transfer_target_{target}.json"))["by_seed"]
        Xs_full = np.concatenate([data[s]["X"] for s in sources])
        ys_full = np.concatenate([data[s]["y"] for s in sources])
        for seed in seeds:
            sp = sp_all[str(seed)]
            te = np.isin(data[target]["sid"], sp["test"])
            Xte, yte = data[target]["X"][te], data[target]["y"][te]
            tgt_tr = np.isin(data[target]["sid"], sp["fractions"]["100pct"]["train"])
            Xt_unlab = data[target]["X"][tgt_tr]
            rng = np.random.default_rng(seed)
            nval = max(2, int(0.15 * len(ys_full)))
            vi = rng.choice(len(ys_full), nval, replace=False)
            ti = np.setdiff1d(np.arange(len(ys_full)), vi)
            for variant in variants:
                fn = pred_dir / f"{target}__seed{seed}__{variant}.csv"
                if fn.exists():
                    continue
                t0 = time.time()
                sc = StandardScaler().fit(Xs_full[ti])
                Xtr_s, ytr_s = sc.transform(Xs_full[ti]), ys_full[ti]
                Xte_s = sc.transform(Xte)
                if variant in ("naive", "classweight"):
                    m = LogisticRegression(max_iter=2000, C=1.0, random_state=seed,
                                           class_weight=("balanced" if variant == "classweight" else None))
                    m.fit(Xtr_s, ytr_s)
                    p_train = m.predict_proba(Xtr_s)[:, 1]; p = m.predict_proba(Xte_s)[:, 1]
                elif variant == "coral":
                    A = coral_fit(Xtr_s, sc.transform(Xt_unlab))
                    Xa = coral_apply(Xtr_s, A)
                    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
                    m.fit(Xa, ytr_s)
                    p_train = m.predict_proba(Xa)[:, 1]; p = m.predict_proba(coral_apply(Xte_s, A))[:, 1]
                else:
                    model = train_dann(Xtr_s, ytr_s, sc.transform(Xt_unlab), sc.transform(Xs_full[vi]), ys_full[vi],
                                       device, seed, epochs, patience)
                    p_train = predict(model, Xtr_s, device); p = predict(model, Xte_s, device)
                thr = choose_threshold(ytr_s, p_train)
                pd.DataFrame({"target": target, "sources": "+".join(sources), "variant": variant, "seed": seed,
                              "trial_id": data[target]["tid"][te], "subject_id": data[target]["sid"][te],
                              "y_true": yte, "y_prob": p, "threshold_train": thr,
                              "y_pred": (p >= thr).astype(int)}).to_csv(fn, index=False)
                mt = compute_all(yte, p, thr)
                quick.append({"target": target, "variant": variant, "seed": seed, "mcc": mt["mcc"],
                              "roc_auc": mt["roc_auc"], "bal_acc": mt["balanced_accuracy"]})
                log.info(f"target={target:16s} {variant:12s} seed={seed} n={len(yte):4d} "
                         f"MCC={mt['mcc']:+.3f} AUC={mt['roc_auc']:.3f} thr={thr:.3f} ({time.time()-t0:.0f}s)")
    if quick:
        q = pd.DataFrame(quick)
        q.to_csv(pred_dir / f"quicklook_{ts}.csv", index=False)
        log.info("\n" + q.groupby(["target", "variant"])[["mcc", "roc_auc", "bal_acc"]].mean().round(3).to_string())
    log.info(f"[done] {pred_dir}")


if __name__ == "__main__":
    main()
