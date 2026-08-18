#!/usr/bin/env python
"""Experiment 4 - main pretraining + fine-tuning experiment.
Shared encoder (36->64->32) pretrained on the two source datasets with SEPARATE dataset-specific heads
and task-balanced batches. Source heads are then discarded and a NEW target head is added.
Fine-tuning uses 10/25/50/100% of TARGET TRAINING PARTICIPANTS (nested subsets, frozen splits).
Conditions: scratch, pretrained_frozen, pretrained_finetuned, coral_finetuned, dann_finetuned, full_reference."""
import argparse, json, sys, logging, datetime, time, copy
from pathlib import Path
import numpy as np, yaml, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.tabular_v2 import TabularEncoder, Head, EncHead
from src.models.coral_dann_v2 import coral_fit, coral_apply, DomainDiscriminator
from src.models.train_v2 import seed_all
from src.evaluation.metrics_v2 import compute_all, choose_threshold

DATASETS = ["neuma", "restaurant_logo", "ds007406"]
FRACTIONS = ["10pct", "25pct", "50pct", "100pct"]
CONDITIONS = ["scratch", "pretrained_frozen", "pretrained_finetuned", "coral_finetuned", "dann_finetuned", "full_reference"]


def load_features(root, name, feat):
    f = np.load(root / name / f"{name}_features_{feat}.npz", allow_pickle=True)
    return {"X": f["X"].astype(np.float32), "y": f["y"].astype(int),
            "sid": f["subject_ids"].astype(str), "tid": f["trial_ids"].astype(str)}


def pretrain_shared(src, device, seed, epochs, patience, in_dim=36):
    """Shared encoder + one head per source dataset. Task-balanced: equal batch from each source per step."""
    seed_all(seed)
    enc = TabularEncoder(in_dim).to(device)
    heads = {n: Head(enc.emb_dim).to(device) for n in src}
    params = list(enc.parameters()) + [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    T, crits, V = {}, {}, {}
    g = torch.Generator().manual_seed(seed)
    for n, d in src.items():
        rng = np.random.default_rng(seed + 17)
        nv = max(2, int(0.15 * len(d["y"])))
        vi = rng.choice(len(d["y"]), nv, replace=False)
        ti = np.setdiff1d(np.arange(len(d["y"])), vi)
        T[n] = (torch.tensor(d["X"][ti]).to(device), torch.tensor(d["y"][ti], dtype=torch.long).to(device))
        V[n] = (torch.tensor(d["X"][vi]).to(device), torch.tensor(d["y"][vi], dtype=torch.long).to(device))
        cnt = np.bincount(d["y"][ti], minlength=2)
        crits[n] = nn.CrossEntropyLoss(weight=torch.tensor(len(ti) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device))
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        enc.train(); [h.train() for h in heads.values()]
        losses = []
        for n, (Xn, yn) in T.items():
            idx = torch.randperm(len(yn), generator=g)[:64]
            if len(idx) < 2:
                continue
            losses.append(crits[n](heads[n](enc(Xn[idx])), yn[idx]))
        if not losses:
            break
        opt.zero_grad(); (sum(losses) / len(losses)).backward(); opt.step()
        enc.eval(); [h.eval() for h in heads.values()]
        with torch.no_grad():
            vl = float(sum(crits[n](heads[n](enc(V[n][0])), V[n][1]) for n in V) / len(V))
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return state if state else {k: v.detach().clone() for k, v in enc.state_dict().items()}


def pretrain_dann(src, Xt_unlab, device, seed, epochs, patience, in_dim=36):
    seed_all(seed)
    enc = TabularEncoder(in_dim).to(device)
    heads = {n: Head(enc.emb_dim).to(device) for n in src}
    disc = DomainDiscriminator(enc.emb_dim, 2, 0.5).to(device)
    params = list(enc.parameters()) + [p for h in heads.values() for p in h.parameters()] + list(disc.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    dom_crit = nn.CrossEntropyLoss()
    Xt_t = torch.tensor(Xt_unlab).to(device)
    T, crits = {}, {}
    g = torch.Generator().manual_seed(seed)
    for n, d in src.items():
        T[n] = (torch.tensor(d["X"]).to(device), torch.tensor(d["y"], dtype=torch.long).to(device))
        cnt = np.bincount(d["y"], minlength=2)
        crits[n] = nn.CrossEntropyLoss(weight=torch.tensor(len(d["y"]) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device))
    for ep in range(epochs):
        enc.train(); [h.train() for h in heads.values()]; disc.train()
        losses = []
        for n, (Xn, yn) in T.items():
            idx = torch.randperm(len(yn), generator=g)[:64]
            if len(idx) < 2:
                continue
            zs = enc(Xn[idx])
            losses.append(crits[n](heads[n](zs), yn[idx]))
            k = min(len(idx), len(Xt_t))
            zt = enc(Xt_t[torch.randperm(len(Xt_t), generator=g)[:k]])
            losses.append(0.5 * (dom_crit(disc(zs), torch.zeros(len(zs), dtype=torch.long, device=device))
                                 + dom_crit(disc(zt), torch.ones(len(zt), dtype=torch.long, device=device))))
        if not losses:
            break
        opt.zero_grad(); (sum(losses) / len(losses)).backward(); opt.step()
    return {k: v.detach().clone() for k, v in enc.state_dict().items()}


def finetune(enc_state, Xtr, ytr, Xva, yva, device, seed, epochs, patience, freeze, in_dim=36):
    seed_all(seed)
    enc = TabularEncoder(in_dim).to(device)
    if enc_state is not None:
        enc.load_state_dict(enc_state)
    if freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
    head = Head(enc.emb_dim).to(device)
    params = list(head.parameters()) + ([] if freeze else list(enc.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=2)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device))
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva).to(device); yv = torch.tensor(yva, dtype=torch.long).to(device)
    best, state, bad, g = np.inf, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        enc.train() if not freeze else enc.eval()
        head.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(); crit(head(enc(Xt[idx].to(device))), yt[idx].to(device)).backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            vl = float(crit(head(enc(Xv)), yv))
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
    m = EncHead(enc, head)
    with torch.no_grad():
        p_tr = torch.softmax(m(Xt.to(device)), 1)[:, 1].cpu().numpy()
    return m, p_tr


def predict(m, X, device):
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32).to(device)), 1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--features", default="v2")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pretrain-epochs", type=int, default=300)
    ap.add_argument("--finetune-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    root = ROOT / (cfg["output_root"] + ("_smoke" if a.smoke else ""))
    splits_dir = root / "splits"
    suffix = "" if a.features == "v2" else "_" + a.features.replace("_v2", "")
    out = ROOT / a.output_dir
    pred_dir = out / "predictions" / (("exp4_transfer_smoke" if a.smoke else "exp4_transfer") + suffix)
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / f"07_transfer_{ts}.log"), logging.StreamHandler()])
    log = logging.getLogger("e4")
    device = torch.device(f"cuda:{a.gpu if a.gpu < torch.cuda.device_count() else 0}" if torch.cuda.is_available() else "cpu")
    pt_ep = 5 if a.smoke else a.pretrain_epochs
    ft_ep = 3 if a.smoke else a.finetune_epochs
    pat = 2 if a.smoke else a.patience
    seeds = a.seeds[:1] if a.smoke else a.seeds
    fracs = ["10pct", "100pct"] if a.smoke else FRACTIONS
    conds = ["scratch", "pretrained_finetuned"] if a.smoke else CONDITIONS
    data = {n: load_features(root, n, a.features) for n in DATASETS}
    log.info(f"device={device} features={a.features} seeds={seeds}")
    quick = []
    for target in DATASETS:
        sources = [s for s in DATASETS if s != target]
        sp_all = json.load(open(splits_dir / f"transfer_target_{target}.json"))["by_seed"]
        d = data[target]
        for seed in seeds:
            sp = sp_all[str(seed)]
            te = np.isin(d["sid"], sp["test"])
            Xte, yte = d["X"][te], d["y"][te]
            tgt100 = np.isin(d["sid"], sp["fractions"]["100pct"]["train"])
            src = {s: data[s] for s in sources}
            t0 = time.time()
            enc_plain = pretrain_shared(src, device, seed, pt_ep, pat, d["X"].shape[1])
            enc_dann = pretrain_dann(src, d["X"][tgt100], device, seed, min(pt_ep, 100), pat, d["X"].shape[1])
            Xs_all = np.concatenate([data[s]["X"] for s in sources])
            sc_src = StandardScaler().fit(Xs_all)
            A = coral_fit(sc_src.transform(Xs_all), sc_src.transform(d["X"][tgt100]))
            src_coral = {s: {"X": coral_apply(sc_src.transform(data[s]["X"]), A), "y": data[s]["y"]} for s in sources}
            enc_coral = pretrain_shared(src_coral, device, seed, pt_ep, pat, d["X"].shape[1])
            log.info(f"[{target} seed={seed}] pretraining done ({time.time()-t0:.0f}s)")
            for frac in fracs:
                fr = sp["fractions"][frac]
                trm = np.isin(d["sid"], fr["train"]); vam = np.isin(d["sid"], fr["val"])
                if not trm.any():
                    continue
                Xtr, ytr = d["X"][trm], d["y"][trm]
                Xva = d["X"][vam] if vam.any() else Xtr[:2]
                yva = d["y"][vam] if vam.any() else ytr[:2]
                if len(np.unique(ytr)) < 2:
                    log.warning(f"  {frac}: single-class train, skip"); continue
                for cond in conds:
                    fn = pred_dir / f"{target}__seed{seed}__{frac}__{cond}.csv"
                    if fn.exists():
                        continue
                    t1 = time.time()
                    Xtr_u, ytr_u, Xva_u, yva_u, Xte_u = Xtr, ytr, Xva, yva, Xte
                    if cond == "scratch":
                        m, p_tr = finetune(None, Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, False, d["X"].shape[1])
                    elif cond == "pretrained_frozen":
                        m, p_tr = finetune(copy.deepcopy(enc_plain), Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, True, d["X"].shape[1])
                    elif cond == "pretrained_finetuned":
                        m, p_tr = finetune(copy.deepcopy(enc_plain), Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, False, d["X"].shape[1])
                    elif cond == "coral_finetuned":
                        Xtr_u = coral_apply(sc_src.transform(Xtr), A); Xva_u = coral_apply(sc_src.transform(Xva), A); Xte_u = coral_apply(sc_src.transform(Xte), A)
                        m, p_tr = finetune(copy.deepcopy(enc_coral), Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, False, d["X"].shape[1])
                    elif cond == "dann_finetuned":
                        m, p_tr = finetune(copy.deepcopy(enc_dann), Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, False, d["X"].shape[1])
                    else:
                        f100 = sp["fractions"]["100pct"]
                        t100 = np.isin(d["sid"], f100["train"]); v100 = np.isin(d["sid"], f100["val"])
                        Xtr_u, ytr_u = d["X"][t100], d["y"][t100]
                        Xva_u = d["X"][v100] if v100.any() else Xtr_u[:2]
                        yva_u = d["y"][v100] if v100.any() else ytr_u[:2]
                        m, p_tr = finetune(None, Xtr_u, ytr_u, Xva_u, yva_u, device, seed, ft_ep, pat, False, d["X"].shape[1])
                    p = predict(m, Xte_u, device)
                    thr = choose_threshold(ytr_u, p_tr)
                    pd.DataFrame({"target": target, "seed": seed, "fraction": frac, "condition": cond,
                                  "n_train_participants": len(fr["train"]) if cond != "full_reference" else len(sp["fractions"]["100pct"]["train"]),
                                  "trial_id": d["tid"][te], "subject_id": d["sid"][te], "y_true": yte,
                                  "y_prob": p, "threshold_train": thr, "y_pred": (p >= thr).astype(int)}).to_csv(fn, index=False)
                    mt = compute_all(yte, p, thr)
                    quick.append({"target": target, "seed": seed, "fraction": frac, "condition": cond,
                                  "mcc": mt["mcc"], "roc_auc": mt["roc_auc"], "bal_acc": mt["balanced_accuracy"]})
                    log.info(f"  {target:16s} seed={seed} {frac:7s} {cond:22s} MCC={mt['mcc']:+.3f} AUC={mt['roc_auc']:.3f} ({time.time()-t1:.0f}s)")
    if quick:
        q = pd.DataFrame(quick)
        q.to_csv(pred_dir / f"quicklook_{ts}.csv", index=False)
        log.info("\n" + q.groupby(["target", "fraction", "condition"])[["mcc", "roc_auc"]].mean().round(3).to_string())
    log.info(f"[done] {pred_dir}")


if __name__ == "__main__":
    main()
