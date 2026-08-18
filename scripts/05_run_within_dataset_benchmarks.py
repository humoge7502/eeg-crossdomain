#!/usr/bin/env python
"""Experiment 2 — correct within-dataset benchmark on frozen subject-wise folds.
Models: logreg, svm_rbf, xgboost, mlp (36-d features); eegnet, deepconvnet (windows->trial). Reuses Exp1 subject-wise predictions where present."""
import argparse, json, sys, logging, datetime, time, shutil, importlib
from pathlib import Path
import numpy as np, yaml, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from src.models.tabular_v2 import TabularEncoder, Head, EncHead
from src.models.train_v2 import seed_all
from src.evaluation.metrics_v2 import compute_all, choose_threshold, choose_threshold, choose_threshold
E1 = importlib.import_module("04_run_leakage_audit")

def train_mlp(Xtr, ytr, Xva, yva, device, seed, max_epochs, patience):
    seed_all(seed); model = EncHead(TabularEncoder(Xtr.shape[1]), Head()).to(device)
    cnt = np.bincount(ytr, minlength=2); w = torch.tensor(len(ytr) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long); Xv, yv = torch.tensor(Xva, dtype=torch.float32).to(device), torch.tensor(yva, dtype=torch.long).to(device)
    best, state, bad, g = np.inf, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2: continue
            opt.zero_grad(); loss = crit(model(Xt[idx].to(device)), yt[idx].to(device)); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(crit(model(Xv), yv)) if len(yva) else 0.0
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if state: model.load_state_dict(state)
    model.eval(); return model, {"epochs_run": ep + 1, "best_val_loss": float(best)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml"); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--output-dir", default="outputs"); ap.add_argument("--datasets", nargs="+", default=["neuma", "restaurant_logo", "ds007406"])
    ap.add_argument("--models", nargs="+", default=["logreg", "svm_rbf", "xgboost", "mlp", "eegnet", "deepconvnet"]); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(); cfg = yaml.safe_load(open(ROOT/a.config)); root = ROOT/(cfg["output_root"] + ("_smoke" if a.smoke else "")); splits = root/"splits"
    seeds = [a.seed] if a.seed is not None else a.seeds; out = ROOT/a.output_dir; tag = "exp2_within_smoke" if a.smoke else "exp2_within"; pred_dir = out/"predictions"/tag; pred_dir.mkdir(parents=True, exist_ok=True)
    e1_dir = out/"predictions"/("exp1_leakage_smoke" if a.smoke else "exp1_leakage"); ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); (out/"logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out/"logs"/f"05_within_{ts}.log"), logging.StreamHandler()]); log = logging.getLogger("e2")
    device = torch.device(f"cuda:{a.gpu if a.gpu < torch.cuda.device_count() else 0}" if torch.cuda.is_available() else "cpu"); json.dump({"args": vars(a), "config": cfg, "timestamp": ts}, open(pred_dir/f"run_config_{ts}.json", "w"), indent=1, default=str)
    from xgboost import XGBClassifier
    quick = []
    for name in a.datasets:
        d = E1.load(root, name); SW = json.load(open(splits/f"within_subjectwise_{name}.json"))["folds_by_seed"]
        for seed in seeds:
            for fold in SW[str(seed)][: (2 if a.smoke else None)]:
                fi = fold["fold"]; tr, va, te, wtr, wva, wte = E1.masks_from_fold(d, fold, "subjectwise")
                for model_name in a.models:
                    fn = pred_dir/f"{name}__subjectwise__{model_name}__seed{seed}__fold{fi}.csv"
                    if fn.exists(): continue
                    src = e1_dir/f"{name}__subjectwise__{model_name}__seed{seed}__fold{fi}.csv"
                    if src.exists(): shutil.copy(src, fn); log.info(f"{name} {model_name} seed={seed} fold={fi}: reused Exp1 prediction"); continue
                    if model_name in ("logreg", "svm_rbf", "eegnet", "deepconvnet") and not a.smoke: continue   # produced by Exp1; re-run 05 after Exp1 finishes to copy them
                    t0 = time.time(); info = {}
                    if model_name in ("logreg", "svm_rbf"): p, info = E1.run_classical(model_name, d, tr, va, te, d["y"], seed); tids = d["tid"][te]
                    elif model_name == "xgboost":
                        sc = StandardScaler().fit(d["X"][tr]); pos = max(1, int(d["y"][tr].sum())); spw = (len(d["y"][tr]) - pos) / pos
                        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, random_state=seed, n_jobs=8, eval_metric="logloss", early_stopping_rounds=30 if va.any() else None)
                        if va.any(): m.fit(sc.transform(d["X"][tr]), d["y"][tr], eval_set=[(sc.transform(d["X"][va]), d["y"][va])], verbose=False); info = {"best_iteration": int(m.best_iteration)}
                        else: m.fit(sc.transform(d["X"][tr]), d["y"][tr])
                        p = m.predict_proba(sc.transform(d["X"][te]))[:, 1]; tids = d["tid"][te]; info = {**info, "p_train": m.predict_proba(sc.transform(d["X"][tr]))[:, 1]}; info = {**info, "p_train": m.predict_proba(sc.transform(d["X"][tr]))[:, 1]}; info = {**info, "p_train": m.predict_proba(sc.transform(d["X"][tr]))[:, 1]}
                    elif model_name == "mlp":
                        sc = StandardScaler().fit(d["X"][tr]); Xva = sc.transform(d["X"][va]) if va.any() else sc.transform(d["X"][tr])[:2]; yva = d["y"][va] if va.any() else d["y"][tr][:2]
                        m, info = train_mlp(sc.transform(d["X"][tr]), d["y"][tr], Xva, yva, device, seed, 3 if a.smoke else 200, 2 if a.smoke else 20)
                        with torch.no_grad():
                            p = torch.softmax(m(torch.tensor(sc.transform(d["X"][te]), dtype=torch.float32).to(device)), 1)[:, 1].cpu().numpy()
                            info = {**info, "p_train": torch.softmax(m(torch.tensor(sc.transform(d["X"][tr]), dtype=torch.float32).to(device)), 1)[:, 1].cpu().numpy()}
                        tids = d["tid"][te]
                    else: tids, p, info = E1.run_deep(model_name, d, wtr, wva, wte, d["wy"], seed, device, a.smoke)
                    pos = {t: i for i, t in enumerate(d["tid"])}; idx = np.array([pos[t] for t in tids])
                    rows = pd.DataFrame({"dataset": name, "protocol": "subjectwise", "model": model_name, "seed": seed, "fold": fi, "subject_id": d["sid"][idx], "trial_id": tids, "y_true": d["y"][idx], "y_prob": p, "y_pred": (p >= 0.5).astype(int),
                                         "n_train_subjects": len(np.unique(d["sid"][tr])), "n_test_subjects": len(np.unique(d["sid"][te]))})
                    p_train = info.pop("p_train", None); train_tids = info.pop("train_tids", d["tid"][tr] if p_train is not None else None)
                    if p_train is not None:
                        tpos = np.array([pos[t] for t in train_tids]); ytrn = y_tr[tpos] if "y_tr" in dir() else d["y"][tpos]
                        thr = choose_threshold(ytrn, p_train); pd.DataFrame({"trial_id": train_tids, "subject_id": d["sid"][tpos], "y_true": ytrn, "y_prob": p_train}).to_csv(str(fn).replace(".csv", "__train.csv"), index=False)
                    else: thr = 0.5
                    rows["threshold_train"] = thr; rows["y_pred_thr"] = (rows.y_prob >= thr).astype(int)
                    rows.to_csv(fn, index=False); mt = compute_all(rows.y_true.values, rows.y_prob.values, thr)
                    quick.append({"dataset": name, "model": model_name, "seed": seed, "fold": fi, "mcc": mt["mcc"], "roc_auc": mt["roc_auc"], "bal_acc": mt["balanced_accuracy"], **{f"train_{k}": v for k, v in info.items()}})
                    log.info(f"{name:15s} {model_name:11s} seed={seed} fold={fi} n_test={mt['n']:4d} MCC={mt['mcc']:+.3f} AUC={mt['roc_auc']:.3f} balAcc={mt['balanced_accuracy']:.3f} ({time.time()-t0:.0f}s)")
    if quick:
        q = pd.DataFrame(quick); q.to_csv(pred_dir/f"quicklook_{ts}.csv", index=False); log.info("\n" + q.groupby(["dataset", "model"])[["mcc", "roc_auc", "bal_acc"]].mean().round(3).to_string())
    log.info(f"[done] {pred_dir}")

if __name__ == "__main__": main()
