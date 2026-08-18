#!/usr/bin/env python
"""Experiment 1 — leakage audit. Subject-wise vs epoch-wise splits, plus permuted-label control.
Models: logreg, svm_rbf (36-d features); eegnet, deepconvnet (1s/250Hz windows, trial-aggregated).
Decision thresholds chosen on training subjects only. Saves test + train predictions."""
import argparse, json, csv, sys, logging, datetime, time
from pathlib import Path
import numpy as np, yaml, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.deep_v2 import EEGNet, DeepConvNet
from src.models.train_v2 import ZScore, train_torch, predict_proba, aggregate_windows_to_trials, seed_all
from src.evaluation.metrics_v2 import compute_all, choose_threshold


def load(root, name, feat="v2"):
    f = np.load(root/name/f"{name}_features_{feat}.npz", allow_pickle=True)
    w = np.load(root/name/f"{name}_windows_common_v2.npz", allow_pickle=True)
    return {"X": f["X"].astype(np.float32), "y": f["y"].astype(int),
            "sid": f["subject_ids"].astype(str), "tid": f["trial_ids"].astype(str),
            "W": w["X"].astype(np.float32), "wy": w["y"].astype(int),
            "wsid": w["subject_ids"].astype(str), "wtid": w["trial_ids"].astype(str)}


def masks_from_fold(d, fold, protocol):
    if protocol == "subjectwise":
        tr = np.isin(d["sid"], fold["train"]); va = np.isin(d["sid"], fold["val"]); te = np.isin(d["sid"], fold["test"])
        wtr = np.isin(d["wsid"], fold["train"]); wva = np.isin(d["wsid"], fold["val"]); wte = np.isin(d["wsid"], fold["test"])
    else:
        tr = np.isin(d["tid"], fold["train_trial_ids"]); va = np.isin(d["tid"], fold["val_trial_ids"]); te = np.isin(d["tid"], fold["test_trial_ids"])
        wtr = np.isin(d["wtid"], fold["train_trial_ids"]); wva = np.isin(d["wtid"], fold["val_trial_ids"]); wte = np.isin(d["wtid"], fold["test_trial_ids"])
    return tr, va, te, wtr, wva, wte


def permute_within_subject(y, sid, rng):
    y2 = y.copy()
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        y2[m] = y[rng.permutation(m)]
    return y2


def run_classical(model_name, d, tr, va, te, y_use, seed):
    sc = StandardScaler().fit(d["X"][tr])
    Xtr, Xte = sc.transform(d["X"][tr]), sc.transform(d["X"][te])
    if model_name == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)
    else:
        n_min = int(np.bincount(y_use[tr], minlength=2).min())
        cv = 3 if n_min >= 3 else 2
        m = CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
            method="sigmoid", cv=cv, ensemble=False)
    m.fit(Xtr, y_use[tr])
    p_train = m.predict_proba(Xtr)[:, 1]
    p_test = m.predict_proba(Xte)[:, 1]
    return p_test, p_train, d["tid"][tr]


def run_deep(model_name, d, wtr, wva, wte, wy_use, seed, device, smoke):
    z = ZScore().fit(d["W"][wtr])
    Wtr = z.transform(d["W"][wtr]); Wva = z.transform(d["W"][wva]); Wte = z.transform(d["W"][wte])
    n_ch, n_t = d["W"].shape[1], d["W"].shape[2]
    seed_all(seed)
    net = EEGNet(n_ch, n_t) if model_name == "eegnet" else DeepConvNet(n_ch, n_t)
    yva = wy_use[wva] if wva.any() else wy_use[wtr][:1]
    Wva_fit = Wva if wva.any() else Wtr[:1]
    net, info = train_torch(net, Wtr, wy_use[wtr], Wva_fit, yva, device,
                            max_epochs=3 if smoke else 100, patience=2 if smoke else 15, seed=seed)
    p_win_te = predict_proba(net, Wte, device)
    te_tids, p_test = aggregate_windows_to_trials(p_win_te, d["wtid"][wte])
    p_win_tr = predict_proba(net, Wtr, device)
    tr_tids, p_train = aggregate_windows_to_trials(p_win_tr, d["wtid"][wtr])
    return te_tids, p_test, tr_tids, p_train, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--datasets", nargs="+", default=["neuma", "restaurant_logo", "ds007406"])
    ap.add_argument("--models", nargs="+", default=["logreg", "svm_rbf", "eegnet", "deepconvnet"])
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--features", default="v2", help="v2 (raw) or persubj_v2 (per-participant z-scored)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT/a.config))
    root = ROOT/(cfg["output_root"] + ("_smoke" if a.smoke else ""))
    splits = root/"splits"
    seeds = [a.seed] if a.seed is not None else a.seeds
    out = ROOT/a.output_dir
    suffix = "" if a.features == "v2" else "_" + a.features.replace("_v2", "")
    tag = ("exp1_leakage_smoke" if a.smoke else "exp1_leakage") + suffix
    pred_dir = out/"predictions"/tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out/"logs").mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out/"logs"/f"04_leakage_{ts}.log"),
                                  logging.StreamHandler()])
    log = logging.getLogger("e1")
    device = torch.device(f"cuda:{a.gpu if a.gpu < torch.cuda.device_count() else 0}"
                          if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} seeds={seeds} smoke={a.smoke}")
    json.dump({"args": vars(a), "config": cfg, "timestamp": ts, "device": str(device)},
              open(pred_dir/f"run_config_{ts}.json", "w"), indent=1, default=str)
    quick = []
    for name in a.datasets:
        d = load(root, name, a.features)
        if not (splits/f"within_subjectwise_{name}.json").exists():
            log.warning(f"no splits for {name}"); continue
        SW = json.load(open(splits/f"within_subjectwise_{name}.json"))["folds_by_seed"]
        EW = json.load(open(splits/f"within_epochwise_{name}.json"))["folds_by_seed"]
        tid_pos = {t: i for i, t in enumerate(d["tid"])}
        for seed in seeds:
            for protocol, folds in (("subjectwise", SW[str(seed)]),
                                    ("epochwise", EW[str(seed)]),
                                    ("permuted_subjectwise", SW[str(seed)])):
                for fold in folds[:(2 if a.smoke else None)]:
                    fi = fold["fold"]
                    for model_name in a.models:
                        fn = pred_dir/f"{name}__{protocol}__{model_name}__seed{seed}__fold{fi}.csv"
                        if a.resume and fn.exists():
                            continue
                        t0 = time.time()
                        base = "subjectwise" if protocol != "epochwise" else "epochwise"
                        tr, va, te, wtr, wva, wte = masks_from_fold(d, fold, base)
                        y_use = d["y"].copy()
                        wy_use = d["wy"].copy()
                        rng = np.random.default_rng(seed * 100003 + fi)
                        if protocol == "permuted_subjectwise":
                            y_use = permute_within_subject(d["y"], d["sid"], rng)
                            lm = {t: yy for t, yy in zip(d["tid"], y_use)}
                            wy_use = np.array([lm[t] for t in d["wtid"]])
                        if model_name in ("logreg", "svm_rbf"):
                            p_test, p_train, train_tids = run_classical(model_name, d, tr, va, te, y_use, seed)
                            te_tids = d["tid"][te]
                        else:
                            te_tids, p_test, train_tids, p_train, _ = run_deep(
                                model_name, d, wtr, wva, wte, wy_use, seed, device, a.smoke)
                        # threshold from training subjects only
                        tr_idx = np.array([tid_pos[t] for t in train_tids])
                        thr = choose_threshold(y_use[tr_idx], p_train)
                        # save train file
                        pd.DataFrame({"trial_id": train_tids, "subject_id": d["sid"][tr_idx],
                                      "y_true": y_use[tr_idx], "y_prob": p_train}
                                     ).to_csv(str(fn).replace(".csv", "__train.csv"), index=False)
                        # save test file
                        te_idx = np.array([tid_pos[t] for t in te_tids])
                        rows = pd.DataFrame({"dataset": name, "protocol": protocol, "model": model_name,
                                             "seed": seed, "fold": fi, "subject_id": d["sid"][te_idx],
                                             "trial_id": te_tids, "y_true": d["y"][te_idx],
                                             "y_prob": p_test, "threshold_train": thr,
                                             "y_pred": (p_test >= thr).astype(int),
                                             "n_train_subjects": int(np.isin(d["sid"], fold.get("train", [])).sum() > 0 and len(set(d["sid"][tr]))),
                                             "n_test_subjects": len(set(d["sid"][te_idx]))})
                        rows.to_csv(fn, index=False)
                        mt = compute_all(rows.y_true.values, rows.y_prob.values, thr)
                        quick.append({"dataset": name, "protocol": protocol, "model": model_name,
                                      "seed": seed, "fold": fi, "mcc": mt["mcc"],
                                      "roc_auc": mt["roc_auc"], "bal_acc": mt["balanced_accuracy"],
                                      "threshold": thr, "n_test": mt["n"]})
                        log.info(f"{name:15s} {protocol:20s} {model_name:11s} seed={seed} fold={fi} "
                                 f"n_test={mt['n']:4d} MCC={mt['mcc']:+.3f} AUC={mt['roc_auc']:.3f} "
                                 f"balAcc={mt['balanced_accuracy']:.3f} thr={thr:.3f} ({time.time()-t0:.0f}s)")
    if quick:
        q = pd.DataFrame(quick)
        q.to_csv(pred_dir/f"quicklook_{ts}.csv", index=False)
        log.info("\n" + q.groupby(["dataset", "protocol", "model"])[["mcc", "roc_auc", "bal_acc"]].mean().round(3).to_string())
    log.info(f"[done] predictions in {pred_dir}")


if __name__ == "__main__":
    main()
