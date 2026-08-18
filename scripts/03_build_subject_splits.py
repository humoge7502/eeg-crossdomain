#!/usr/bin/env python
"""Phase 3: build and freeze participant-level splits for all experiments.
Outputs (data/processed_v2/splits/): 
  within_subjectwise_<ds>.json   : K stratified-group folds (by subject); each fold lists train/val/test subject IDs
  within_epochwise_<ds>.json     : K epoch-wise stratified folds (LEAKY, leakage audit only) — trial_ids
  transfer_target_<ds>.json      : per seed, held-out test participants + train-participant subsets at 10/25/50/100%
  splits_manifest.csv            : one row per (dataset, protocol, seed, fold, split, subject)
Splitting is by participant everywhere except the explicitly-named epoch-wise leakage protocol.
"""
import argparse, json, csv, sys, random
from pathlib import Path
import numpy as np, yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
ROOT = Path(__file__).resolve().parent.parent

def subj_table(y, sid):
    subs = np.array(sorted(set(sid))); pos_rate = np.array([y[sid == s].mean() for s in subs]); n = np.array([(sid == s).sum() for s in subs])
    return subs, pos_rate, n

def group_folds(y, sid, k, seed):
    """StratifiedGroupKFold over trials with subject groups; returns list of (train_subjects, test_subjects)."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed); out = []
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups=sid):
        out.append((sorted(set(sid[tr])), sorted(set(sid[te]))))
    return out

def carve_val(train_subs, y, sid, frac, seed):
    """Hold out ~frac of TRAIN subjects for validation/early stopping (subject-level, stratified by subject pos-rate)."""
    rng = np.random.default_rng(seed); subs = np.array(train_subs)
    if len(subs) < 4: return list(subs), []
    pr = np.array([y[sid == s].mean() for s in subs]); order = np.argsort(pr, kind="stable")
    n_val = max(1, int(round(frac * len(subs)))); pick = set(rng.choice(order, size=n_val, replace=False).tolist()) if n_val < len(subs) else set()
    val = sorted(subs[list(pick)]); tr = sorted(s for s in subs if s not in val); return tr, val

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=None); ap.add_argument("--k", type=int, default=5); ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.00])
    a = ap.parse_args(); cfg = yaml.safe_load(open(ROOT/a.config))
    root = ROOT/(cfg["output_root"] + ("_smoke" if a.smoke else "")); out = Path(a.output_dir) if a.output_dir else root/"splits"; out.mkdir(parents=True, exist_ok=True)
    seeds = [a.seed] if a.seed is not None else a.seeds; rows = []; summary = {}
    for name in ["neuma", "restaurant_logo", "ds007406"]:
        f = root/name/f"{name}_features_v2.npz"
        if not f.exists(): print(f"[{name}] missing {f}, skip"); continue
        d = np.load(f, allow_pickle=True); y = d["y"].astype(int); sid = d["subject_ids"].astype(str); tid = d["trial_ids"].astype(str)
        subs, pr, n = subj_table(y, sid); k = min(a.k, len(subs)); summary[name] = {"n_subjects": int(len(subs)), "n_trials": int(len(y)), "k": k, "protocols": {}}
        # 1. subject-wise K folds (one per seed) with subject-level validation carved from train
        sw = {}
        for seed in seeds:
            folds = []
            for fi, (tr, te) in enumerate(group_folds(y, sid, k, seed)):
                tr2, val = carve_val(tr, y, sid, a.val_frac, seed * 1000 + fi)
                assert not (set(tr2) & set(te)) and not (set(val) & set(te)) and not (set(tr2) & set(val))
                folds.append({"fold": fi, "train": tr2, "val": val, "test": te, "n_test_trials": int(np.isin(sid, te).sum()), "test_pos": int(y[np.isin(sid, te)].sum())})
                for split, ss in (("train", tr2), ("val", val), ("test", te)):
                    for s in ss: rows.append({"dataset": name, "protocol": "within_subjectwise", "seed": seed, "fold": fi, "split": split, "subject_id": s})
            sw[str(seed)] = folds
        json.dump({"dataset": name, "protocol": "within_subjectwise", "k": k, "val_frac": a.val_frac, "seeds": seeds, "folds_by_seed": sw}, open(out/f"within_subjectwise_{name}.json", "w"), indent=1)
        # 2. epoch-wise K folds (LEAKY; leakage audit only). Same k, same seeds; splitting unit = trial.
        ew = {}
        for seed in seeds:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed); folds = []
            for fi, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                # val carved epoch-wise from train, so the leak is exactly the same in early stopping
                rng = np.random.default_rng(seed * 1000 + fi); tr = rng.permutation(tr); nv = max(1, int(round(a.val_frac * len(tr))))
                folds.append({"fold": fi, "train_trial_ids": tid[tr[nv:]].tolist(), "val_trial_ids": tid[tr[:nv]].tolist(), "test_trial_ids": tid[te].tolist(),
                              "n_test_subjects_also_in_train": int(len(set(sid[te]) & set(sid[tr])))})
            ew[str(seed)] = folds
        json.dump({"dataset": name, "protocol": "within_epochwise_LEAKY", "k": k, "seeds": seeds, "folds_by_seed": ew, "warning": "trials of the same participant appear in train and test by design; use only for the leakage audit"}, open(out/f"within_epochwise_{name}.json", "w"), indent=1)
        # 3. transfer target splits: per seed, one held-out test set of participants (~1/k) and nested train fractions (participants!)
        tf = {}
        for seed in seeds:
            tr_all, te = group_folds(y, sid, k, seed)[0]; tr_all = list(tr_all)
            rng = np.random.default_rng(seed + 7919); order = rng.permutation(tr_all).tolist(); fr = {}
            for frac in a.fractions:
                m = max(2, int(round(frac * len(order)))) if frac < 1 else len(order); sub = sorted(order[:m])   # nested: smaller subsets ⊂ larger
                tr2, val = carve_val(sub, y, sid, a.val_frac, seed * 31 + int(frac * 100))
                fr[f"{int(frac*100)}pct"] = {"train": tr2, "val": val, "n_train_participants": len(sub)}
                for s in tr2: rows.append({"dataset": name, "protocol": f"transfer_{int(frac*100)}pct", "seed": seed, "fold": 0, "split": "train", "subject_id": s})
                for s in val: rows.append({"dataset": name, "protocol": f"transfer_{int(frac*100)}pct", "seed": seed, "fold": 0, "split": "val", "subject_id": s})
            for s in te: rows.append({"dataset": name, "protocol": "transfer_test", "seed": seed, "fold": 0, "split": "test", "subject_id": s})
            tf[str(seed)] = {"test": te, "fractions": fr}
        json.dump({"dataset": name, "protocol": "transfer_target", "seeds": seeds, "fractions": a.fractions, "by_seed": tf}, open(out/f"transfer_target_{name}.json", "w"), indent=1)
        summary[name]["protocols"] = {"within_subjectwise": {"k": k, "test_subjects_per_fold": [len(fd["test"]) for fd in sw[str(seeds[0])]]}, "within_epochwise": {"k": k},
                                     "transfer": {"n_test_participants": len(tf[str(seeds[0])]["test"]), "train_participants_by_fraction": {kk: v["n_train_participants"] for kk, v in tf[str(seeds[0])]["fractions"].items()}}}
        print(f"[{name}] subjects={len(subs)} trials={len(y)} k={k} | subject-wise test sizes {summary[name]['protocols']['within_subjectwise']['test_subjects_per_fold']} | transfer test={summary[name]['protocols']['transfer']['n_test_participants']} train@fractions={summary[name]['protocols']['transfer']['train_participants_by_fraction']}")
    with open(out/"splits_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "protocol", "seed", "fold", "split", "subject_id"]); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump({"seeds": seeds, "k": a.k, "val_frac": a.val_frac, "fractions": a.fractions, "summary": summary}, open(out/"splits_summary.json", "w"), indent=1)
    print(f"[ok] wrote splits to {out}")

if __name__ == "__main__": main()
