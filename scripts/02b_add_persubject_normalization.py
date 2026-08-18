#!/usr/bin/env python
"""Phase 2b: add a per-participant normalized feature set.
Each participant's features are z-scored using ONLY that participant's own trials.
No labels are used -> not label leakage. It IS transductive (test participants' unlabelled
trials are used to compute their own statistics); this is recorded in the manifest and must
be declared in the manuscript. The original (un-normalized) feature files are left untouched.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parent.parent

def per_subject_z(X, sid, eps=1e-12):
    Z = X.copy().astype(np.float32); stats = {}
    for s in np.unique(sid):
        m = sid == s
        mu = X[m].mean(0); sd = X[m].std(0)
        Z[m] = (X[m] - mu) / (sd + eps)
        stats[str(s)] = {"n_trials": int(m.sum()), "mean_of_sd": float(sd.mean())}
    return Z, stats

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT/a.config))
    root = ROOT/(cfg["output_root"] + ("_smoke" if a.smoke else ""))
    report = {"method": "per-participant z-score over that participant's own trials",
              "labels_used": False, "transductive": True,
              "note": "test participants' unlabelled trials contribute to their own mean/sd", "datasets": {}}
    for name in ["neuma", "restaurant_logo", "ds007406"]:
        src = root/name/f"{name}_features_v2.npz"
        if not src.exists():
            print(f"[{name}] missing {src}"); continue
        d = np.load(src, allow_pickle=True)
        X, sid = d["X"].astype(np.float32), d["subject_ids"].astype(str)
        Z, stats = per_subject_z(X, sid)
        out = root/name/f"{name}_features_persubj_v2.npz"
        np.savez_compressed(out, X=Z, X_raw=X, X_abs=d["X_abs"], y=d["y"], subject_ids=sid,
                            trial_ids=d["trial_ids"], feature_names=d["feature_names"], dataset=name)
        report["datasets"][name] = {"n_trials": int(len(X)), "n_subjects": int(len(np.unique(sid))),
                                    "finite": bool(np.isfinite(Z).all()),
                                    "between_subject_var_before": float(np.var([X[sid == s].mean(0) for s in np.unique(sid)], axis=0).mean()),
                                    "between_subject_var_after": float(np.var([Z[sid == s].mean(0) for s in np.unique(sid)], axis=0).mean())}
        r = report["datasets"][name]
        print(f"[{name}] {r['n_trials']} trials, {r['n_subjects']} subjects, finite={r['finite']}, "
              f"between-subject variance {r['between_subject_var_before']:.4f} -> {r['between_subject_var_after']:.6f}")
    json.dump(report, open(root/"persubj_normalization_report.json", "w"), indent=1)
    print(f"[ok] wrote *_features_persubj_v2.npz and {root}/persubj_normalization_report.json")

if __name__ == "__main__":
    main()
