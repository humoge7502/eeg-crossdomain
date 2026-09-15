#!/usr/bin/env python
"""Restaurant-Logo motor-confound control (PROJECT_STATUS.md item 2).

The keypress differs by class (A vs P -> Recognized vs NotRecognized) and epochs
are stimulus-locked over [0, 3] s, so post-keypress motor activity is inside the
analysis window. This script quantifies how much of the dataset's decodability
survives when the window is truncated to end BEFORE the median keypress.

Design (pre-registered here, one global cutoff, no per-trial RT leakage):
  * cutoff = median RT over ALL trials pooled across subjects (a single global
    constant; a per-trial window would smuggle RT into the features).
  * For every trial, rebuild the stimulus-locked epoch over [0, cutoff) s.
  * Recompute the same 36-d band-power features (src.features.bandpower_v2).
  * Within-subject LOSOCV, same protocol as the leakage audit (script 04):
    logreg + calibrated svm_rbf, scaler/weights from train subjects only,
    threshold chosen on train predictions only (choose_threshold). Headline
    metrics: kappa (calibrated probs) + ROC-AUC.
  * Family-level within-subject permutation control: N_PERM times, shuffle
    labels within each subject, re-run the whole LOSOCV with fast
    decision-function models, and compare mean-fold AUC. AUC is invariant to
    the monotone sigmoid calibration, so observed and null AUC use the same
    statistic (decision-function ranking). p = (1 + #{null >= obs}) / (N+1).

Outputs results/rl_motor_confound_report.md + .json. CPU-only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.features.bandpower_v2 import compute_features
from src.evaluation.metrics_v2 import compute_all, choose_threshold

SHARED = ["Fz", "Cz", "Pz", "F3", "F4", "O1", "O2"]  # the 7 shared channels, same as all other analyses (F3/F4 substituted for RL)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
TOTAL_RANGE = (1, 45)
EPOCH_TMAX = 3.0


def load_rl_epochs(native_path):
    d = np.load(native_path, allow_pickle=True)
    ch = [str(c) for c in d["ch_names"]]
    assert ch == SHARED, f"unexpected channels: {ch}"
    return {"X": d["X"].astype(np.float64), "y": d["y"].astype(int),
            "sid": d["subject_ids"].astype(str), "tid": d["trial_ids"].astype(str),
            "sfreq": float(d["sfreq"]), "ch_names": ch}


def rebuild_features(d, manifest_csv, cutoff_s):
    """Rebuild stimulus-locked epochs over [0, cutoff_s) and recompute the
    36-d features. Returns features plus the share of trials whose keypress
    falls entirely outside the truncated window (rt >= cutoff)."""
    sfreq = d["sfreq"]
    n_samp = int(round(cutoff_s * sfreq))
    man = pd.read_csv(manifest_csv).set_index("trial_id")
    m = man.loc[list(d["tid"])]  # raises if trial_ids ever drift apart
    ep_trunc = d["X"][:, :, :n_samp]
    X_rel, X_abs, feat_names, _ = compute_features(ep_trunc, sfreq, SHARED, BANDS, TOTAL_RANGE)
    clean_frac = float((m["rt_s"].values >= cutoff_s).mean())
    return {"X": X_rel, "y": d["y"], "sid": d["sid"], "tid": d["tid"],
            "clean_frac": clean_frac}


def _fit_predict(model_name, Xtr, ytr, Xte, seed):
    """Returns (p_test_calibrated, score_test, p_train_calibrated).
    score_test is the monotone-invariant ranking score (decision function)."""
    if model_name == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=1.0, random_state=seed)
        m.fit(Xtr, ytr)
        return (m.predict_proba(Xte)[:, 1], m.decision_function(Xte),
                m.predict_proba(Xtr)[:, 1])
    n_min = int(np.bincount(ytr, minlength=2).min())
    cv = 3 if n_min >= 3 else 2
    m = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=seed), method="sigmoid", cv=cv, ensemble=False)
    m.fit(Xtr, ytr)
    # decision function of the underlying (uncalibrated) SVM:
    cal_svcs = [c for c in m.calibrated_classifiers_]
    # CalibratedClassifierCV(ensemble=False) holds ONE fitted svc
    base = m.calibrated_classifiers_[0].estimator
    return (m.predict_proba(Xte)[:, 1], base.decision_function(Xte),
            m.predict_proba(Xtr)[:, 1])


def losocv(d, model_name, seed):
    """Within-subject LOSOCV as in scripts/04: scaler, class weights and
    threshold from train subjects only. Per-fold kappa (calibrated) + AUC
    (decision function; identical to calibrated AUC by monotonicity)."""
    X, y, sid = d["X"], d["y"], d["sid"]
    sids = np.unique(sid)
    kappas, aucs, folds = [], [], []
    for te_s in sids:
        te = sid == te_s
        tr = ~te
        if len(np.unique(y[tr])) < 2 or y[te].sum() in (0, int(te.sum())):
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        p_te, s_te, p_tr = _fit_predict(model_name, Xtr, y[tr], Xte, seed)
        thr = choose_threshold(y[tr], p_tr)
        kappas.append(compute_all(y[te], p_te, thr)["kappa"])
        aucs.append(roc_auc_score(y[te], s_te))
        folds.append(str(te_s))
    return {"kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "per_fold_kappa": [float(k) for k in kappas],
            "per_fold_auc": [float(a) for a in aucs], "folds": folds}


def _fast_score(model_name, Xtr, ytr, Xte, seed):
    """Uncalibrated model for permutation nulls (same model class)."""
    if model_name == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=1.0, random_state=seed)
    else:
        m = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                random_state=seed)
    m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def permute_within_subject(y, sid, rng):
    y2 = y.copy()
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        y2[idx] = y[rng.permutation(idx)]
    return y2


def permutation_p(d, model_name, n_perm, seed):
    """Family-level within-subject permutation test on mean-fold AUC,
    decision-function statistic for both observed and null."""
    X, y, sid = d["X"], d["y"], d["sid"]
    sids = np.unique(sid)

    def mean_auc(yy):
        vals = []
        for te_s in sids:
            te = sid == te_s
            tr = ~te
            if len(np.unique(yy[tr])) < 2 or yy[te].sum() in (0, int(te.sum())):
                continue
            sc = StandardScaler().fit(X[tr])
            s = _fast_score(model_name, sc.transform(X[tr]), yy[tr],
                            sc.transform(X[te]), seed)
            vals.append(roc_auc_score(yy[te], s))
        return float(np.mean(vals))

    obs = mean_auc(y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = mean_auc(permute_within_subject(y, sid, rng))
        if (i + 1) % 25 == 0:
            print(f"      perm {i + 1}/{n_perm}", flush=True)
    p = float((1 + int((null >= obs).sum())) / (n_perm + 1))
    return {"observed_mean_auc_df": obs, "p_perm": p,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "n_perm": n_perm}


def write_report(res, cutoff, n_trials, n_subj, median_rt, clean_frac, out_md, n_perm):
    lines = [
        "# Restaurant-Logo motor-confound control",
        "",
        "The keypress differs by class and epochs are stimulus-locked [0, 3] s,",
        "so the post-keypress motor period is inside the window. Control:",
        "truncate every epoch to [0, cutoff) s - strictly BEFORE the median",
        "keypress - recompute the same 36-d band-power features, and re-run the",
        "identical within-subject LOSOCV protocol.",
        "",
        f"- trials: {n_trials}, subjects: {n_subj}",
        f"- cutoff = median RT pooled across subjects: {median_rt:.3f} s",
        f"- trials whose keypress falls outside the truncated window (rt >= cutoff): {clean_frac * 100:.1f}%",
        f"- full window: [0, 3.0) s | truncated: [0, {cutoff:.3f}) s ({cutoff / EPOCH_TMAX * 100:.1f}%)",
        f"- permutation: {n_perm} family-level within-subject shuffles, p on mean-fold AUC",
        "",
        "| Window | Model | kappa (mean +/- std) | ROC-AUC | p_perm (AUC) |",
        "|---|---|---|---|---|",
    ]
    for proto in ("full", "truncated"):
        for model in ("logreg", "svm_rbf"):
            s = res[proto][model]["losocv"]
            p = res[proto][model]["perm"]["p_perm"]
            lines.append(
                f"| {proto} | {model} | {s['kappa_mean']:.4f} +/- {s['kappa_std']:.4f} | "
                f"{s['auc_mean']:.4f} | {p:.4f} |")
    lines += [
        "",
        "## Reading this table",
        "",
        "- If the FULL window decodes but the TRUNCATED window does not (kappa",
        "  collapses to ~0, p_perm ~ chance), most of the decodable signal is",
        "  motor/post-keypress activity: treat this dataset's contribution as",
        "  confounded and say so in the writeup.",
        "- If the TRUNCATED window still decodes at a similar kappa with",
        "  p_perm < 0.05, the signal survives removal of the motor period and",
        "  the motor confound does not explain the within-dataset result.",
        "- Either way, report BOTH windows side by side; do not silently switch",
        "  windows between experiments.",
    ]
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/rl_motor_confound_report.md")
    args = ap.parse_args()

    npz = ROOT / "data/processed_v2/restaurant_logo/restaurant_logo_epochs7_native_v2.npz"
    man = ROOT / "data/processed_v2/restaurant_logo/restaurant_logo_trial_manifest_v2.csv"
    d = load_rl_epochs(npz)
    median_rt = float(np.median(pd.read_csv(man)["rt_s"].values))
    n_trials, n_subj = len(d["y"]), len(np.unique(d["sid"]))
    print(f"[motor] trials={n_trials} subjects={n_subj} median_rt={median_rt:.3f}s")

    trunc = rebuild_features(d, man, median_rt)
    full = rebuild_features(d, man, EPOCH_TMAX)
    print(f"[motor] truncated features: {trunc['X'].shape}, "
          f"clean_trials={trunc['clean_frac'] * 100:.1f}%")

    res = {}
    for proto, dd in (("full", full), ("truncated", trunc)):
        res[proto] = {}
        for model in ("logreg", "svm_rbf"):
            print(f"[motor] {proto} / {model} ...", flush=True)
            res[proto][model] = {
                "losocv": losocv(dd, model, args.seed),
                "perm": permutation_p(dd, model, args.n_perm, args.seed),
            }
            s = res[proto][model]
            print(f"    kappa={s['losocv']['kappa_mean']:.4f} "
                  f"auc={s['losocv']['auc_mean']:.4f} "
                  f"p={s['perm']['p_perm']:.4f}", flush=True)

    out_md = ROOT / args.out
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_report(res, median_rt, n_trials, n_subj, median_rt,
                 trunc["clean_frac"], out_md, args.n_perm)
    out_md.with_suffix(".json").write_text(json.dumps(
        {"median_rt_s": median_rt, "n_trials": n_trials, "n_subjects": n_subj,
         "clean_trial_frac": trunc["clean_frac"], "results": res},
        indent=2, default=float))
    print(f"[motor] report -> {out_md}")


if __name__ == "__main__":
    main()
