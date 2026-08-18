#!/usr/bin/env python
"""Diagnostic: is the harmonized 7ch/1s representation destroying signal, and is there ANY decodable
information in these recordings? Controls:
  A) native-channel, full-duration band power vs harmonized 36-d, same subject-wise folds
  B) positive control: Restaurant-Logo Basal(rest) vs Test(task) -- must be decodable if data are sound
  C) subject-identity probe: can features identify participants (upper bound on usable information)
"""
import argparse, json, sys, logging, datetime
from pathlib import Path
import numpy as np, yaml, pandas as pd, mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features.bandpower_v2 import compute_features
from src.evaluation.metrics_v2 import compute_all, choose_threshold
mne.set_log_level("ERROR")
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def native_features(name, paths, cfg):
    """Band power on ALL native channels over the FULL trial (no harmonization, no truncation)."""
    raw_root = Path(paths["datasets"][name]["confirmed"])
    X, y, sid = [], [], []
    if name == "ds007406":
        for sub in sorted(raw_root.glob("sub-*")):
            f = sub/"eeg"/f"{sub.name}_task-extremeversustraditionalvideos_eeg.set"
            if not f.exists(): continue
            epo = mne.io.read_epochs_eeglab(str(f)); ev = pd.read_csv(str(f).replace("_eeg.set", "_events.tsv"), sep="\t")
            lab = [1 if str(v).strip().lower() == "extreme" else 0 for v in ev["value"] if str(v).strip().lower() in ("extreme", "traditional")]
            d = epo.get_data(); ch = list(epo.ch_names)
            xr, xa, names, _ = compute_features(d, epo.info["sfreq"], ch, BANDS, (1, 40), faa_pair=("F3", "F4"))
            X.append(xr); y += lab[:len(xr)]; sid += [sub.name]*len(xr)
    else:
        sheet = None
        sys.path.insert(0, str(ROOT)); from src.data.loaders_v2 import rl_sheet_labels, load_rl_subject
        sheet = rl_sheet_labels(raw_root/"ExperimentResults.xlsx")
        for n in range(1, 16):
            f = raw_root/f"Subject {n}"/f"Sub{n}_Test.set"
            if not f.exists(): continue
            r = load_rl_subject(f, {"tmin_s": 0.0, "tmax_s": 3.0}, [1.0, 45.0], 60, 1.0, sheet)
            xr, xa, names, _ = compute_features(r["eeg"].astype(np.float64), r["sfreq"], r["ch_names"], BANDS, (1, 40), faa_pair=("C3", "C4"))
            X.append(xr); y += r["y"].tolist(); sid += [r["subject_id"]]*len(r["y"])
    return np.concatenate(X), np.array(y), np.array(sid)


def rest_vs_task(paths):
    """Positive control: Restaurant-Logo Basal (rest) vs Test (task), 3 s epochs, same feature pipeline."""
    root = Path(paths["datasets"]["restaurant_logo"]["confirmed"])
    X, y, sid = [], [], []
    for n in range(1, 16):
        for cond, lab in (("Basal", 0), ("Test", 1)):
            f = root/f"Subject {n}"/f"Sub{n}_{cond}.set"
            if not f.exists(): continue
            raw = mne.io.read_raw_eeglab(str(f), preload=True); sf = raw.info["sfreq"]; d = raw.get_data()
            w = int(3*sf); nw = d.shape[1]//w
            ep = np.stack([d[:, i*w:(i+1)*w] for i in range(nw)])
            xr, _, _, _ = compute_features(ep, sf, list(raw.ch_names), BANDS, (1, 40), faa_pair=("C3", "C4"))
            X.append(xr); y += [lab]*len(xr); sid += [f"S{n}"]*len(xr)
    return np.concatenate(X), np.array(y), np.array(sid)


def subject_identity_probe(root, name):
    """Can a model identify WHICH participant a trial came from? Upper bound on information content."""
    f = np.load(root/name/f"{name}_features_v2.npz", allow_pickle=True)
    X, sid = f["X"], f["subject_ids"].astype(str)
    subs, ycode = np.unique(sid, return_inverse=True)
    sc = StandardScaler().fit_transform(X)
    m = LogisticRegression(max_iter=1000)
    from sklearn.model_selection import StratifiedKFold
    sk = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(m, sc, ycode, cv=sk, scoring="accuracy").mean()
    return float(acc), float(1.0/len(subs)), len(subs)


def eval_subjectwise(X, y, sid, seed=0, k=5):
    k = min(k, len(np.unique(sid)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    ys, ps = [], []
    for tr, te in sgkf.split(X, y, groups=sid):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(sc.transform(X[tr]), y[tr])
        ps.append(m.predict_proba(sc.transform(X[te]))[:, 1]); ys.append(y[te])
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    return compute_all(y_all, p_all, 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml")
    ap.add_argument("--output-dir", default="outputs/statistics")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT/a.config)); paths = yaml.safe_load(open(ROOT/cfg["paths_config"]))
    root = ROOT/cfg["output_root"]; out = ROOT/a.output_dir; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
    log = logging.getLogger("diag"); res = {}
    log.info("=== A) native channels + full duration vs harmonized 7ch/1s (logreg, subject-wise 5-fold) ===")
    for name in ["ds007406", "restaurant_logo"]:
        h = np.load(root/name/f"{name}_features_v2.npz", allow_pickle=True)
        m_h = eval_subjectwise(h["X"], h["y"].astype(int), h["subject_ids"].astype(str))
        Xn, yn, sn = native_features(name, paths, cfg)
        m_n = eval_subjectwise(Xn, yn, sn)
        log.info(f"{name:16s} harmonized(36d): MCC={m_h['mcc']:+.3f} AUC={m_h['roc_auc']:.3f} | native({Xn.shape[1]}d, full trial): MCC={m_n['mcc']:+.3f} AUC={m_n['roc_auc']:.3f}")
        res[f"A_{name}"] = {"harmonized": m_h, "native": m_n, "native_dim": int(Xn.shape[1])}
    log.info("\n=== B) POSITIVE CONTROL: Restaurant-Logo rest(Basal) vs task(Test) ===")
    Xr, yr, sr = rest_vs_task(paths); m_r = eval_subjectwise(Xr, yr, sr)
    log.info(f"rest vs task: n={len(yr)} MCC={m_r['mcc']:+.3f} AUC={m_r['roc_auc']:.3f} balAcc={m_r['balanced_accuracy']:.3f}  <-- should be HIGH if data+pipeline are sound")
    res["B_rest_vs_task"] = m_r
    log.info("\n=== C) subject-identity probe (can features tell participants apart?) ===")
    for name in ["neuma", "restaurant_logo", "ds007406"]:
        acc, chance, ns = subject_identity_probe(root, name)
        log.info(f"{name:16s} {ns} participants: identity accuracy={acc:.3f} (chance={chance:.3f}, ratio={acc/chance:.1f}x)")
        res[f"C_{name}"] = {"accuracy": acc, "chance": chance, "n_subjects": ns}
    json.dump(res, open(out/"diagnostics.json", "w"), indent=1, default=str)
    log.info(f"\n[ok] {out}/diagnostics.json")


if __name__ == "__main__":
    main()
