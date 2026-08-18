#!/usr/bin/env python
"""Statistics & tables from saved predictions (never from logged numbers).
For each experiment dir under outputs/predictions/: pooled metrics per (dataset, protocol, model, seed) and per participant,
bootstrap CIs over participants, paired comparisons on identical participants, Holm correction, W/T/L.
Exp1 extra: leakage inflation (epochwise - subjectwise) and permuted-label null."""
import argparse, glob, json, sys, itertools, datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.evaluation.metrics_v2 import compute_all
from src.statistics.paired_v2 import bootstrap_ci, paired_compare, holm
METRICS = ["mcc", "kappa", "balanced_accuracy", "pr_auc", "roc_auc", "macro_f1", "sensitivity", "specificity", "ece"]

def load_preds(pdir):
    fs = sorted(glob.glob(str(pdir/"*__*__*__seed*__fold*.csv")))
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

def participant_metrics(df):
    """One row per (dataset, protocol, model, seed, subject): metrics on that participant's held-out trials (pooled over folds within seed — each subject is tested once per seed)."""
    rows = []
    for (ds, pr, mo, se, su), g in df.groupby(["dataset", "protocol", "model", "seed", "subject_id"]):
        m = compute_all(g.y_true.values, g.y_prob.values); rows.append({"dataset": ds, "protocol": pr, "model": mo, "seed": se, "subject_id": su, "n_trials": len(g), "single_class": int(g.y_true.nunique() < 2), **{k: m[k] for k in METRICS + ["accuracy"]}})
    return pd.DataFrame(rows)

def pooled_metrics(df):
    rows = []
    for (ds, pr, mo, se), g in df.groupby(["dataset", "protocol", "model", "seed"]):
        m = compute_all(g.y_true.values, g.y_prob.values); rows.append({"dataset": ds, "protocol": pr, "model": mo, "seed": se, "n_trials": len(g), "n_subjects": g.subject_id.nunique(), **{k: m[k] for k in METRICS + ["accuracy"]}})
    return pd.DataFrame(rows)

def summarize(part, pooled, metric="mcc"):
    """Per (dataset, protocol, model): mean over seeds of pooled metric; participant-level mean with bootstrap CI (participants x seeds averaged per participant first)."""
    rows = []
    for (ds, pr, mo), g in part.groupby(["dataset", "protocol", "model"]):
        # participant score = mean over seeds (so seeds are not counted as independent participants); single-class participants excluded for MCC/AUC-type metrics
        gp = g[g.single_class == 0].groupby("subject_id")[metric].mean(); mean, lo, hi = bootstrap_ci(gp.values)
        pl = pooled[(pooled.dataset == ds) & (pooled.protocol == pr) & (pooled.model == mo)]
        rows.append({"dataset": ds, "protocol": pr, "model": mo, "metric": metric, "n_participants": int(len(gp)), "n_single_class_excluded": int(g[g.single_class == 1].subject_id.nunique()),
                     "participant_mean": mean, "ci_lo": lo, "ci_hi": hi, "pooled_mean_over_seeds": float(pl[metric].mean()), "pooled_sd_over_seeds": float(pl[metric].std(ddof=0)), "n_seeds": int(pl.seed.nunique())})
    return pd.DataFrame(rows)

def paired_table(part, key_cols, metric="mcc", within=("dataset",), compare_over="model", fixed=None):
    """Paired comparisons between all pairs of `compare_over` levels, on identical participants (mean over seeds per participant)."""
    rows = []
    for keys, g in part.groupby(list(within)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if fixed:
            for k, v in fixed.items(): g = g[g[k] == v]
        g = g[g.single_class == 0]; piv = g.groupby(["subject_id", compare_over])[metric].mean().unstack(compare_over)
        for a, b in itertools.combinations(piv.columns, 2):
            r = paired_compare(piv[a].values, piv[b].values); rows.append({**dict(zip(within, keys)), "metric": metric, "a": a, "b": b, **r})
    t = pd.DataFrame(rows)
    if len(t):
        for fam, idx in t.groupby(list(within)).groups.items(): t.loc[idx, "wilcoxon_p_holm"] = holm(t.loc[idx, "wilcoxon_p"].values); t.loc[idx, "perm_p_holm"] = holm(t.loc[idx, "perm_p"].values)
    return t

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default="outputs/statistics"); ap.add_argument("--experiments", nargs="+", default=["exp1_leakage", "exp2_within", "exp3_lodo", "exp4_transfer"]); ap.add_argument("--metrics", nargs="+", default=["mcc", "roc_auc", "balanced_accuracy", "pr_auc"])
    a = ap.parse_args(); out = ROOT/a.output_dir; out.mkdir(parents=True, exist_ok=True); ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); report = {"timestamp": ts, "experiments": {}}
    for exp in a.experiments:
        pdir = ROOT/"outputs/predictions"/(exp + ("_smoke" if a.smoke else "")); df = load_preds(pdir)
        if df is None: print(f"[{exp}] no predictions yet"); continue
        part = participant_metrics(df); pooled = pooled_metrics(df); part.to_csv(out/f"{exp}_participant_metrics.csv", index=False); pooled.to_csv(out/f"{exp}_pooled_metrics.csv", index=False)
        summ = pd.concat([summarize(part, pooled, m) for m in a.metrics]); summ.to_csv(out/f"{exp}_summary.csv", index=False)
        n_files = len(glob.glob(str(pdir/"*fold*.csv"))); report["experiments"][exp] = {"n_prediction_files": n_files, "n_rows": int(len(df)), "seeds": sorted(df.seed.unique().tolist()), "models": sorted(df.model.unique().tolist()), "protocols": sorted(df.protocol.unique().tolist())}
        print(f"\n===== {exp}: {n_files} prediction files, {len(df)} predictions, seeds={sorted(df.seed.unique())} =====")
        show = summ[summ.metric == "mcc"].pivot_table(index=["dataset", "model"], columns="protocol", values="participant_mean").round(3); print(show.to_string())
        if exp == "exp1_leakage":
            # leakage inflation: paired over participants (each participant has a subject-wise score and an epoch-wise score for the same model)
            rows = []
            for (ds, mo), g in part[part.single_class == 0].groupby(["dataset", "model"]):
                piv = g.groupby(["subject_id", "protocol"])["mcc"].mean().unstack("protocol")
                if {"subjectwise", "epochwise"} <= set(piv.columns):
                    r = paired_compare(piv["epochwise"].values, piv["subjectwise"].values); rows.append({"dataset": ds, "model": mo, "comparison": "epochwise - subjectwise (MCC)", **r})
                if {"subjectwise", "permuted_subjectwise"} <= set(piv.columns):
                    r = paired_compare(piv["subjectwise"].values, piv["permuted_subjectwise"].values); rows.append({"dataset": ds, "model": mo, "comparison": "subjectwise - permuted_null (MCC)", **r})
                if {"epochwise", "permuted_subjectwise"} <= set(piv.columns):
                    r = paired_compare(piv["epochwise"].values, piv["permuted_subjectwise"].values); rows.append({"dataset": ds, "model": mo, "comparison": "epochwise - permuted_null (MCC)", **r})
            if rows:
                lk = pd.DataFrame(rows)
                for c, g in lk.groupby("comparison").groups.items(): lk.loc[g, "wilcoxon_p_holm"] = holm(lk.loc[g, "wilcoxon_p"].values)
                lk.to_csv(out/"exp1_leakage_inflation.csv", index=False); print("\n-- leakage inflation (participant-paired) --"); print(lk[["dataset", "model", "comparison", "n_pairs", "mean_a", "mean_b", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "wilcoxon_p_holm", "cohen_dz", "wins", "ties", "losses"]].round(3).to_string(index=False))
        else:
            for m in a.metrics:
                pt = paired_table(part, None, metric=m, within=("dataset", "protocol"), compare_over="model")
                if len(pt): pt.to_csv(out/f"{exp}_paired_models_{m}.csv", index=False)
            pt = paired_table(part, None, metric="mcc", within=("dataset", "protocol"), compare_over="model")
            if len(pt): print("\n-- paired model comparisons (MCC, Holm within dataset) --"); print(pt[["dataset", "a", "b", "n_pairs", "mean_a", "mean_b", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p_holm", "cliffs_delta", "wins", "ties", "losses"]].round(3).to_string(index=False))
    json.dump(report, open(out/f"stats_report_{ts}.json", "w"), indent=1); print(f"\n[ok] tables in {out}")

if __name__ == "__main__": main()
