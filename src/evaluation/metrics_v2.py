"""Metrics recomputable from saved predictions. All functions take y_true, y_prob (P(class 1)), optional threshold."""
import numpy as np
from sklearn.metrics import (matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, average_precision_score, roc_auc_score, f1_score, recall_score)

def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def compute_all(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yh = (p >= thr).astype(int); out = {"n": int(len(y)), "pos_rate": float(y.mean()), "threshold": float(thr)}
    two = len(np.unique(y)) == 2
    out["mcc"] = float(matthews_corrcoef(y, yh)) if two else float("nan"); out["kappa"] = float(cohen_kappa_score(y, yh)) if two else float("nan")
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, yh)) if two else float("nan")
    out["pr_auc"] = float(average_precision_score(y, p)) if two else float("nan"); out["roc_auc"] = float(roc_auc_score(y, p)) if two else float("nan")
    out["macro_f1"] = float(f1_score(y, yh, average="macro")); out["sensitivity"] = float(recall_score(y, yh, pos_label=1)) if (y == 1).any() else float("nan")
    out["specificity"] = float(recall_score(y, yh, pos_label=0)) if (y == 0).any() else float("nan"); out["accuracy"] = float((y == yh).mean()); out["ece"] = ece(y, p)
    return out

def per_participant(df, thr=0.5, min_n=2):
    """df: columns subject_id, y_true, y_prob. Returns dict subject -> metrics (nan-safe for single-class subjects)."""
    res = {}
    for s, g in df.groupby("subject_id"):
        if len(g) < min_n: continue
        res[s] = compute_all(g["y_true"].values, g["y_prob"].values, thr)
    return res


def choose_threshold(y_train, p_train, grid=None):
    """Threshold maximising balanced accuracy on TRAINING (or validation) predictions. Never uses test data."""
    y = np.asarray(y_train).astype(int); p = np.asarray(p_train, float)
    if len(np.unique(y)) < 2 or len(p) == 0: return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))) if grid is None else grid; best, bt = -1, 0.5
    for t in grid:
        yh = (p >= t).astype(int); ba = 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())
        if ba > best + 1e-12: best, bt = ba, float(t)
    return bt


def choose_threshold(y_train, p_train, grid=None):
    """Threshold maximising balanced accuracy on TRAINING (or validation) predictions. Never uses test data."""
    y = np.asarray(y_train).astype(int); p = np.asarray(p_train, float)
    if len(np.unique(y)) < 2 or len(p) == 0: return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))) if grid is None else grid; best, bt = -1, 0.5
    for t in grid:
        yh = (p >= t).astype(int); ba = 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())
        if ba > best + 1e-12: best, bt = ba, float(t)
    return bt


def choose_threshold(y_train, p_train, grid=None):
    """Threshold maximising balanced accuracy on TRAINING (or validation) predictions. Never uses test data."""
    y = np.asarray(y_train).astype(int); p = np.asarray(p_train, float)
    if len(np.unique(y)) < 2 or len(p) == 0: return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))) if grid is None else grid; best, bt = -1, 0.5
    for t in grid:
        yh = (p >= t).astype(int); ba = 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())
        if ba > best + 1e-12: best, bt = ba, float(t)
    return bt


def choose_threshold(y_train, p_train, grid=None):
    """Threshold maximising balanced accuracy on TRAINING (or validation) predictions. Never uses test data."""
    y = np.asarray(y_train).astype(int); p = np.asarray(p_train, float)
    if len(np.unique(y)) < 2 or len(p) == 0: return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))) if grid is None else grid; best, bt = -1, 0.5
    for t in grid:
        yh = (p >= t).astype(int); ba = 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())
        if ba > best + 1e-12: best, bt = ba, float(t)
    return bt


def choose_threshold(y_train, p_train, grid=None):
    """Threshold maximising balanced accuracy on TRAINING (or validation) predictions. Never uses test data."""
    y = np.asarray(y_train).astype(int); p = np.asarray(p_train, float)
    if len(np.unique(y)) < 2 or len(p) == 0: return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))) if grid is None else grid; best, bt = -1, 0.5
    for t in grid:
        yh = (p >= t).astype(int); ba = 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())
        if ba > best + 1e-12: best, bt = ba, float(t)
    return bt
