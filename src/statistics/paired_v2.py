"""Participant-level statistics. Units of analysis are participants; trials are never treated as independent."""
import numpy as np
from scipy import stats

def bootstrap_ci(values, n_boot=5000, seed=0, alpha=0.05, stat=np.nanmean):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed); bs = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return float(stat(v)), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))

def paired_permutation_p(diff, n_perm=10000, seed=0):
    d = np.asarray(diff, float); d = d[np.isfinite(d)]
    if len(d) == 0: return np.nan
    rng = np.random.default_rng(seed); obs = abs(d.mean()); signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))

def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) == 0: return np.nan
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum(); return float((gt - lt) / (len(a) * len(b)))

def paired_compare(a, b, seed=0):
    """a, b: arrays aligned on the same participants. Returns dict with mean diff, CI, wilcoxon p, perm p, effect sizes, W/T/L."""
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]; d = a - b
    out = {"n_pairs": int(len(d)), "mean_a": float(a.mean()) if len(d) else np.nan, "mean_b": float(b.mean()) if len(d) else np.nan}
    md, lo, hi = bootstrap_ci(d, seed=seed); out.update({"mean_diff": md, "ci_lo": lo, "ci_hi": hi})
    try: out["wilcoxon_p"] = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if (len(d) >= 5 and np.any(d != 0)) else np.nan
    except Exception: out["wilcoxon_p"] = np.nan
    out["perm_p"] = paired_permutation_p(d, seed=seed); sd = d.std(ddof=1) if len(d) > 1 else np.nan
    out["cohen_dz"] = float(d.mean() / sd) if sd and sd > 0 else np.nan; out["cliffs_delta"] = cliffs_delta(a, b)
    out["wins"] = int((d > 1e-9).sum()); out["ties"] = int((np.abs(d) <= 1e-9).sum()); out["losses"] = int((d < -1e-9).sum()); return out

def holm(pvals):
    p = np.asarray(pvals, float); n = np.isfinite(p).sum(); adj = np.full_like(p, np.nan); order = np.argsort(np.where(np.isfinite(p), p, np.inf)); running = 0.0
    for rank, i in enumerate(order[:n]):
        running = max(running, (n - rank) * p[i]); adj[i] = min(1.0, running)
    return adj
