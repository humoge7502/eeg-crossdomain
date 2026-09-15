"""
stats_utils.py

Shared statistical machinery for the cross-domain audit.

Deliberately mirrors the conventions already established in the two prior
papers (NeuroClick, INS-HDGS-CMT):
  - subject/dataset-level paired comparisons, never pooled-only comparisons
  - two-sided Wilcoxon signed-rank as the default paired test, with a
    permutation-test fallback when the paired-sample count is small
    (small n is exactly the risk you flagged for this project, so the
    permutation path is the one actually used throughout the audit)
  - Holm-Bonferroni correction within each declared family of tests
  - bootstrap confidence intervals reported alongside every point estimate

# VERIFY: no dependency beyond numpy/scipy. If scipy is unavailable on the
# Brev image, wilcoxon_paired() falls back to the permutation test only.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def bootstrap_ci(values, n_boot: int = 10000, ci: float = 0.95, seed: int = 42):
    """
    Percentile bootstrap CI on the mean of `values`.

    Returns (mean, lo, hi). Matches the "Student-t / bootstrap CI on
    participant-level differences" reporting style used in both prior papers.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lo), float(hi)


def permutation_test_paired(a, b, n_perm: int = 10000, seed: int = 42):
    """
    Two-sided paired permutation test on (a - b), via random sign-flips.

    This is the test of record for small-n comparisons in this project
    (the honest choice given 10-42 subjects per dataset): it makes no
    distributional assumption and degrades gracefully at small n, unlike
    the asymptotic approximation Wilcoxon relies on below ~20 pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= np.abs(observed)) + 1) / (n_perm + 1)
    return float(observed), float(p)


def wilcoxon_paired(a, b):
    """
    Two-sided Wilcoxon signed-rank test, zero-differences discarded
    (matching the convention stated explicitly in both prior papers).
    Falls back to the permutation test if scipy is unavailable or if
    n < 10, where the Wilcoxon normal approximation is unreliable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]
    if not _HAVE_SCIPY or len(diff) < 10:
        _, p = permutation_test_paired(a, b)
        return p
    try:
        _, p = _scipy_wilcoxon(diff)
        return float(p)
    except ValueError:
        _, p = permutation_test_paired(a, b)
        return p


def holm_correction(pvalues):
    """
    Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    same order as the input. Matches Holm (1979), as cited in both papers.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (n - rank) * pvalues[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def rank_biserial_paired(a, b):
    """
    Matched-pairs rank-biserial correlation, the effect-size measure used
    alongside Wilcoxon p-values in both prior papers.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return float("nan")
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    return float((n_pos - n_neg) / (n_pos + n_neg))


def summarize_comparison(a, b, label: str = ""):
    """
    One-stop summary for a single paired comparison: mean difference,
    bootstrap CI, raw Wilcoxon/permutation p, and rank-biserial effect size.
    Returns a dict; collect these across a family and Holm-correct together
    with holm_correction() before reporting — never report a raw p alone.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    mean_diff, lo, hi = bootstrap_ci(diff)
    p_raw = wilcoxon_paired(a, b)
    effect = rank_biserial_paired(a, b)
    return {
        "label": label,
        "n_pairs": int(np.sum(~np.isnan(diff))),
        "mean_diff": mean_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_raw": p_raw,
        "rank_biserial": effect,
    }
