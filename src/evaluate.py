import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0), "cohen_kappa": cohen_kappa_score(y_true, y_pred)}
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics

def permutation_test(y_true, y_pred, n_permutations=1000, metric_fn=accuracy_score, seed=42):
    rng = np.random.default_rng(seed)
    observed = metric_fn(y_true, y_pred)
    permuted_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_true)
        permuted_scores[i] = metric_fn(shuffled, y_pred)
    p_value = float(np.mean(permuted_scores >= observed))
    return {"observed_score": float(observed), "permuted_mean": float(permuted_scores.mean()), "permuted_std": float(permuted_scores.std()), "p_value": p_value, "n_permutations": n_permutations, "significant_at_0.05": p_value < 0.05}

def summarize_lodo_results(results):
    all_metric_keys = results[0]["metrics"].keys()
    summary = {}
    for key in all_metric_keys:
        values = [r["metrics"][key] for r in results if not np.isnan(r["metrics"][key])]
        summary[key] = {"mean": float(np.mean(values)) if values else float("nan"), "std": float(np.std(values)) if values else float("nan"), "per_fold": [r["metrics"][key] for r in results]}
    return summary
