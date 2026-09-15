import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
from src.utils import load_config, ensure_dir, project_root, set_seed

DATASET_NAMES = ["neuma", "restaurant_logo", "ds007406"]


def load_dataset(name):
    p = project_root() / "data" / "processed_v2" / name / f"{name}_windows_common_v2.npz"
    d = np.load(p, allow_pickle=True)
    return d["X"].astype(np.float64), d["y"].astype(np.int64), d["subject_ids"]


def permute_labels_within_subject(y, groups, rng):
    y_perm = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def run_lodo_fold(X_train, y_train, X_test, y_test):
    pipeline = make_pipeline(Covariances(estimator="oas"), MDM())
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return cohen_kappa_score(y_test, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()
    held_out = args.dataset
    N_PERMUTATIONS = args.n_permutations

    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    results_dir = ensure_dir(project_root() / "results")

    print(f"[{held_out}] Loading data...", flush=True)
    train_names = [k for k in DATASET_NAMES if k != held_out]
    train_data = [load_dataset(k) for k in train_names]
    X_train = np.concatenate([d[0] for d in train_data], axis=0)
    y_train = np.concatenate([d[1] for d in train_data], axis=0)
    X_test, y_test, test_groups = load_dataset(held_out)
    print(f"[{held_out}] train={len(y_train)}, test={len(y_test)}", flush=True)

    t0 = time.time()
    observed_kappa = run_lodo_fold(X_train, y_train, X_test, y_test)
    print(f"[{held_out}] Observed kappa: {observed_kappa:.4f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[{held_out}] Running {N_PERMUTATIONS} within-subject permutations...", flush=True)
    rng = np.random.default_rng(cfg["training"]["seed"])
    perm_kappas = []
    t_start = time.time()
    for i in range(N_PERMUTATIONS):
        y_test_shuffled = permute_labels_within_subject(y_test, test_groups, rng)
        k = run_lodo_fold(X_train, y_train, X_test, y_test_shuffled)
        perm_kappas.append(k)
        if (i + 1) % 50 == 0:
            elapsed_min = (time.time() - t_start) / 60
            print(f"  [{held_out}] permutation {i+1}/{N_PERMUTATIONS} (elapsed: {elapsed_min:.1f} min)", flush=True)

    perm_kappas = np.array(perm_kappas)
    p_value = float((1 + np.sum(perm_kappas >= observed_kappa)) / (1 + N_PERMUTATIONS))
    total_min = (time.time() - t_start) / 60

    result = {
        "dataset": held_out,
        "observed_kappa": float(observed_kappa),
        "null_mean": float(perm_kappas.mean()),
        "null_std": float(perm_kappas.std()),
        "null_95ci": [float(np.percentile(perm_kappas, 2.5)), float(np.percentile(perm_kappas, 97.5))],
        "p_value": p_value,
        "p_value_formula": "(1 + count(perm_kappa >= observed_kappa)) / (1 + N_PERMUTATIONS)",
        "n_permutations": N_PERMUTATIONS,
        "significant_at_0.05": p_value < 0.05,
        "wall_time_minutes": total_min,
    }
    with open(results_dir / f"mdm_{held_out}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{held_out}] FINAL: kappa={observed_kappa:.4f}, p={p_value:.4f}, "
          f"null mean/std={perm_kappas.mean():.4f}/{perm_kappas.std():.4f}", flush=True)
    print(f"Saved -> results/mdm_{held_out}_results.json", flush=True)


if __name__ == "__main__":
    main()
