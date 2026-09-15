import sys, json
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
    return cohen_kappa_score(y_test, preds), preds


def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    results_dir = ensure_dir(project_root() / "results")

    train_names = ["neuma", "restaurant_logo"]
    train_data = [load_dataset(k) for k in train_names]
    X_train = np.concatenate([d[0] for d in train_data], axis=0)
    y_train = np.concatenate([d[1] for d in train_data], axis=0)
    X_test, y_test, test_groups = load_dataset("ds007406")

    print("=" * 60 + "\n1. PERMUTATION-SEED STABILITY CHECK\n" + "=" * 60)
    observed_kappa, _ = run_lodo_fold(X_train, y_train, X_test, y_test)
    print(f"Observed kappa (deterministic): {observed_kappa:.4f}")

    for seed in [42, 123, 999]:
        rng = np.random.default_rng(seed)
        perm_kappas = []
        for i in range(300):
            y_shuffled = permute_labels_within_subject(y_test, test_groups, rng)
            k, _ = run_lodo_fold(X_train, y_train, X_test, y_shuffled)
            perm_kappas.append(k)
        perm_kappas = np.array(perm_kappas)
        p_value = float((1 + np.sum(perm_kappas >= observed_kappa)) / (1 + 300))
        print(f"  seed={seed}: null mean/std={perm_kappas.mean():.4f}/{perm_kappas.std():.4f}, p={p_value:.4f}")

    print("\n" + "=" * 60 + "\n2. LEAVE-ONE-SUBJECT-OUT SENSITIVITY (ds007406)\n" + "=" * 60)
    unique_subjects = np.unique(test_groups)
    print(f"ds007406 has {len(unique_subjects)} subjects: {list(unique_subjects)}")
    loo_kappas = {}
    for held_subject in unique_subjects:
        keep_mask = test_groups != held_subject
        X_test_loo, y_test_loo = X_test[keep_mask], y_test[keep_mask]
        kappa_loo, _ = run_lodo_fold(X_train, y_train, X_test_loo, y_test_loo)
        loo_kappas[str(held_subject)] = float(kappa_loo)
        print(f"  Excluding subject {held_subject}: kappa={kappa_loo:.4f} (full kappa was {observed_kappa:.4f})")

    loo_values = np.array(list(loo_kappas.values()))
    print(f"\nLOO kappa range: [{loo_values.min():.4f}, {loo_values.max():.4f}], mean={loo_values.mean():.4f}")
    print(f"Full-sample kappa: {observed_kappa:.4f}")

    result = {
        "observed_kappa_full": float(observed_kappa),
        "leave_one_subject_out_kappas": loo_kappas,
        "loo_min": float(loo_values.min()),
        "loo_max": float(loo_values.max()),
        "loo_mean": float(loo_values.mean()),
    }
    with open(results_dir / "mdm_ds007406_robustness.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> results/mdm_ds007406_robustness.json")


if __name__ == "__main__":
    main()
