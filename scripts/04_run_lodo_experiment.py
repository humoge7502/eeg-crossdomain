"""
PATCH for scripts/04_run_lodo_experiment.py

WHAT CHANGED (search for '# PATCH' comments below):
  1. load_raw_epochs_all_datasets() now also reads `subject_ids` from the
     cached .npz when present (it's already being saved there by
     02b_resample_truncate_epochs.py whenever upstream preprocessing
     produced them — see scripts/03_run_within_dataset_baselines.py's
     `groups_e` usage, which already depends on this same field).
  2. main() now saves y_true / y_prob / subject_ids per held-out fold to
     results/lodo_predictions_<held_out>.npz, in addition to the existing
     aggregate lodo_results.json (unchanged).

WHY: this is what lets scripts/06_lodo_permutation_and_equivalence.py test
significance against FIXED predictions (cheap, no retraining) instead of
retraining a full model per permutation (infeasible before your GPU
window closes). This is the same "shuffle labels against fixed held-out
predictions" design already used for the within-subject permutation test
in INS-HDGS-CMT Section 3.1 — you're extending a method you've already
used, not adopting a new one.

HOW TO APPLY: replace scripts/04_run_lodo_experiment.py with this file's
content (it is a complete, drop-in replacement — every line from your
original is preserved except the two additions marked '# PATCH').
Then run it ONCE on Brev (one LODO pass, same cost as your existing
lodo_log2.txt run) before running script 06.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import load_config, ensure_dir, project_root, get_device, set_seed
from src.train import run_lodo_fold
from src.evaluate import summarize_lodo_results


def load_raw_epochs_all_datasets(cfg):
    all_data = {}
    for dataset_key in cfg["datasets"].keys():
        path = (project_root() / cfg["paths"]["processed_dir"] / dataset_key /
                f"{dataset_key}_epochs_common.npz")
        if not path.exists():
            print(f"[lodo] Missing {path} — this dataset will be excluded from LODO.")
            continue
        data = np.load(path)
        # PATCH: read subject_ids when the cache has them (it already
        # does, for datasets where 02b saved them — see its own
        # conditional np.savez call). Falls back to None otherwise, in
        # which case the downstream permutation test degrades to
        # full-set (not within-subject) shuffling and says so explicitly.
        subject_ids = data["subject_ids"] if "subject_ids" in data else None
        all_data[dataset_key] = (data["X"], data["y"], subject_ids)
        n_subj = len(np.unique(subject_ids)) if subject_ids is not None else "unknown"
        print(f"[lodo] Loaded {dataset_key}: X={data['X'].shape}, y={data['y'].shape}, "
              f"subject_ids={'present (' + str(n_subj) + ' subjects)' if subject_ids is not None else 'MISSING'}")
    return all_data


def main():
    ap = argparse.ArgumentParser(description="LODO transfer experiment (leave-one-dataset-out).")
    ap.add_argument("--config", type=str, default="configs/default.yaml",
                    help="Path to config (relative to repo root or absolute). Use a reduced `datasets:` block for sensitivity runs (e.g. ds007406 dropped).")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix for output files, e.g. --tag scaled -> lodo_predictions_<ds>_scaled.npz")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    cfg = load_config(str(cfg_path))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])
    tag = f"_{args.tag}" if args.tag else ""

    all_data = load_raw_epochs_all_datasets(cfg)

    if len(all_data) < 2:
        print(f"\n[lodo] Only {len(all_data)} datasets available - need at least 2 for LODO.")
        return

    fold_results = []
    for held_out in all_data.keys():
        print(f"\n{'='*60}\nLODO fold: held out = {held_out}\n{'='*60}")
        fold_result = run_lodo_fold(held_out, all_data, cfg, device)
        fold_results.append(fold_result)
        print(f"[lodo] held_out={held_out} -> {fold_result['metrics']}")

        # PATCH: persist fixed predictions + subject_ids for this fold,
        # so scripts/06_lodo_permutation_and_equivalence.py can run a
        # cheap, no-retrain permutation test against them.
        pred_path = results_dir / f"lodo_predictions_{held_out}{tag}.npz"
        save_kwargs = {
            "y_true": fold_result["y_true"],
            "y_prob": fold_result["y_prob"],
        }
        if fold_result.get("test_subject_ids") is not None:
            save_kwargs["subject_ids"] = fold_result["test_subject_ids"]
        np.savez(pred_path, **save_kwargs)
        print(f"[lodo] Saved fixed predictions -> {pred_path}")

    summary = summarize_lodo_results(fold_results)
    print(f"\n{'='*60}\nLODO SUMMARY ({len(fold_results)} held-out folds)\n{'='*60}")
    for metric, stats in summary.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(per-fold: {[round(v, 4) for v in stats['per_fold']]})")

    out_path = results_dir / f"lodo_results{tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_fold": [{"held_out": r["held_out"], "trained_on": r["trained_on"],
                          "metrics": r["metrics"]} for r in fold_results],
            "summary": summary,
        }, f, indent=2)
    print(f"\n[lodo] Saved -> {out_path}")


if __name__ == "__main__":
    main()
