import sys
import json
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
        all_data[dataset_key] = (data["X"], data["y"])
        print(f"[lodo] Loaded {dataset_key}: X={data['X'].shape}, y={data['y'].shape}")
    return all_data

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    results_dir = ensure_dir(project_root() / cfg["paths"]["results_dir"])

    all_data = load_raw_epochs_all_datasets(cfg)

    if len(all_data) < 3:
        print(f"\n[lodo] Only {len(all_data)}/3 datasets available.")
        return

    fold_results = []
    for held_out in all_data.keys():
        print(f"\n{'='*60}\nLODO fold: held out = {held_out}\n{'='*60}")
        fold_result = run_lodo_fold(held_out, all_data, cfg, device)
        fold_results.append(fold_result)
        print(f"[lodo] held_out={held_out} -> {fold_result['metrics']}")

    summary = summarize_lodo_results(fold_results)
    print(f"\n{'='*60}\nLODO SUMMARY (mean +/- std across 3 held-out folds)\n{'='*60}")
    for metric, stats in summary.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(per-fold: {[round(v, 4) for v in stats['per_fold']]})")

    out_path = results_dir / "lodo_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_fold": [{"held_out": r["held_out"], "trained_on": r["trained_on"],
                          "metrics": r["metrics"]} for r in fold_results],
            "summary": summary,
        }, f, indent=2)
    print(f"\n[lodo] Saved -> {out_path}")

if __name__ == "__main__":
    main()
