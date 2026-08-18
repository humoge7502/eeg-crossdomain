import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import load_config, ensure_dir, project_root
from src.datasets import load_neuma_subject, load_restaurant_logo_subject, load_ds007406_subject
from src.harmonize import build_harmonized_features

def get_raw_epochs_array(epochs_obj, target_channels):
    from src.harmonize import map_to_shared_channels
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    mapped = map_to_shared_channels(epochs_obj, target_channels, True)
    data = mapped.get_data()
    return data

def process_neuma(cfg):
    raw_dir = project_root() / "data" / "raw" / "neuma"
    xdf_files = sorted(raw_dir.glob("S*.xdf"))
    print(f"[neuma] Found {len(xdf_files)} subject files")

    all_features, all_labels, all_subjects = [], [], []
    all_raw_epochs, all_raw_labels = [], []

    for xdf_path in xdf_files:
        try:
            result = load_neuma_subject(str(xdf_path))
            features = build_harmonized_features(result["epochs"], cfg)
            raw_arr = get_raw_epochs_array(result["epochs"], cfg["harmonization"]["shared_channels"])

            all_features.append(features)
            all_labels.append(result["labels"])
            all_subjects.extend([result["subject_id"]] * len(result["labels"]))
            all_raw_epochs.append(raw_arr)
            all_raw_labels.append(result["labels"])

            print(f"  [neuma] {result['subject_id']}: {features.shape[0]} epochs")
        except Exception as e:
            print(f"  [neuma] FAILED on {xdf_path.name}: {e}")
            continue

    if not all_features:
        return

    out_dir = ensure_dir(project_root() / "data" / "processed" / "neuma")
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    np.savez(out_dir / "neuma_harmonized.npz", X=X, y=y, subject_ids=np.array(all_subjects))
    print(f"[neuma] Saved harmonized features: {X.shape} -> {out_dir}/neuma_harmonized.npz")

    min_len = min(a.shape[2] for a in all_raw_epochs)
    raw_X = np.concatenate([a[:, :, :min_len] for a in all_raw_epochs], axis=0)
    raw_y = np.concatenate(all_raw_labels, axis=0)
    raw_X = raw_X[:, np.newaxis, :, :]
    raw_subjects = np.array(all_subjects)
    np.savez(out_dir / "neuma_epochs.npz", X=raw_X, y=raw_y, subject_ids=raw_subjects)
    print(f"[neuma] Saved raw epochs: {raw_X.shape} -> {out_dir}/neuma_epochs.npz")

def process_restaurant_logo(cfg):
    raw_dir = project_root() / "data" / "raw" / "restaurant_logo" / "Neuromarketing"
    subject_dirs = sorted(raw_dir.glob("Subject *"))
    print(f"[restaurant_logo] Found {len(subject_dirs)} subject dirs")

    all_features, all_labels, all_subjects = [], [], []
    all_raw_epochs, all_raw_labels = [], []

    for subj_dir in subject_dirs:
        subj_num = "".join(c for c in subj_dir.name if c.isdigit())
        set_path = subj_dir / f"Sub{subj_num}_Test.set"
        if not set_path.exists():
            continue
        try:
            result = load_restaurant_logo_subject(str(set_path))
            features = build_harmonized_features(result["epochs"], cfg)
            raw_arr = get_raw_epochs_array(result["epochs"], cfg["harmonization"]["shared_channels"])

            all_features.append(features)
            all_labels.append(result["labels"])
            all_subjects.extend([result["subject_id"]] * len(result["labels"]))
            all_raw_epochs.append(raw_arr)
            all_raw_labels.append(result["labels"])

            print(f"  [restaurant_logo] {result['subject_id']}: {features.shape[0]} epochs")
        except Exception as e:
            print(f"  [restaurant_logo] FAILED on {subj_dir.name}: {e}")
            continue

    if not all_features:
        return

    out_dir = ensure_dir(project_root() / "data" / "processed" / "restaurant_logo")
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    np.savez(out_dir / "restaurant_logo_harmonized.npz", X=X, y=y, subject_ids=np.array(all_subjects))
    print(f"[restaurant_logo] Saved harmonized features: {X.shape}")

    min_len = min(a.shape[2] for a in all_raw_epochs)
    raw_X = np.concatenate([a[:, :, :min_len] for a in all_raw_epochs], axis=0)
    raw_y = np.concatenate(all_raw_labels, axis=0)
    raw_X = raw_X[:, np.newaxis, :, :]
    raw_subjects = np.array(all_subjects)
    np.savez(out_dir / "restaurant_logo_epochs.npz", X=raw_X, y=raw_y, subject_ids=raw_subjects)
    print(f"[restaurant_logo] Saved raw epochs: {raw_X.shape}")

def process_ds007406(cfg):
    raw_dir = project_root() / "data" / "raw" / "ds007406" / "open Neuro"
    subject_dirs = sorted(raw_dir.glob("sub-*"))
    print(f"[ds007406] Found {len(subject_dirs)} subject dirs")

    all_features, all_labels, all_subjects = [], [], []
    all_raw_epochs, all_raw_labels = [], []

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        set_path = subj_dir / "eeg" / f"{subj_id}_task-extremeversustraditionalvideos_eeg.set"
        if not set_path.exists():
            continue
        try:
            result = load_ds007406_subject(str(set_path))
            features = build_harmonized_features(result["epochs"], cfg)
            raw_arr = get_raw_epochs_array(result["epochs"], cfg["harmonization"]["shared_channels"])

            all_features.append(features)
            all_labels.append(result["labels"])
            all_subjects.extend([result["subject_id"]] * len(result["labels"]))
            all_raw_epochs.append(raw_arr)
            all_raw_labels.append(result["labels"])

            print(f"  [ds007406] {result['subject_id']}: {features.shape[0]} epochs")
        except Exception as e:
            print(f"  [ds007406] FAILED on {subj_dir.name}: {e}")
            continue

    if not all_features:
        return

    out_dir = ensure_dir(project_root() / "data" / "processed" / "ds007406")
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    np.savez(out_dir / "ds007406_harmonized.npz", X=X, y=y, subject_ids=np.array(all_subjects))
    print(f"[ds007406] Saved harmonized features: {X.shape}")

    min_len = min(a.shape[2] for a in all_raw_epochs)
    raw_X = np.concatenate([a[:, :, :min_len] for a in all_raw_epochs], axis=0)
    raw_y = np.concatenate(all_raw_labels, axis=0)
    raw_X = raw_X[:, np.newaxis, :, :]
    raw_subjects = np.array(all_subjects)
    np.savez(out_dir / "ds007406_epochs.npz", X=raw_X, y=raw_y, subject_ids=raw_subjects)
    print(f"[ds007406] Saved raw epochs: {raw_X.shape}")

def main():
    cfg = load_config(str(project_root() / "configs/default.yaml"))
    print("="*60 + "\nProcessing NeuMa (all 42 subjects)\n" + "="*60)
    process_neuma(cfg)
    print("\n" + "="*60 + "\nProcessing Restaurant-Logo (all 15 subjects)\n" + "="*60)
    process_restaurant_logo(cfg)
    print("\n" + "="*60 + "\nProcessing ds007406 (all 10 subjects)\n" + "="*60)
    process_ds007406(cfg)
    print("\n" + "="*60 + "\nALL DATASETS PROCESSED\n" + "="*60)

if __name__ == "__main__":
    main()
