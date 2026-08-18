import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import resample
from src.utils import project_root, ensure_dir

TARGET_SFREQ = 250.0
TARGET_DURATION_SEC = 1.0
TARGET_SAMPLES = int(TARGET_SFREQ * TARGET_DURATION_SEC)

DATASET_NATIVE_SFREQ = {
    "neuma": 300.0,
    "restaurant_logo": 250.0,
    "ds007406": 256.0,
}

def resample_and_truncate(X, native_sfreq):
    n_epochs, n_1, n_channels, n_timepoints = X.shape
    if native_sfreq == TARGET_SFREQ:
        X_resampled = X
    else:
        target_len_full = int(round(n_timepoints * TARGET_SFREQ / native_sfreq))
        X_resampled = resample(X, target_len_full, axis=3)
    available_samples = X_resampled.shape[3]
    if available_samples < TARGET_SAMPLES:
        raise ValueError(f"Only {available_samples} samples available, need {TARGET_SAMPLES}")
    return X_resampled[:, :, :, :TARGET_SAMPLES].astype(np.float32)

def process_dataset(name):
    processed_dir = project_root() / "data" / "processed" / name
    epochs_path = processed_dir / f"{name}_epochs.npz"
    if not epochs_path.exists():
        print(f"[{name}] SKIPPED - {epochs_path} not found")
        return

    data = np.load(epochs_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    subject_ids = data["subject_ids"] if "subject_ids" in data else None
    native_sfreq = DATASET_NATIVE_SFREQ[name]

    print(f"[{name}] Original: X={X.shape} at {native_sfreq}Hz, subject_ids present: {subject_ids is not None}")
    X_common = resample_and_truncate(X, native_sfreq)
    print(f"[{name}] Resampled+truncated: X={X_common.shape}")

    out_path = processed_dir / f"{name}_epochs_common.npz"
    if subject_ids is not None:
        np.savez(out_path, X=X_common, y=y, subject_ids=subject_ids)
    else:
        np.savez(out_path, X=X_common, y=y)
        print(f"[{name}] WARNING: no subject_ids found — group-wise CV will not be possible for this dataset")
    print(f"[{name}] Saved -> {out_path}")

def main():
    for name in ["neuma", "restaurant_logo", "ds007406"]:
        process_dataset(name)
        print()

if __name__ == "__main__":
    main()
