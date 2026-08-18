import mne
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ds007406" / "open Neuro"
TASK_NAME = "extremeversustraditionalvideos"

def process_subject(subject_num):
    subj_id = f"sub-{subject_num:03d}"
    subj_dir = RAW_DIR / subj_id / "eeg"
    set_path = subj_dir / f"{subj_id}_task-{TASK_NAME}_eeg.set"
    events_path = subj_dir / f"{subj_id}_task-{TASK_NAME}_events.tsv"

    if not set_path.exists() or not events_path.exists():
        return None

    epochs_obj = mne.io.read_epochs_eeglab(str(set_path), verbose=False)
    sfreq = epochs_obj.info["sfreq"]
    channel_names = epochs_obj.ch_names

    events_df = pd.read_csv(events_path, sep="\t")
    labels_list = []
    for _, row in events_df.iterrows():
        label_text = str(row["value"]).strip().lower()
        if label_text not in ("extreme", "traditional"):
            continue
        labels_list.append(1 if label_text == "extreme" else 0)

    epochs_arr = epochs_obj.get_data()

    if len(labels_list) != epochs_arr.shape[0]:
        print(f"  WARNING: {len(labels_list)} labels vs {epochs_arr.shape[0]} epochs — truncating")
        n = min(len(labels_list), epochs_arr.shape[0])
        labels_list = labels_list[:n]
        epochs_arr = epochs_arr[:n]

    if len(labels_list) == 0:
        return None

    return {
        "subject_id": subj_id,
        "eeg": epochs_arr,
        "labels": np.array(labels_list, dtype=np.int64),
        "channels": channel_names,
        "sfreq": sfreq,
    }

def build_all_subjects():
    results = []
    for subj_num in range(1, 11):
        subj_id = f"sub-{subj_num:03d}"
        print(f"Processing {subj_id}...")
        result = process_subject(subj_num)
        if result is None:
            print(f"  SKIPPED {subj_id} (missing files or no valid epochs)")
            continue
        n_extreme = result["labels"].sum()
        n_total = len(result["labels"])
        print(f"  {subj_id}: {n_total} epochs, {n_extreme} Extreme / {n_total-n_extreme} Traditional, "
              f"EEG shape {result['eeg'].shape}")
        results.append(result)
    return results

if __name__ == "__main__":
    results = build_all_subjects()
    print(f"\nTotal subjects processed: {len(results)}")
    total_epochs = sum(len(r["labels"]) for r in results)
    print(f"Total epochs across all subjects: {total_epochs}")

    out_dir = Path("data/processed/ds007406_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        np.savez(out_dir / f"{r['subject_id']}_real.npz",
                 eeg=r["eeg"], labels=r["labels"])
    print(f"Saved per-subject files to {out_dir}/")
