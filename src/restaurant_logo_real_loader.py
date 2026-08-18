import mne
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "restaurant_logo" / "Neuromarketing"
EPOCH_TMIN = 0.0
EPOCH_TMAX = 3.0

def load_recognition_labels():
    df = pd.read_excel(RAW_DIR / "ExperimentResults.xlsx")
    blank_rows = df[df["Logos"].isna()].index.tolist()
    first_blank = blank_rows[0] if blank_rows else len(df)
    df = df.loc[:first_blank - 1].copy()
    df["Logos"] = df["Logos"].astype(int)
    return df

def process_subject(subject_num, labels_df):
    subj_dir = RAW_DIR / f"Subject {subject_num}"
    set_path = subj_dir / f"Sub{subject_num}_Test.set"
    if not set_path.exists():
        return None

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    onsets = [a["onset"] for a in raw.annotations if a["description"] == "OVTK_StimulationId_ExperimentStart"]
    onsets = sorted(onsets)

    subj_col = f"S{subject_num}"
    if subj_col not in labels_df.columns:
        return None

    logo_labels = dict(zip(labels_df["Logos"], labels_df[subj_col]))
    logo_order = sorted(logo_labels.keys())

    n_trials = min(len(onsets), len(logo_order))

    data = raw.get_data()
    times = raw.times

    epochs_list, labels_list, logo_ids = [], [], []
    for i in range(n_trials):
        onset = onsets[i]
        start_sample = int((onset + EPOCH_TMIN) * sfreq)
        end_sample = int((onset + EPOCH_TMAX) * sfreq)
        if end_sample > data.shape[1]:
            continue
        seg = data[:, start_sample:end_sample]
        logo_id = logo_order[i]
        label = logo_labels[logo_id]
        if pd.isna(label):
            continue
        epochs_list.append(seg)
        labels_list.append(int(label))
        logo_ids.append(logo_id)

    if not epochs_list:
        return None

    min_len = min(s.shape[1] for s in epochs_list)
    epochs_arr = np.stack([s[:, :min_len] for s in epochs_list], axis=0)

    return {
        "subject_id": f"S{subject_num}",
        "eeg": epochs_arr,
        "labels": np.array(labels_list, dtype=np.int64),
        "logo_ids": np.array(logo_ids, dtype=np.int64),
        "channels": raw.ch_names,
        "sfreq": sfreq,
    }

def build_all_subjects():
    labels_df = load_recognition_labels()
    results = []
    for subj_num in range(1, 16):
        print(f"Processing S{subj_num}...")
        result = process_subject(subj_num, labels_df)
        if result is None:
            print(f"  SKIPPED S{subj_num} (missing file or no valid trials)")
            continue
        n_rec = result["labels"].sum()
        n_total = len(result["labels"])
        print(f"  S{subj_num}: {n_total} trials, {n_rec} Recognized / {n_total-n_rec} NotRecognized, "
              f"EEG shape {result['eeg'].shape}")
        results.append(result)
    return results

if __name__ == "__main__":
    results = build_all_subjects()
    print(f"\nTotal subjects processed: {len(results)}")
    total_trials = sum(len(r["labels"]) for r in results)
    print(f"Total trials across all subjects: {total_trials}")

    out_dir = Path("data/processed/restaurant_logo_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        np.savez(out_dir / f"{r['subject_id']}_real.npz",
                 eeg=r["eeg"], labels=r["labels"], logo_ids=r["logo_ids"])
    print(f"Saved per-subject files to {out_dir}/")
