import mne
import numpy as np
import pyxdf
import scipy.io as sio
import pandas as pd
from pathlib import Path

NEUMA_EEG_CHANNEL_ORDER = ["P3","C3","F3","Fz","F4","C4","P4","Cz","Pz","Fp1","Fp2",
                          "T3","T5","O1","O2","F7","F8","A2","T6","T4"]
NEUMA_SCALE_X = 3000 / 1920
NEUMA_SCALE_Y = 1688 / 1080
NEUMA_MIN_DWELL_SAMPLES = 3
NEUMA_EPOCH_PRE_SEC = 0.2
NEUMA_EPOCH_POST_SEC = 1.0
NEUMA_SFREQ = 300.0

def _neuma_load_bounding_boxes(dep_dir: Path):
    boxes_per_page = {}
    for page in range(1, 7):
        mat = sio.loadmat(str(dep_dir / "BoundingBox_Coordinates" / f"BoundingBoxPage_{page}.mat"))
        roi_list = mat["ROI_list"][0]
        boxes_per_page[page] = [np.array(b).flatten() for b in roi_list]
    return boxes_per_page

def _neuma_load_product_descriptions(dep_dir: Path):
    mat = sio.loadmat(str(dep_dir / "Leaflet_Product_Descriptions.mat"))
    arr = mat["Product_Descriptions"]
    names = {}
    for page_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            product_number = page_idx * 24 + col_idx + 1
            names[product_number] = str(arr[page_idx, col_idx][0])
    return names

def _neuma_parse_selected_products(xlsx_path: Path):
    df = pd.read_excel(xlsx_path)
    selected = set()
    for col in ["Q76", "Q77", "Q78"]:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            digits = "".join(c for c in str(val).strip() if c.isdigit())
            if digits:
                selected.add(int(digits))
    return selected

def _neuma_get_page_windows(marker_stream):
    events = marker_stream["time_series"]
    times = marker_stream["time_stamps"]
    windows = []
    current_page, current_start = None, None
    for i, ev in enumerate(events):
        text = ev[0]
        if text.startswith("Category:IMG=ID:FYLLADIO_"):
            page_num = int(text.split("FYLLADIO_")[1].split(".")[0])
            if current_page is not None:
                windows.append((current_page, current_start, times[i]))
            current_page, current_start = page_num, times[i]
        elif text == "EOE" and current_page is not None:
            windows.append((current_page, current_start, times[i]))
            current_page = None
    return windows

def _neuma_find_product_dwells(page_num, t_start, t_end, pos_times, pos_vals, boxes):
    mask = (pos_times >= t_start) & (pos_times <= t_end)
    times_win = pos_times[mask]
    pos_win = pos_vals[mask]

    img_x = pos_win[:, 0] * NEUMA_SCALE_X
    img_y = pos_win[:, 1] * NEUMA_SCALE_Y

    box_idx_per_sample = np.full(len(img_x), -1, dtype=int)
    for i, (px, py) in enumerate(zip(img_x, img_y)):
        for b_idx, (bx, by, bw, bh) in enumerate(boxes):
            if bx <= px <= bx + bw and by <= py <= by + bh:
                box_idx_per_sample[i] = b_idx
                break

    dwells = []
    current_box, run_start_idx = None, None
    for i, b in enumerate(box_idx_per_sample):
        if b != current_box:
            if current_box is not None and current_box != -1:
                run_len = i - run_start_idx
                if run_len >= NEUMA_MIN_DWELL_SAMPLES:
                    product_number = (page_num - 1) * 24 + current_box + 1
                    dwells.append((product_number, times_win[run_start_idx], times_win[i - 1]))
            current_box, run_start_idx = b, i
    if current_box is not None and current_box != -1:
        run_len = len(box_idx_per_sample) - run_start_idx
        if run_len >= NEUMA_MIN_DWELL_SAMPLES:
            product_number = (page_num - 1) * 24 + current_box + 1
            dwells.append((product_number, times_win[run_start_idx], times_win[-1]))
    return dwells

def _neuma_extract_eeg_epoch(eeg_times, eeg_data, chan_indices, onset_time):
    start_t = onset_time - NEUMA_EPOCH_PRE_SEC
    end_t = onset_time + NEUMA_EPOCH_POST_SEC
    mask = (eeg_times >= start_t) & (eeg_times <= end_t)
    if mask.sum() < 10:
        return None
    return eeg_data[mask][:, chan_indices]

def load_neuma_subject(xdf_path: str) -> dict:
    xdf_path = Path(xdf_path)
    subject_id = xdf_path.stem
    raw_dir = xdf_path.parent
    dep_dir = raw_dir / "Dependencies" / "Dependencies"
    xlsx_path = raw_dir / f"{subject_id}.xlsx"

    boxes_per_page = _neuma_load_bounding_boxes(dep_dir)
    product_names = _neuma_load_product_descriptions(dep_dir)

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    by_name = {s["info"]["name"][0]: s for s in streams}

    eeg_stream = by_name["WS-default"]
    eeg_labels_raw = [c["label"][0] for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    chan_indices = [eeg_labels_raw.index(ch) for ch in NEUMA_EEG_CHANNEL_ORDER if ch in eeg_labels_raw]
    used_channels = [ch for ch in NEUMA_EEG_CHANNEL_ORDER if ch in eeg_labels_raw]

    eeg_times = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"])

    marker_stream = by_name["MyMarkerStream3"]
    page_windows = _neuma_get_page_windows(marker_stream)

    pos_stream = by_name["MousePosition"]
    pos_times = np.array(pos_stream["time_stamps"])
    pos_vals = np.array(pos_stream["time_series"])

    selected_products = _neuma_parse_selected_products(xlsx_path)

    epochs_list, labels, product_ids = [], [], []
    for page_num, t_start, t_end in page_windows:
        boxes = boxes_per_page[page_num]
        dwells = _neuma_find_product_dwells(page_num, t_start, t_end, pos_times, pos_vals, boxes)
        for product_number, dwell_start, dwell_end in dwells:
            seg = _neuma_extract_eeg_epoch(eeg_times, eeg_data, chan_indices, dwell_start)
            if seg is None:
                continue
            label = 1 if product_number in selected_products else 0
            epochs_list.append(seg)
            labels.append(label)
            product_ids.append(product_number)

    if not epochs_list:
        raise ValueError(f"No valid product-level epochs extracted for {subject_id}")

    max_len = max(s.shape[0] for s in epochs_list)
    padded = np.zeros((len(epochs_list), max_len, len(used_channels)), dtype=np.float32)
    for i, s in enumerate(epochs_list):
        padded[i, :s.shape[0], :] = s

    info = mne.create_info(ch_names=used_channels, sfreq=NEUMA_SFREQ, ch_types="eeg")
    n_epochs, n_times, n_ch = padded.shape
    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.array(labels, dtype=int),
    ])
    epochs_data = np.transpose(padded, (0, 2, 1))
    epochs_obj = mne.EpochsArray(epochs_data, info, events=events, tmin=-NEUMA_EPOCH_PRE_SEC, verbose=False)

    return {
        "subject_id": subject_id,
        "epochs": epochs_obj,
        "labels": np.array(labels, dtype=int),
        "product_ids": np.array(product_ids, dtype=int),
        "event_id": {"NoBuy": 0, "Buy": 1},
        "channel_names": used_channels,
        "dataset": "neuma",
    }

RESTAURANT_LOGO_EPOCH_TMIN = 0.0
RESTAURANT_LOGO_EPOCH_TMAX = 3.0

def _restaurant_logo_load_recognition_labels(raw_dir: Path):
    df = pd.read_excel(raw_dir / "ExperimentResults.xlsx")
    blank_rows = df[df["Logos"].isna()].index.tolist()
    first_blank = blank_rows[0] if blank_rows else len(df)
    df = df.loc[:first_blank - 1].copy()
    df["Logos"] = df["Logos"].astype(int)
    return df

def load_restaurant_logo_subject(set_path: str) -> dict:
    set_path = Path(set_path)
    subject_num = int("".join(c for c in set_path.parent.name if c.isdigit()))
    raw_dir = set_path.parent.parent

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    channel_names = raw.ch_names

    onsets = sorted([a["onset"] for a in raw.annotations
                      if a["description"] == "OVTK_StimulationId_ExperimentStart"])

    labels_df = _restaurant_logo_load_recognition_labels(raw_dir)
    subj_col = f"S{subject_num}"
    logo_labels = dict(zip(labels_df["Logos"], labels_df[subj_col]))
    logo_order = sorted(logo_labels.keys())
    n_trials = min(len(onsets), len(logo_order))

    data = raw.get_data()
    epochs_list, labels_list = [], []
    for i in range(n_trials):
        onset = onsets[i]
        start_sample = int((onset + RESTAURANT_LOGO_EPOCH_TMIN) * sfreq)
        end_sample = int((onset + RESTAURANT_LOGO_EPOCH_TMAX) * sfreq)
        if end_sample > data.shape[1]:
            continue
        label = logo_labels[logo_order[i]]
        if pd.isna(label):
            continue
        epochs_list.append(data[:, start_sample:end_sample])
        labels_list.append(int(label))

    if not epochs_list:
        raise ValueError(f"No valid trials extracted for subject {subject_num}")

    min_len = min(s.shape[1] for s in epochs_list)
    epochs_arr = np.stack([s[:, :min_len] for s in epochs_list], axis=0)

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    n_ep = epochs_arr.shape[0]
    events = np.column_stack([
        np.arange(n_ep) * epochs_arr.shape[2],
        np.zeros(n_ep, dtype=int),
        np.array(labels_list, dtype=int),
    ])
    epochs_obj = mne.EpochsArray(epochs_arr, info, events=events, tmin=RESTAURANT_LOGO_EPOCH_TMIN, verbose=False)

    return {
        "subject_id": f"S{subject_num}",
        "epochs": epochs_obj,
        "labels": np.array(labels_list, dtype=int),
        "event_id": {"NotRecognized": 0, "Recognized": 1},
        "channel_names": channel_names,
        "dataset": "restaurant_logo",
    }

def load_ds007406_subject(set_path: str) -> dict:
    import pandas as pd
    set_path = Path(set_path)
    subj_id = set_path.name.split("_task-")[0]
    events_path = set_path.parent / f"{subj_id}_task-extremeversustraditionalvideos_events.tsv"

    epochs_obj = mne.io.read_epochs_eeglab(str(set_path), verbose=False)
    channel_names = epochs_obj.ch_names

    events_df = pd.read_csv(events_path, sep="\t")
    labels_list = []
    for _, row in events_df.iterrows():
        label_text = str(row["value"]).strip().lower()
        if label_text not in ("extreme", "traditional"):
            continue
        labels_list.append(1 if label_text == "extreme" else 0)

    n = min(len(labels_list), len(epochs_obj))
    epochs_obj = epochs_obj[:n]
    labels_arr = np.array(labels_list[:n], dtype=int)

    events = np.column_stack([
        np.arange(n) * epochs_obj.get_data().shape[2],
        np.zeros(n, dtype=int),
        labels_arr,
    ])
    epochs_obj.events = events
    epochs_obj.event_id = {"Traditional": 0, "Extreme": 1}

    return {
        "subject_id": subj_id,
        "epochs": epochs_obj,
        "labels": labels_arr,
        "event_id": {"Traditional": 0, "Extreme": 1},
        "channel_names": channel_names,
        "dataset": "ds007406",
    }

LOADERS = {
    "neuma": load_neuma_subject,
    "restaurant_logo": load_restaurant_logo_subject,
    "ds007406": load_ds007406_subject,
}
