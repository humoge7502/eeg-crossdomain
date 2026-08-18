import pyxdf
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "neuma"
DEP_DIR = RAW_DIR / "Dependencies" / "Dependencies"

EEG_CHANNEL_ORDER = ["P3","C3","F3","Fz","F4","C4","P4","Cz","Pz","Fp1","Fp2",
                     "T3","T5","O1","O2","F7","F8","A2","T6","T4"]
EXCLUDE_CHANNELS = {"X1","X2","X3","TRG"}
SCALE_X = 3000 / 1920
SCALE_Y = 1688 / 1080
MIN_DWELL_SAMPLES = 3
EPOCH_PRE_SEC = 0.2
EPOCH_POST_SEC = 1.0
SFREQ_EEG = 300.0

def load_bounding_boxes():
    boxes_per_page = {}
    for page in range(1, 7):
        mat = sio.loadmat(str(DEP_DIR / "BoundingBox_Coordinates" / f"BoundingBoxPage_{page}.mat"))
        roi_list = mat["ROI_list"][0]
        boxes_per_page[page] = [np.array(b).flatten() for b in roi_list]
    return boxes_per_page

def load_product_descriptions():
    mat = sio.loadmat(str(DEP_DIR / "Leaflet_Product_Descriptions.mat"))
    arr = mat["Product_Descriptions"]
    names = {}
    for page_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            product_number = page_idx * 24 + col_idx + 1
            names[product_number] = str(arr[page_idx, col_idx][0])
    return names

def parse_selected_products(xlsx_path):
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

def get_page_windows(marker_stream):
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

def find_product_dwells(page_num, t_start, t_end, pos_times, pos_vals, boxes):
    mask = (pos_times >= t_start) & (pos_times <= t_end)
    times_win = pos_times[mask]
    pos_win = pos_vals[mask]

    img_x = pos_win[:, 0] * SCALE_X
    img_y = pos_win[:, 1] * SCALE_Y

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
                if run_len >= MIN_DWELL_SAMPLES:
                    product_number = (page_num - 1) * 24 + current_box + 1
                    dwells.append((product_number, times_win[run_start_idx], times_win[i - 1]))
            current_box, run_start_idx = b, i
    if current_box is not None and current_box != -1:
        run_len = len(box_idx_per_sample) - run_start_idx
        if run_len >= MIN_DWELL_SAMPLES:
            product_number = (page_num - 1) * 24 + current_box + 1
            dwells.append((product_number, times_win[run_start_idx], times_win[-1]))
    return dwells

def extract_eeg_epoch(eeg_times, eeg_data, chan_indices, onset_time):
    start_t = onset_time - EPOCH_PRE_SEC
    end_t = onset_time + EPOCH_POST_SEC
    mask = (eeg_times >= start_t) & (eeg_times <= end_t)
    if mask.sum() < 10:
        return None
    seg = eeg_data[mask][:, chan_indices]
    return seg

def process_subject(subject_id, boxes_per_page, product_names):
    xdf_path = RAW_DIR / f"{subject_id}.xdf"
    xlsx_path = RAW_DIR / f"{subject_id}.xlsx"
    if not xdf_path.exists() or not xlsx_path.exists():
        return None

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    by_name = {s["info"]["name"][0]: s for s in streams}

    eeg_stream = by_name["WS-default"]
    eeg_labels_raw = [c["label"][0] for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    chan_indices = [eeg_labels_raw.index(ch) for ch in EEG_CHANNEL_ORDER if ch in eeg_labels_raw]
    used_channels = [ch for ch in EEG_CHANNEL_ORDER if ch in eeg_labels_raw]

    eeg_times = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"])

    marker_stream = by_name["MyMarkerStream3"]
    page_windows = get_page_windows(marker_stream)

    pos_stream = by_name["MousePosition"]
    pos_times = np.array(pos_stream["time_stamps"])
    pos_vals = np.array(pos_stream["time_series"])

    selected_products = parse_selected_products(xlsx_path)

    samples, labels, product_ids, product_names_list = [], [], [], []

    for page_num, t_start, t_end in page_windows:
        boxes = boxes_per_page[page_num]
        dwells = find_product_dwells(page_num, t_start, t_end, pos_times, pos_vals, boxes)
        for product_number, dwell_start, dwell_end in dwells:
            seg = extract_eeg_epoch(eeg_times, eeg_data, chan_indices, dwell_start)
            if seg is None:
                continue
            label = 1 if product_number in selected_products else 0
            samples.append(seg)
            labels.append(label)
            product_ids.append(product_number)
            product_names_list.append(product_names.get(product_number, "UNKNOWN"))

    if not samples:
        return None

    max_len = max(s.shape[0] for s in samples)
    padded = np.zeros((len(samples), max_len, len(used_channels)), dtype=np.float32)
    for i, s in enumerate(samples):
        padded[i, :s.shape[0], :] = s

    return {
        "subject_id": subject_id,
        "eeg": padded,
        "labels": np.array(labels, dtype=np.int64),
        "product_ids": np.array(product_ids, dtype=np.int64),
        "product_names": np.array(product_names_list, dtype=object),
        "channels_used": used_channels,
    }

def build_all_subjects():
    boxes_per_page = load_bounding_boxes()
    product_names = load_product_descriptions()

    xdf_files = sorted(RAW_DIR.glob("S*.xdf"))
    all_results = []
    for xdf_file in xdf_files:
        subject_id = xdf_file.stem
        print(f"Processing {subject_id}...")
        result = process_subject(subject_id, boxes_per_page, product_names)
        if result is None:
            print(f"  SKIPPED {subject_id} (missing stream or no valid epochs)")
            continue
        n_buy = result["labels"].sum()
        n_total = len(result["labels"])
        print(f"  {subject_id}: {n_total} product epochs, {n_buy} Buy / {n_total - n_buy} NoBuy, "
              f"EEG shape {result['eeg'].shape}")
        all_results.append(result)

    return all_results

if __name__ == "__main__":
    results = build_all_subjects()
    print(f"\nTotal subjects processed: {len(results)}")
    total_epochs = sum(len(r["labels"]) for r in results)
    print(f"Total product-level epochs across all subjects: {total_epochs}")

    out_path = Path("data/processed/neuma_real")
    out_path.mkdir(parents=True, exist_ok=True)
    for r in results:
        np.savez(out_path / f"{r['subject_id']}_real.npz",
                 eeg=r["eeg"], labels=r["labels"],
                 product_ids=r["product_ids"], product_names=r["product_names"])
    print(f"Saved per-subject real epoch files to {out_path}/")
