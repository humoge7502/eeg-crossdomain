import pyxdf
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("/home/nvidia/24PHD1314/Quality-Aware-Consumer-Response/data/neuma_raw")
DEP_DIR = RAW_DIR / "Dependencies" / "Dependencies"

REASON_LETTERS = set("ABCDEFGHIJ")

def load_bounding_boxes():
    boxes_per_page = {}
    for page in range(1, 7):
        mat = sio.loadmat(str(DEP_DIR / "BoundingBox_Coordinates" / f"BoundingBoxPage_{page}.mat"))
        roi_list = mat["ROI_list"][0]
        boxes = [np.array(b).flatten() for b in roi_list]
        boxes_per_page[page] = boxes
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
        for val in df[col].dropna():
            val = str(val).strip()
            digits = "".join(c for c in val if c.isdigit())
            if digits:
                selected.add(int(digits))
    return selected

def load_streams(xdf_path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    by_name = {s["info"]["name"][0]: s for s in streams}
    return by_name

def get_page_windows(marker_stream):
    events = marker_stream["time_series"]
    times = marker_stream["time_stamps"]
    windows = []
    current_page = None
    current_start = None
    for i, ev in enumerate(events):
        text = ev[0]
        if text.startswith("Category:IMG=ID:FYLLADIO_"):
            page_num = int(text.split("FYLLADIO_")[1].split(".")[0])
            if current_page is not None:
                windows.append((current_page, current_start, times[i]))
            current_page = page_num
            current_start = times[i]
        elif text == "EOE" and current_page is not None:
            windows.append((current_page, current_start, times[i]))
            current_page = None
    return windows

def transform_to_image_coords(x, y):
    img_x = (x + 1920) * (3000 / 1920)
    img_y = y * (1688 / 1080)
    return img_x, img_y

def point_in_box(px, py, box):
    bx, by, bw, bh = box
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)

def validate_subject(subject_id, boxes_per_page):
    xdf_path = RAW_DIR / f"{subject_id}.xdf"
    streams = load_streams(xdf_path)

    marker_stream = streams["MyMarkerStream3"]
    mouse_pos_stream = streams["MousePosition"]

    page_windows = get_page_windows(marker_stream)
    pos_times = np.array(mouse_pos_stream["time_stamps"])
    pos_vals = np.array(mouse_pos_stream["time_series"])

    total_points = 0
    inside_points = 0

    for page_num, t_start, t_end in page_windows:
        mask = (pos_times >= t_start) & (pos_times <= t_end)
        page_positions = pos_vals[mask]
        boxes = boxes_per_page[page_num]

        for x, y in page_positions:
            img_x, img_y = transform_to_image_coords(x, y)
            total_points += 1
            for box in boxes:
                if point_in_box(img_x, img_y, box):
                    inside_points += 1
                    break

    fraction = inside_points / total_points if total_points > 0 else 0.0
    print(f"{subject_id}: {inside_points}/{total_points} mouse samples landed inside a product box ({fraction:.1%})")
    return fraction

if __name__ == "__main__":
    boxes_per_page = load_bounding_boxes()
    names = load_product_descriptions()
    print("Sample product names:", {k: names[k] for k in [1, 16, 34, 65, 144]})

    selected = parse_selected_products(RAW_DIR / "S01.xlsx")
    print("S01 selected product numbers:", sorted(selected))
    print("S01 selected product names:", [names.get(p, "UNKNOWN") for p in sorted(selected)])

    validate_subject("S01", boxes_per_page)
