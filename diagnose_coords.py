import pyxdf
import scipy.io as sio
import numpy as np
from pathlib import Path

RAW_DIR = Path("/home/nvidia/24PHD1314/Quality-Aware-Consumer-Response/data/neuma_raw")
DEP_DIR = RAW_DIR / "Dependencies" / "Dependencies"

mat = sio.loadmat(str(DEP_DIR / "BoundingBox_Coordinates" / "BoundingBoxPage_1.mat"))
roi_list = mat["ROI_list"][0]
boxes = [np.array(b).flatten() for b in roi_list]
all_x = [b[0] for b in boxes] + [b[0]+b[2] for b in boxes]
all_y = [b[1] for b in boxes] + [b[1]+b[3] for b in boxes]
print("Bounding boxes: x range", min(all_x), "to", max(all_x), " y range", min(all_y), "to", max(all_y))

streams, _ = pyxdf.load_xdf(str(RAW_DIR / "S01.xdf"))
by_name = {s["info"]["name"][0]: s for s in streams}

marker_stream = by_name["MyMarkerStream3"]
events = marker_stream["time_series"]
times = marker_stream["time_stamps"]

page1_start, page1_end = None, None
for i, ev in enumerate(events):
    text = ev[0]
    if text == "Category:IMG=ID:FYLLADIO_1.tif=Type:Leaflet_Images_1" and page1_start is None:
        page1_start = times[i]
    elif page1_start is not None and page1_end is None and (text.startswith("Category:IMG") or text == "EOE"):
        page1_end = times[i]
        break

print("Page 1 window:", page1_start, "to", page1_end)

pos_stream = by_name["MousePosition"]
pos_times = np.array(pos_stream["time_stamps"])
pos_vals = np.array(pos_stream["time_series"])
mask = (pos_times >= page1_start) & (pos_times <= page1_end)
page1_positions = pos_vals[mask]

print("Raw mouse positions during page 1: x range", page1_positions[:,0].min(), "to", page1_positions[:,0].max())
print("Raw mouse positions during page 1: y range", page1_positions[:,1].min(), "to", page1_positions[:,1].max())
print("Number of samples in this window:", len(page1_positions))
print("First 10 raw positions:", page1_positions[:10])

x_shifted = page1_positions[:,0] + 1920
y_shifted = page1_positions[:,1]
print("Shifted-only (no scale) x range:", x_shifted.min(), "to", x_shifted.max())
print("Shifted-only (no scale) y range:", y_shifted.min(), "to", y_shifted.max())
