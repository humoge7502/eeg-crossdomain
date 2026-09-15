import pyxdf
import scipy.io as sio
import numpy as np
from pathlib import Path

RAW_DIR = Path("/home/nvidia/24PHD1314/Quality-Aware-Consumer-Response/data/neuma_raw")
DEP_DIR = RAW_DIR / "Dependencies" / "Dependencies"

mat = sio.loadmat(str(DEP_DIR / "BoundingBox_Coordinates" / "BoundingBoxPage_1.mat"))
roi_list = mat["ROI_list"][0]
boxes = [np.array(b).flatten() for b in roi_list]

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

pos_stream = by_name["MousePosition"]
pos_times = np.array(pos_stream["time_stamps"])
pos_vals = np.array(pos_stream["time_series"])
mask = (pos_times >= page1_start) & (pos_times <= page1_end)
page1_positions = pos_vals[mask]

scale_x = 3000 / 1920
scale_y = 1688 / 1080

img_x = page1_positions[:,0] * scale_x
img_y = page1_positions[:,1] * scale_y

print("Transformed (scale-only) x range:", img_x.min(), "to", img_x.max())
print("Transformed (scale-only) y range:", img_y.min(), "to", img_y.max())

inside_count = 0
for px, py in zip(img_x, img_y):
    for bx, by, bw, bh in boxes:
        if bx <= px <= bx+bw and by <= py <= by+bh:
            inside_count += 1
            break

print(f"Inside a product box: {inside_count}/{len(img_x)} ({inside_count/len(img_x):.1%})")

for idx, (bx, by, bw, bh) in enumerate(boxes):
    print(f"  box {idx}: x=[{bx:.0f},{bx+bw:.0f}] y=[{by:.0f},{by+bh:.0f}]")
