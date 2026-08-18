import sys, numpy as np; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.harmonize_v2 import plan_substitutions, apply_substitutions
from src.features.bandpower_v2 import compute_features, BAND_ORDER
from src.data.loaders_v2 import pair_stimuli_with_presses
SHARED = ["Fz","Cz","Pz","F3","F4","O1","O2"]; BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,40)}

def test_substitution_plan_records_distances():
    plan = plan_substitutions(["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"], SHARED)
    d = {p["target"]: p for p in plan}
    assert d["F3"]["method"] == "present" and d["F3"]["distance_m"] == 0.0
    assert d["Fz"]["sources"] == ["AF3","AF4"] and d["Fz"]["distance_m"] > 0
    assert d["Pz"]["sources"] == ["P7","P8"]
    assert all(p["sources"] for p in plan)

def test_substitution_apply_shape_and_mean():
    chs = ["C4","Cz","Fz","C3","Pz","PO7","Oz","PO8"]; plan = plan_substitutions(chs, SHARED)
    x = np.arange(8*10, dtype=float).reshape(8,10); out = apply_substitutions(x, chs, plan)
    assert out.shape == (7,10)
    f3 = [p for p in plan if p["target"]=="F3"][0]; assert f3["sources"] == ["C3"]
    assert np.allclose(out[3], x[chs.index("C3")])

def test_bandpower_alpha_peak_and_faa_sign():
    sf, T = 250.0, 750; t = np.arange(T)/sf; rng = np.random.default_rng(0)
    ep = rng.normal(0, 1e-6, (2, 7, T)); ep[:, 4, :] += 20e-6*np.sin(2*np.pi*10*t)  # alpha at F4 only
    Xr, Xa, names, meta = compute_features(ep, sf, SHARED, BANDS, (1,40))
    assert Xr.shape == (2,36) and len(names) == 36 and np.isfinite(Xr).all()
    i = SHARED.index("F4")*5 + BAND_ORDER.index("alpha"); assert Xr[0, i] == Xr[0, SHARED.index("F4")*5:SHARED.index("F4")*5+5].max()
    assert (Xr[:, -1] > 0).all()  # FAA = ln(F4) - ln(F3) positive

def test_pairing_uses_marker_before_press_and_ignores_fixation_markers():
    markers = [3,6,9,12,15,18]; presses = [10.4,13.4,18.7]; vals = [("A",1),("A",1),("P",0)]
    tr = pair_stimuli_with_presses(markers, presses, vals, 3.0)
    assert [x["onset"] for x in tr] == [9,12,18] and [x["press"][1] for x in tr] == [1,1,0]
    assert abs(tr[0]["rt_s"] - 1.4) < 1e-9

def test_lcs_alignment_handles_extra_and_missing_presses():
    from src.data.loaders_v2 import align_press_to_sheet
    sheet = [1,1,1,0,0,0,0,1,1,1]
    extra = [1]+sheet; pairs = align_press_to_sheet(extra, sheet)
    assert len(pairs) == 10 and all(extra[i] == sheet[j] for i, j in pairs)          # spurious extra press: all 10 sheet rows recovered
    assert [j for _, j in pairs] == list(range(10))
    missing = sheet[:3]+sheet[4:]; pairs = align_press_to_sheet(missing, sheet)
    assert len(pairs) == 9 and all(missing[i] == sheet[j] for i, j in pairs)          # one missed press: 9 recovered
    assert [i for i, _ in pairs] == list(range(9))
    wrong = [1-x for x in sheet]; assert len(align_press_to_sheet(wrong, sheet)) < 10  # inverted labels align poorly
