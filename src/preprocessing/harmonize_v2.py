"""Channel harmonization with fully recorded substitutions (v2)."""
import numpy as np, mne
mne.set_log_level("ERROR")
_POS = None
_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

def positions():
    global _POS
    if _POS is None:
        _POS = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return _POS

def apply_car(data, ch_names, exclude=()):
    idx = [i for i, c in enumerate(ch_names) if c not in exclude]
    return data - data[..., idx, :].mean(axis=-2, keepdims=True)

def plan_substitutions(ch_names, shared, tie_tol_m=0.002):
    pos = positions(); plan = []
    for tgt in shared:
        if tgt in ch_names:
            plan.append({"target": tgt, "sources": [tgt], "distance_m": 0.0, "method": "present"}); continue
        cands = [c for c in ch_names if c not in shared and _ALIAS.get(c, c) in pos]
        if tgt not in pos or not cands:
            plan.append({"target": tgt, "sources": [], "distance_m": None, "method": "UNAVAILABLE"}); continue
        d = {c: float(np.linalg.norm(pos[_ALIAS.get(c, c)] - pos[tgt])) for c in cands}
        dmin = min(d.values()); srcs = sorted(c for c in cands if d[c] <= dmin + tie_tol_m)
        plan.append({"target": tgt, "sources": srcs, "distance_m": round(dmin, 4),
                     "method": "nearest_neighbour" if len(srcs) == 1 else "mean_of_equidistant_neighbours"})
    return plan

def apply_substitutions(data, ch_names, plan):
    out = []
    for p in plan:
        if not p["sources"]: raise ValueError(f"cannot harmonize channel {p['target']}")
        idx = [ch_names.index(s) for s in p["sources"]]
        out.append(data[..., idx, :].mean(axis=-2))
    return np.stack(out, axis=-2)
