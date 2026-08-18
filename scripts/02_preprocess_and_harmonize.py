#!/usr/bin/env python
"""Phase 2: build v2 harmonized representations for NeuMa, Restaurant-Logo, ds007406 (read-only on raw; writes data/processed_v2)."""
import argparse, json, csv, sys, logging, datetime, random
from fractions import Fraction
from pathlib import Path
import numpy as np, yaml
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.preprocessing.harmonize_v2 import plan_substitutions, apply_substitutions
from src.features.bandpower_v2 import compute_features, BAND_ORDER
from src.data import loaders_v2 as L

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_v2.yaml"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 subjects per dataset -> <output_root>_smoke")
    ap.add_argument("--output-dir", default=None); ap.add_argument("--datasets", nargs="+", default=["neuma", "restaurant_logo", "ds007406"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT/a.config)); paths = yaml.safe_load(open(ROOT/cfg["paths_config"]))
    seed = a.seed if a.seed is not None else cfg["seed"]; random.seed(seed); np.random.seed(seed)
    out_root = ROOT/(a.output_dir or (cfg["output_root"] + ("_smoke" if a.smoke else ""))); out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); (ROOT/"outputs/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(ROOT/f"outputs/logs/02_preprocess_v2_{ts}{'_smoke' if a.smoke else ''}.log"), logging.StreamHandler()])
    log = logging.getLogger("p2"); log.info(f"config={a.config} seed={seed} smoke={a.smoke} out={out_root}")
    shared = cfg["shared_channels"]; band = cfg["preprocessing"]["bandpass_hz"]; bands = {k: tuple(v) for k, v in cfg["features"]["bands_hz"].items()}
    subs_rows, part_rows, summary = [], [], {"config": cfg, "seed": seed, "smoke": a.smoke, "timestamp": ts, "datasets": {}}
    for name in a.datasets:
        raw = Path(paths["datasets"][name]["confirmed"]); ecfg = cfg["epochs"][name]; line = cfg["line_freq_hz"][name] if cfg["preprocessing"]["notch"] else None
        us = float(cfg["preprocessing"]["unit_scale_to_volts"][name]); car_ex = cfg["preprocessing"].get("car_exclude", {}).get(name, [])
        if name == "neuma": files = sorted(raw.glob("S*.xdf")); loader = lambda f: L.load_neuma_subject(f, ecfg, band, line, us, car_ex)
        elif name == "restaurant_logo":
            files = sorted(raw.glob("Subject */Sub*_Test.set"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()))); sheet = L.rl_sheet_labels(raw/"ExperimentResults.xlsx")
            loader = lambda f: L.load_rl_subject(f, ecfg, band, line, us, sheet)
        else: files = sorted(raw.glob("sub-*/eeg/*_eeg.set")); loader = lambda f: L.load_ds_subject(f, ecfg, band, line, us)
        if a.smoke: files = files[:2]
        log.info(f"[{name}] {len(files)} subject files from {raw}")
        Xr, Xa, E, Y, SID, TID, META, W, WY, WS, WT, WI = [], [], [], [], [], [], [], [], [], [], [], []
        plan_logged = None; feat_names = None; sfreq0 = None; failures = []
        for f in files:
            try:
                r = loader(f)
                plan = plan_substitutions(r["ch_names"], shared, cfg["substitution"]["tie_tolerance_m"])
                if plan_logged is None or plan != plan_logged:
                    for p in plan: subs_rows.append({"dataset": name, "subject_id": r["subject_id"], **p, "sources": "+".join(p["sources"])})
                    plan_logged = plan
                ep7 = apply_substitutions(r["eeg"].astype(np.float64), r["ch_names"], plan)
                xr, xa, feat_names, fm = compute_features(ep7, r["sfreq"], shared, bands, tuple(cfg["features"]["total_power_range_hz"]))
                fr = Fraction(cfg["common_windows"]["sfreq"], int(round(r["sfreq"]))); ep250 = resample_poly(ep7, fr.numerator, fr.denominator, axis=-1) if fr != 1 else ep7
                wl = int(round(cfg["common_windows"]["window_s"] * cfg["common_windows"]["sfreq"])); nw = ep250.shape[-1] // wl
                for i in range(len(r["y"])):
                    for w in range(nw):
                        W.append(ep250[i, :, w*wl:(w+1)*wl].astype(np.float32)); WY.append(r["y"][i]); WS.append(r["subject_id"]); WT.append(r["trial_ids"][i]); WI.append(w)
                Xr.append(xr); Xa.append(xa); E.append(ep7.astype(np.float32)); Y.append(r["y"]); SID += [r["subject_id"]] * len(r["y"]); TID += r["trial_ids"]
                for tid, m, yy in zip(r["trial_ids"], r["meta"], r["y"]): META.append({"dataset": name, "subject_id": r["subject_id"], "trial_id": tid, "y": int(yy), **m})
                sfreq0 = r["sfreq"]
                pr = {"dataset": name, "subject_id": r["subject_id"], "file": str(f), "n_trials": len(r["y"]), "n_pos": int(r["y"].sum()), "n_neg": int(len(r["y"]) - r["y"].sum()),
                      "sfreq": r["sfreq"], "n_native_ch": len(r["ch_names"]), "native_channels": "|".join(r["ch_names"]), "n_windows": nw * len(r["y"]), "features_finite": bool(np.isfinite(xr).all()), **{f"note_{k}": v for k, v in r["notes"].items()}}
                part_rows.append(pr); log.info(f"  {r['subject_id']}: {pr['n_trials']} trials ({pr['n_pos']}+/{pr['n_neg']}-) notes={r['notes']}")
            except Exception as e:
                failures.append({"dataset": name, "file": str(f), "error": str(e)[:300]}); log.exception(f"  FAILED {f.name}: {e!r}")
        if not Y: log.error(f"[{name}] nothing loaded"); continue
        d = out_root/name; d.mkdir(parents=True, exist_ok=True)
        y = np.concatenate(Y); sid = np.array(SID); tid = np.array(TID)
        np.savez_compressed(d/f"{name}_features_v2.npz", X=np.concatenate(Xr), X_abs=np.concatenate(Xa), y=y, subject_ids=sid, trial_ids=tid, feature_names=np.array(feat_names), dataset=name)
        np.savez_compressed(d/f"{name}_epochs7_native_v2.npz", X=np.concatenate(E), y=y, subject_ids=sid, trial_ids=tid, ch_names=np.array(shared), sfreq=sfreq0)
        np.savez_compressed(d/f"{name}_windows_common_v2.npz", X=np.stack(W), y=np.array(WY), subject_ids=np.array(WS), trial_ids=np.array(WT), window_idx=np.array(WI), ch_names=np.array(shared), sfreq=cfg["common_windows"]["sfreq"])
        with open(d/f"{name}_trial_manifest_v2.csv", "w", newline="") as fh:
            keys = sorted({k for m in META for k in m}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(m) for m in META]
        summary["datasets"][name] = {"n_subjects": int(len(np.unique(sid))), "n_trials": int(len(y)), "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}, "n_windows": len(W),
                                     "feature_dim": int(np.concatenate(Xr).shape[1]), "native_sfreq": sfreq0, "substitution_plan": plan_logged, "failures": failures}
        log.info(f"[{name}] DONE subjects={summary['datasets'][name]['n_subjects']} trials={len(y)} classes={summary['datasets'][name]['class_counts']} windows={len(W)} failures={len(failures)}")
    for fn, rows in (("participant_manifest_v2.csv", part_rows), ("channel_substitutions_v2.csv", subs_rows)):
        with open(out_root/fn, "w", newline="") as fh:
            keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    (out_root/"feature_spec_v2.json").write_text(json.dumps({"feature_order": f"{len(shared)} channels {shared} x bands {BAND_ORDER} (channel-major) + FAA", **summary}, indent=2, default=str))
    (ROOT/"outputs/tables").mkdir(parents=True, exist_ok=True)
    if not a.smoke:
        import shutil; shutil.copy(out_root/"participant_manifest_v2.csv", ROOT/"outputs/tables/participant_manifest_v2.csv"); shutil.copy(out_root/"channel_substitutions_v2.csv", ROOT/"outputs/tables/channel_substitutions_v2.csv")
    log.info("SUMMARY " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "substitution_plan"} for k, v in summary["datasets"].items()}, default=str))
    log.info("SUBSTITUTIONS " + json.dumps({k: v["substitution_plan"] for k, v in summary["datasets"].items()}, default=str))

if __name__ == "__main__": main()
