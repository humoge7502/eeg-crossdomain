#!/usr/bin/env python
"""Read-only dataset audit for NeuMa, Restaurant-Logo, ds007406.
Reports EVERY candidate location listed in the config without choosing between them.
Also inspects existing processed .npz files (subject_ids present? label counts? shapes?).
Outputs: dataset_manifest.json, dataset_level.csv, participant_level.csv, processed_summary.csv, channel_substitution_placeholder.
"""
import argparse, json, csv, hashlib, zipfile, sys, random, datetime, glob, re
from pathlib import Path
from collections import Counter
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audit.parsing import (parse_neuma_subject_id, parse_bids_subject_id, parse_generic_subject_id,
                               ds007406_label_from_trial_type, count_labels)

def sha256(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def ext_hist(files): return dict(Counter(Path(f).suffix.lower() or "<none>" for f in files))

def audit_zip(zpath, smoke):
    z = zipfile.ZipFile(zpath); names = z.namelist()
    return {"type": "zip", "n_entries": len(names), "ext_hist": ext_hist(names), "sample_entries": names[:30],
            "testzip_ok": (z.testzip() is None) if not smoke else "skipped(smoke)"}

def audit_neuma(root, cfg, smoke, do_hash):
    root = Path(root); files = sorted(root.glob(cfg["eeg_glob"]))
    parts, rows = [], []
    for f in files[: (2 if smoke else None)]:
        sid = parse_neuma_subject_id(f.name); size = f.stat().st_size
        row = {"participant_id": sid, "file": str(f), "size_bytes": size, "sha256": sha256(f) if do_hash else ""}
        try:
            import pyxdf
            streams, hdr = pyxdf.load_xdf(str(f), dejitter_timestamps=False)
            row["xdf_streams"] = [s["info"]["name"][0] for s in streams]
            row["n_streams"] = len(streams)
        except Exception as e:
            row["xdf_error"] = str(e)[:200]
        rows.append(row)
    all_ids = [parse_neuma_subject_id(f.name) for f in files]
    aux = {g: len(list(root.glob(g))) for g in cfg.get("aux_globs", [])}
    return {"n_eeg_files": len(files), "participant_ids": all_ids, "duplicate_ids": [k for k,v in Counter(all_ids).items() if v>1],
            "aux_file_counts": aux, "note": "NeuMa Buy/NoBuy labels come from questionnaire xlsx (Q76-78) + click/mouse streams; label counts require the labelling pipeline and are NOT computed here."}, rows

def audit_ds007406(root, cfg, smoke, do_hash):
    root = Path(root); rows = []; missing = []
    ptsv = root/"participants.tsv"; desc = root/"dataset_description.json"
    subs = sorted(p for p in root.glob("sub-*") if p.is_dir())
    for s in subs[: (2 if smoke else None)]:
        sid = s.name; eeg = s/"eeg"; base = None
        sets = list(eeg.glob("*_eeg.set"))
        for suf in cfg["required_suffixes"]:
            if not list(eeg.glob(f"*{suf}")): missing.append({"participant_id": sid, "missing": suf})
        labels = []; n_events = 0; trial_types = Counter()
        ev = list(eeg.glob("*_events.tsv"))
        if ev:
            import pandas as pd
            df = pd.read_csv(ev[0], sep="\t"); n_events = len(df)
            col = "trial_type" if "trial_type" in df.columns else next((c for c in df.columns if "type" in c.lower() or "cond" in c.lower()), None)
            if col is not None:
                trial_types = Counter(df[col].astype(str)); labels = [ds007406_label_from_trial_type(t) for t in df[col]]
        row = {"participant_id": sid, "file": str(sets[0]) if sets else "", "n_events": n_events,
               "trial_type_counts": dict(trial_types), "label_counts": count_labels(labels),
               "fdt_present": bool(list(eeg.glob("*.fdt"))), "size_bytes": sets[0].stat().st_size if sets else 0,
               "sha256": sha256(sets[0]) if (do_hash and sets) else ""}
        rows.append(row)
    ids = [s.name for s in subs]
    return {"n_subject_dirs": len(subs), "participant_ids": ids, "duplicate_ids": [k for k,v in Counter(ids).items() if v>1],
            "participants_tsv": ptsv.exists(), "dataset_description": desc.exists(), "missing_required_files": missing}, rows

def audit_generic(root, cfg, smoke, do_hash):
    root = Path(root); files = sorted(p for p in root.glob(cfg["eeg_glob"]) if p.is_file())
    ids = [parse_generic_subject_id(f.name, cfg["participant_regex"]) for f in files]
    rows = [{"participant_id": i, "file": str(f), "size_bytes": f.stat().st_size, "sha256": sha256(f) if do_hash else ""}
            for i, f in list(zip(ids, files))[: (10 if smoke else None)]]
    return {"n_files": len(files), "ext_hist": ext_hist(files), "sample_files": [str(f) for f in files[:25]],
            "participant_ids": sorted(set(i for i in ids if i)), "n_unmatched_files": sum(i is None for i in ids),
            "note": "Restaurant-Logo raw format/labels not yet confirmed; audit lists files only."}, rows

def audit_processed(pdir, name, patterns):
    out = []
    for pat in patterns:
        p = Path(pdir)/name/pat.format(name=name)
        r = {"dataset": name, "file": str(p), "exists": p.exists()}
        if p.exists():
            try:
                d = np.load(p, allow_pickle=True); r["keys"] = list(d.keys())
                for k in d.keys():
                    r[f"shape_{k}"] = list(d[k].shape)
                if "y" in d: r["label_counts"] = dict(Counter(d["y"].tolist()))
                if "subject_ids" in d:
                    r["n_subjects"] = int(len(np.unique(d["subject_ids"]))); r["subject_ids"] = sorted(map(str, np.unique(d["subject_ids"])))
                else: r["WARNING"] = "no subject_ids -> subject-wise splitting impossible from this file"
                if "X" in d:
                    X = d["X"]; r["X_finite"] = bool(np.isfinite(X).all()); r["X_std"] = float(np.std(X))
            except Exception as e: r["error"] = str(e)[:200]
        out.append(r)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paths_audit.yaml"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="limit to first 2 participants per candidate, no hashing")
    ap.add_argument("--output-dir", default="outputs/audit"); ap.add_argument("--hash", action="store_true", help="sha256 every EEG file (slow)")
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    cfg = yaml.safe_load(open(a.config)); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    do_hash = a.hash and not a.smoke
    manifest = {"timestamp": datetime.datetime.now().isoformat(), "config": a.config, "seed": a.seed, "smoke": a.smoke, "datasets": {}}
    ds_rows, part_rows = [], []
    for name, dcfg in cfg["datasets"].items():
        manifest["datasets"][name] = {"label_task": dcfg["label_task"], "confirmed": dcfg.get("confirmed"), "candidates": []}
        for cand in dcfg["candidates"]:
            c = {"path": cand, "exists": Path(cand).exists()}
            if not c["exists"]: manifest["datasets"][name]["candidates"].append(c); ds_rows.append({"dataset": name, "candidate": cand, "exists": False}); continue
            try:
                if Path(cand).suffix.lower() == ".zip": summary, rows = audit_zip(cand, a.smoke), []
                elif name == "neuma": summary, rows = audit_neuma(cand, dcfg, a.smoke, do_hash)
                elif dcfg.get("bids"): summary, rows = audit_ds007406(cand, dcfg, a.smoke, do_hash)
                else: summary, rows = audit_generic(cand, dcfg, a.smoke, do_hash)
            except Exception as e: summary, rows = {"error": str(e)[:300]}, []
            c.update(summary); manifest["datasets"][name]["candidates"].append(c)
            ds_rows.append({"dataset": name, "candidate": cand, "exists": True, "n_participants": len(summary.get("participant_ids", [])),
                            "duplicates": len(summary.get("duplicate_ids", [])), "missing_files": len(summary.get("missing_required_files", [])), "error": summary.get("error", "")})
            for r in rows: part_rows.append({"dataset": name, "candidate": cand, **{k: (json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()}})
        manifest["datasets"][name]["n_existing_candidates"] = sum(c["exists"] for c in manifest["datasets"][name]["candidates"])
        manifest["datasets"][name]["processed"] = audit_processed(cfg["processed_dir"], name, cfg["processed_files"])
    (out/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    def wcsv(p, rows):
        keys = sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    wcsv(out/"dataset_level.csv", ds_rows); wcsv(out/"participant_level.csv", part_rows)
    proc = [{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for n in cfg["datasets"] for r in manifest["datasets"][n]["processed"]]
    wcsv(out/"processed_summary.csv", proc)
    print(json.dumps(manifest, indent=2, default=str)[:20000])
    print("\n=== SUMMARY ===")
    for r in ds_rows: print(r)
    print("\n=== PROCESSED (subject_ids present? label counts?) ===")
    for r in proc: print({k: r[k] for k in r if k in ("dataset","file","exists","n_subjects","label_counts","WARNING","error","shape_X")})
    print(f"\n[ok] wrote {out}/dataset_manifest.json, dataset_level.csv, participant_level.csv, processed_summary.csv")
    for n, d in manifest["datasets"].items():
        if d["n_existing_candidates"] > 1 and not d.get("confirmed"): print(f"[CONFIRM NEEDED] {n}: {d['n_existing_candidates']} candidate locations exist — set `confirmed:` in {a.config}")

if __name__ == "__main__": main()
