#!/usr/bin/env python
"""Read-only environment audit. Writes JSON + CSV to --output-dir."""
import argparse, json, os, platform, shutil, subprocess, sys, importlib, datetime, random
from pathlib import Path
import numpy as np

PKGS = ["numpy","scipy","pandas","sklearn","xgboost","torch","mne","pyxdf","pyriemann","statsmodels","yaml","matplotlib","seaborn","pytest","h5py","mat73","tqdm"]

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERROR: {e}"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paths_audit.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default="outputs/audit")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rep = {"timestamp": datetime.datetime.now().isoformat(), "cwd": os.getcwd(), "python": sys.version,
           "executable": sys.executable, "virtual_env": os.environ.get("VIRTUAL_ENV"), "conda_prefix": os.environ.get("CONDA_PREFIX"),
           "platform": platform.platform(), "cpu_count": os.cpu_count(), "seed": a.seed, "smoke": a.smoke, "config": a.config}
    du = shutil.disk_usage(os.getcwd()); rep["disk_gb"] = {"total": round(du.total/1e9,1), "free": round(du.free/1e9,1)}
    try:
        du2 = shutil.disk_usage("/mnt"); rep["disk_mnt_gb"] = {"total": round(du2.total/1e9,1), "free": round(du2.free/1e9,1)}
    except Exception: pass
    pk = {}
    for m in PKGS:
        try: pk[m] = getattr(importlib.import_module(m), "__version__", "present")
        except Exception as e: pk[m] = f"MISSING ({type(e).__name__})"
    rep["packages"] = pk
    try:
        import torch
        rep["torch"] = {"cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
                        "device_count": torch.cuda.device_count(),
                        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as e: rep["torch"] = {"error": str(e)}
    rep["nvidia_smi"] = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader")
    rep["git"] = {"inside": sh("git rev-parse --is-inside-work-tree"), "head": sh("git rev-parse HEAD 2>&1"), "status_lines": len(sh("git status --short").splitlines())}
    (out/"environment_audit.json").write_text(json.dumps(rep, indent=2))
    import csv
    with open(out/"environment_packages.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["package","version"]); [w.writerow([k,v]) for k,v in pk.items()]
    print(json.dumps(rep, indent=2)); print(f"\n[ok] wrote {out/'environment_audit.json'} and {out/'environment_packages.csv'}")

if __name__ == "__main__": main()
