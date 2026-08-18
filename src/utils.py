import random
import yaml
import numpy as np
from pathlib import Path

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_device(preferred: str = "cuda"):
    import torch
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] Using CPU.")
    return dev

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
