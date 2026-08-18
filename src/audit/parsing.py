"""Pure, testable helpers for participant-ID and label parsing used by the read-only dataset audit."""
import re
from pathlib import Path
from typing import Optional, Dict

DS007406_LABEL_MAP = {"extreme": 1, "traditional": 0}

def parse_neuma_subject_id(filename: str) -> Optional[str]:
    """'S01.xdf' -> 'S01'; returns None if it does not match."""
    m = re.match(r"^(S\d+)\.xdf$", Path(filename).name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def parse_bids_subject_id(path: str) -> Optional[str]:
    """'.../sub-003/eeg/sub-003_task-x_eeg.set' -> 'sub-003'."""
    m = re.search(r"(sub-\d+)", str(path))
    return m.group(1) if m else None

def parse_generic_subject_id(filename: str, regex: str) -> Optional[str]:
    m = re.search(regex, Path(filename).name)
    if not m:
        return None
    digits = m.group(1)
    return f"P{int(digits):03d}"

def ds007406_label_from_trial_type(trial_type: str) -> Optional[int]:
    """Map a BIDS events.tsv trial_type string to 1=extreme, 0=traditional, None=other/unknown."""
    t = str(trial_type).strip().lower()
    for key, val in DS007406_LABEL_MAP.items():
        if key in t:
            return val
    return None

def count_labels(labels) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for l in labels:
        k = "unlabelled" if l is None else str(int(l))
        out[k] = out.get(k, 0) + 1
    return out
