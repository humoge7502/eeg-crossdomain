import json, glob, sys; from pathlib import Path
import pytest, numpy as np
ROOT = Path(__file__).resolve().parent.parent
@pytest.mark.parametrize("f", glob.glob(str(ROOT/"data/processed_v2/splits/within_subjectwise_*.json")))
def test_no_subject_overlap_and_full_coverage(f):
    j = json.load(open(f)); ds = j["dataset"]; d = np.load(ROOT/f"data/processed_v2/{ds}/{ds}_features_v2.npz", allow_pickle=True); subs = set(d["subject_ids"].astype(str))
    for seed, folds in j["folds_by_seed"].items():
        tests = []
        for fd in folds:
            tr, va, te = set(fd["train"]), set(fd["val"]), set(fd["test"])
            assert not (tr & te) and not (va & te) and not (tr & va); tests += fd["test"]
        assert sorted(tests) == sorted(subs)   # every subject tested exactly once per seed
@pytest.mark.parametrize("f", glob.glob(str(ROOT/"data/processed_v2/splits/transfer_target_*.json")))
def test_transfer_fractions_nested_and_disjoint_from_test(f):
    j = json.load(open(f))
    for seed, s in j["by_seed"].items():
        te = set(s["test"]); prev = set()
        for k in ["10pct", "25pct", "50pct", "100pct"]:
            cur = set(s["fractions"][k]["train"]) | set(s["fractions"][k]["val"])
            assert not (cur & te) and prev <= cur; prev = cur
