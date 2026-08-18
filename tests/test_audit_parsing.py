import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audit.parsing import *

def test_neuma_ids():
    assert parse_neuma_subject_id("S01.xdf") == "S01"
    assert parse_neuma_subject_id("/x/y/s12.XDF") == "S12"
    assert parse_neuma_subject_id("Neuma_RawDataset_Info.pdf") is None

def test_bids_ids():
    assert parse_bids_subject_id("/d/sub-003/eeg/sub-003_task-a_eeg.set") == "sub-003"
    assert parse_bids_subject_id("participants.tsv") is None

def test_generic_ids():
    assert parse_generic_subject_id("subject_7_logo.csv", r"(?i)(?:sub|subj|subject|s|p)[-_ ]?(\d+)") == "P007"
    assert parse_generic_subject_id("readme.txt", r"(?i)subject[-_ ]?(\d+)") is None

def test_ds007406_labels():
    assert ds007406_label_from_trial_type("extreme_video") == 1
    assert ds007406_label_from_trial_type("Traditional") == 0
    assert ds007406_label_from_trial_type("fixation") is None
    assert count_labels([1, 0, 0, None]) == {"1": 1, "0": 2, "unlabelled": 1}
