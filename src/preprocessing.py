import mne
import numpy as np

def preprocess_raw(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    pcfg = cfg["preprocessing"]
    raw = raw.copy()
    raw.load_data(verbose=False)
    if pcfg.get("resample_to"):
        raw.resample(pcfg["resample_to"], verbose=False)
    raw.filter(l_freq=pcfg["bandpass_low"], h_freq=pcfg["bandpass_high"], fir_design="firwin", verbose=False)
    if pcfg.get("notch_freq"):
        raw.notch_filter(freqs=pcfg["notch_freq"], verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    n_ch = len(raw.ch_names)
    n_components = min(pcfg["ica_n_components"], n_ch - 1)
    if n_components >= 2:
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=cfg["training"]["seed"], verbose=False)
        ica.fit(raw, verbose=False)
        frontal_candidates = [ch for ch in raw.ch_names if ch.lower().startswith(("fp", "af", "f"))]
        if frontal_candidates:
            try:
                eog_indices, _ = ica.find_bads_eog(raw, ch_name=frontal_candidates[0], verbose=False)
                ica.exclude = eog_indices
            except Exception:
                pass
        ica.apply(raw, verbose=False)
    else:
        print(f"[preprocess] Only {n_ch} channels — skipping ICA.")
    return raw

def epoch_from_events(raw, events, event_id, tmin, tmax, baseline=(None, 0)):
    if len(events) == 0:
        raise ValueError("No events found — check event extraction in the dataset loader.")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False)
    return epochs
