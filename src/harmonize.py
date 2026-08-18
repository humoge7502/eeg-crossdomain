import numpy as np
import mne
from scipy.signal import welch

_trapz = getattr(np, "trapezoid", None) or np.trapz

def map_to_shared_channels(epochs, shared_channels, nearest_neighbor_fallback=True):
    present = [ch for ch in shared_channels if ch in epochs.ch_names]
    missing = [ch for ch in shared_channels if ch not in epochs.ch_names]
    if not missing:
        return epochs.copy().pick(shared_channels)
    if not nearest_neighbor_fallback:
        print(f"[harmonize] WARNING: channels {missing} absent — only using {present}")
        return epochs.copy().pick(present)
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    rename_map = {}
    for miss_ch in missing:
        if miss_ch not in pos:
            continue
        target_xyz = pos[miss_ch]
        candidates = [ch for ch in epochs.ch_names if ch in pos and ch not in shared_channels]
        if not candidates:
            continue
        dists = {ch: np.linalg.norm(pos[ch] - target_xyz) for ch in candidates}
        nearest = min(dists, key=dists.get)
        rename_map[nearest] = miss_ch
        print(f"[harmonize] missing '{miss_ch}' -> substituting nearest electrode '{nearest}' (distance {dists[nearest]:.3f} m)")
    ep = epochs.copy()
    if rename_map:
        ep.rename_channels(rename_map)
    final_present = [ch for ch in shared_channels if ch in ep.ch_names]
    return ep.pick(final_present)

def band_power_features(epochs, bands, sfreq):
    data = epochs.get_data()
    n_epochs, n_channels, n_samples = data.shape
    nperseg = min(256, n_samples)
    feature_rows = []
    for ep_idx in range(n_epochs):
        row = []
        for ch_idx in range(n_channels):
            freqs, psd = welch(data[ep_idx, ch_idx], fs=sfreq, nperseg=nperseg)
            for band_name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                band_power = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
                row.append(band_power)
        feature_rows.append(row)
    return np.array(feature_rows)

def frontal_asymmetry_index(epochs, pairs, band=(8, 13), sfreq=None):
    sfreq = sfreq or epochs.info["sfreq"]
    data = epochs.get_data()
    n_epochs = data.shape[0]
    out = np.zeros((n_epochs, len(pairs)))
    for pair_idx, (left_ch, right_ch) in enumerate(pairs):
        if left_ch not in epochs.ch_names or right_ch not in epochs.ch_names:
            out[:, pair_idx] = np.nan
            continue
        li = epochs.ch_names.index(left_ch)
        ri = epochs.ch_names.index(right_ch)
        for ep_idx in range(n_epochs):
            f_l, p_l = welch(data[ep_idx, li], fs=sfreq, nperseg=min(256, data.shape[-1]))
            f_r, p_r = welch(data[ep_idx, ri], fs=sfreq, nperseg=min(256, data.shape[-1]))
            mask_l = (f_l >= band[0]) & (f_l <= band[1])
            mask_r = (f_r >= band[0]) & (f_r <= band[1])
            power_l = _trapz(p_l[mask_l], f_l[mask_l]) + 1e-12
            power_r = _trapz(p_r[mask_r], f_r[mask_r]) + 1e-12
            out[ep_idx, pair_idx] = np.log(power_r) - np.log(power_l)
    return out

def build_harmonized_features(epochs, cfg):
    hcfg = cfg["harmonization"]
    mapped = map_to_shared_channels(epochs, hcfg["shared_channels"], hcfg["nearest_neighbor_fallback"])
    bp = band_power_features(mapped, hcfg["bands"], sfreq=mapped.info["sfreq"])
    asym = frontal_asymmetry_index(mapped, hcfg["asymmetry_pairs"], sfreq=mapped.info["sfreq"])
    return np.concatenate([bp, asym], axis=1)
