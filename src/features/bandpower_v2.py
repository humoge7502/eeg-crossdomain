"""Band-power feature extraction (v2). Input epochs in volts, shape [n, ch, T]."""
import numpy as np
from scipy.signal import welch
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]

def compute_features(ep, sfreq, ch_names, bands, total_range, faa_pair=("F3", "F4"), eps=1e-30):
    n, c, T = ep.shape
    nperseg = int(min(T, round(2 * sfreq))); nov = nperseg // 2
    f, psd = welch(ep, fs=sfreq, window="hann", nperseg=nperseg, noverlap=nov, axis=-1)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return np.trapezoid(psd[..., m], f[m], axis=-1) if m.sum() > 1 else np.zeros((n, c))
    absp = np.stack([bp(*bands[b]) for b in BAND_ORDER], axis=-1)
    total = bp(*total_range)[..., None]
    rel = absp / np.maximum(total, eps)
    X_rel = np.log10(np.maximum(rel, 1e-12)).reshape(n, c * len(BAND_ORDER))
    X_abs = np.log10(np.maximum(absp, eps)).reshape(n, c * len(BAND_ORDER))
    li, ri, ai = ch_names.index(faa_pair[0]), ch_names.index(faa_pair[1]), BAND_ORDER.index("alpha")
    faa = np.log(np.maximum(absp[:, ri, ai], eps)) - np.log(np.maximum(absp[:, li, ai], eps))
    names = [f"{ch}_{b}" for ch in ch_names for b in BAND_ORDER] + [f"FAA_{faa_pair[1]}_minus_{faa_pair[0]}_alpha"]
    meta = {"nperseg": nperseg, "noverlap": nov, "n_freqs": int(len(f)), "freq_res_hz": float(f[1] - f[0])}
    return (np.concatenate([X_rel, faa[:, None]], 1).astype(np.float32),
            np.concatenate([X_abs, faa[:, None]], 1).astype(np.float32), names, meta)
