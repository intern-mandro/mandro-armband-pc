"""
features.py
===========
EMG feature extraction.

Two feature families, concatenated by the pipeline (FEATURE_MODE = "concat"):
  - Classic features : per-channel time-domain and frequency-domain features
  - TSD features      : Temporal-Spatial Descriptors (Khushaba et al., 2017)
"""

import numpy as np


# =============================================================================
# Classic time-domain features
# =============================================================================

def MAV(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value - shape (C,)"""
    return np.mean(np.abs(window), axis=0)


def RMS(window: np.ndarray) -> np.ndarray:
    """Root Mean Square - shape (C,)"""
    return np.sqrt(np.mean(window ** 2, axis=0))


def WL(window: np.ndarray) -> np.ndarray:
    """Waveform Length - shape (C,)"""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def ZC(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Zero Crossing count - shape (C,)"""
    signs = np.sign(window - threshold)
    crossings = np.diff(signs, axis=0)
    return np.sum(crossings != 0, axis=0).astype(float)


def SSC(window: np.ndarray) -> np.ndarray:
    """Slope Sign Change count - shape (C,)"""
    diff1 = np.diff(window, axis=0)
    signs = np.sign(diff1)
    changes = np.diff(signs, axis=0)
    return np.sum(changes != 0, axis=0).astype(float)


def MaxAV(window: np.ndarray) -> np.ndarray:
    """Maximum Absolute Value - shape (C,)"""
    return np.max(np.abs(window), axis=0)


def STD(window: np.ndarray) -> np.ndarray:
    """Standard Deviation - shape (C,)"""
    return np.std(window, axis=0)


# =============================================================================
# Frequency-domain features
# =============================================================================

def _freq_features(window: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute 5 frequency-domain features for each channel.

    Returns
    -------
    np.ndarray  shape (5 * C,)
        [MeanPower, TotalPower, MeanFreq, MedianFreq, PeakFreq] x channels
    """
    N, C = window.shape
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    fft_power = np.abs(np.fft.rfft(window, axis=0)) ** 2  # (N//2+1, C)

    features = []
    for ch in range(C):
        p = fft_power[:, ch]
        total = p.sum() + 1e-12
        mean_pow   = p.mean()
        total_pow  = total
        mean_freq  = np.sum(freq * p) / total
        cumsum     = np.cumsum(p)
        med_idx    = min(np.searchsorted(cumsum, p.sum() / 2), len(freq) - 1)
        med_freq   = freq[med_idx]
        peak_freq  = freq[np.argmax(p)]
        features.extend([mean_pow, total_pow, mean_freq, med_freq, peak_freq])

    return np.array(features)


# =============================================================================
# Classic feature extraction
# =============================================================================

def extract_features(
    X: np.ndarray,
    sampling_frequency_EMG: int = 900,
) -> np.ndarray:
    """
    Extract MAV, MaxAV, STD, RMS, WL, SSC + 5 frequency-domain features
    for each window.

    Parameters
    ----------
    X : np.ndarray  shape (n_windows, N_samples, C_channels)
    sampling_frequency_EMG : int

    Returns
    -------
    np.ndarray  shape (n_windows, n_features)
        n_features = (6 time + 5 freq) x C_channels = 11 x C
    """
    all_features = []
    for window in X:
        time_feats = np.concatenate([
            MAV(window),
            MaxAV(window),
            STD(window),
            RMS(window),
            WL(window),
            SSC(window),
        ])
        freq_feats = _freq_features(window, sampling_frequency_EMG)
        all_features.append(np.concatenate([time_feats, freq_feats]))
    return np.array(all_features)


def get_feature_labels(n_channels: int) -> list[str]:
    """
    Names in the EXACT order produced by extract_features.

    Time-domain features are grouped BY FEATURE (mav_ch0..3, then maxav_ch0..3, ...)
    Frequency-domain features are grouped BY CHANNEL (mpow_ch0, tpow_ch0, mfreq_ch0,
    medfreq_ch0, peakfreq_ch0, then mpow_ch1, tpow_ch1, ...).

    This matches the layout produced by:
      np.concatenate([MAV, MaxAV, STD, RMS, WL, SSC])   <- grouped by feature
      + _freq_features                                   <- grouped by channel
    """
    time_names = ['mav', 'maxav', 'std', 'rms', 'wl', 'ssc']
    freq_names = ['mpow', 'tpow', 'mfreq', 'medfreq', 'peakfreq']
    labels = []
    # Time-domain: grouped by feature
    for feat in time_names:
        for ch in range(n_channels):
            labels.append(f"{feat}_ch{ch}")
    # Frequency-domain: grouped by channel
    for ch in range(n_channels):
        for feat in freq_names:
            labels.append(f"{feat}_ch{ch}")
    return labels


# =============================================================================
# TSD features - Temporal-Spatial Descriptors (Khushaba et al., 2017)
# =============================================================================

def _cosine_sim_elementwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise (channel-wise) cosine similarity."""
    num = np.sum(a * b, axis=0)
    den = (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)) + 1e-12
    return num / den


def compute_tdd(window: np.ndarray) -> np.ndarray:
    """
    Temporal Difference Descriptor (normalized discrete difference).

    Parameters
    ----------
    window : np.ndarray  shape (N, C)

    Returns
    -------
    np.ndarray  shape (N-1, C)
    """
    diff = np.diff(window, axis=0)
    norm = np.linalg.norm(diff, axis=0, keepdims=True) + 1e-12
    return diff / norm


def compute_tsd(
    window: np.ndarray,
    lam: float = 0.1,
) -> np.ndarray:
    """
    Temporal-Spatial Descriptor (TSD) for one window.

    Parameters
    ----------
    window : np.ndarray  shape (N, C)
    lam    : float       regularization parameter lambda

    Returns
    -------
    np.ndarray  shape (C*(C+1)//2 + C,)
    """
    N, C = window.shape
    cov = np.cov(window.T) + lam * np.eye(C)
    idx = np.triu_indices(C)
    cov_feats = cov[idx]
    energy = np.mean(window ** 2, axis=0)

    return np.concatenate([cov_feats, energy])


def extract_features_tsd(
    X: np.ndarray,
    sampling_frequency_EMG: int = 900,
    win_ms: int = 115,
    inc_ms: int = 57,
    lam: float = 0.1,
) -> np.ndarray:
    """
    Extract TSD features for each window of X.

    Parameters
    ----------
    X                       : np.ndarray  shape (n_windows, N_samples, C_channels)
    sampling_frequency_EMG  : int
    win_ms                  : int  internal sub-window size (ms)
    inc_ms                  : int  sub-window increment (ms)
    lam                     : float

    Returns
    -------
    np.ndarray  shape (n_windows, n_tsd_features)
    """
    win_samples = int(sampling_frequency_EMG * win_ms / 1000)
    inc_samples = int(sampling_frequency_EMG * inc_ms / 1000)

    all_features = []
    for window in X:
        N, C = window.shape
        sub_features = []

        start = 0
        while start + win_samples <= N:
            sub = window[start: start + win_samples]
            sub_features.append(compute_tsd(sub, lam=lam))
            start += inc_samples

        if sub_features:
            feat = np.mean(sub_features, axis=0)
        else:
            feat = compute_tsd(window, lam=lam)

        all_features.append(feat)

    return np.array(all_features)


def get_tsd_feature_labels(n_channels: int) -> list[str]:
    """
    Names of TSD features in the same order as extract_features_tsd.

    Returns
    -------
    list[str]  length = C*(C+1)//2 + C
    """
    labels = []
    for i in range(n_channels):
        for j in range(i, n_channels):
            labels.append(f"tsd_cov_{i}{j}")
    for i in range(n_channels):
        labels.append(f"tsd_energy_ch{i}")
    return labels
