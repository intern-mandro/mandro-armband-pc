# BP_filter
# compute_enveloppe
# calc_fft_power
# remove_delay_seconds   # new function

"""
preprocessing.py
================
Filtering, rectification, envelope extraction, and EMG signal normalization.
"""

import numpy as np
import pandas as pd

from scipy.signal import butter


# ─────────────────────────────────────────────────────────────────────────────
# Band-pass filtering
# ─────────────────────────────────────────────────────────────────────────────

from scipy.signal import lfilter, lfilter_zi


def BP_filter(
    df: pd.DataFrame,
    lowcut: float = 35.0,
    highcut: float = 300.0,
    sampling_frequency_EMG: int = 900,
    order: int = 4,
    labels: list | None = None,
) -> np.ndarray:
    """
    Causal Butterworth band-pass filter (matches Teensy real-time behavior).
    Single forward pass via lfilter; lfilter_zi initialization reduces
    startup transient.
    """
    if labels is None:
        labels = [c for c in df.columns
                  if c.startswith('Raw_CH') or c.startswith('Ch')]

    nyq = sampling_frequency_EMG / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')

    signal = df[labels].values.astype(float)
    filtered = np.zeros_like(signal)
    zi_template = lfilter_zi(b, a)

    for ch in range(signal.shape[1]):
        zi = zi_template * signal[0, ch]
        filtered[:, ch], _ = lfilter(b, a, signal[:, ch], zi=zi)

    return filtered


def compute_enveloppe(
    signal: np.ndarray,
    size: int = 10,
) -> np.ndarray:
    """
    Causal moving-average envelope: y[t] uses only signal[t-size+1 : t+1].
    Matches a sliding-window mean implemented with a circular buffer in C++.
    """
    envelope = np.zeros_like(signal, dtype=float)
    kernel = np.ones(size) / size

    for ch in range(signal.shape[1]):
        full = np.convolve(signal[:, ch], kernel, mode='full')
        envelope[:, ch] = full[:signal.shape[0]]

    return envelope


def calc_fft_power(
    signal: np.ndarray,
    sampling_frequency_EMG: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of a multi-channel signal.

    Parameters
    ----------
    signal : np.ndarray shape (N, C)

    sampling_frequency_EMG : int

    Returns
    -------
    freq : np.ndarray shape (N//2+1,)

    fft_power : np.ndarray shape (N//2+1, C)
    """
    N = signal.shape[0]

    freq = np.fft.rfftfreq(
        N,
        1.0 / sampling_frequency_EMG
    )

    fft_val = np.fft.rfft(
        signal,
        axis=0
    )

    fft_power = np.abs(fft_val) ** 2

    return freq, fft_power


def remove_delay_seconds(
    df: pd.DataFrame,
    delay_s: float = 5.0,
    timestamp_col: str = 'Timestamp',
) -> pd.DataFrame:
    """
    Remove the first `delay_s` seconds from a DataFrame
    (startup artifacts).

    Parameters
    ----------
    df : pd.DataFrame

    delay_s : float
        Delay to remove (seconds).

    timestamp_col : str
        Name of the time column.

    Returns
    -------
    Filtered and re-indexed pd.DataFrame.
    """
    t0 = df[timestamp_col].iloc[0]

    mask = df[timestamp_col] >= (t0 + delay_s)

    return df[mask].reset_index(drop=True)


def normalize_signal(
    signal: np.ndarray,
    method: str = 'minmax',
) -> np.ndarray:
    """
    Normalize a signal channel by channel.

    Parameters
    ----------
    signal : np.ndarray shape (N, C)

    method : str
        'minmax' → [0, 1]
        'zscore' → mean 0, standard deviation 1.

    Returns
    -------
    np.ndarray shape (N, C)
    """
    out = np.zeros_like(
        signal,
        dtype=float
    )

    for ch in range(signal.shape[1]):

        col = signal[:, ch].astype(float)

        if method == 'minmax':

            col_min = col.min()
            col_max = col.max()

            denom = (
                col_max - col_min
                if col_max != col_min
                else 1.0
            )

            out[:, ch] = (
                (col - col_min) / denom
            )

        elif method == 'zscore':

            mu = col.mean()
            sigma = col.std()

            out[:, ch] = (
                (col - mu)
                / (sigma if sigma > 0 else 1.0)
            )

        else:
            raise ValueError(
                f"Unknown method: '{method}' "
                f"(use 'minmax' or 'zscore')"
            )

    return out