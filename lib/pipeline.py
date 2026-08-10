# to create
# preprocess_pipeline
# fit_scaler
# apply_scaler
# fit_lda
# apply_lda
# core of the project

"""
pipeline.py
===========
High-level functions encapsulating the EMG pipeline steps
(preprocessing → features → scaler → LDA → saving).
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)

from lib.preprocessing import (
    BP_filter,
    compute_enveloppe,
    remove_delay_seconds
)


from lib.features.features import (
    extract_features,
    extract_features_tsd
)



# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(
    df: pd.DataFrame,
    labels: list[str],
    sampling_frequency_EMG: int = 900,
    lowcut: float = 35.0,
    highcut: float = 300.0,
    delay_s: float = 5.0,
    kernel_ratio: float = 0.1,
    window_size: int | None = None,
    warmup_samples: int = 100,
) -> pd.DataFrame:

    df = remove_delay_seconds(df, delay_s=delay_s)

    filtered = BP_filter(
        df, lowcut=lowcut, highcut=highcut,
        sampling_frequency_EMG=sampling_frequency_EMG,
        labels=labels,
    )
    df = df.copy()
    df[labels] = filtered
    df[labels] = np.abs(df[labels])

    kernel_size = int(window_size * kernel_ratio) if window_size else 12
    envelope = compute_enveloppe(df[labels].values, size=kernel_size)
    df[labels] = envelope

    if warmup_samples > 0:
        df = df.iloc[warmup_samples:].reset_index(drop=True)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_pipeline(
    X: np.ndarray,
    sampling_frequency_EMG: int = 900,
    mode: str = "concat",
    tsd_win_ms: int = 115,
    tsd_inc_ms: int = 57,
    tsd_lam: float = 0.1,
) -> np.ndarray:
    """
    Extract features according to the selected mode.

    Parameters
    ----------
    X : np.ndarray shape (n_windows, N_samples, C_channels)
    sampling_frequency_EMG : int
    mode : str 'classic' | 'tsd' | 'concat'
    tsd_win_ms, tsd_inc_ms, tsd_lam : TSD parameters

    Returns
    -------
    np.ndarray shape (n_windows, n_features)
    """
    if mode == "classic":

        return extract_features(
            X,
            sampling_frequency_EMG
        )

    elif mode == "tsd":

        return extract_features_tsd(
            X,
            sampling_frequency_EMG,
            win_ms=tsd_win_ms,
            inc_ms=tsd_inc_ms,
            lam=tsd_lam
        )

    elif mode == "concat":

        classic = extract_features(
            X,
            sampling_frequency_EMG
        )

        tsd = extract_features_tsd(
            X,
            sampling_frequency_EMG,
            win_ms=tsd_win_ms,
            inc_ms=tsd_inc_ms,
            lam=tsd_lam
        )

        return np.concatenate(
            [classic, tsd],
            axis=1
        )

    else:
        raise ValueError(
            f"Unknown mode: '{mode}' "
            f"(use 'classic', 'tsd' or 'concat')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scaler
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(
    X_train: np.ndarray
) -> StandardScaler:
    """
    Train a StandardScaler on X_train.
    """
    scaler = StandardScaler()

    scaler.fit(X_train)

    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X: np.ndarray
) -> np.ndarray:
    """
    Apply a pre-trained scaler on X.
    """
    return scaler.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────────────────────────────────────

def fit_lda(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LinearDiscriminantAnalysis:
    """
    Train an LDA on (X_train, y_train).
    """
    lda = LinearDiscriminantAnalysis()

    lda.fit(X_train, y_train)

    return lda


def apply_lda(
    lda: LinearDiscriminantAnalysis,
    X: np.ndarray
) -> np.ndarray:
    """
    Project X into the LDA space.
    """
    return lda.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load complete pipeline
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline(
    pipeline_dict: dict,
    name: str,
    folder: str = "models/scalers",
) -> None:
    """
    Save a pipeline dictionary (scaler, lda, etc.) as pkl.

    Parameters
    ----------
    pipeline_dict : dict e.g. {'scaler': scaler, 'lda': lda}
    name : str file name (without extension)
    folder : str
    """
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{name}.pkl")

    with open(path, 'wb') as f:
        pkl.dump(pipeline_dict, f)

    print(f"Pipeline saved → {path}")


def load_pipeline(
    name: str,
    folder: str = "models/scalers"
) -> dict:
    """
    Load a pipeline dictionary from a pkl file.

    Returns
    -------
    dict
    """
    path = os.path.join(folder, f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline not found: {path}"
        )

    with open(path, 'rb') as f:
        return pkl.load(f)