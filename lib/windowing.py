# get_n_windows
# get_windows
# get_action_windows
# split_train_test_set
# split_train_test_val
# split_train_val

"""
windowing.py
============
Window extraction, label assignment,
and train/test/validation splitting utilities.
"""

import numpy as np
import pandas as pd
import random


def get_n_windows(
    df: pd.DataFrame,
    window_size: int
) -> int:
    """
    Return the number of complete windows
    in a DataFrame.
    """
    return len(df) // window_size


def get_windows(
    df: pd.DataFrame,
    window_size: int
) -> list[pd.DataFrame]:
    """
    Split a DataFrame into non-overlapping windows.

    Parameters
    ----------
    df : pd.DataFrame

    window_size : int

    Returns
    -------
    list[pd.DataFrame]
    """
    n = get_n_windows(
        df,
        window_size
    )

    return [
        df.iloc[
            i * window_size:
            (i + 1) * window_size
        ].reset_index(drop=True)

        for i in range(n)
    ]


def get_action_windows(
    window: pd.DataFrame,
    action_col: str = 'Action'
) -> int:
    """
    Return the majority label
    of a window.

    Parameters
    ----------
    window : pd.DataFrame

    action_col : str

    Returns
    -------
    int
        Majority label.
    """
    return int(
        window[action_col].mode()[0]
    )


def split_train_test_set(
    takes: list,
    test_ratio: float = 0.2,
    random_state: int | None = None,
) -> tuple[list, list]:
    """
    Split a list of takes into
    training and test sets.

    Parameters
    ----------
    takes : list

    test_ratio : float

    random_state : int | None

    Returns
    -------
    (training_takes, test_takes)
    """
    if random_state is not None:
        random.seed(random_state)

    shuffled = list(takes)

    random.shuffle(shuffled)

    n_test = max(
        1,
        int(len(shuffled) * test_ratio)
    )

    return (
        shuffled[n_test:],
        shuffled[:n_test]
    )


def split_train_test_val(
    takes: list,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int | None = None,
) -> tuple[list, list, list]:
    """
    Split a list of takes into
    training, validation, and test sets.

    Parameters
    ----------
    takes : list

    test_ratio : float

    val_ratio : float

    random_state : int | None

    Returns
    -------
    (training_takes, validation_takes, test_takes)
    """
    if random_state is not None:
        random.seed(random_state)

    shuffled = list(takes)

    random.shuffle(shuffled)

    n = len(shuffled)

    n_test = max(
        1,
        int(n * test_ratio)
    )

    n_val = max(
        1,
        int(n * val_ratio)
    )

    test_takes = shuffled[:n_test]

    val_takes = shuffled[
        n_test: n_test + n_val
    ]

    train_takes = shuffled[
        n_test + n_val:
    ]

    return (
        train_takes,
        val_takes,
        test_takes
    )


def split_train_val(
    emg: np.ndarray,
    labels: np.ndarray,
    val_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Split feature arrays into
    training and validation sets
    using random shuffling.

    Parameters
    ----------
    emg : np.ndarray
        Shape (N, F)

    labels : np.ndarray
        Shape (N,)

    val_size : float

    random_state : int | None

    Returns
    -------
    (
        emg_train,
        labels_train,
        emg_val,
        labels_val
    )
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = emg.shape[0]

    N_val = int(N * val_size)

    idx = np.random.permutation(N)

    return (
        emg[idx[N_val:]],
        labels[idx[N_val:]],

        emg[idx[:N_val]],
        labels[idx[:N_val]],
    )