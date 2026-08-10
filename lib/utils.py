# get_spike_index
# format_number_list
# get_activation_code

"""
utils.py
========
Utility functions: spike detection, activation codes,
formatting utilities, and random seed setup.
"""

import numpy as np
import random


def get_spike_index(
    signal: np.ndarray,
    threshold: float = 50.0,
    baseline: float = 25.0,
) -> int | None:
    """
    Return the index of the first spike exceeding the threshold.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array (single channel amplitude).

    threshold : float
        Detection threshold.

    baseline : float
        Signal must first fall below this value before
        a spike can be detected.

    Returns
    -------
    int | None
        Index of the detected spike, or None if not found.
    """
    for i in range(1, len(signal)):

        if (
            signal[i] > threshold
            and signal[i - 1] <= baseline
        ):
            return i

    return None


def get_activation_code(layer) -> str:
    """
    Convert a Keras activation function
    into a C++ export string code.

    Parameters
    ----------
    layer : tf.keras.layers.Dense

    Returns
    -------
    str
        Example: 'RELU', 'SOFTMAX'
    """
    activation_map = {
        'sigmoid': 'SIGMOID',
        'relu': 'RELU',
        'tanh': 'TANH',
        'linear': 'LINEAR',
        'softmax': 'SOFTMAX',
    }

    name = layer.activation.__name__.lower()

    return activation_map.get(
        name,
        'UNKNOWN'
    )


def format_number_list(numbers) -> str:
    """
    Format a list of numbers as a C++ initializer list string.

    Returns
    -------
    str
        Example:
        '{\\n1.0, 2.0\\n}'
    """
    return (
        "{\n"
        + ", ".join(f"{float(x)}" for x in numbers)
        + "\n}"
    )


def set_random_seed(
    seed: int = 42
) -> None:
    """
    Set all random seeds
    (Python, NumPy, TensorFlow).

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

    except ImportError:
        pass

    print(f"Random seed set to {seed}")


def shift_delay(
    df,
    spike_threshold: float = 50.0,
    baseline: float = 25.0,
    offset_correction: int = 200,
    labels: list | None = None,
):
    """
    Correct EMG signal delay by aligning the data
    on the first detected spike.

    Parameters
    ----------
    df : pd.DataFrame

    spike_threshold : float

    baseline : float

    offset_correction : int
        Additional offset correction in samples.

    labels : list[str] | None
        Signal column names.

    Returns
    -------
    pd.DataFrame
        Shifted and reindexed DataFrame.
    """
    if labels is None:

        labels = [
            c for c in df.columns
            if c.startswith('Raw_CH')
            or c.startswith('Ch')
        ]

    # Detect spike on the first channel
    signal = df[labels[0]].values

    spike_idx = get_spike_index(
        signal,
        threshold=spike_threshold,
        baseline=baseline
    )

    if spike_idx is None:

        print(
            "  ⚠ No spike detected, "
            "delay correction skipped."
        )

        return df

    shift = max(
        0,
        spike_idx - offset_correction
    )

    return df.iloc[shift:].reset_index(drop=True)