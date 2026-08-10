# build_model
# to create
# build_cnn
# build_lstm
# build_svm
# build_rf

"""
models.py
=========
Model architecture builders (NN, CNN, SVM, Random Forest).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Dense neural network (Keras)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    input_shape: int = 114,
    output_shape: int = 4,
    n_hidden_layers: int = 1,
    n_units: int = 32,
    hidden_activation: str = 'relu',
    output_activation: str = 'softmax',
    optimizer: str = 'adam',
    learning_rate: float = 0.0001,
    loss: str = 'categorical_crossentropy',
    metrics: list | None = None,
    kernel_regularizer=None,
    epochs: int = 100,  # kept for KerasClassifier compatibility
):
    """
    Build a dense Keras neural network.

    Parameters
    ----------
    input_shape : int
    output_shape : int number of classes
    n_hidden_layers : int
    n_units : int
    hidden_activation : str
    output_activation : str
    optimizer : str | keras optimizer
    learning_rate : float
    loss : str
    metrics : list
    kernel_regularizer : keras regularizer | None
    epochs : int not used here, kept for KerasClassifier

    Returns
    -------
    tf.keras.Sequential
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

    except ImportError:
        raise ImportError(
            "TensorFlow is not installed. "
            "Install it with: pip install tensorflow"
        )

    if metrics is None:
        metrics = ['accuracy']

    model = keras.Sequential()

    model.add(
        keras.layers.InputLayer(
            input_shape=(input_shape,)
        )
    )

    for _ in range(n_hidden_layers):

        model.add(
            keras.layers.Dense(
                n_units,
                activation=hidden_activation,
                kernel_regularizer=kernel_regularizer,
            )
        )

    model.add(
        keras.layers.Dense(
            output_shape,
            activation=output_activation
        )
    )

    if optimizer == 'adam':

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1-D CNN
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(
    input_shape: tuple = (115, 4),
    output_shape: int = 4,
    n_filters: int = 32,
    kernel_size: int = 3,
    n_conv_layers: int = 2,
    n_dense_units: int = 64,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
):
    """
    Build a 1-D CNN for EMG classification
    on windowed raw signals.

    Parameters
    ----------
    input_shape : tuple (window_size, n_channels)
    output_shape : int number of classes
    n_filters : int
    kernel_size : int
    n_conv_layers : int
    n_dense_units : int
    optimizer : str
    learning_rate : float

    Returns
    -------
    tf.keras.Sequential
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

    except ImportError:
        raise ImportError("TensorFlow is not installed.")

    model = keras.Sequential()

    model.add(
        keras.layers.InputLayer(
            input_shape=input_shape
        )
    )

    for _ in range(n_conv_layers):

        model.add(
            keras.layers.Conv1D(
                n_filters,
                kernel_size,
                activation='relu',
                padding='same'
            )
        )

        model.add(
            keras.layers.MaxPooling1D(
                pool_size=2
            )
        )

    model.add(keras.layers.Flatten())

    model.add(
        keras.layers.Dense(
            n_dense_units,
            activation='relu'
        )
    )

    model.add(
        keras.layers.Dense(
            output_shape,
            activation='softmax'
        )
    )

    if optimizer == 'adam':

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SVM
# ─────────────────────────────────────────────────────────────────────────────

def build_svm(
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    probability: bool = True,
):
    """
    Build a scikit-learn SVM.

    Parameters
    ----------
    C : float
    kernel : str
    gamma : str | float
    probability : bool enables calibration for predict_proba

    Returns
    -------
    sklearn.svm.SVC
    """
    from sklearn.svm import SVC

    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability
    )


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
):
    """
    Build a scikit-learn Random Forest.

    Parameters
    ----------
    n_estimators : int
    max_depth : int | None
    random_state : int

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )