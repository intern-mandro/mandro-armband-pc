# grid_search_NN
# rand_search_NN
# build_test_model
# best_of_100
# save_grid

"""
training.py
===========
Training, hyperparameter search, and best model selection.
"""

import numpy as np
import pickle as pkl
import os

from sklearn.metrics import (
    f1_score,
    accuracy_score
)

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from joblib import parallel_backend

from lib.models import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def _make_keras_clf(
    n_classes: int
):
    """
    Create a KerasClassifier with default values.
    """
    try:
        from scikeras.wrappers import (
            KerasClassifier
        )

    except ImportError:
        raise ImportError(
            "Install scikeras: pip install scikeras"
        )

    return KerasClassifier(
        build_fn=build_model,
        input_shape=114,
        output_shape=n_classes,
        n_hidden_layers=2,
        n_units=64,
        hidden_activation='relu',
        output_activation='softmax',
        optimizer='adam',
        learning_rate=0.001,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        kernel_regularizer=None,
        epochs=100,
        batch_size=32,
        verbose=2,
    )


_SCORING = {
    'f1_macro': 'f1_macro',
    'balanced_accuracy': 'balanced_accuracy',
    'accuracy': 'accuracy',
}


def rand_search_NN(
    param_grid: dict,
    emg_train: np.ndarray,
    actions_train_encoded: np.ndarray,
    type_model: str,
    n_iter: int = 45,
    n_classes: int = 4,
    random_state: int = 42,
    save_grid: bool = False,
):
    """
    Random hyperparameter search (RandomizedSearchCV)
    for a Keras neural network.

    Parameters
    ----------
    param_grid : dict

    emg_train : np.ndarray

    actions_train_encoded : np.ndarray
        One-hot encoded labels.

    type_model : str
        Identifier used for saving.

    n_iter : int

    n_classes : int

    random_state : int

    save_grid : bool

    Returns
    -------
    RandomizedSearchCV result
    """
    try:
        import tensorflow as tf

    except ImportError:
        raise ImportError("TensorFlow required.")

    keras_clf = _make_keras_clf(n_classes)

    rand_grid = RandomizedSearchCV(
        estimator=keras_clf,
        n_iter=n_iter,
        param_distributions=param_grid,
        cv=5,
        verbose=2,
        n_jobs=1,
        scoring=_SCORING,
        refit='f1_macro',
        random_state=random_state,
    )

    cp_callback = tf.keras.callbacks.EarlyStopping(
        patience=1,
        verbose=1
    )

    with parallel_backend('loky'):

        result = rand_grid.fit(
            emg_train,
            actions_train_encoded,
            verbose=2,
            callbacks=[cp_callback]
        )

    if save_grid:
        _save_grid(result, type_model)

    return result


def grid_search_NN(
    param_grid: dict,
    emg_train: np.ndarray,
    actions_train_encoded: np.ndarray,
    type_model: str,
    n_classes: int = 4,
    save_grid: bool = False,
):
    """
    Exhaustive grid search (GridSearchCV)
    for a Keras neural network.

    Parameters
    ----------
    Same parameters as rand_search_NN,
    except n_iter and random_state.

    Returns
    -------
    GridSearchCV result
    """
    try:
        import tensorflow as tf

    except ImportError:
        raise ImportError("TensorFlow required.")

    keras_clf = _make_keras_clf(n_classes)

    grid = GridSearchCV(
        estimator=keras_clf,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=1,
        scoring=_SCORING,
        refit='f1_macro',
    )

    cp_callback = tf.keras.callbacks.EarlyStopping(
        patience=1,
        verbose=1
    )

    with parallel_backend('loky'):

        result = grid.fit(
            emg_train,
            actions_train_encoded,
            verbose=2,
            callbacks=[cp_callback]
        )

    if save_grid:
        _save_grid(result, type_model)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build and evaluate a single model
# ─────────────────────────────────────────────────────────────────────────────

def build_test_model(
    emg_train: np.ndarray,
    actions_train_encoded: np.ndarray,
    emg_test: np.ndarray,
    actions_test_encoded: np.ndarray,
    actions_test: np.ndarray,
    grid_search,
):
    """
    Train a model using the best hyperparameters
    and evaluate it on the test set.

    Parameters
    ----------
    emg_train, actions_train_encoded :
        Training data

    emg_test, actions_test_encoded :
        Test data (one-hot)

    actions_test :
        Integer labels (for metrics)

    grid_search :
        Search result (best_params_ required)

    Returns
    -------
    (model, predictions : np.ndarray)
    """
    model = build_model(
        **grid_search.best_params_
    )

    with parallel_backend('loky'):

        model.fit(
            emg_train,
            actions_train_encoded,
            epochs=grid_search.best_params_['epochs'],
            batch_size=32
        )

    model.evaluate(
        emg_test,
        actions_test_encoded
    )

    predictions = model.predict(
        emg_test
    ).argmax(axis=1)

    print(
        f"Test Accuracy : "
        f"{accuracy_score(actions_test, predictions):.4f}"
    )

    print(
        f"F1 Macro      : "
        f"{f1_score(actions_test, predictions, average='macro'):.4f}"
    )

    return model, predictions


def best_of_100(
    emg_train: np.ndarray,
    actions_train_encoded: np.ndarray,
    emg_test: np.ndarray,
    actions_test_encoded: np.ndarray,
    actions_test: np.ndarray,
    grid_search,
    n_iter: int = 100,
) -> tuple[dict, dict, list, list]:
    """
    Train the model `n_iter` times
    and keep the best one (accuracy + F1).

    Returns
    -------
    (top_accuracy, top_f1, accuracy_list, f1_list)
    """
    accuracy_list = []
    f1_list = []

    top_acc = {
        'best_of_100_model': None,
        'best_of_100_accuracy': 0,
        'best_of_100_f1': 0,
        'best_of_100_index': 0
    }

    top_f1 = {
        'best_of_100_model': None,
        'best_of_100_accuracy': 0,
        'best_of_100_f1': 0,
        'best_of_100_index': 0
    }

    for i in range(n_iter):

        print(f"Iteration {i + 1}/{n_iter}")

        model, pred = build_test_model(
            emg_train,
            actions_train_encoded,
            emg_test,
            actions_test_encoded,
            actions_test,
            grid_search
        )

        acc = accuracy_score(
            actions_test,
            pred
        )

        f1 = f1_score(
            actions_test,
            pred,
            average='macro'
        )

        accuracy_list.append(acc)
        f1_list.append(f1)

        if acc > top_acc['best_of_100_accuracy']:

            top_acc.update({
                'best_of_100_model': model,
                'best_of_100_accuracy': acc,
                'best_of_100_f1': f1,
                'best_of_100_index': i
            })

        if f1 > top_f1['best_of_100_f1']:

            top_f1.update({
                'best_of_100_model': model,
                'best_of_100_accuracy': acc,
                'best_of_100_f1': f1,
                'best_of_100_index': i
            })

    return (
        top_acc,
        top_f1,
        accuracy_list,
        f1_list
    )


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    class_weight: dict | None = None,
):
    """
    Train a Keras model with callbacks (EarlyStopping).

    Parameters
    ----------
    model : tf.keras.Model

    X_train, y_train :
        Training data (y must be one-hot encoded)

    X_val, y_val :
        Validation data (optional)

    epochs, batch_size : int

    Returns
    -------
    history : tf.keras.callbacks.History
    """
    try:
        import tensorflow as tf

    except ImportError:
        raise ImportError("TensorFlow required.")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    validation_data = (
        (X_val, y_val)
        if X_val is not None
        else None
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Internal saving utility
# ─────────────────────────────────────────────────────────────────────────────

def _save_grid(
    grid,
    type_model: str,
    folder: str = "results/experiments"
) -> None:

    os.makedirs(folder, exist_ok=True)

    path = os.path.join(
        folder,
        f"NN_grid_search_{type_model}.pkl"
    )

    with open(path, 'wb') as f:
        pkl.dump(grid, f)

    print(f"Grid search saved → {path}")


# Public alias for compatibility
save_grid = _save_grid