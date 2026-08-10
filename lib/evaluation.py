# conf_matrix_plot
# create_heatmap_data
# create_parameter_heatmaps
# to add
# evaluate_model
# plot_lda_projection
# plot_pca_projection

"""
evaluation.py
=============
Model evaluation, confusion matrices, benchmarking, and LDA/PCA projection
visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Generic evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _show_or_skip():
    """Call plt.show() only when running on an interactive backend.
    When MPLBACKEND=Agg (e.g. subprocess from PyQt), skips display
    to avoid the spurious exit-code-1 on process termination."""
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    if _mpl.get_backend().lower() not in ("agg", "pdf", "ps", "svg"):
        _plt.show()
    else:
        _plt.close('all')



def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    actions: dict | None = None,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a model on a test set and display metrics.

    Parameters
    ----------
    model          : model with predict() method
    X_test         : np.ndarray
    y_test         : np.ndarray integer labels
    actions        : dict {name: code} for display
    model_name     : str

    Returns
    -------
    dict {'accuracy': float, 'f1_macro': float, 'report': str}
    """
    pred = model.predict(X_test)

    # Keras case (softmax output)
    if pred.ndim == 2:
        pred = pred.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average='macro')

    target_names = list(actions.keys()) if actions else None
    report = classification_report(y_test, pred, target_names=target_names)

    print(f"\n{'─'*50}")
    print(f" {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  F1 Macro  : {f1:.4f}  ({f1*100:.2f} %)")
    print(f"\n{report}")

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'report': report,
        'predictions': pred
    }


def conf_matrix_plot(
    model,
    model_name: str,
    emg_test: np.ndarray,
    actions_test: np.ndarray,
    actions: dict | None = None,
    save_path: str | None = None,
) -> None:
    """
    Display the confusion matrix of a Keras model.

    Parameters
    ----------
    model         : tf.keras.Model
    model_name    : str
    emg_test      : np.ndarray
    actions_test  : np.ndarray integer labels
    actions       : dict {name: code}
    save_path     : str | None path to save the figure
    """
    pred = model.predict(emg_test)
    pred = pred.argmax(axis=1)

    cm = confusion_matrix(actions_test, pred)
    tick_labels = list(actions.keys()) if actions else None

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='rocket_r',
        xticklabels=tick_labels,
        yticklabels=tick_labels
    )

    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    _show_or_skip()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def create_heatmap_data(grid_results: dict) -> pd.DataFrame:
    """Build a DataFrame from grid/random search results."""
    params_df = pd.DataFrame(grid_results['params'])

    params_df['mean_accuracy'] = grid_results.get(
        'mean_test_accuracy',
        grid_results.get('mean_test_balanced_accuracy', np.nan)
    )

    params_df['mean_f1_macro'] = grid_results['mean_test_f1_macro']

    return params_df


def create_parameter_heatmaps(
    params_df: pd.DataFrame,
    param_combinations: list[tuple],
    name_plot: str,
    figsize: tuple = (20, 12),
) -> None:
    """
    Generate performance heatmaps for each parameter pair.

    Parameters
    ----------
    params_df          : DataFrame (from create_heatmap_data)
    param_combinations : list of (x_param, y_param)
    name_plot          : str global title
    figsize            : tuple
    """
    metrics = ['mean_accuracy', 'mean_f1_macro']
    metric_names = ['Accuracy', 'F1 Macro']

    n_combos = len(param_combinations)

    fig, axes = plt.subplots(2, n_combos, figsize=figsize)
    fig.suptitle(f'Hyperparameter Heatmaps — {name_plot}')

    vmin = params_df[metrics].min().min()
    vmax = params_df[metrics].max().max()

    for m_idx, metric in enumerate(metrics):

        for p_idx, (x_param, y_param) in enumerate(param_combinations):

            pivot = params_df.pivot_table(
                index=y_param,
                columns=x_param,
                values=metric,
                aggfunc='mean'
            )

            ax = axes[m_idx, p_idx]

            sns.heatmap(
                pivot,
                annot=True,
                cmap='Spectral',
                fmt='.3f',
                ax=ax,
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(
                f'{metric_names[m_idx]}: {y_param} vs {x_param}'
            )

    plt.tight_layout()
    _show_or_skip()

    for metric, label in zip(metrics, metric_names):

        top_cols = ['n_hidden_layers', 'n_units', 'optimizer', metric]
        top_cols = [c for c in top_cols if c in params_df.columns]

        print(f"\nTop 5 by {label}:")
        print(params_df.nlargest(5, metric)[top_cols])


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    actions: dict | None = None,
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Parameters
    ----------
    models  : dict {name: model}
    X_test  : np.ndarray
    y_test  : np.ndarray
    actions : dict {name: code}

    Returns
    -------
    pd.DataFrame columns: ['model', 'accuracy', 'f1_macro']
    """
    rows = []

    for name, model in models.items():

        res = evaluate_model(
            model,
            X_test,
            y_test,
            actions,
            model_name=name
        )

        rows.append({
            'model': name,
            'accuracy': res['accuracy'],
            'f1_macro': res['f1_macro']
        })

    return pd.DataFrame(rows).sort_values(
        'f1_macro',
        ascending=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Projection visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_lda_projection(
    X_lda: np.ndarray,
    y: np.ndarray,
    actions: dict | None = None,
    title: str = "LDA Projection",
    save_path: str | None = None,
) -> None:
    """
    Display a 2-D LDA projection colored by class.

    Parameters
    ----------
    X_lda    : np.ndarray shape (N, ≥2)
    y        : np.ndarray integer labels
    actions  : dict {name: code}
    title    : str
    save_path: str | None
    """
    label_map = {v: k for k, v in actions.items()} if actions else {}

    plt.figure(figsize=(8, 6))

    for cls in np.unique(y):

        mask = y == cls
        lbl = label_map.get(cls, str(cls))

        plt.scatter(
            X_lda[mask, 0],
            X_lda[mask, 1],
            label=lbl,
            alpha=0.6,
            s=15
        )

    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    _show_or_skip()


def plot_pca_projection(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    actions: dict | None = None,
    title: str = "PCA Projection",
    save_path: str | None = None,
) -> None:
    """
    Reduce X using PCA and display the 2-D projection colored by class.

    Parameters
    ----------
    X            : np.ndarray shape (N, F)
    y            : np.ndarray integer labels
    n_components : int
    actions      : dict {name: code}
    title        : str
    save_path    : str | None
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)
    exp_var = pca.explained_variance_ratio_

    label_map = {v: k for k, v in actions.items()} if actions else {}

    plt.figure(figsize=(8, 6))

    for cls in np.unique(y):

        mask = y == cls
        lbl = label_map.get(cls, str(cls))

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=lbl,
            alpha=0.6,
            s=15
        )

    plt.xlabel(f'PC1 ({exp_var[0]*100:.1f} %)')
    plt.ylabel(f'PC2 ({exp_var[1]*100:.1f} %)')

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    _show_or_skip()