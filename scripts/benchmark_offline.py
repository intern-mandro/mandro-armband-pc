"""
benchmark_offline.py
────────────────────
Offline evaluation of the trained model.
Outputs a JSON file with all metrics (accuracy, confusion matrix,
classification report, model topology, dataset stats).

Usage:
    python scripts/benchmark_offline.py [--label MY_LABEL]

The output JSON is saved in benchmarks/benchmark_<label>.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score
)
import tensorflow as tf
import joblib

from lib.data_loader import load_datasets_new_format
from lib.pipeline import preprocess_pipeline, extract_features_pipeline, apply_scaler
from lib.windowing import get_windows, get_action_windows, split_train_val
from lib.configs import (
    DATA_RAW, DATA_TEST_RAW, ACTIONS, LABELS, SAMPLING_FREQ, WINDOW_SIZE,
    LOWCUT, HIGHCUT, DELAY_S, KERNEL_RATIO,
    TSD_WIN_MS, TSD_INC_MS, TSD_LAM,
    RANDOM_STATE, VAL_RATIO, N_CLASSES,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default=None,
                   help="Label for this benchmark (e.g. 'sherylann_4cl'). "
                        "Default: name of the project root folder.")
    p.add_argument("--model-path",
                   default="models/trained/model_causal_concat58.keras")
    p.add_argument("--scaler-path",
                   default="models/scalers/scaler_causal_concat58.pkl")
    p.add_argument("--output-dir", default="benchmarks")
    return p.parse_args()


def load_and_process(folder):
    """Load CSVs, preprocess, window, extract features. Returns (X, y)."""
    print(f"  Loading {folder} ...")
    takes = load_datasets_new_format(folder, ACTIONS, LABELS)
    print(f"    {len(takes)} takes loaded")

    processed = []
    min_samples = int(DELAY_S * SAMPLING_FREQ) + 200
    for df in takes:
        if len(df) < min_samples:
            continue
        processed.append(preprocess_pipeline(
            df, labels=LABELS, sampling_frequency_EMG=SAMPLING_FREQ,
            lowcut=LOWCUT, highcut=HIGHCUT,
            delay_s=DELAY_S, kernel_ratio=KERNEL_RATIO,
            window_size=WINDOW_SIZE, warmup_samples=100))

    X_list, y_list = [], []
    for df in processed:
        for w in get_windows(df, WINDOW_SIZE):
            if (w['Action'] == -1).any():
                continue
            X_list.append(w[LABELS].values)
            y_list.append(get_action_windows(w))

    X_raw = np.array(X_list)
    y = np.array(y_list)
    X_feat = extract_features_pipeline(
        X_raw, SAMPLING_FREQ, mode='concat',
        tsd_win_ms=TSD_WIN_MS, tsd_inc_ms=TSD_INC_MS, tsd_lam=TSD_LAM)
    return X_feat, y


def evaluate(model, scaler, X_feat, y_true, class_names):
    """Return dict of metrics."""
    X_std = apply_scaler(scaler, X_feat)

    t0 = time.perf_counter()
    y_pred = np.argmax(model.predict(X_std, verbose=0), axis=1)
    inference_time_s = time.perf_counter() - t0
    per_sample_ms = 1000.0 * inference_time_s / len(y_true)

    labels_arg = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels_arg)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', labels=labels_arg, zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=labels_arg,
        target_names=class_names, output_dict=True, zero_division=0)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference_total_s": float(inference_time_s),
        "inference_per_sample_ms": float(per_sample_ms),
        "class_distribution": {
            class_names[i]: int(np.sum(y_true == i))
            for i in range(len(class_names))
        },
    }


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    label = args.label or project_root.name.replace(' ', '_')
    print(f"\n══════ Benchmark: {label} ══════")

    # Load model and scaler
    print(f"\n── Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    print(f"── Loading scaler: {args.scaler_path}")
    scaler_data = joblib.load(args.scaler_path)
    scaler = scaler_data['scaler'] if isinstance(scaler_data, dict) else scaler_data

    # Class names from ACTIONS
    inv_actions = {v: k for k, v in ACTIONS.items()}
    class_names = [inv_actions[i] for i in range(N_CLASSES)]
    print(f"   Classes: {class_names}")
    print(f"   Topology: {[l.units for l in model.layers if hasattr(l, 'units')]}")
    print(f"   Total params: {model.count_params()}")

    # Evaluate on BATCH1 val (random 20% split, reproducible)
    print(f"\n── BATCH1 val evaluation")
    X_b1, y_b1 = load_and_process(DATA_RAW)
    _, _, X_val, y_val = split_train_val(
        X_b1, y_b1, val_size=VAL_RATIO, random_state=RANDOM_STATE)
    print(f"    Val set: {len(y_val)} windows")
    val_metrics = evaluate(model, scaler, X_val, y_val, class_names)
    print(f"    Val accuracy: {val_metrics['accuracy']:.4f}")

    # Evaluate on BATCH2 holdout
    print(f"\n── BATCH2 holdout evaluation")
    X_b2, y_b2 = load_and_process(DATA_TEST_RAW)
    print(f"    Holdout: {len(y_b2)} windows")
    test_metrics = evaluate(model, scaler, X_b2, y_b2, class_names)
    print(f"    Test accuracy: {test_metrics['accuracy']:.4f}")

    # Assemble final dict
    output = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "model_path": args.model_path,
        "n_classes": N_CLASSES,
        "class_names": class_names,
        "topology": [int(l.units) for l in model.layers if hasattr(l, 'units')],
        "n_params": int(model.count_params()),
        "n_features": 58,
        "window_size": WINDOW_SIZE,
        "sampling_freq": SAMPLING_FREQ,
        "data_train": str(DATA_RAW),
        "data_test": str(DATA_TEST_RAW),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"benchmark_{label}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to {out_path}")
    print(f"\nSummary:")
    print(f"  Val:  {val_metrics['accuracy']*100:.2f}%  F1={val_metrics['f1_macro']*100:.2f}%")
    print(f"  Test: {test_metrics['accuracy']*100:.2f}%  F1={test_metrics['f1_macro']*100:.2f}%")


if __name__ == "__main__":
    main()