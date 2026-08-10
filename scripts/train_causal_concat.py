import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from lib.data_loader import load_datasets_new_format
from lib.pipeline import (
    preprocess_pipeline, extract_features_pipeline,
    fit_scaler, apply_scaler, save_pipeline,
)
from lib.windowing import (
    get_windows, get_action_windows, split_train_val,
)
from lib.models import build_model
from lib.training import train_model
from lib.evaluation import evaluate_model, conf_matrix_plot
from lib.configs import (
    DATA_RAW, DATA_TEST_RAW,
    ACTIONS, LABELS, SAMPLING_FREQ, WINDOW_SIZE,
    LOWCUT, HIGHCUT, DELAY_S, KERNEL_RATIO,
    TSD_WIN_MS, TSD_INC_MS, TSD_LAM,
    RANDOM_STATE, VAL_RATIO, N_CLASSES,
)
from tensorflow.keras.utils import to_categorical


def preprocess_takes(takes, label):
    print(f"\n── Preprocessing {label} ({len(takes)} takes)")
    processed = []
    for df in takes:
        df_p = preprocess_pipeline(
            df, labels=LABELS,
            sampling_frequency_EMG=SAMPLING_FREQ,
            lowcut=LOWCUT, highcut=HIGHCUT,
            delay_s=DELAY_S, kernel_ratio=KERNEL_RATIO,
            window_size=WINDOW_SIZE,
            warmup_samples=100,
        )
        processed.append(df_p)
    return processed


def takes_to_arrays(takes_list):
    X_list, y_list = [], []
    for df in takes_list:
        windows = get_windows(df, WINDOW_SIZE)
        for w in windows:
            if (w['Action'] == -1).any():
                continue
            X_list.append(w[LABELS].values)
            y_list.append(get_action_windows(w))
    return np.array(X_list), np.array(y_list)


print("══════════════════════════════════════════════════════")
print(" Loading BATCH1 (train + val)")
print("══════════════════════════════════════════════════════")
takes_train = load_datasets_new_format(DATA_RAW, ACTIONS, LABELS)

print("\n══════════════════════════════════════════════════════")
print(" Loading BATCH2 (holdout test)")
print("══════════════════════════════════════════════════════")
takes_test = load_datasets_new_format(DATA_TEST_RAW, ACTIONS, LABELS)

train_processed = preprocess_takes(takes_train, "BATCH1")
test_processed  = preprocess_takes(takes_test,  "BATCH2")

print("\n── Windowing")
X_trainval_raw, y_trainval = takes_to_arrays(train_processed)
X_test_raw,     y_test     = takes_to_arrays(test_processed)

print(f"  BATCH1 windows: {len(X_trainval_raw)}")
print(f"  BATCH2 windows: {len(X_test_raw)}")

print("\n── Feature extraction (concat = classic + TSD)")
X_trainval = extract_features_pipeline(
    X_trainval_raw, SAMPLING_FREQ, mode="concat",
    tsd_win_ms=TSD_WIN_MS, tsd_inc_ms=TSD_INC_MS, tsd_lam=TSD_LAM,
)
X_test = extract_features_pipeline(
    X_test_raw, SAMPLING_FREQ, mode="concat",
    tsd_win_ms=TSD_WIN_MS, tsd_inc_ms=TSD_INC_MS, tsd_lam=TSD_LAM,
)

assert X_trainval.shape[1] > 0, f"Expected positive features, got {X_trainval.shape[1]}"
print(f"  Feature vector size: {X_trainval.shape[1]} ✓")

print("\n── Train / val split (random shuffle within BATCH1)")
X_train, y_train, X_val, y_val = split_train_val(
    X_trainval, y_trainval,
    val_size=VAL_RATIO, random_state=RANDOM_STATE,
)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test (BATCH2): {len(X_test)}")

print("\n── Fit scaler on TRAIN only (avoids data leakage)")
scaler = fit_scaler(X_train)
X_train_s = apply_scaler(scaler, X_train)
X_val_s   = apply_scaler(scaler, X_val)
X_test_s  = apply_scaler(scaler, X_test)

y_train_oh = to_categorical(y_train, num_classes=N_CLASSES)
y_val_oh   = to_categorical(y_val,   num_classes=N_CLASSES)

print("\n── Building Dense NN")
model = build_model(
    input_shape=X_train_s.shape[1],
    output_shape=N_CLASSES,
    n_hidden_layers=2,
    n_units=64,
    learning_rate=0.001,
)
model.summary()

print("\n── Training")
history = train_model(
    model, X_train_s, y_train_oh,
    X_val=X_val_s, y_val=y_val_oh,
    epochs=200, batch_size=32,
)

print("\n══════════════════════════════════════════════════════")
print(" Evaluation on BATCH2 (inter-session generalization)")
print("══════════════════════════════════════════════════════")
results_test = evaluate_model(model, X_test_s, y_test, ACTIONS,
                              "Causal_DenseNN (BATCH2 test)")

print("\n── Sanity check: evaluation on VAL (BATCH1 split)")
results_val = evaluate_model(model, X_val_s, y_val, ACTIONS,
                             "Causal_DenseNN (BATCH1 val)")

conf_matrix_plot(model, "Causal_DenseNN_BATCH2",
                 X_test_s, y_test, ACTIONS)

# Scaler name mirrors the model name (export_teensy_headers.py expects this convention).
# Default keeps the historical name when running from CLI without env var.
import os as _os
_model_name = _os.environ.get(
    "EMG_MODEL_OUTPUT_NAME",
    f"model_causal_concat{X_train_s.shape[1]}.keras",
)
_scaler_name = (
    _model_name.replace("model_", "scaler_", 1).replace(".keras", "")
    if _model_name.startswith("model_")
    else f"scaler_causal_concat{X_train_s.shape[1]}"
)
Path("models/trained").mkdir(parents=True, exist_ok=True)
Path("models/scalers").mkdir(parents=True, exist_ok=True)
save_pipeline({'scaler': scaler}, _scaler_name,
              folder="models/scalers")
# Output model name: read from EMG_MODEL_OUTPUT_NAME env var if set,
# fall back to the generic name used by the CLI workflow.
import os as _os
_model_output_name = _os.environ.get(
    "EMG_MODEL_OUTPUT_NAME",
    f"model_causal_concat{X_train_s.shape[1]}.keras",
)
model.save(f"models/trained/{_model_output_name}")
print(f"\n   Output model saved as: models/trained/{_model_output_name}")

# Save the offline confusion matrix next to the model (for the app's viewer).
try:
    import csv as _csv
    from sklearn.metrics import confusion_matrix as _cm
    _idx2name = {v: k for k, v in ACTIONS.items()}
    _names = [str(_idx2name.get(i, i)) for i in range(N_CLASSES)]
    _yp = model.predict(X_test_s, verbose=0).argmax(1)
    _M = _cm(y_test, _yp, labels=list(range(N_CLASSES)))
    _stem = (_model_output_name[:-6] if _model_output_name.endswith(".keras")
             else _model_output_name)
    _cpath = f"models/trained/{_stem}.confusion_offline.csv"
    with open(_cpath, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow([""] + _names)
        for _i in range(N_CLASSES):
            _w.writerow([_names[_i]] + [int(v) for v in _M[_i]])
    print(f"   saved offline confusion: {_cpath}")
except Exception as _e:
    print(f"   (offline confusion not saved: {_e})")

print(f"\n══════════════════════════════════════════════════════")
print(f" Done")
print(f"   BATCH1 val accuracy: {results_val['accuracy']:.4f} | F1: {results_val['f1_macro']:.4f}")
print(f"   BATCH2 test accuracy: {results_test['accuracy']:.4f} | F1: {results_test['f1_macro']:.4f}")
print(f"══════════════════════════════════════════════════════")