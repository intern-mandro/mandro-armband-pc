"""
train_multisession.py
=====================
Multi-session training with an HONEST metric and NO dedicated test batch.

For the subject inferred from EMG_DATA_RAW, this:
  1. Auto-discovers ALL its session folders  <subject>_BATCH<N>  (e.g. S006_BATCH1, S006_BATCH2).
     Override with EMG_DATA_SESSIONS="dirA:dirB" if you want to pick sessions manually.
  2. Reports an honest accuracy via LEAVE-ONE-SESSION-OUT cross-validation
     (each fold trains on all sessions but one, tests on the held-out session
      = true inter-session / new-placement generalization, no leakage).
     If only ONE session exists, falls back to GroupKFold-by-take and WARNS
     loudly that the number is intra-session only (does NOT reflect live use).
  3. Trains the FINAL deployed model on ALL sessions and saves model + scaler.

It ignores EMG_DATA_TEST_RAW entirely (no S003 fallback).
"""
import sys, os, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from lib.data_loader import load_datasets_new_format
from lib.pipeline import (
    preprocess_pipeline, extract_features_pipeline,
    fit_scaler, apply_scaler, save_pipeline,
)
from lib.windowing import get_windows, get_action_windows, split_train_val
from lib.models import build_model
from lib.training import train_model
from lib.configs import (
    DATA_RAW, ACTIONS, LABELS, SAMPLING_FREQ, WINDOW_SIZE,
    LOWCUT, HIGHCUT, DELAY_S, KERNEL_RATIO,
    TSD_WIN_MS, TSD_INC_MS, TSD_LAM,
    RANDOM_STATE, VAL_RATIO, N_CLASSES,
)
from tensorflow.keras.utils import to_categorical


def preprocess_one(df):
    return preprocess_pipeline(
        df, labels=LABELS, sampling_frequency_EMG=SAMPLING_FREQ,
        lowcut=LOWCUT, highcut=HIGHCUT, delay_s=DELAY_S,
        kernel_ratio=KERNEL_RATIO, window_size=WINDOW_SIZE,
        warmup_samples=100,
    )


def take_to_arrays(df):
    X_list, y_list = [], []
    for w in get_windows(df, WINDOW_SIZE):
        if (w['Action'] == -1).any():
            continue
        X_list.append(w[LABELS].values)
        y_list.append(get_action_windows(w))
    if not X_list:
        return np.empty((0,)), np.empty((0,))
    return np.array(X_list), np.array(y_list)


def features(Xr):
    return extract_features_pipeline(
        Xr, SAMPLING_FREQ, mode="concat",
        tsd_win_ms=TSD_WIN_MS, tsd_inc_ms=TSD_INC_MS, tsd_lam=TSD_LAM,
    )


def discover_sessions():
    raw = os.path.normpath(os.environ.get("EMG_DATA_RAW", DATA_RAW))
    subj = re.sub(r"_BATCH\d+.*$", "", os.path.basename(raw))
    override = os.environ.get("EMG_DATA_SESSIONS")
    if override:
        dirs = [os.path.normpath(d.strip()) for d in override.split(os.pathsep) if d.strip()]
    else:
        parent = os.path.dirname(raw)
        dirs = sorted(
            d for d in glob.glob(os.path.join(parent, f"{subj}_BATCH*"))
            if re.match(rf"^{re.escape(subj)}_BATCH\d+$", os.path.basename(d))
            and os.path.isdir(d)
        )
    return subj, dirs


def cat(parts):
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def fit_train_predict(train_parts, X_test):
    Xtr_full, ytr_full = cat(train_parts)
    Xtr, ytr, Xv, yv = split_train_val(
        Xtr_full, ytr_full, val_size=VAL_RATIO, random_state=RANDOM_STATE)
    sc = fit_scaler(Xtr)
    model = build_model(input_shape=Xtr.shape[1], output_shape=N_CLASSES,
                        n_hidden_layers=2, n_units=64, learning_rate=0.001)
    train_model(model, apply_scaler(sc, Xtr), to_categorical(ytr, N_CLASSES),
                X_val=apply_scaler(sc, Xv), y_val=to_categorical(yv, N_CLASSES),
                epochs=200, batch_size=32)
    return model.predict(apply_scaler(sc, X_test), verbose=0).argmax(1)


subj, session_dirs = discover_sessions()
if not session_dirs:
    raise SystemExit(f"No '{subj}_BATCH<N>' session folders found.")
print("══════════════════════════════════════════════════════")
print(f" Subject {subj}: {len(session_dirs)} session(s)")
for d in session_dirs:
    print("   -", os.path.basename(d))
print("══════════════════════════════════════════════════════")

session_takes = {}
for d in session_dirs:
    name = os.path.basename(d)
    feats = []
    for df in load_datasets_new_format(d, ACTIONS, LABELS):
        Xr, y = take_to_arrays(preprocess_one(df))
        if len(Xr):
            feats.append((features(Xr), y))
    session_takes[name] = feats
    print(f"  {name}: {len(feats)} takes, {sum(len(y) for _, y in feats)} windows")

names = list(session_takes)
conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
fold_accs, fold_f1 = [], []

if len(names) >= 2:
    print("\n══════════════════════════════════════════════════════")
    print(" Honest metric: LEAVE-ONE-SESSION-OUT CV")
    print("══════════════════════════════════════════════════════")
    for held in names:
        train_parts = [t for n in names if n != held for t in session_takes[n]]
        Xte, yte = cat(session_takes[held])
        pred = fit_train_predict(train_parts, Xte)
        a = accuracy_score(yte, pred); f = f1_score(yte, pred, average="macro")
        fold_accs.append(a); fold_f1.append(f)
        conf += confusion_matrix(yte, pred, labels=list(range(N_CLASSES)))
        print(f"  hold out {held}: acc={a:.4f}  f1={f:.4f}")
    metric_name = "leave-one-session-out (inter-session)"
else:
    print("\n⚠️  ONLY ONE SESSION for this subject.")
    print("⚠️  Falling back to GroupKFold-by-take = INTRA-session estimate.")
    print("⚠️  This does NOT reflect live / new-placement performance.")
    print("⚠️  Record a 2nd session (re-don the bracelet) for an honest number.")
    parts = session_takes[names[0]]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
    for tr, te in GroupKFold(n_splits=min(5, len(parts))).split(X, y, groups):
        pred = fit_train_predict([(X[tr], y[tr])], X[te])
        fold_accs.append(accuracy_score(y[te], pred))
        fold_f1.append(f1_score(y[te], pred, average="macro"))
        conf += confusion_matrix(y[te], pred, labels=list(range(N_CLASSES)))
    metric_name = "GroupKFold-by-take (INTRA-session, optimistic)"

cv_acc = float(np.mean(fold_accs)); cv_f1 = float(np.mean(fold_f1))
print(f"\n=== {metric_name} ===")
print(f"accuracy: {cv_acc:.4f} +/- {np.std(fold_accs):.4f}")
print(f"f1 macro: {cv_f1:.4f} +/- {np.std(fold_f1):.4f}")

idx2name = {v: k for k, v in ACTIONS.items()}
rec = conf.diagonal() / conf.sum(axis=1).clip(min=1)
print("\nper-class recall:")
for i in range(N_CLASSES):
    print(f"   {str(idx2name.get(i, i)):<12} {rec[i]:.2f}")

print("\n══════════════════════════════════════════════════════")
print(" Final model: train on ALL sessions (deployment)")
print("══════════════════════════════════════════════════════")
all_parts = [t for n in names for t in session_takes[n]]
X_all, y_all = cat(all_parts)
Xtr, ytr, Xv, yv = split_train_val(
    X_all, y_all, val_size=VAL_RATIO, random_state=RANDOM_STATE)
scaler = fit_scaler(Xtr)
model = build_model(input_shape=Xtr.shape[1], output_shape=N_CLASSES,
                    n_hidden_layers=2, n_units=64, learning_rate=0.001)
train_model(model, apply_scaler(scaler, Xtr), to_categorical(ytr, N_CLASSES),
            X_val=apply_scaler(scaler, Xv), y_val=to_categorical(yv, N_CLASSES),
            epochs=200, batch_size=32)

_model_name = os.environ.get(
    "EMG_MODEL_OUTPUT_NAME", f"model_causal_concat{X_all.shape[1]}.keras")
_scaler_name = (_model_name.replace("model_", "scaler_", 1).replace(".keras", "")
                if _model_name.startswith("model_")
                else f"scaler_causal_concat{X_all.shape[1]}")
Path("models/trained").mkdir(parents=True, exist_ok=True)
Path("models/scalers").mkdir(parents=True, exist_ok=True)
save_pipeline({'scaler': scaler}, _scaler_name, folder="models/scalers")
model.save(f"models/trained/{_model_name}")
print(f"   saved model : models/trained/{_model_name}")

# Save the offline (cross-validation) confusion matrix next to the model.
try:
    import csv as _csv
    _idx2name = {v: k for k, v in ACTIONS.items()}
    _names = [str(_idx2name.get(i, i)) for i in range(N_CLASSES)]
    _stem = _model_name[:-6] if _model_name.endswith(".keras") else _model_name
    _cpath = f"models/trained/{_stem}.confusion_offline.csv"
    with open(_cpath, "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow([""] + _names)
        for _i in range(N_CLASSES):
            _w.writerow([_names[_i]] + [int(v) for v in conf[_i]])
    print(f"   saved offline confusion: {_cpath}")
except Exception as _e:
    print(f"   (offline confusion not saved: {_e})")
print(f"   saved scaler: models/scalers/{_scaler_name}.pkl")

# Bridge line so the current app (phase2 regex 'BATCH2 test accuracy:') still
# extracts an accuracy. This now reports the HONEST CV accuracy, not a holdout.
print(f"BATCH2 test accuracy: {cv_acc:.4f}")
print(f"\nDone. Honest {metric_name} accuracy: {cv_acc:.4f}")