"""
config.py — Global configuration for the EMG project
====================================================
Centralizes all shared parameters between notebooks and scripts.

The CSV files are recorded in 6-class mode (rest, flexion, extension, close,
supination, pronation). The training mode (N_CLASSES_MODE) controls which
labels are actually used:
  - 4 classes: rest, flexion, extension, close (supination/pronation rows ignored)
  - 6 classes: all six gestures

The filtering is automatic: labels not present in the ACTIONS dict are mapped to
Action=-1 by data_loader and excluded downstream.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV folders contain 6-class recordings; the same data is reused for both 4cl and 6cl
# DATA_RAW and DATA_TEST_RAW can be overridden via environment variables,
# which is how the desktop app (client_app/phase2_training.py) injects
# the subject + batch selected in the UI.
# Falls back to S003 PRONATION defaults when no env var is set, keeping
# the CLI workflow (python scripts/train_causal_concat.py) backwards compatible.
DATA_RAW      = os.environ.get(
    "EMG_DATA_RAW",
    os.path.join(ROOT_DIR, "data", "S003_BATCH1_PRONATION"),
)
DATA_TEST_RAW = os.environ.get(
    "EMG_DATA_TEST_RAW",
    os.path.join(ROOT_DIR, "data", "S003_BATCH2_PRONATION"),
)

DATA_PROC    = os.path.join(ROOT_DIR, "data", "processed")
DATA_FEAT    = os.path.join(ROOT_DIR, "data", "features")
DATA_OFFLINE = os.path.join(ROOT_DIR, "data", "offline")
MODELS_DIR   = os.path.join(ROOT_DIR, "models", "trained")
EXPORT_DIR   = os.path.join(ROOT_DIR, "models", "exported")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
FIRMWARE_DIR = os.path.join(ROOT_DIR, "firmware", "teensy")

# ─── Acquisition ────────────────────────────────────────────────────────
SAMPLING_FREQ   = 1200   # Hz (BLE notify ~60 pkt/s * 20 samples/pkt)
WINDOW_SIZE     = 128
WINDOW_TIME     = WINDOW_SIZE / SAMPLING_FREQ
N_CHANNELS      = 8
LABELS          = [f"Raw_CH{i}" for i in range(N_CHANNELS)]
COLORS          = ['red', 'green', 'blue', 'orange', 'purple', 'magenta']

# ─── Preprocessing ──────────────────────────────────────────────────────
LOWCUT          = 35.0
HIGHCUT         = 300.0
FILTER_ORDER    = 4
DELAY_S         = 5.0
KERNEL_RATIO    = 0.10

# ─── Classes — switch here to retrain in 4cl or 6cl mode ────────────────
# Change this single value to control the whole pipeline.
GESTURE_SETS = {
    "4cl": ["rest", "flexion", "extension", "close"],
    "6cl": ["rest", "flexion", "extension", "close", "supination", "pronation"],
    "rps": ["idle", "rock", "paper", "scissors"],
}

# Active set. Overridable via env var (EMG_GESTURE_SET) so the capture UI and
# the training subprocess agree on which gestures are in play. Default keeps
# the previous behaviour (6-class).
GESTURE_SET = os.environ.get("EMG_GESTURE_SET", "6cl")
if GESTURE_SET not in GESTURE_SETS:
    raise ValueError(
        f"EMG_GESTURE_SET must be one of {list(GESTURE_SETS)}, got {GESTURE_SET!r}"
    )

# Backward-compatible aliases (other modules still import these names).
GESTURES_4CL = GESTURE_SETS["4cl"]
GESTURES_6CL = GESTURE_SETS["6cl"]

ACTIONS = {g: i for i, g in enumerate(GESTURE_SETS[GESTURE_SET])}
N_CLASSES = len(ACTIONS)
N_CLASSES_MODE = N_CLASSES   # backward-compat alias

# ─── Features ───────────────────────────────────────────────────────────
FEATURE_MODE = "concat"
TSD_WIN_MS   = 80
TSD_INC_MS   = 40
TSD_LAM      = 0.1

# ─── Training ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
VAL_RATIO    = 0.20
TEST_RATIO   = 0.20

# ─── Gesture-set helpers (single source of truth) ───────────────────────
# The gesture set (4cl / 6cl / rps) is a PROPERTY of the recorded data and of
# the trained model, not a manual switch. These helpers let every part of the
# pipeline resolve the set the same way, without relying on EMG_GESTURE_SET.
import csv as _csv
import glob as _glob


def gestures_for(set_name):
    """Ordered gesture names for a set (e.g. 'rps' -> [idle, rock, paper, scissors])."""
    return list(GESTURE_SETS[set_name])


def actions_for(set_name):
    """{gesture: class_index} for a set."""
    return {g: i for i, g in enumerate(GESTURE_SETS[set_name])}


def n_classes_for(set_name):
    return len(GESTURE_SETS[set_name])


def detect_gesture_set(labels):
    """Infer the set name from a collection of recorded labels, else None.

    Distinctive markers, most specific first: rock/paper/scissors/idle -> rps;
    supination/pronation -> 6cl; flexion/extension/close -> 4cl.
    """
    obs = {str(l).strip().lower() for l in labels}
    if obs & {"rock", "paper", "scissors", "idle"}:
        return "rps"
    if obs & {"supination", "pronation"}:
        return "6cl"
    if obs & {"flexion", "extension", "close"}:
        return "4cl"
    return None


def gesture_set_of_folder(folder):
    """Detect the gesture set of a data/<subject>_BATCH*/ folder from its CSV
    Label column. Returns as soon as a distinctive label is seen (fast)."""
    # An explicit stamp (written by the logger at capture time) wins.
    try:
        meta = os.path.join(folder, "gesture_set.txt")
        if os.path.isfile(meta):
            with open(meta, encoding="utf-8") as f:
                v = f.read().strip().lower()
            if v in GESTURE_SETS:
                return v
    except Exception:
        pass
    try:
        csvs = sorted(_glob.glob(os.path.join(folder, "*.csv")))
    except Exception:
        return None
    labels = set()
    for path in csvs:
        try:
            with open(path, newline="") as f:
                reader = _csv.DictReader(f)
                if not reader.fieldnames or "Label" not in reader.fieldnames:
                    continue
                for r in reader:
                    v = r.get("Label")
                    if v:
                        labels.add(str(v).strip().lower())
                        # rps / 6cl markers are unique -> decidable at once.
                        # 4cl is NOT (flexion is in 6cl too), so don't early-
                        # return on it: a 6cl take has flexion before supination.
                        if labels & {"rock", "paper", "scissors", "idle"}:
                            return "rps"
                        if labels & {"supination", "pronation"}:
                            return "6cl"
        except Exception:
            continue
    return detect_gesture_set(labels)


def set_from_model_name(path):
    """Extract the gesture-set token from a model filename, else None.

    e.g. model_S013_rps_20260625.keras -> 'rps'; model_S012_6cl_... -> '6cl'.
    """
    base = os.path.basename(str(path or "")).lower()
    for s in GESTURE_SETS:
        if f"_{s}_" in base or f"_{s}." in base:
            return s
    return None