"""Post-acquisition viewer — 8-channel EMG session.

Standalone usage: python view_single_take.py path/to/file.csv
From dashboard: view_last_take(file_path)

CSV expected format:
    Time(ms), Raw_CH0..7, Amp_CH0..7, Label, Window
"""

import matplotlib
import sys

if sys.platform == "darwin":
    matplotlib.use("MacOSX")
elif sys.platform == "win32":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("TkAgg")

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

# ── Default configuration ────────────────────────────────────────────────

N_CH = 8
LABELS = [f"Raw_CH{i}" for i in range(N_CH)]
COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
    "#3498db", "#9b59b6", "#e84393", "#fd79a8",
]

# Mapping label string → numeric Action (for overlay)
ACTIONS = {
    "none":       0,
    "pause":      0,
    "rest":       1,
    "flexion":    2,
    "extension":  3,
    "close":      4,
    "supination": 5,
    "pronation":  6,
}


# ── Loading ─────────────────────────────────────────────────────────────

def load_csv(file_path: str):
    """Load CSV and return (DataFrame, metadata dict)."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Label" in df.columns and df["Label"].dtype == object:
        df["Action"] = df["Label"].map(ACTIONS).fillna(0)
    elif "Action" not in df.columns:
        df["Action"] = 0

    basename = os.path.basename(file_path)
    parts = basename.split("_")

    # Time axis: use Time(ms) column if available
    if "Time(ms)" in df.columns:
        duration_s = (df["Time(ms)"].iloc[-1] - df["Time(ms)"].iloc[0]) / 1000.0
    else:
        duration_s = len(df) / 900.0  # fallback

    meta = {
        "basename": basename,
        "date_str": parts[0] if len(parts) >= 1 else "N/A",
        "time_str": parts[1] if len(parts) >= 2 else "N/A",
        "duration": round(duration_s, 2),
        "n_samples": len(df),
    }

    print(f"File loaded: {basename}")
    print(f"   Duration: {meta['duration']} s  |  Samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    if "Label" in df.columns:
        print(f"   Labels found: {sorted(df['Label'].unique())}")

    return df, meta


# ── Main visualization ──────────────────────────────────────────────────

def plot_EMG_data(df: pd.DataFrame, title: str,
                  labels=None, colors=None):
    if labels is None:
        labels = LABELS
    if colors is None:
        colors = COLORS

    # Time axis in seconds
    if "Time(ms)" in df.columns:
        t = df["Time(ms)"].values / 1000.0
    else:
        t = np.arange(len(df)) / 900.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Top: raw signals
    ax_sig = axes[0]
    for lbl, color in zip(labels, colors):
        if lbl in df.columns:
            ax_sig.plot(t, df[lbl], label=lbl, color=color, linewidth=0.7, alpha=0.85)

    ax_sig.set_title("Raw signals")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.legend(loc="upper right", ncol=4, fontsize=8)

    # Bottom: actions timeline (numeric)
    ax_act = axes[1]
    if "Label" in df.columns:
        # Color-coded background by label
        unique_labels = df["Label"].unique()
        label_to_y = {lbl: i for i, lbl in enumerate(sorted(unique_labels))}

        # Draw vertical color bands
        for lbl in unique_labels:
            mask = df["Label"] == lbl
            if mask.sum() == 0:
                continue
            indices = np.where(mask.values)[0]
            # Find contiguous regions
            splits = np.where(np.diff(indices) > 1)[0]
            starts = np.concatenate([[indices[0]], indices[splits + 1]])
            ends = np.concatenate([indices[splits], [indices[-1]]])
            for s, e in zip(starts, ends):
                if s < len(t) and e < len(t):
                    ax_act.axvspan(t[s], t[e], ymin=0.0, ymax=1.0,
                                    alpha=0.3, color=_label_color(lbl))

        # Plot label text at midpoint of each block
        last_lbl = None
        block_start = 0
        for i, lbl in enumerate(df["Label"].values):
            if lbl != last_lbl:
                if last_lbl is not None and (i - block_start) > 5:
                    mid_t = t[(block_start + i) // 2]
                    ax_act.text(mid_t, 0.5, last_lbl, ha="center", va="center",
                                  fontsize=9, fontweight="bold")
                block_start = i
                last_lbl = lbl
        # Last block
        if last_lbl is not None and (len(df) - block_start) > 5:
            mid_t = t[(block_start + len(df) - 1) // 2]
            ax_act.text(mid_t, 0.5, last_lbl, ha="center", va="center",
                          fontsize=9, fontweight="bold")

    ax_act.set_ylim(0, 1)
    ax_act.set_yticks([])
    ax_act.set_xlabel("Time [s]")
    ax_act.set_title("Protocol labels")

    plt.tight_layout()
    plt.show(block=True)


def _label_color(label):
    """Distinct color per label for the background band."""
    colormap = {
        "rest":       "#95a5a6",
        "flexion":    "#3498db",
        "extension":  "#e67e22",
        "close":      "#27ae60",
        "supination": "#9b59b6",
        "pronation":  "#e84393",
        "pause":      "#ecf0f1",
        "none":       "#ecf0f1",
    }
    return colormap.get(label, "#bdc3c7")


# ── Quick statistics ────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame):
    if "Label" in df.columns:
        print("\nLabel distribution:")
        for lbl, cnt in df["Label"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"  {lbl:<12} : {cnt:6d} samples  ({pct:.1f} %)")

    print("\nChannel statistics (raw signal):")
    present = [c for c in LABELS if c in df.columns]
    if present:
        print(df[present].describe().round(2))


# ── Main entry point ────────────────────────────────────────────────────

def view_last_take(file_path: str, show_stats=True):
    df, meta = load_csv(file_path)
    title = f"{meta['basename']}  —  {meta['duration']} s"

    plot_EMG_data(df, title)

    if show_stats:
        print_stats(df)


# ── Direct execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Path to CSV file: ").strip()
    view_last_take(path)
