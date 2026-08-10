"""
take_viewer.py
==============
In-app viewer for the last recorded take. Rebuilds the same kind of plot as
view_single_take.py (raw signals + protocol-label bands) but embeds the
matplotlib figure inside a Qt dialog via the QtAgg backend, instead of
launching a separate process.

If a per-channel baseline_std array (from the calibration screen) is supplied,
the dialog analyses the resting segments of the take and warns the user if the
noise level is significantly above the personal baseline.

Usage:
    TakeViewerDialog(csv_path, parent, baseline_std=array).exec()
"""

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QFrame,
)
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor

try:
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg, NavigationToolbar2QT,
    )
    _DEPS_OK = True
    _DEPS_ERR = ""
except Exception as exc:
    _DEPS_OK = False
    _DEPS_ERR = str(exc)


N_CH = 8
RAW_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
    "#3498db", "#9b59b6", "#e84393", "#fd79a8",
]
LABEL_COLORS = {
    "rest": "#95a5a6", "flexion": "#3498db", "extension": "#e67e22",
    "close": "#27ae60", "supination": "#9b59b6", "pronation": "#e84393",
    "pause": "#dfe6e9", "none": "#dfe6e9",
}

# A channel's resting noise must exceed this multiple of its calibration
# baseline_std before it is flagged as noisy.
NOISE_FACTOR = 3.0

# At least this fraction of channels must be noisy to trigger a "redo" warning.
NOISY_CH_FRAC = 0.25   # e.g. 2 / 8 channels


# ── Noise assessment ──────────────────────────────────────────────────────────

def assess_take_noise(df, baseline_std):
    """Compare resting-segment noise in *df* against *baseline_std*.

    Parameters
    ----------
    df : pd.DataFrame
        CSV loaded from the take file. Must contain Raw_CH0..Raw_CH7 and Label.
    baseline_std : np.ndarray, shape (N_CH,)
        Per-channel std measured during calibration at rest.

    Returns
    -------
    dict with keys:
        "noisy_channels"  : list[int]   channels that exceeded the threshold
        "take_std"        : np.ndarray  per-channel std during rest in this take
        "threshold"       : np.ndarray  NOISE_FACTOR × baseline_std
        "verdict"         : "ok" | "warn" | "bad"
        "message"         : str         human-readable summary
    """
    rest_labels = {"pause", "rest"}
    raw_cols = [f"Raw_CH{i}" for i in range(N_CH) if f"Raw_CH{i}" in df.columns]
    n = len(raw_cols)

    # Extract resting rows
    if "Label" in df.columns:
        mask = df["Label"].astype(str).isin(rest_labels)
        rest_df = df.loc[mask, raw_cols]
    else:
        rest_df = df[raw_cols]   # no label column → use everything

    if rest_df.empty or baseline_std is None or len(baseline_std) < n:
        return {
            "noisy_channels": [],
            "take_std": np.zeros(n),
            "threshold": np.zeros(n),
            "verdict": "ok",
            "message": "",
        }

    take_std  = rest_df.values.std(axis=0)          # (n,)
    threshold = NOISE_FACTOR * baseline_std[:n]      # (n,)
    noisy     = [i for i in range(n) if take_std[i] > threshold[i]]

    frac = len(noisy) / n if n > 0 else 0.0

    if frac == 0.0:
        verdict = "ok"
        message = "✓ Good take — resting noise within the personal baseline."
    elif frac <= NOISY_CH_FRAC:
        verdict = "warn"
        ch_str  = ", ".join(f"CH{i}" for i in noisy)
        message = (
            f"⚠ Slightly noisy resting segments on {ch_str}. "
            "You may continue, but consider redoing this take."
        )
    else:
        verdict = "bad"
        ch_str  = ", ".join(f"CH{i}" for i in noisy)
        message = (
            f"✗ Too much noise during rest on {ch_str}. "
            "Please redo this recording."
        )

    return {
        "noisy_channels": noisy,
        "take_std":  take_std,
        "threshold": threshold,
        "verdict":   verdict,
        "message":   message,
    }


# ── Figure builder ────────────────────────────────────────────────────────────

def build_take_figure(csv_path):
    df = pd.read_csv(csv_path)

    if "Time(ms)" in df.columns and len(df):
        t = df["Time(ms)"].values / 1000.0
    else:
        t = np.arange(len(df)) / 900.0

    fig = Figure(figsize=(11, 6.5))
    ax_sig = fig.add_subplot(2, 1, 1)
    ax_act = fig.add_subplot(2, 1, 2, sharex=ax_sig)

    raw_cols = [c for c in (f"Raw_CH{i}" for i in range(N_CH)) if c in df.columns]
    for i, col in enumerate(raw_cols):
        ax_sig.plot(t, df[col], color=RAW_COLORS[i % len(RAW_COLORS)],
                    linewidth=0.7, alpha=0.85, label=col)
    ax_sig.set_title("Raw signals")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.grid(True, alpha=0.3)
    if raw_cols:
        ax_sig.legend(loc="upper right", ncol=4, fontsize=8)

    if "Label" in df.columns and len(df):
        labels = df["Label"].astype(str).values
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                lbl     = labels[start]
                end_idx = min(i - 1, len(t) - 1)
                if start <= end_idx:
                    ax_act.axvspan(t[start], t[end_idx],
                                   color=LABEL_COLORS.get(lbl, "#bdc3c7"),
                                   alpha=0.35)
                    if (i - start) > 5:
                        mid_idx = min((start + i) // 2, len(t) - 1)
                        ax_act.text(t[mid_idx], 0.5, lbl, ha="center",
                                    va="center", fontsize=9, fontweight="bold")
                start = i

    ax_act.set_ylim(0, 1)
    ax_act.set_yticks([])
    ax_act.set_xlabel("Time [s]")
    ax_act.set_title("Protocol labels")
    fig.tight_layout()
    return fig, pd.read_csv(csv_path)   # return df so the caller can assess it


# ── Toolbar icon helper ───────────────────────────────────────────────────────

def _whiten_toolbar(toolbar):
    """Recolour matplotlib navigation-toolbar icons to white for dark theme."""
    size = toolbar.iconSize()
    if not size.isValid() or size.isEmpty():
        size = QSize(24, 24)
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        src = icon.pixmap(size)
        if src.isNull():
            continue
        white = QPixmap(src.size())
        white.fill(QColor(0, 0, 0, 0))
        painter = QPainter(white)
        painter.drawPixmap(0, 0, src)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(white.rect(), QColor("white"))
        painter.end()
        action.setIcon(QIcon(white))
    toolbar.setStyleSheet(
        "QToolBar { background:#1a2236; border:none; }"
        "QToolButton { background:transparent; padding:3px; }"
        "QToolButton:hover { background:#2a3550; border-radius:4px; }"
        "QToolBar QLabel { color:#e6e9f2; }")


# ── Noise banner helper ───────────────────────────────────────────────────────

def _noise_banner(verdict, message):
    """Return a styled QLabel banner for the given verdict."""
    palette = {
        "ok":   ("#16321f", "#2ed573", "#2ed573"),
        "warn": ("#332a16", "#ffdd59", "#ffdd59"),
        "bad":  ("#3a1a1a", "#ff6b6b", "#ff6b6b"),
    }
    bg, fg, border = palette.get(verdict, palette["ok"])
    lbl = QLabel(message)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"QLabel {{ background:{bg}; color:{fg};"
        f" border-left:4px solid {border}; border-radius:8px;"
        " padding:12px 16px; font-size:14px; font-weight:700; }}")
    return lbl


# ── Dialog ────────────────────────────────────────────────────────────────────

class TakeViewerDialog(QDialog):
    RESULT_CONTINUE = 1
    RESULT_REDO = 2
    def __init__(self, csv_path, parent=None, baseline_std=None):
        super().__init__(parent)
        self.setWindowTitle(f"Last take — {os.path.basename(csv_path)}")
        self.resize(1000, 720)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        if not _DEPS_OK:
            lay.addWidget(QLabel(
                "matplotlib / pandas are required to display the take.\n"
                f"({_DEPS_ERR})\nInstall with: pip install matplotlib pandas"))
            self._add_close(lay)
            return

        try:
            fig, df = build_take_figure(csv_path)
        except Exception as exc:
            lay.addWidget(QLabel(f"Could not display this take:\n{exc}"))
            self._add_close(lay)
            return

        # ── Noise verdict banner ──────────────────────────────────────────
        if baseline_std is not None:
            result = assess_take_noise(df, baseline_std)
            if result["verdict"] != "ok" or True:   # always show banner
                lay.addWidget(_noise_banner(result["verdict"], result["message"]))

                # Per-channel detail line when there are noisy channels
                if result["noisy_channels"]:
                    import numpy as np
                    lines = []
                    for i in result["noisy_channels"]:
                        ts = result["take_std"][i]
                        th = result["threshold"][i]
                        lines.append(
                            f"CH{i}: noise σ={ts:.2f}  (threshold {th:.2f})")
                    detail = QLabel("  |  ".join(lines))
                    detail.setStyleSheet(
                        "color:#ff6b6b; font-size:11px; border:none;"
                        " padding:2px 16px;")
                    lay.addWidget(detail)
        else:
            # No calibration data available — neutral info message
            info = QLabel(
                "ℹ No calibration baseline available. "
                "Run the calibration step to enable noise detection.")
            info.setStyleSheet(
                "QLabel { background:#16203a; color:#9fb3ff;"
                " border-left:4px solid #2a3550; border-radius:8px;"
                " padding:10px 16px; font-size:13px; }")
            lay.addWidget(info)

        # ── Plot ─────────────────────────────────────────────────────────
        canvas  = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        _whiten_toolbar(toolbar)
        lay.addWidget(toolbar)
        lay.addWidget(canvas, 1)
        self._add_close(lay)

    def _add_close(self, lay):
        from PyQt6.QtWidgets import QHBoxLayout, QPushButton
        row = QHBoxLayout()
        row.addStretch(1)
        btn_redo = QPushButton("Redo this take")
        btn_redo.setMinimumHeight(40)
        btn_redo.setStyleSheet(
            "QPushButton { background:#ff6b6b; color:#1a0a0a;"
            " font-size:13px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 22px; }")
        btn_redo.clicked.connect(lambda: self.done(self.RESULT_REDO))
        row.addWidget(btn_redo)
        btn_cont = QPushButton("Continue →")
        btn_cont.setMinimumHeight(40)
        btn_cont.setDefault(True)
        btn_cont.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020;"
            " font-size:13px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 22px; }")
        btn_cont.clicked.connect(lambda: self.done(self.RESULT_CONTINUE))
        row.addWidget(btn_cont)
        row.addStretch(1)
        lay.addLayout(row)
