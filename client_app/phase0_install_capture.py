"""
phase0_install_capture.py
=========================
Phase 0 - Install capture firmware.

Before recording any EMG data, the bracelet must run firmware that
streams raw EMG over BLE. That's the same hybrid firmware Phase 3 flashes
(exo_armband_hybrid_6clf) - it streams raw EMG unconditionally, and only
runs inference once a model has been loaded (which hasn't happened yet
at this point in a fresh workflow). There is no separate "raw-only"
sketch anymore; see lib/firmware_uploader.install_raw_firmware().

This screen gives the user two options:

  - Install capture firmware : compiles and flashes the (unified) sketch
    via lib/firmware_uploader.install_raw_firmware(). Takes ~2-3 minutes.
    Also perfectly fine to skip this and use Phase 3's "Install on
    bracelet" instead - they flash the same firmware.
  - Skip : moves to Phase 1 immediately, assuming the bracelet already
    runs it (e.g. flashed earlier from this screen or from Phase 3).

After a successful install OR a skip, the user moves to Phase 1 by
pressing the global "Next" button.
"""

import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Make the project root importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib import firmware_uploader as fu  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────

class RawFlashWorker(QThread):
    """Runs firmware_uploader.install_raw_firmware() in a thread."""

    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed      = pyqtSignal(str)

    def __init__(self, port=None, parent=None):
        super().__init__(parent)
        self.port = port

    def run(self):
        try:
            fu.install_raw_firmware(
                port=self.port,
                on_progress=lambda msg: self.progress.emit(msg),
            )
            self.finished_ok.emit()
        except fu.FirmwareUploaderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Phase 0 widget
# ─────────────────────────────────────────────────────────────────────

class Phase0InstallCapture(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._detected_port = None
        self._install_done = False
        self._build_ui()
        # Auto-probe on open
        self._refresh_port(silent=True)

    # ── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(14)

        # Title
        title = QLabel("Phase 0 - Install capture firmware")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        intro = QLabel(
            "Before recording, the bracelet must run the capture firmware\n"
            "(same firmware Phase 3 uses - install it here or there, either works).\n"
            "Install it now, or skip if you already did it previously.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        outer.addWidget(intro)

        # ── Info card ──
        info_card = QFrame()
        info_card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:8px; }")
        info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(18, 14, 18, 14)
        info_lay.setSpacing(8)

        port_row = QHBoxLayout()
        port_row.setSpacing(8)
        self.lbl_port = QLabel("Detected port: (searching\u2026)")
        self.lbl_port.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        port_row.addWidget(self.lbl_port, stretch=1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(
            "QPushButton { background:#2a3550; color:#e5ebff;"
            " border:1px solid #3a4570; border-radius:6px;"
            " padding:6px 14px; font-size:12px; }"
            "QPushButton:hover { background:#3a4570; }")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        port_row.addWidget(self.btn_refresh)
        info_lay.addLayout(port_row)
        outer.addWidget(info_card)

        # ── Action buttons row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_install = QPushButton("\u25b6  Install capture firmware")
        self.btn_install.setMinimumHeight(48)
        self.btn_install.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020;"
            " font-size:14px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 22px; }"
            "QPushButton:hover { background:#3ee685; }"
            "QPushButton:disabled { background:#3a4570; color:#7c87a8; }")
        self.btn_install.clicked.connect(self._on_install_clicked)
        btn_row.addWidget(self.btn_install, stretch=2)

        self.btn_skip = QPushButton("Skip  \u2014  already installed")
        self.btn_skip.setMinimumHeight(48)
        self.btn_skip.setStyleSheet(
            "QPushButton { background:#2a3550; color:#e5ebff;"
            " font-size:13px; font-weight:700; border:1px solid #3a4570;"
            " border-radius:8px; padding:10px 22px; }"
            "QPushButton:hover { background:#3a4570; }")
        self.btn_skip.clicked.connect(self._on_skip_clicked)
        btn_row.addWidget(self.btn_skip, stretch=1)

        outer.addLayout(btn_row)

        # ── Progress bar (indeterminate while running) ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:6px; height:10px; }"
            "QProgressBar::chunk { background:#2ed573; border-radius:5px; }")
        outer.addWidget(self.progress_bar)

        # ── Log ──
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "QTextEdit { background:#0e1422; color:#a8b3d4;"
            " border:1px solid #2a3550; border-radius:6px;"
            " font-family: 'Menlo', 'Monaco', monospace; font-size:11px; }")
        self.log.setMinimumHeight(180)
        f = QFont("Menlo")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(f)
        outer.addWidget(self.log, stretch=1)

        # Status hint
        self.status_label = QLabel(
            "Ready. Plug in the bracelet, then click Install or Skip.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color:#9aa6c8; font-size:12px; border:none; padding:4px 0;")
        outer.addWidget(self.status_label)

        self._update_button_state()

    # ── State helpers ──────────────────────────────────────────────
    def _update_button_state(self):
        is_running = self.worker is not None and self.worker.isRunning()
        self.btn_install.setEnabled(bool(self._detected_port) and not is_running)
        self.btn_skip.setEnabled(not is_running)
        self.btn_refresh.setEnabled(not is_running)

        if is_running:
            self.btn_install.setText("\u23f3  Installing\u2026 do not unplug")
        else:
            self.btn_install.setText("\u25b6  Install capture firmware")

    def _refresh_port(self, silent=False):
        try:
            self._detected_port = fu.detect_bracelet_port()
        except fu.FirmwareUploaderError as exc:
            self._detected_port = None
            if not silent:
                self._append_log(f"[detect] {exc}")
        if self._detected_port:
            self.lbl_port.setText(f"Detected port: {self._detected_port}")
        else:
            self.lbl_port.setText(
                "Detected port: (no bracelet found \u2014 plug it in, then Refresh)")
        self._update_button_state()

    def _append_log(self, msg):
        self.log.append(msg)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── Event handlers ─────────────────────────────────────────────
    def _on_refresh_clicked(self):
        self._append_log("[detect] Searching for bracelet\u2026")
        self._refresh_port(silent=False)
        if self._detected_port:
            self._append_log(f"[detect] Found: {self._detected_port}")
        else:
            self._append_log(
                "[detect] No bracelet found. Is it plugged in and powered on?")

    def _on_install_clicked(self):
        if not self._detected_port:
            self._append_log(
                "[error] No bracelet detected. Plug it in and click Refresh.")
            return

        self.log.clear()
        self._append_log("Starting raw capture firmware install\u2026")
        self._append_log(f"  Port: {self._detected_port}")
        self._append_log("")

        self.progress_bar.setRange(0, 0)  # indeterminate
        self.status_label.setText(
            "Compiling and flashing\u2026 ~2-3 minutes. Do not unplug.")

        self.worker = RawFlashWorker(port=self._detected_port)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_flash_ok)
        self.worker.failed.connect(self._on_flash_failed)
        self.worker.start()
        self._update_button_state()

    def _on_skip_clicked(self):
        """User says the bracelet already runs the raw firmware. Move on."""
        self._jump_to_phase1()

    def _on_progress(self, msg):
        self._append_log(msg)

    def _on_flash_ok(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._append_log("")
        self._append_log(
            "\u2705 Capture firmware installed. You can now move to Phase 1.")
        self.status_label.setText("Done. Click Next to start recording.")
        self._install_done = True
        self.worker = None
        self._update_button_state()

    def _on_flash_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log("\u274c Install failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self._append_log("")
        self._append_log(
            "You can fix the issue (replug the bracelet, etc.) and click Install again.")
        self.status_label.setText("Install failed. See log above.")
        self.worker = None
        self._refresh_port(silent=True)
        self._update_button_state()

    def _jump_to_phase1(self):
        """Bubble up to AppWindow and ask it to navigate to Phase 1."""
        parent = self.parent()
        while parent is not None and not hasattr(parent, "stack"):
            parent = parent.parent()
        if parent is not None and hasattr(parent, "stack"):
            try:
                parent.stack.setCurrentIndex(1)  # Phase 1 (now index 1)
                if hasattr(parent, "update_navigation"):
                    parent.update_navigation()
            except Exception:
                pass
