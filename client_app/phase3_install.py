"""
Phase 3 — Install model on the bracelet.

Replaces the old three-step UI (generate headers / open Arduino IDE / confirm)
with a single auto-flash flow powered by lib.firmware_uploader. The user
clicks one button and watches the pipeline run end-to-end.

Public API (unchanged for back-compat with load_session wiring):
    Phase3Install.set_model_path(path)
"""

import asyncio
import os
import sys
from pathlib import Path

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

# Make the project root importable so we can pull in lib.firmware_uploader
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib import firmware_uploader as fu  # noqa: E402
from lib import ble_weights as bw  # noqa: E402
import ble_selection  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────

class FlashWorker(QThread):
    """Runs firmware_uploader.upload_model() in a background thread.

    The full pipeline is: export headers -> copy into sketch -> compile
    -> flash. Each progress line from the underlying subprocess is
    re-emitted as a Qt signal so the UI can append it to the log.
    """
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed      = pyqtSignal(str)

    def __init__(self, model_path, port=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.port       = port

    def run(self):
        try:
            fu.upload_model(
                Path(self.model_path),
                port=self.port,
                on_progress=lambda msg: self.progress.emit(msg),
            )
            self.finished_ok.emit()
        except fu.FirmwareUploaderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class BleWeightsWorker(QThread):
    """Runs lib.ble_weights' serialize+send pipeline in a background thread.

    Unlike FlashWorker (full USB reflash), this only pushes the trained
    weights to a bracelet that is ALREADY running firmware with the BLE
    weights-receive characteristic (see firmware/esp32/exo_armband_hybrid_6clf).
    """
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed      = pyqtSignal(str)

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path

    def run(self):
        try:
            asyncio.run(self._run_async())
            self.finished_ok.emit()
        except bw.BleWeightsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")

    async def _run_async(self):
        address = ble_selection.get_selected()
        if not address:
            self.progress.emit("Scanning for bracelet...")
            try:
                from bleak import BleakScanner
            except ImportError:
                raise bw.BleWeightsError("bleak is not installed")
            target = None
            for attempt in range(1, 4):
                self.progress.emit(f"Scanning for bracelet... (attempt {attempt}/3)")
                devices = await BleakScanner.discover(timeout=5.0)
                target = next((d for d in devices if d.name == ble_selection.DEVICE_NAME), None)
                if target is not None:
                    break
                if attempt < 3:
                    await asyncio.sleep(1.0)
            if target is None:
                raise bw.BleWeightsError(
                    f"Bracelet '{ble_selection.DEVICE_NAME}' not found after 3 attempts")
            address = target.address

        self.progress.emit(f"Exporting weights from {os.path.basename(self.model_path)}...")
        payload = bw.serialize_weights(self.model_path)
        frame = bw.build_frame(payload)
        self.progress.emit(f"Payload: {len(payload)} bytes ({len(frame)} bytes on the wire)")

        await bw.send_weights_ble(
            address, frame, on_progress=lambda msg: self.progress.emit(msg))


# ─────────────────────────────────────────────────────────────────────
# Phase 3 widget
# ─────────────────────────────────────────────────────────────────────

class Phase3Install(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.ble_worker = None
        self._model_path = None
        self._detected_port = None
        self._build_ui()
        # Probe the bracelet on open so the user sees the port right away
        self._refresh_port(silent=True)

    # ── Public API kept for backward compatibility with load_session ──
    def set_model_path(self, path):
        """Set the .keras model used by the next 'Install' click.

        Called by the start menu's 'Load an old session' flow. Pass None
        to clear and fall back to whatever the export script's default is.
        """
        self._model_path = path or None
        self._update_info_labels()
        self._update_button_state()

    # ── UI construction ──────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(14)

        # Title
        title = QLabel("Install the model on your bracelet")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        intro = QLabel(
            "Click Install and the bracelet will be re-flashed automatically.\n"
            "No need to open Arduino IDE.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        outer.addWidget(intro)

        # ── Info card: model + port ──
        info_card = self._make_card()
        info_lay = QVBoxLayout(info_card)
        info_lay.setContentsMargins(18, 14, 18, 14)
        info_lay.setSpacing(8)

        self.lbl_model = QLabel("Selected model: (none — pick one from the Load screen first)")
        self.lbl_model.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        self.lbl_model.setWordWrap(True)
        info_lay.addWidget(self.lbl_model)

        port_row = QHBoxLayout()
        port_row.setSpacing(8)
        self.lbl_port = QLabel("Detected port: (searching...)")
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

        # ── Install button ──
        self.btn_install = QPushButton("\u25b6  Install on bracelet")
        self.btn_install.setMinimumHeight(48)
        self.btn_install.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020;"
            " font-size:15px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 28px; }"
            "QPushButton:hover { background:#3ee685; }"
            "QPushButton:disabled { background:#3a4570; color:#7c87a8; }")
        self.btn_install.clicked.connect(self._on_install_clicked)
        outer.addWidget(self.btn_install)

        # ── BLE weights-only path (bracelet must already be running the
        #    BLE-capable firmware once flashed via the button above) ──
        ble_row = QHBoxLayout()
        ble_row.setSpacing(8)

        self.btn_choose_bracelet = QPushButton("Choose bracelet")
        self.btn_choose_bracelet.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-weight:700;"
            " border:1px solid #2a3550; border-radius:8px; padding:8px; }"
            "QPushButton:hover { border:1px solid #3d4d75; }")
        self.btn_choose_bracelet.clicked.connect(self._choose_bracelet)
        ble_row.addWidget(self.btn_choose_bracelet)

        self.btn_send_ble = QPushButton("\U0001F4E1  Send weights over BLE (no reflash)")
        self.btn_send_ble.setMinimumHeight(40)
        self.btn_send_ble.setStyleSheet(
            "QPushButton { background:#3a73ff; color:#ffffff;"
            " font-size:13px; font-weight:700; border:none;"
            " border-radius:8px; padding:10px 18px; }"
            "QPushButton:hover { background:#4a85ff; }"
            "QPushButton:disabled { background:#2a3550; color:#7c87a8; }")
        self.btn_send_ble.clicked.connect(self._on_send_weights_clicked)
        ble_row.addWidget(self.btn_send_ble, stretch=1)
        outer.addLayout(ble_row)

        ble_hint = QLabel(
            "Use this after the bracelet is already running firmware with BLE "
            "weight support (flash it once with the button above first).")
        ble_hint.setWordWrap(True)
        ble_hint.setStyleSheet("color:#7c87a8; font-size:11px; border:none;")
        outer.addWidget(ble_hint)

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

        # ── Log area ──
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "QTextEdit { background:#0e1422; color:#a8b3d4;"
            " border:1px solid #2a3550; border-radius:6px;"
            " font-family: 'Menlo', 'Monaco', monospace; font-size:11px; }")
        self.log.setMinimumHeight(180)
        # Use a fixed-width font for log lines so progress bytes/% align
        f = QFont("Menlo")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(f)
        outer.addWidget(self.log, stretch=1)

        # ── Continue button (hidden until success) ──
        self.btn_continue = QPushButton("Continue \u2192")
        self.btn_continue.setMinimumHeight(40)
        self.btn_continue.setStyleSheet(
            "QPushButton { background:#3a73ff; color:#ffffff;"
            " font-size:13px; font-weight:700; border:none;"
            " border-radius:8px; padding:10px 22px; }"
            "QPushButton:hover { background:#4a85ff; }")
        self.btn_continue.clicked.connect(self._on_continue_clicked)
        self.btn_continue.hide()
        outer.addWidget(self.btn_continue)

        self._update_button_state()

    def _make_card(self):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:8px; }")
        return card

    # ── State helpers ────────────────────────────────────────────────
    def _update_info_labels(self):
        if self._model_path:
            name = os.path.basename(self._model_path)
            self.lbl_model.setText(f"Selected model: {name}")
        else:
            self.lbl_model.setText(
                "Selected model: (none \u2014 pick one from the Load screen first)")

        if self._detected_port:
            self.lbl_port.setText(f"Detected port: {self._detected_port}")
        else:
            self.lbl_port.setText(
                "Detected port: (no bracelet found \u2014 plug it in then click Refresh)")

    def _update_button_state(self):
        usb_running = self.worker is not None and self.worker.isRunning()
        ble_running = self.ble_worker is not None and self.ble_worker.isRunning()
        any_running = usb_running or ble_running

        ready = bool(self._model_path) and bool(self._detected_port) and not any_running
        self.btn_install.setEnabled(ready)
        if usb_running:
            self.btn_install.setText("\u23f3  Installing... do not unplug")
        else:
            self.btn_install.setText("\u25b6  Install on bracelet")

        ble_ready = bool(self._model_path) and not any_running
        self.btn_send_ble.setEnabled(ble_ready)
        self.btn_choose_bracelet.setEnabled(not any_running)
        if ble_running:
            self.btn_send_ble.setText("\u23f3  Sending over BLE...")
        else:
            self.btn_send_ble.setText("\U0001F4E1  Send weights over BLE (no reflash)")

    def _refresh_port(self, silent=False):
        try:
            self._detected_port = fu.detect_bracelet_port()
        except fu.FirmwareUploaderError as exc:
            self._detected_port = None
            if not silent:
                self._append_log(f"[detect] {exc}")
        self._update_info_labels()
        self._update_button_state()

    def _append_log(self, msg):
        self.log.append(msg)
        # Auto-scroll to the bottom
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── Event handlers ───────────────────────────────────────────────
    def _on_refresh_clicked(self):
        self._append_log("[detect] Searching for bracelet...")
        self._refresh_port(silent=False)
        if self._detected_port:
            self._append_log(f"[detect] Found: {self._detected_port}")
        else:
            self._append_log(
                "[detect] No bracelet found. Is it plugged in and powered on?")

    def _on_install_clicked(self):
        if not self._model_path:
            self._append_log(
                "[error] No model selected. Pick a session from the Load screen first.")
            return
        if not self._detected_port:
            self._append_log(
                "[error] No bracelet detected. Plug it in and click Refresh.")
            return

        # Clear UI and start
        self.log.clear()
        self._append_log("Starting install pipeline...")
        self._append_log(f"  Model: {os.path.basename(self._model_path)}")
        self._append_log(f"  Port:  {self._detected_port}")
        self._append_log("")

        # Indeterminate progress bar while running
        self.progress_bar.setRange(0, 0)
        self.btn_continue.hide()

        self.worker = FlashWorker(self._model_path, port=self._detected_port)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_flash_ok)
        self.worker.failed.connect(self._on_flash_failed)
        self.worker.start()
        self._update_button_state()

    def _on_progress(self, msg):
        self._append_log(msg)

    def _on_flash_ok(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._append_log("")
        self._append_log("\u2705 Install complete. The bracelet has rebooted with the new model.")
        self.btn_continue.show()
        self.worker = None
        self._update_button_state()

    def _on_flash_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log(f"\u274c Install failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self._append_log("")
        self._append_log("You can fix the issue (replug the bracelet, etc.) and click Install again.")
        self.worker = None
        # Refresh port in case it changed during the failed flash
        self._refresh_port(silent=True)
        self._update_button_state()

    # ── BLE weights-only path ──────────────────────────────────────────
    def _choose_bracelet(self):
        from ble_selection import BraceletSelectorDialog
        BraceletSelectorDialog(self).exec()

    def _on_send_weights_clicked(self):
        if not self._model_path:
            self._append_log(
                "[error] No model selected. Pick a session from the Load screen first.")
            return

        self.log.clear()
        self._append_log("Starting BLE weights transfer...")
        self._append_log(f"  Model: {os.path.basename(self._model_path)}")
        self._append_log("")

        self.progress_bar.setRange(0, 0)
        self.btn_continue.hide()

        self.ble_worker = BleWeightsWorker(self._model_path)
        self.ble_worker.progress.connect(self._on_progress)
        self.ble_worker.finished_ok.connect(self._on_ble_ok)
        self.ble_worker.failed.connect(self._on_ble_failed)
        self.ble_worker.start()
        self._update_button_state()

    def _on_ble_ok(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._append_log("")
        self._append_log("✅ Weights sent. The bracelet has rebooted with the new model.")
        self.btn_continue.show()
        self.ble_worker = None
        self._update_button_state()

    def _on_ble_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log("❌ BLE transfer failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self._append_log("")
        self._append_log(
            "Make sure the bracelet is powered on and running BLE-capable "
            "firmware (flash it once with 'Install on bracelet' first).")
        self.ble_worker = None
        self._update_button_state()

    def _on_continue_clicked(self):
        # Bubble up to the parent AppWindow to navigate to Phase 4
        parent = self.parent()
        while parent is not None and not hasattr(parent, "stack"):
            parent = parent.parent()
        if parent is not None and hasattr(parent, "stack"):
            try:
                parent.stack.setCurrentIndex(4)  # Phase 4 = Verify (was 3 before Phase 0)
                if hasattr(parent, "update_navigation"):
                    parent.update_navigation()
            except Exception:
                pass
