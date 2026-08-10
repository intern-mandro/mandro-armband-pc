"""
phase4_verify.py
================
Phase 4 - Live verification (BLE).

The bracelet (flashed with the user's model) sends predictions over BLE
on a dedicated characteristic. Each notification is a UTF-8 string:
    "classname|l0|l1|l2|l3|l4|l5"

A background QThread scans for the bracelet, connects, subscribes to the
prediction characteristic and forwards each detected gesture name to the
UI label.
"""

import asyncio

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    BleakScanner = None
    BleakClient = None


DEVICE_NAME = "ESP32S3_FAST_BLE"

import ble_selection
CHAR_UUID_PRED = "abcd1234-5678-1234-5678-abcdef123457"


class PredictionWorker(QThread):
    """Connects to the bracelet over BLE and listens for prediction notifications."""

    gesture_detected = pyqtSignal(str)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        if BleakScanner is None:
            self.failed.emit("bleak is not installed")
            return
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.failed.emit(str(exc))

    async def _run_async(self):
        selected = ble_selection.get_selected()
        if selected:
            target_address = selected
        else:
            self.status.emit("Scanning for bracelet...")
            try:
                devices = await BleakScanner.discover(timeout=5.0)
            except Exception as exc:
                self.failed.emit(f"BLE scan failed: {exc}")
                return
            target = next((d for d in devices if d.name == DEVICE_NAME), None)
            if target is None:
                self.failed.emit(f"Bracelet '{DEVICE_NAME}' not found")
                return
            target_address = target.address

        self.status.emit(f"Connecting to {target_address}...")

        def on_notify(_sender, data):
            try:
                text = bytes(data).decode("utf-8", errors="replace")
            except Exception:
                return
            parts = text.split("|")
            if not parts:
                return
            name = parts[0].strip()
            if name:
                self.gesture_detected.emit(text)

        try:
            async with BleakClient(target_address) as client:
                await client.start_notify(CHAR_UUID_PRED, on_notify)
                self.status.emit("Connected - waiting for predictions...")
                while not self._stop_requested:
                    await asyncio.sleep(0.1)
                try:
                    await client.stop_notify(CHAR_UUID_PRED)
                except Exception:
                    pass
        except Exception as exc:
            self.failed.emit(f"BLE connection failed: {exc}")


class Phase4Verify(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.session_row_index = None
        self._gesture_set = None
        self._gestures, self._name2idx = self._load_gestures()
        self._build_ui()

    def _load_gestures(self, gset=None):
        """Active gesture names (by class index) + firmware-name -> index map.

        If `gset` is given (resolved from the model/session), relabelling
        follows that set. Otherwise it falls back to configs.ACTIONS."""
        try:
            from lib.configs import ACTIONS, GESTURE_SETS
        except Exception:
            try:
                from configs import ACTIONS, GESTURE_SETS
            except Exception:
                return None, {}
        if gset and gset in GESTURE_SETS:
            names = list(GESTURE_SETS[gset])
        else:
            names = [None] * len(ACTIONS)
            for g, i in ACTIONS.items():
                if isinstance(i, int) and 0 <= i < len(names):
                    names[i] = g
        name2idx = {}
        for setname in ("6cl", "4cl", "rps"):
            for i, g in enumerate(GESTURE_SETS.get(setname, [])):
                name2idx.setdefault(str(g).lower(), i)
        return names, name2idx

    def set_gesture_set(self, gset):
        """Force the gesture set used to relabel predictions (from the model)."""
        if gset:
            self._gesture_set = gset
            self._gestures, self._name2idx = self._load_gestures(gset)
            if hasattr(self, "set_label"):
                self.set_label.setText(f"Model gesture set: {gset.upper()}")
            print(f"Phase 4 gesture set: {gset}")

    def set_session_row_index(self, row_index):
        """Resolve the gesture set from a session row (set by AppWindow/main)."""
        self.session_row_index = row_index
        if row_index is None:
            return
        try:
            import sessions_registry
            gset = sessions_registry.get_gesture_set(row_index)
        except Exception:
            gset = None
        if gset:
            self.set_gesture_set(gset)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(20)

        title = QLabel("Test your model")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        info = QLabel(
            "Put the bracelet on and turn it on, then click Start to see\n"
            "your gestures recognized live - wirelessly.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; padding:20px; color:#c5cce0;"
            " font-size:13px; line-height:1.6; }")
        outer.addWidget(info)

        self.choose_button = QPushButton("Choose bracelet")
        self.choose_button.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-weight:700;"
            " border:1px solid #2a3550; border-radius:8px; padding:8px; }"
            "QPushButton:hover { border:1px solid #3d4d75; }")
        self.choose_button.clicked.connect(self._choose_bracelet)
        outer.addWidget(self.choose_button)

        self.toggle_button = QPushButton("Start verification")
        self.toggle_button.setMinimumHeight(50)
        self.toggle_button.setStyleSheet(self._btn_style(False))
        self.toggle_button.clicked.connect(self._on_toggle)
        outer.addWidget(self.toggle_button)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color:#9fb3ff; font-size:13px; border:none;")
        outer.addWidget(self.status_label)

        self.set_label = QLabel("")
        self.set_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_label.setStyleSheet(
            "color:#2ed573; font-size:13px; font-weight:700; border:none;")
        outer.addWidget(self.set_label)

        self.gesture_label = QLabel("--")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_label.setStyleSheet(
            "QLabel { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; padding:40px; color:#2ed573;"
            " font-size:48px; font-weight:800; }")
        outer.addWidget(self.gesture_label, stretch=1)

    def _btn_style(self, running):
        if running:
            return ("QPushButton { background:#ff6b6b; color:#1a0a0a;"
                    " font-size:15px; font-weight:800; border:none;"
                    " border-radius:10px; padding:12px; }")
        return ("QPushButton { background:#2ed573; color:#0a1020;"
                " font-size:15px; font-weight:800; border:none;"
                " border-radius:10px; padding:12px; }")

    def _choose_bracelet(self):
        from ble_selection import BraceletSelectorDialog
        BraceletSelectorDialog(self).exec()

    def _on_toggle(self):
        if self.worker is None:
            self.status_label.setText("Starting BLE scan...")
            self.worker = PredictionWorker()
            self.worker.gesture_detected.connect(self._on_gesture)
            self.worker.status.connect(self.status_label.setText)
            self.worker.failed.connect(self._on_failed)
            self.worker.start()
            self.toggle_button.setText("Stop verification")
            self.toggle_button.setStyleSheet(self._btn_style(True))
        else:
            self.status_label.setText("Stopping...")
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
            self.toggle_button.setText("Start verification")
            self.toggle_button.setStyleSheet(self._btn_style(False))
            self.status_label.setText("Stopped")
            self.gesture_label.setText("--")

    def _on_gesture(self, text):
        parts = text.split("|")
        raw = parts[0].strip() if parts else ""
        name = raw
        if self._gestures:
            idx = None
            try:
                logits = [float(x) for x in parts[1:] if x.strip() != ""]
                logits = logits[:len(self._gestures)]
                if logits:
                    idx = max(range(len(logits)), key=lambda k: logits[k])
            except (ValueError, IndexError):
                idx = None
            if idx is None:
                idx = self._name2idx.get(raw.lower())
            if idx is not None and 0 <= idx < len(self._gestures) and self._gestures[idx]:
                name = self._gestures[idx]
        self.gesture_label.setText(name.capitalize())

    def _on_failed(self, error):
        self.status_label.setText(f"Error: {error}")
        if self.worker is not None:
            self.worker = None
            self.toggle_button.setText("Start verification")
            self.toggle_button.setStyleSheet(self._btn_style(False))

    def stop(self):
        """Stop the BLE worker and release the connection (idempotent)."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
            self.toggle_button.setText("Start verification")
            self.toggle_button.setStyleSheet(self._btn_style(False))
            self.status_label.setText("Stopped")
            self.gesture_label.setText("--")

    def hideEvent(self, event):
        # Leaving Phase 4 must free the single BLE link so Phase 5 can connect.
        self.stop()
        super().hideEvent(event)