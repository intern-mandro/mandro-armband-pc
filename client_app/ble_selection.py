"""
ble_selection.py
================
Lets the user pick a specific BLE device when several are in range. Shows
every nearby BLE device (unfiltered) - useful for debugging when the target
isn't showing up under its expected name (e.g. right after a fresh flash or
with an unusual advertising state). Entries whose name matches the expected
one are marked so the real target is still easy to spot among the noise.
Distinguished by name (if any) + address + signal strength (RSSI).

Originally bracelet-only (BraceletSelectorDialog); generalized into
DeviceSelectorDialog so the robotic hand can reuse the same scan-and-pick UI
(see pair_hand_screen.py) instead of typing a MAC/name in by hand. Bracelet
and hand selections are stored separately - picking one never clears the
other.

The chosen address is remembered for the session (module-level). Callers
fall back to "first device matching the expected name" if nothing was
explicitly chosen.

Note: on macOS the address is a per-host UUID that can change between
sessions, so the choice is per-session by design.
"""
import asyncio

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

try:
    from bleak import BleakScanner
except ImportError:
    BleakScanner = None

DEVICE_NAME = "ESP32S3_FAST_BLE"

_selected_address = None       # bracelet
_selected_hand_address = None  # robotic hand


def get_selected():
    return _selected_address


def set_selected(address):
    global _selected_address
    _selected_address = address


def clear_selected():
    global _selected_address
    _selected_address = None


def get_selected_hand():
    return _selected_hand_address


def set_selected_hand(address):
    global _selected_hand_address
    _selected_hand_address = address


def clear_selected_hand():
    global _selected_hand_address
    _selected_hand_address = None


class _ScanWorker(QThread):
    done = pyqtSignal(list)      # list of (name, address, rssi)
    failed = pyqtSignal(str)

    def run(self):
        if BleakScanner is None:
            self.failed.emit("bleak is not installed")
            return
        try:
            found = asyncio.run(self._scan())
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(found)

    async def _scan(self):
        discovered = await BleakScanner.discover(timeout=5.0, return_adv=True)
        items = []
        for address, (dev, adv) in discovered.items():
            name = dev.name or (adv.local_name if adv else None) or ""
            # No name filter - show every BLE device in range so the user can
            # debug a target that isn't advertising under its expected name.
            items.append((name, address, adv.rssi if adv else -999))
        items.sort(key=lambda t: t[2], reverse=True)   # strongest signal first
        return items


class DeviceSelectorDialog(QDialog):
    """Generic "scan and pick a BLE device" dialog.

    `expected_name` + `match_mode` only control the ★ hint next to matching
    entries - every nearby device is still listed, so a target that isn't
    advertising under the expected name can still be found and picked.
    `match_mode="exact"` requires an exact name match (bracelet); "prefix"
    does a case-insensitive startswith (hand - its module name can vary).

    `get_selected_fn`/`set_selected_fn` point at whichever module-level slot
    (bracelet vs hand) this instance should read/write.
    """

    def __init__(self, expected_name, title, get_selected_fn, set_selected_fn,
                 match_mode="exact", parent=None):
        super().__init__(parent)
        self._expected_name = expected_name
        self._match_mode = match_mode
        self._get_selected = get_selected_fn
        self._set_selected = set_selected_fn

        self.setWindowTitle(title)
        self.setStyleSheet("QDialog { background:#0e1422; }")
        self.setMinimumSize(560, 460)
        self._worker = None

        outer = QVBoxLayout(self)
        self._title = QLabel("Scanning for devices...")
        self._title.setStyleSheet(
            "color:#ffffff; font-size:14px; font-weight:800; border:none;")
        outer.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        self._holder_w = QWidget()
        self._holder = QVBoxLayout(self._holder_w)
        self._holder.setSpacing(6)
        scroll.setWidget(self._holder_w)
        outer.addWidget(scroll, 1)

        row = QHBoxLayout()
        self._rescan_btn = QPushButton("Rescan")
        self._rescan_btn.setStyleSheet(self._btn_grey())
        self._rescan_btn.clicked.connect(self._start_scan)
        row.addWidget(self._rescan_btn)

        self._any_btn = QPushButton("Use any (first found)")
        self._any_btn.setStyleSheet(self._btn_grey())
        self._any_btn.clicked.connect(self._use_any)
        row.addWidget(self._any_btn)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(self._btn_grey())
        close_btn.clicked.connect(self.reject)
        row.addWidget(close_btn)
        outer.addLayout(row)

        self._start_scan()

    def _matches(self, name):
        if self._match_mode == "exact":
            return name == self._expected_name
        return name.lower().startswith(self._expected_name.lower())

    def _btn_grey(self):
        return ("QPushButton { background:#2a3550; color:#e6e9f2; border:none;"
                " border-radius:6px; padding:8px 14px; }")

    def _clear_list(self):
        while self._holder.count():
            w = self._holder.takeAt(0).widget()
            if w is not None:
                w.setParent(None)

    def _start_scan(self):
        self._clear_list()
        self._title.setText("Scanning for devices...")
        self._rescan_btn.setEnabled(False)
        self._worker = _ScanWorker()
        self._worker.done.connect(self._on_results)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, msg):
        self._rescan_btn.setEnabled(True)
        self._title.setText(f"Scan failed: {msg}")

    def _on_results(self, items):
        self._rescan_btn.setEnabled(True)
        self._clear_list()
        if not items:
            self._title.setText(
                "No BLE devices found nearby. Check Bluetooth is on and rescan.")
            return
        current = self._get_selected()
        self._title.setText(
            f"{len(items)} BLE device(s) found - closest first "
            f"(★ = expected '{self._expected_name}'):")
        for name, address, rssi in items:
            mark = "  (current)" if address == current else ""
            star = "★ " if self._matches(name) else ""
            display_name = name or "(no name)"
            btn = QPushButton(f"{star}{display_name}    RSSI {rssi} dBm    {address}{mark}")
            btn.setStyleSheet(
                "QPushButton { background:#1a2236; color:#e6e9f2; text-align:left;"
                " border:1px solid #2a3550; border-radius:6px; padding:10px 12px; }"
                "QPushButton:hover { border:1px solid #3d4d75; }")
            btn.clicked.connect(lambda _=False, a=address: self._choose(a))
            self._holder.addWidget(btn)

    def _choose(self, address):
        self._set_selected(address)
        self.accept()

    def _use_any(self):
        self._set_selected(None)
        self.accept()


class BraceletSelectorDialog(DeviceSelectorDialog):
    def __init__(self, parent=None):
        super().__init__(
            DEVICE_NAME, "Choose bracelet",
            get_selected, set_selected,
            match_mode="exact", parent=parent)


class HandSelectorDialog(DeviceSelectorDialog):
    def __init__(self, expected_name, parent=None):
        super().__init__(
            expected_name, "Choose robotic hand",
            get_selected_hand, set_selected_hand,
            match_mode="prefix", parent=parent)
