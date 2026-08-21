"""
Pair the robotic hand's BLE MAC into the bracelet's NVS storage.

Standalone menu screen (routed from start_menu.StartMenu), separate from
the Phase 3 install/flash flow since pairing is a one-off BLE-device setup
step, not part of the model-install pipeline.
"""

import asyncio
import os
import sys
import time

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Make the project root importable so we can pull in lib.ble_hand_pairing
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib import ble_hand_pairing as hp  # noqa: E402
import ble_diagnostics as bd  # noqa: E402
import ble_selection  # noqa: E402

# Local aliases so the rest of this file (predates the extraction into
# ble_diagnostics.py) doesn't need renaming everywhere.
_scan_for = bd.scan_for
_wait_visible = bd.wait_visible
_connected = bd.connected


# ─────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────

class PairingWorker(QThread):
    """Connects to the bracelet AND the robotic hand at the same time (PC
    as the hub for both links), writes the hand's MAC into the bracelet's
    NVS pairing storage (see lib.ble_hand_pairing) over the bracelet link,
    then reads it back over that same link to confirm the bracelet
    actually stored it.

    Order is load-bearing, not arbitrary:
      1. Connect to the bracelet FIRST. That alone flips it into SLAVE
         mode (exo_armband_hybrid.ino modeTick(): deviceConnected==true),
         which makes it call forceDisconnectChipsen() and let go of its
         own MASTER-mode link to the hand - only after that does the hand
         start advertising again (a connected BLE peripheral doesn't
         advertise).
      2. Only then scan for and connect to the hand directly. Doing this
         before step 1 would race the bracelet for the hand's single
         connection slot instead of the bracelet handing it over cleanly.
      3. With both links open (bracelet + hand, PC as the common center),
         write the pairing packet and verify over the bracelet link.
      4. Disconnect hand, then bracelet. Once the bracelet sees
         deviceConnected==false again it goes back to MASTER mode and
         reconnects to the hand on its own.

    The hand is found by scanning for a BLE advertising-name prefix. The
    user can type this in on the screen (in case the hand's BLE module
    changes) - it defaults to lib.ble_hand_pairing.HAND_NAME_PREFIX
    ("chipsen", the module in current use) when left blank.
    """
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed      = pyqtSignal(str)

    def __init__(self, hand_name=None, parent=None):
        super().__init__(parent)
        self.hand_name = hand_name.strip() if hand_name and hand_name.strip() \
            else hp.HAND_NAME_PREFIX
        self._t0 = None

    def _log(self, msg):
        """Prefix every progress line with elapsed time since this pairing
        attempt started, so a slow/stuck step is visible directly in the
        log instead of having to guess from wall-clock gaps."""
        elapsed = time.monotonic() - self._t0
        self.progress.emit(f"[+{elapsed:5.1f}s] {msg}")

    def run(self):
        self._t0 = time.monotonic()
        try:
            asyncio.run(self._run_async())
            self._log(f"Done - total {time.monotonic() - self._t0:.1f}s")
            self.finished_ok.emit()
        except (hp.BlePairingError, bd.BleConnectError) as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            self.failed.emit(str(exc))
        except Exception as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            # str(exc) can be empty for some exceptions (asyncio/TimeoutError
            # with no message is a common one) - always include the type so
            # the log never shows a blank "Unexpected error: " with no clue
            # what actually happened.
            self.failed.emit(f"Unexpected error: {type(exc).__name__}: {exc}")

    async def _run_async(self):
        armband_address = ble_selection.get_selected()
        if not armband_address:
            self._log("No bracelet pre-selected - scanning for it")
            armband_address = await _scan_for(
                ble_selection.DEVICE_NAME, "bracelet", self._log, exact=True)
        else:
            self._log(f"Using pre-selected bracelet {armband_address}")

        t_connect = time.monotonic()
        self._log(f"Connecting to bracelet at {armband_address}...")
        async with _connected(armband_address, "bracelet") as armband_client:
            self._log(
                f"Bracelet connected in {time.monotonic() - t_connect:.2f}s "
                "(SLAVE mode - it released the hand). Scanning for the hand...")

            hand_address = ble_selection.get_selected_hand()
            if hand_address:
                self._log(f"Using pre-selected hand {hand_address}")
                # A pre-selected address skips the scan-by-name loop below,
                # which otherwise doubles as a buffer that gives the hand
                # time to notice the bracelet let go of it and resume
                # advertising. Without that buffer, connecting immediately
                # can race the hand's own reconnect delay - confirm it's
                # actually advertising first instead of hitting connect()
                # blind (that raced state is what a connect attempt that
                # just hangs for the full timeout instead of failing fast
                # looks like).
                await _wait_visible(hand_address, "hand", self._log)
            else:
                # The hand's RSSI has been noticeably weaker than the
                # bracelet's in testing (around -80dBm+, borderline for
                # reliable BLE discovery), so it gets more/longer attempts
                # than the default.
                hand_address = await _scan_for(
                    self.hand_name, "robotic hand", self._log,
                    attempts=6, timeout=6.0)

            t_connect = time.monotonic()
            self._log(f"Connecting to hand at {hand_address}...")
            async with _connected(hand_address, "hand", timeout_s=25.0) as hand_client:
                self._log(
                    f"Hand connected in {time.monotonic() - t_connect:.2f}s - "
                    "bracelet and hand both linked to the PC now. "
                    "Sending pairing packet...")

                sent = hp.mac_str_to_bytes(hand_address)
                self._log(f"MAC to store: {hp.mac_bytes_to_str(sent)} "
                          f"(raw bytes: {sent.hex(' ').upper()})")

                t_send = time.monotonic()
                await hp.send_pair_ble(
                    armband_address, hand_address,
                    client=armband_client,
                    on_progress=self._log)
                self._log(f"Write + ack round-trip took {time.monotonic() - t_send:.2f}s")

                self._log("Verifying the bracelet actually stored it...")
                t_read = time.monotonic()
                stored = await hp.read_pair_ble(armband_address, client=armband_client)
                self._log(
                    f"Read-back took {time.monotonic() - t_read:.2f}s - "
                    f"got {stored.hex(' ').upper() or '(empty)'}")

                if stored != sent:
                    # Ambiguous whether this is a real mismatch or just NVS
                    # flash commit not having landed yet by the time we
                    # read - re-read after a short delay to tell those
                    # apart instead of guessing.
                    self._log(
                        "Mismatch on first read-back - re-reading after "
                        "1s in case the bracelet's NVS write just hadn't "
                        "committed yet...")
                    await asyncio.sleep(1.0)
                    t_reread = time.monotonic()
                    stored2 = await hp.read_pair_ble(armband_address, client=armband_client)
                    self._log(
                        f"Second read-back took {time.monotonic() - t_reread:.2f}s - "
                        f"got {stored2.hex(' ').upper() or '(empty)'}"
                        + (" (same as first - not a timing race)"
                           if stored2 == stored else
                           " (DIFFERENT from first - was a timing race, now matches)"
                           if stored2 == sent else
                           " (different from first, still doesn't match sent)"))
                    stored = stored2

        if stored == sent:
            self._log(f"Confirmed: bracelet stored {hp.mac_bytes_to_str(stored)}")
        elif not stored:
            raise hp.BlePairingError(
                "Bracelet acked the pairing but the stored value reads back empty.")
        elif stored == sent[::-1]:
            raise hp.BlePairingError(
                f"Stored MAC is byte-reversed ({hp.mac_bytes_to_str(stored)}) "
                f"vs sent ({hp.mac_bytes_to_str(sent)}).")
        else:
            raise hp.BlePairingError(
                f"Stored MAC ({hp.mac_bytes_to_str(stored)}) doesn't match "
                f"what was sent ({hp.mac_bytes_to_str(sent)}).")


class CheckPairWorker(QThread):
    """Reads back whatever MAC is currently stored in the bracelet's NVS
    pairing storage, without writing anything. NVS only ever holds one
    MAC at a time (a single "master_mac" key - see PairCharCallbacks in
    the bracelet .ino's onWrite()/onRead()), not a history/list of past
    pairings - so this reports "one currently-paired hand" or "none"."""
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal(str)   # "" means NVS is empty (never paired)
    failed      = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = None

    def _log(self, msg):
        elapsed = time.monotonic() - self._t0
        self.progress.emit(f"[+{elapsed:5.1f}s] {msg}")

    def run(self):
        self._t0 = time.monotonic()
        try:
            result = asyncio.run(self._run_async())
            self._log(f"Done - total {time.monotonic() - self._t0:.1f}s")
            self.finished_ok.emit(result)
        except (hp.BlePairingError, bd.BleConnectError) as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            self.failed.emit(str(exc))
        except Exception as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            # str(exc) can be empty for some exceptions (asyncio/TimeoutError
            # with no message is a common one) - always include the type so
            # the log never shows a blank "Unexpected error: " with no clue
            # what actually happened.
            self.failed.emit(f"Unexpected error: {type(exc).__name__}: {exc}")

    async def _run_async(self) -> str:
        armband_address = ble_selection.get_selected()
        if not armband_address:
            armband_address = await _scan_for(
                ble_selection.DEVICE_NAME, "bracelet", self._log, exact=True)

        t_connect = time.monotonic()
        self._log(f"Connecting to bracelet at {armband_address}...")
        async with _connected(armband_address, "bracelet") as armband_client:
            self._log(f"Connected in {time.monotonic() - t_connect:.2f}s")

            self._log("Reading stored MAC from NVS...")
            stored = await hp.read_pair_ble(armband_address, client=armband_client)

        return hp.mac_bytes_to_str(stored) if stored else ""


class ClearPairWorker(QThread):
    """Erases the bracelet's stored master MAC (see
    lib.ble_hand_pairing.clear_pair_ble). After this, the bracelet's
    MASTER mode won't scan for or connect to any hand until it's paired
    again from this screen."""
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed      = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = None

    def _log(self, msg):
        elapsed = time.monotonic() - self._t0
        self.progress.emit(f"[+{elapsed:5.1f}s] {msg}")

    def run(self):
        self._t0 = time.monotonic()
        try:
            asyncio.run(self._run_async())
            self._log(f"Done - total {time.monotonic() - self._t0:.1f}s")
            self.finished_ok.emit()
        except (hp.BlePairingError, bd.BleConnectError) as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            self.failed.emit(str(exc))
        except Exception as exc:
            self._log(f"Failed after {time.monotonic() - self._t0:.1f}s total")
            # str(exc) can be empty for some exceptions (asyncio/TimeoutError
            # with no message is a common one) - always include the type so
            # the log never shows a blank "Unexpected error: " with no clue
            # what actually happened.
            self.failed.emit(f"Unexpected error: {type(exc).__name__}: {exc}")

    async def _run_async(self):
        armband_address = ble_selection.get_selected()
        if not armband_address:
            armband_address = await _scan_for(
                ble_selection.DEVICE_NAME, "bracelet", self._log, exact=True)

        t_connect = time.monotonic()
        self._log(f"Connecting to bracelet at {armband_address}...")
        async with _connected(armband_address, "bracelet") as armband_client:
            self._log(f"Connected in {time.monotonic() - t_connect:.2f}s")

            await hp.clear_pair_ble(
                armband_address, client=armband_client,
                on_progress=self._log)


# ─────────────────────────────────────────────────────────────────────
# Screen widget
# ─────────────────────────────────────────────────────────────────────

class PairHandScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.pairing_worker = None
        self.check_worker = None
        self.clear_worker = None
        self._build_ui()
        self._refresh_bracelet_label()
        self._refresh_hand_label()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(14)

        title = QLabel("Pair the robotic hand")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        intro = QLabel(
            "Scans for the robotic hand and the bracelet, then sends the "
            "hand's BLE MAC so the bracelet can auto-reconnect to it once "
            "the PC disconnects. Both devices must be powered on and in range.")
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        outer.addWidget(intro)

        # ── Bracelet selection (reuses the same picker as Phase 3's "Choose
        #    bracelet" - scan-and-pick, not free text, since there's already
        #    a proven dialog for this and it avoids MAC/name typos). ──
        bracelet_row = QHBoxLayout()
        bracelet_row.setSpacing(8)
        self.lbl_bracelet = QLabel("Bracelet: (any 'ESP32S3_FAST_BLE' found)")
        self.lbl_bracelet.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        bracelet_row.addWidget(self.lbl_bracelet, stretch=1)
        self.btn_choose_bracelet = QPushButton("Choose bracelet")
        self.btn_choose_bracelet.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-weight:700;"
            " border:1px solid #2a3550; border-radius:8px; padding:6px 14px; }"
            "QPushButton:hover { border:1px solid #3d4d75; }")
        self.btn_choose_bracelet.clicked.connect(self._on_choose_bracelet_clicked)
        bracelet_row.addWidget(self.btn_choose_bracelet)
        outer.addLayout(bracelet_row)

        # ── Hand selection: scan-and-pick (default, like the bracelet above)
        #    plus a free-text name field for typing it in directly - same
        #    "pick from a dialog, or type it" pattern as the bracelet, just
        #    with both options visible since the hand's BLE module name
        #    isn't as fixed as the bracelet's. Choose hand wins if set. ──
        hand_pick_row = QHBoxLayout()
        hand_pick_row.setSpacing(8)
        self.lbl_hand = QLabel("Hand: (auto-scan by name below)")
        self.lbl_hand.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        hand_pick_row.addWidget(self.lbl_hand, stretch=1)
        self.btn_choose_hand = QPushButton("Choose hand")
        self.btn_choose_hand.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-weight:700;"
            " border:1px solid #2a3550; border-radius:8px; padding:6px 14px; }"
            "QPushButton:hover { border:1px solid #3d4d75; }")
        self.btn_choose_hand.clicked.connect(self._on_choose_hand_clicked)
        hand_pick_row.addWidget(self.btn_choose_hand)
        outer.addLayout(hand_pick_row)

        # Hand BLE advertising-name prefix (case-insensitive), used for
        # auto-scan when no device was picked above. Defaults to
        # lib.ble_hand_pairing.HAND_NAME_PREFIX when left blank.
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_label = QLabel("Hand device name:")
        name_label.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        name_row.addWidget(name_label)
        self.hand_name_input = QLineEdit()
        self.hand_name_input.setPlaceholderText(hp.HAND_NAME_PREFIX)
        self.hand_name_input.setStyleSheet(
            "QLineEdit { background:#1a2236; color:#e5ebff;"
            " border:1px solid #3a4570; border-radius:6px; padding:6px 10px; }")
        name_row.addWidget(self.hand_name_input, stretch=1)
        outer.addLayout(name_row)

        self.btn_pair_hand = QPushButton("\U0001F91D  Pair robotic hand")
        self.btn_pair_hand.setMinimumHeight(48)
        self.btn_pair_hand.setStyleSheet(
            "QPushButton { background:#ff9f43; color:#0a1020;"
            " font-size:15px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 28px; }"
            "QPushButton:hover { background:#ffb066; }"
            "QPushButton:disabled { background:#3a4570; color:#7c87a8; }")
        self.btn_pair_hand.clicked.connect(self._on_pair_hand_clicked)
        outer.addWidget(self.btn_pair_hand)

        # ── Currently-stored MAC (NVS holds exactly one, not a list/history) ──
        self.lbl_stored_mac = QLabel("Stored hand MAC: (unknown - click Check below)")
        self.lbl_stored_mac.setWordWrap(True)
        self.lbl_stored_mac.setStyleSheet(
            "color:#e5ebff; font-size:13px; border:none; padding:4px 0;")
        outer.addWidget(self.lbl_stored_mac)

        self.btn_check_mac = QPushButton("\U0001F50D  Check stored MAC")
        self.btn_check_mac.setMinimumHeight(36)
        self.btn_check_mac.setStyleSheet(
            "QPushButton { background:#2a3550; color:#e5ebff; font-weight:700;"
            " border:1px solid #3a4570; border-radius:8px; padding:8px 18px; }"
            "QPushButton:hover { background:#3a4570; }"
            "QPushButton:disabled { background:#1a2236; color:#7c87a8; }")
        self.btn_check_mac.clicked.connect(self._on_check_mac_clicked)
        outer.addWidget(self.btn_check_mac)

        self.btn_clear_mac = QPushButton("\U0001F5D1  Clear stored MAC")
        self.btn_clear_mac.setMinimumHeight(36)
        self.btn_clear_mac.setStyleSheet(
            "QPushButton { background:#2a3550; color:#ff6b6b; font-weight:700;"
            " border:1px solid #3a4570; border-radius:8px; padding:8px 18px; }"
            "QPushButton:hover { background:#3a4570; }"
            "QPushButton:disabled { background:#1a2236; color:#7c87a8; }")
        self.btn_clear_mac.clicked.connect(self._on_clear_mac_clicked)
        outer.addWidget(self.btn_clear_mac)

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
        self.log.setMinimumHeight(240)
        f = QFont("Menlo")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(f)
        outer.addWidget(self.log, stretch=1)

    def _append_log(self, msg):
        self.log.append(msg)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_button_state(self):
        pairing_running = self.pairing_worker is not None and self.pairing_worker.isRunning()
        check_running = self.check_worker is not None and self.check_worker.isRunning()
        clear_running = self.clear_worker is not None and self.clear_worker.isRunning()
        any_running = pairing_running or check_running or clear_running

        self.btn_pair_hand.setEnabled(not any_running)
        self.btn_pair_hand.setText(
            "⏳  Pairing..." if pairing_running else "\U0001F91D  Pair robotic hand")

        self.btn_check_mac.setEnabled(not any_running)
        self.btn_check_mac.setText(
            "⏳  Checking..." if check_running else "\U0001F50D  Check stored MAC")

        self.btn_clear_mac.setEnabled(not any_running)
        self.btn_clear_mac.setText(
            "⏳  Clearing..." if clear_running else "\U0001F5D1  Clear stored MAC")

        self.btn_choose_bracelet.setEnabled(not any_running)
        self.btn_choose_hand.setEnabled(not any_running)

    def _refresh_bracelet_label(self):
        address = ble_selection.get_selected()
        if address:
            self.lbl_bracelet.setText(f"Bracelet: {address}")
        else:
            self.lbl_bracelet.setText("Bracelet: (any 'ESP32S3_FAST_BLE' found)")

    def _on_choose_bracelet_clicked(self):
        from ble_selection import BraceletSelectorDialog
        BraceletSelectorDialog(self).exec()
        self._refresh_bracelet_label()

    def _refresh_hand_label(self):
        address = ble_selection.get_selected_hand()
        if address:
            self.lbl_hand.setText(f"Hand: {address}")
        else:
            self.lbl_hand.setText("Hand: (auto-scan by name below)")

    def _on_choose_hand_clicked(self):
        from ble_selection import HandSelectorDialog
        expected_name = self.hand_name_input.text().strip() or hp.HAND_NAME_PREFIX
        HandSelectorDialog(expected_name, self).exec()
        self._refresh_hand_label()

    def _on_pair_hand_clicked(self):
        hand_name = self.hand_name_input.text().strip() or hp.HAND_NAME_PREFIX

        self.log.clear()
        self._append_log(f"Starting robotic hand pairing (looking for '{hand_name}')...")
        self._append_log("")

        self.progress_bar.setRange(0, 0)

        self.pairing_worker = PairingWorker(hand_name=hand_name)
        self.pairing_worker.progress.connect(self._on_progress)
        self.pairing_worker.finished_ok.connect(self._on_pair_ok)
        self.pairing_worker.failed.connect(self._on_pair_failed)
        self.pairing_worker.start()
        self._update_button_state()

    def _on_progress(self, msg):
        self._append_log(msg)

    def _on_pair_ok(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._append_log("")
        self._append_log(
            "✅ Pairing complete — bracelet confirmed it stored the hand's MAC.")
        self.pairing_worker = None
        self._update_button_state()

    def _on_pair_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log("❌ Pairing failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self._append_log("")
        self._append_log(
            "Make sure both the bracelet and the robotic hand are powered "
            "on and in BLE range, then try again.")
        self.pairing_worker = None
        self._update_button_state()

    # ── Check stored MAC ─────────────────────────────────────────────
    def _on_check_mac_clicked(self):
        self._append_log("Checking the bracelet's stored hand MAC...")
        self._append_log("")

        self.progress_bar.setRange(0, 0)

        self.check_worker = CheckPairWorker()
        self.check_worker.progress.connect(self._on_progress)
        self.check_worker.finished_ok.connect(self._on_check_ok)
        self.check_worker.failed.connect(self._on_check_failed)
        self.check_worker.start()
        self._update_button_state()

    def _on_check_ok(self, mac_str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if mac_str:
            self.lbl_stored_mac.setText(f"Stored hand MAC: {mac_str}")
            self._append_log(f"Stored hand MAC: {mac_str}")
        else:
            self.lbl_stored_mac.setText("Stored hand MAC: (none - not paired yet)")
            self._append_log("No hand paired yet - NVS is empty.")
        self.check_worker = None
        self._update_button_state()

    def _on_check_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log("❌ Check failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self.check_worker = None
        self._update_button_state()

    # ── Clear stored MAC ─────────────────────────────────────────────
    def _on_clear_mac_clicked(self):
        reply = QMessageBox.question(
            self,
            "Clear stored MAC?",
            "This erases the hand's MAC from the bracelet's NVS. Until you "
            "pair again, the bracelet won't scan for or connect to any hand "
            "on its own. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._append_log("Clearing the bracelet's stored MAC...")
        self._append_log("")

        self.progress_bar.setRange(0, 0)

        self.clear_worker = ClearPairWorker()
        self.clear_worker.progress.connect(self._on_progress)
        self.clear_worker.finished_ok.connect(self._on_clear_ok)
        self.clear_worker.failed.connect(self._on_clear_failed)
        self.clear_worker.start()
        self._update_button_state()

    def _on_clear_ok(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.lbl_stored_mac.setText("Stored hand MAC: (none - not paired yet)")
        self._append_log("")
        self._append_log("✅ Cleared - the bracelet no longer has a paired hand.")
        self.clear_worker = None
        self._update_button_state()

    def _on_clear_failed(self, err):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._append_log("")
        self._append_log("❌ Clear failed:")
        for line in err.splitlines():
            self._append_log(f"   {line}")
        self.clear_worker = None
        self._update_button_state()
