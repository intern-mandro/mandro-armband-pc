"""BLE worker for EMG dashboard.

Drop-in replacement for the previous `serial_worker.py`. Uses a proper async
queue to ensure that every BLE notification is processed exactly once and in
order, with no race condition between the notify callback and the consumer
loop.

Each BLE notification carries 20 samples * 8 channels = 160 bytes. With the
ESP32S3_FAST_BLE firmware operating at ~64 packets/s, the effective sample
rate is ~1280 Hz.
"""

import asyncio
import time
import numpy as np
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal

import config
import ble_selection
from bleak import BleakScanner, BleakClient


# ── BLE protocol constants ─────────────────────────────────────────────
DEVICE_NAME       = "ESP32S3_FAST_BLE"
CHAR_UUID         = "abcd1234-5678-1234-5678-abcdef123456"
SAMPLE_SIZE       = 8     # bytes per sample (1 byte per channel, 8 channels)
SAMPLES_PER_PKT   = 20
EXPECTED_PKT_SIZE = SAMPLE_SIZE * SAMPLES_PER_PKT   # 160 bytes


# ── Helpers ────────────────────────────────────────────────────────────
def _compute_amp(buf: deque):
    """Peak-to-peak amplitude per channel over the buffered samples."""
    if not buf:
        return None
    arr = np.asarray(buf, dtype=float)
    return arr.max(axis=0) - arr.min(axis=0)


# ── BLE worker ─────────────────────────────────────────────────────────
class SerialWorker(QThread):
    """BLE worker. Keeps the original Qt signal interface so the dashboard
    code does not need to change.
    """

    sig_sample            = pyqtSignal(list, object)  # (raw_vals, amp_vals)
    sig_status            = pyqtSignal(str)
    sig_error             = pyqtSignal(str)
    sig_channel_detected  = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._running   = False
        self._loop      = None
        self._n_mult    = config.N_MULT_DEFAULT

        self._n_samples_amp = int(config.BASE_SAMPLES * self._n_mult)
        self._sample_buf    = deque(maxlen=self._n_samples_amp)
        self._calc_counter  = 0
        self._last_amp      = np.zeros(config.N_CH)

    # ── Public API (kept compatible with the previous serial-based worker) ─
    def configure(self, port=None, baud=None, n_mult=None):
        """Port/baud are ignored (BLE). Only n_mult affects amplitude window."""
        if n_mult is not None:
            self.update_params(n_mult)

    def update_params(self, n_mult):
        self._n_mult = n_mult
        self._n_samples_amp = int(config.BASE_SAMPLES * n_mult)
        self._sample_buf    = deque(maxlen=self._n_samples_amp)
        self._calc_counter  = 0

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            # Schedule stop on the loop thread
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── QThread entry point ────────────────────────────────────────────
    def run(self):
        self._running = True
        try:
            asyncio.run(self._ble_main())
        except Exception as exc:
            if self._running:
                self.sig_error.emit(f"BLE error: {exc}")
        finally:
            self._running = False
            self.sig_status.emit("DISCONNECTED")

    # ── Async core: scan, connect, drain queue ─────────────────────────
    async def _ble_main(self):
        self._loop = asyncio.get_running_loop()

        selected = ble_selection.get_selected()
        if selected:
            target_address = selected
        else:
            self.sig_status.emit("Scanning for ESP32S3_FAST_BLE...")
            try:
                devices = await BleakScanner.discover(timeout=5.0)
            except Exception as exc:
                self.sig_error.emit(f"Scan failed: {exc}")
                return
            target = next((d for d in devices if d.name == DEVICE_NAME), None)
            if target is None:
                self.sig_error.emit(f"Device '{DEVICE_NAME}' not found.")
                return
            target_address = target.address

        self.sig_status.emit(f"Connecting to {target_address}...")

        # Queue used to decouple BLE notify callback from the Qt-emit consumer.
        # Bounded to avoid runaway memory in case the consumer falls behind.
        pkt_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        dropped_pkts = 0

        def on_notify(_sender, data):
            nonlocal dropped_pkts
            if len(data) != EXPECTED_PKT_SIZE:
                return  # malformed packet, ignore
            # Snapshot the buffer (BLE library reuses internal buffers).
            payload = bytes(data)
            try:
                pkt_queue.put_nowait(payload)
            except asyncio.QueueFull:
                dropped_pkts += 1

        try:
            async with BleakClient(target_address) as client:
                await client.start_notify(CHAR_UUID, on_notify)
                self.sig_status.emit(f"CONNECTED: {DEVICE_NAME}")
                self.sig_channel_detected.emit(8)

                last_log = time.perf_counter()
                pkt_count = 0
                sample_count = 0

                while self._running:
                    try:
                        payload = await asyncio.wait_for(pkt_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue

                    pkt_count += 1
                    # Decompose: 20 samples * 8 bytes
                    for i in range(SAMPLES_PER_PKT):
                        s = payload[i * SAMPLE_SIZE : (i + 1) * SAMPLE_SIZE]
                        raw_vals = [float(b) for b in s]

                        # Amplitude window
                        self._sample_buf.append(raw_vals)
                        self._calc_counter += 1
                        if self._calc_counter >= self._n_samples_amp:
                            amp = _compute_amp(self._sample_buf)
                            if amp is not None:
                                self._last_amp = amp
                            self._calc_counter = 0

                        # Emit to the UI (Qt queues the signal across threads).
                        self.sig_sample.emit(raw_vals, self._last_amp)
                        sample_count += 1

                    # Per-second diagnostic log
                    now = time.perf_counter()
                    if now - last_log >= 1.0:
                        rate = sample_count / (now - last_log)
                        pkt_rate = pkt_count / (now - last_log)
                        if dropped_pkts:
                            print(f"[BLE] {rate:.0f} Hz | {pkt_rate:.1f} pkt/s "
                                  f"| dropped={dropped_pkts}")
                        else:
                            print(f"[BLE] {rate:.0f} Hz | {pkt_rate:.1f} pkt/s")
                        sample_count = 0
                        pkt_count = 0
                        dropped_pkts = 0
                        last_log = now

                try:
                    await client.stop_notify(CHAR_UUID)
                except Exception:
                    pass
        except Exception as exc:
            if self._running:
                self.sig_error.emit(f"BLE loop error: {exc}")

    # No-op for compatibility with the old serial cleanup() call.
    def cleanup(self):
        pass
