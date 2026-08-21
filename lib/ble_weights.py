"""
lib/ble_weights.py
===================
BLE weights-only model update: serialize a trained Keras model + its
StandardScaler into the wire format the bracelet's BLE weights
characteristic expects, and push it over BLE with bleak.

Companion to lib/firmware_uploader.py (which flashes the full firmware over
USB). This module never touches USB/arduino-cli - it only talks BLE to a
bracelet already running firmware with the weights-receive characteristic
(firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid).

Wire frame (must match WeightsCharCallbacks::onWrite() in the .ino):
    [4B magic 0xDEADBEEF][4B payload length][payload][4B CRC32 of payload]
    payload = W0 b0 W1 b1 ... means stds, all float32 little-endian.

Public API:
    serialize_weights(keras_path, scaler_path=None) -> bytes   (payload only)
    build_frame(payload: bytes) -> bytes
    parse_frame(frame: bytes) -> bytes                          (for tests)
    async send_weights_ble(address, frame, on_progress=None) -> None

Errors are raised as BleWeightsError with human-readable messages.
"""

from __future__ import annotations

import asyncio
import struct
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from lib.pipeline import load_pipeline


# ─────────────────────────────────────────────────────────────────────
# Wire protocol constants (must match exo_armband_hybrid.ino)
# ─────────────────────────────────────────────────────────────────────

WEIGHTS_MAGIC = 0xDEADBEEF
CHUNK_SIZE = 244   # bytes per BLE write; matches MTU=247 set in firmware
CHAR_UUID_WEIGHTS = "abcd1234-5678-1234-5678-abcdef123458"

ACK_OK_PREFIX = "OK:"


class BleWeightsError(Exception):
    """Raised when any step of the BLE weights-transfer pipeline fails."""


# ─────────────────────────────────────────────────────────────────────
# Scaler path deduction (mirrors scripts/export_teensy_headers.py)
# ─────────────────────────────────────────────────────────────────────

def _deduce_scaler_path(model_path: Path, scaler_folder: str = "models/scalers") -> Path:
    """Given models/trained/model_FOO.keras, return models/scalers/scaler_FOO.pkl"""
    stem = model_path.stem
    if stem.startswith("model_"):
        scaler_name = "scaler_" + stem[len("model_"):]
    else:
        scaler_name = "scaler_" + stem
    return Path(scaler_folder) / f"{scaler_name}.pkl"


# ─────────────────────────────────────────────────────────────────────
# Serialization: .keras + scaler -> raw payload bytes
# ─────────────────────────────────────────────────────────────────────

def serialize_weights(keras_path, scaler_path=None) -> bytes:
    """Extract W0 b0 W1 b1 ... means stds from a trained Keras model + its
    StandardScaler, packed as float32 little-endian - the exact byte layout
    NeuralNet::loadFromLittleFS() expects on the bracelet.

    Raises BleWeightsError if the model/scaler can't be loaded, or if the
    scaler's feature count doesn't match the model's input size.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise BleWeightsError("TensorFlow is required to export weights.")

    keras_path = Path(keras_path)
    if not keras_path.exists():
        raise BleWeightsError(f"Model file not found: {keras_path}")

    if scaler_path is None:
        scaler_path = _deduce_scaler_path(keras_path)
    else:
        scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise BleWeightsError(f"Scaler file not found: {scaler_path}")

    model = tf.keras.models.load_model(keras_path)
    scaler = load_pipeline(scaler_path.stem, folder=str(scaler_path.parent))["scaler"]

    chunks = []
    n_features = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            if n_features is None:
                n_features = w.shape[0]
            chunks.append(w.astype("<f4").flatten())
            chunks.append(b.astype("<f4").flatten())

    if n_features is None:
        raise BleWeightsError(f"No Dense layers found in model: {keras_path}")

    means = np.asarray(scaler.mean_, dtype="<f4")
    stds = np.asarray(scaler.scale_, dtype="<f4")
    if len(means) != n_features:
        raise BleWeightsError(
            f"Scaler feature count ({len(means)}) doesn't match model input "
            f"size ({n_features}) - wrong scaler for this model?")

    chunks.append(means)
    chunks.append(stds)

    return np.concatenate(chunks).tobytes()


# ─────────────────────────────────────────────────────────────────────
# Framing: payload -> magic + length + payload + crc32
# ─────────────────────────────────────────────────────────────────────

def build_frame(payload: bytes) -> bytes:
    """Wrap a raw payload in the wire frame the bracelet expects:
    magic(4) + length(4) + payload + crc32(4), all little-endian."""
    header = struct.pack("<II", WEIGHTS_MAGIC, len(payload))
    crc = struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
    return header + payload + crc


def parse_frame(frame: bytes) -> bytes:
    """Inverse of build_frame(), for round-trip sanity checks. Raises
    BleWeightsError if the magic/length/CRC don't check out. Returns the
    payload bytes."""
    if len(frame) < 12:
        raise BleWeightsError("Frame too short")
    magic, length = struct.unpack("<II", frame[:8])
    if magic != WEIGHTS_MAGIC:
        raise BleWeightsError(f"Bad magic: {magic:#x}")
    payload = frame[8:8 + length]
    if len(payload) != length:
        raise BleWeightsError("Frame shorter than declared length")
    (crc_received,) = struct.unpack("<I", frame[8 + length:12 + length])
    crc_computed = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_received != crc_computed:
        raise BleWeightsError("CRC mismatch")
    return payload


# ─────────────────────────────────────────────────────────────────────
# BLE transfer
# ─────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str], None]


async def send_weights_ble(address: str,
                            frame: bytes,
                            chunk_size: int = CHUNK_SIZE,
                            on_progress: Optional[ProgressCallback] = None,
                            timeout_s: float = 20.0,
                            connect_timeout_s: float = 20.0,
                            slow_chunk_s: float = 0.5,
                            ) -> None:
    """Connect to the bracelet at `address` and push `frame` (from
    build_frame()) to the weights characteristic in write-with-response
    chunks. Waits for the bracelet's 'OK:...' notify before returning.

    The bracelet reboots immediately after acking, which can race the BLE
    stack's own disconnect notification - a disconnect error that arrives
    right after the ack is treated as success, not failure.

    The connect step has an explicit timeout (`connect_timeout_s`) instead
    of relying on bleak's own backend-dependent default - a bad connection
    was once seen to hang far longer than expected before failing with an
    exception whose str() was empty, hiding the real cause. Individual
    chunk writes slower than `slow_chunk_s` are logged (not just the
    throttled every-20-chunks summary), since a BLE link degrading
    mid-transfer shows up as chunks getting progressively slower before
    an outright failure.
    """
    try:
        from bleak import BleakClient
    except ImportError:
        raise BleWeightsError("bleak is not installed.")

    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    ack_event = asyncio.Event()
    ack_holder = {"text": None}

    def on_notify(_sender, data):
        # Log every notify on this characteristic, not just the one that
        # ends up being treated as the ack - if something unexpected
        # arrives (garbled bytes, more than one response) this is the
        # only place it would show up.
        raw = bytes(data)
        text = raw.decode("utf-8", errors="replace").strip()
        _emit(f"  notify: {raw.hex(' ').upper()}  text={text!r}")
        if text.startswith(ACK_OK_PREFIX) or text.startswith("ERR:"):
            ack_holder["text"] = text
            ack_event.set()
        elif text:
            _emit("  (doesn't look like OK:/ERR:, not treating as the ack)")

    t_connect = time.monotonic()
    _emit(f"Connecting to {address}...")
    client = BleakClient(address)
    try:
        async with asyncio.timeout(connect_timeout_s):
            await client.connect()
    except TimeoutError:
        raise BleWeightsError(
            f"Timed out ({connect_timeout_s:.0f}s) connecting to the bracelet at {address}.")
    except Exception as exc:
        raise BleWeightsError(
            f"Failed to connect to the bracelet at {address}: {type(exc).__name__}: {exc}")
    _emit(f"Connected in {time.monotonic() - t_connect:.2f}s")

    try:
        # Subscribe before writing anything, so the ack can never arrive
        # before we're listening for it.
        await client.start_notify(CHAR_UUID_WEIGHTS, on_notify)

        total = len(frame)
        sent = 0
        t_send_start = time.monotonic()
        _emit(f"Sending {total} bytes in {chunk_size}-byte chunks...")
        for offset in range(0, total, chunk_size):
            chunk = frame[offset:offset + chunk_size]
            t_chunk = time.monotonic()
            await client.write_gatt_char(CHAR_UUID_WEIGHTS, chunk, response=True)
            chunk_dt = time.monotonic() - t_chunk
            sent += len(chunk)
            if chunk_dt > slow_chunk_s:
                _emit(f"  slow chunk: {chunk_dt:.2f}s for {len(chunk)} bytes "
                      f"at offset {offset} ({sent}/{total})")
            if sent % (chunk_size * 20) == 0 or sent == total:
                elapsed = time.monotonic() - t_send_start
                rate_kbs = (sent / 1024) / elapsed if elapsed > 0 else 0
                _emit(f"  {sent}/{total} bytes ({sent * 100 // total}%) - "
                      f"{elapsed:.1f}s elapsed, ~{rate_kbs:.1f} KB/s")

        _emit(f"All chunks sent in {time.monotonic() - t_send_start:.2f}s. "
              "Waiting for the bracelet to confirm...")
        t_ack = time.monotonic()
        try:
            await asyncio.wait_for(ack_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise BleWeightsError(
                f"Timed out ({timeout_s:.0f}s) waiting for the bracelet's confirmation. "
                "The transfer may still have gone through - check by "
                "reconnecting and looking at Serial output.")
        _emit(f"  ack notify arrived {time.monotonic() - t_ack:.2f}s after last chunk")

        ack = ack_holder["text"]
        if ack is None or not ack.startswith(ACK_OK_PREFIX):
            raise BleWeightsError(f"Bracelet rejected the weights: {ack}")

        _emit(f"Bracelet confirmed: {ack}")
        try:
            await client.stop_notify(CHAR_UUID_WEIGHTS)
        except Exception:
            pass
    except BleWeightsError:
        raise
    except Exception as exc:
        # The bracelet restarts itself right after acking, which can look
        # like a disconnect/BLE error here even though the transfer
        # succeeded - only treat this as fatal if we never got the ack.
        if ack_holder["text"] is not None and ack_holder["text"].startswith(ACK_OK_PREFIX):
            _emit(f"Bracelet confirmed then disconnected (expected - it's rebooting): "
                  f"{type(exc).__name__}: {exc}")
            return
        raise BleWeightsError(f"BLE transfer failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    _emit("Done. The bracelet is rebooting with the new model.")
