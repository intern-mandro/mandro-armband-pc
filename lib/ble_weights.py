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
                            ) -> None:
    """Connect to the bracelet at `address` and push `frame` (from
    build_frame()) to the weights characteristic in write-with-response
    chunks. Waits for the bracelet's 'OK:...' notify before returning.

    The bracelet reboots immediately after acking, which can race the BLE
    stack's own disconnect notification - a disconnect error that arrives
    right after the ack is treated as success, not failure.
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
        text = bytes(data).decode("utf-8", errors="replace").strip()
        if text:
            ack_holder["text"] = text
            ack_event.set()

    _emit(f"Connecting to {address}...")
    try:
        async with BleakClient(address) as client:
            # Subscribe before writing anything, so the ack can never arrive
            # before we're listening for it.
            await client.start_notify(CHAR_UUID_WEIGHTS, on_notify)

            total = len(frame)
            sent = 0
            _emit(f"Sending {total} bytes in {chunk_size}-byte chunks...")
            for offset in range(0, total, chunk_size):
                chunk = frame[offset:offset + chunk_size]
                await client.write_gatt_char(CHAR_UUID_WEIGHTS, chunk, response=True)
                sent += len(chunk)
                if sent % (chunk_size * 20) == 0 or sent == total:
                    _emit(f"  {sent}/{total} bytes ({sent * 100 // total}%)")

            _emit("Waiting for the bracelet to confirm...")
            try:
                await asyncio.wait_for(ack_event.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                raise BleWeightsError(
                    "Timed out waiting for the bracelet's confirmation. "
                    "The transfer may still have gone through - check by "
                    "reconnecting and looking at Serial output.")

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
            _emit(f"Bracelet confirmed then disconnected (expected - it's rebooting): {exc}")
            return
        raise BleWeightsError(f"BLE transfer failed: {exc}")

    _emit("Done. The bracelet is rebooting with the new model.")
