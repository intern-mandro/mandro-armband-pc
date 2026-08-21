"""
lib/ble_hand_pairing.py
========================
Pairs the robotic hand's BLE MAC address into the bracelet's NVS storage, so
the bracelet's MASTER mode (see docs/armband_mode.png) knows which device to
scan for and connect to once the phone/PC disconnects.

Companion to lib/ble_weights.py - same "connect, write, wait for notify ack"
shape, talking to a different characteristic on the same bracelet.

Wire protocol (must match PairCharCallbacks::onWrite() in the bracelet
.ino, mandro-firmware/firmware/exo_armband_hybrid/exo_armband_hybrid.ino):
    [1B header 0xE0][6B hand MAC, human-readable order][1B XOR checksum]
    checksum = XOR of the 6 MAC bytes only (header excluded).

MAC byte order: sent as-is (not reversed). Confirmed against the firmware's
own comment on masterMac/nativeAddrMatchesHandMac() - the bracelet keeps and
compares masterMac in human-readable order, and only reverses when matching
against a BLE scan's native address.

Public API:
    build_pair_packet(hand_mac: str) -> bytes
    async send_pair_ble(armband_address, hand_mac, on_progress=None) -> str
    async read_pair_ble(armband_address) -> bytes
    async clear_pair_ble(armband_address, on_progress=None) -> str

Errors are raised as BlePairingError with human-readable messages.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────
# Wire protocol constants (must match exo_armband_hybrid.ino)
# ─────────────────────────────────────────────────────────────────────

CHAR_UUID_PAIR = "abcd1234-5678-1234-5678-abcdef123459"
PAIR_HDR = 0xE0
PAIR_PACKET_LEN = 8
PAIR_HDR_CLEAR = 0xE1  # 1-byte standalone write: erase the stored MAC
MAC_BYTE_LEN = 6

ACK_OK_PREFIX = "OK:"

# The robotic hand's BLE module currently advertises under this name prefix
# (case-insensitive). It's the module's own name (CHIPSEN), not the "MARK7"
# product name. This will likely need to become user-selectable once the
# hand's BLE module can vary - see PairingWorker in pair_hand_screen.py.
HAND_NAME_PREFIX = "chipsen"


class BlePairingError(Exception):
    """Raised when any step of the BLE pairing pipeline fails."""


# ─────────────────────────────────────────────────────────────────────
# MAC helpers
# ─────────────────────────────────────────────────────────────────────

def mac_str_to_bytes(address: str) -> bytes:
    """'3C:A5:51:99:BB:5F' -> 6 bytes, human-readable order (no reversal)."""
    parts = address.replace("-", ":").split(":")
    if len(parts) != MAC_BYTE_LEN:
        raise BlePairingError(f"Not a MAC address: {address!r}")
    try:
        return bytes(int(p, 16) for p in parts)
    except ValueError:
        raise BlePairingError(f"Not a MAC address: {address!r}")


def mac_bytes_to_str(data: bytes) -> str:
    return ":".join(f"{b:02X}" for b in data)


# ─────────────────────────────────────────────────────────────────────
# Framing
# ─────────────────────────────────────────────────────────────────────

def build_pair_packet(hand_mac: str) -> bytes:
    """Build the 8-byte pairing packet: header + MAC + XOR checksum."""
    mac = mac_str_to_bytes(hand_mac)
    packet = bytearray(PAIR_PACKET_LEN)
    packet[0] = PAIR_HDR
    packet[1:7] = mac
    chk = 0
    for b in mac:
        chk ^= b
    packet[7] = chk
    return bytes(packet)


def build_clear_packet() -> bytes:
    """Build the 1-byte CLEAR command that erases the bracelet's stored
    master MAC (see PAIR_HDR_CLEAR handling in PairCharCallbacks::onWrite())."""
    return bytes([PAIR_HDR_CLEAR])


# ─────────────────────────────────────────────────────────────────────
# BLE transfer
# ─────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str], None]


async def send_pair_ble(armband_address: str,
                        hand_mac: str,
                        on_progress: Optional[ProgressCallback] = None,
                        timeout_s: float = 10.0,
                        client=None,
                        ) -> str:
    """Write the pairing packet to the bracelet and wait for its ack notify
    (e.g. "OK:PAIR"). Raises BlePairingError on timeout or if the bracelet
    rejects the packet.

    `client`: an already-connected BleakClient to reuse instead of opening
    a new connection. Pass this when the caller needs the bracelet link to
    stay open across multiple calls - e.g. PairingWorker keeps one
    connection open for the whole scan+write+verify sequence, because
    *connecting* to the bracelet is what makes it drop its own MASTER-mode
    link to the hand (see exo_armband_hybrid.ino modeTick()/
    forceDisconnectChipsen()); disconnecting and reconnecting between
    steps would let the bracelet grab the hand back in between.
    If omitted, a new connection is opened and closed just for this call
    (matches lib.ble_weights' style)."""
    try:
        from bleak import BleakClient
    except ImportError:
        raise BlePairingError("bleak is not installed.")

    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    packet = build_pair_packet(hand_mac)

    ack_event = asyncio.Event()
    ack_holder = {"text": None}

    def on_notify(_sender, data):
        # Log every notify on this characteristic, not just the one we end
        # up treating as "the" ack - if something unexpected is arriving
        # (garbled bytes, an unrelated notification, more than one
        # response) this is the only place that would show it.
        raw = bytes(data)
        text = raw.decode("utf-8", errors="replace").strip()
        _emit(f"  notify: {raw.hex(' ').upper()}  text={text!r}")
        if text.startswith(ACK_OK_PREFIX) or text.startswith("ERR:"):
            if ack_holder["text"] is not None:
                _emit(f"  (second ack-shaped notify arrived, ignoring - "
                      f"already had {ack_holder['text']!r})")
                return
            ack_holder["text"] = text
            ack_event.set()
        elif text:
            _emit(f"  (doesn't look like OK:/ERR:, not treating as the ack)")

    async def _do_send(c) -> None:
        t0 = time.monotonic()
        await c.start_notify(CHAR_UUID_PAIR, on_notify)
        _emit(f"  notify subscribed in {time.monotonic() - t0:.2f}s")

        t_write = time.monotonic()
        _emit(f"Sending hand MAC {hand_mac} - packet={packet.hex(' ').upper()}")
        await c.write_gatt_char(CHAR_UUID_PAIR, packet, response=True)
        _emit(f"  write (with response) took {time.monotonic() - t_write:.2f}s")

        t_ack = time.monotonic()
        _emit("Waiting for the bracelet to confirm...")
        try:
            await asyncio.wait_for(ack_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise BlePairingError(
                "Timed out waiting for the bracelet's confirmation.")
        _emit(f"  ack notify arrived {time.monotonic() - t_ack:.2f}s after write")

        try:
            await c.stop_notify(CHAR_UUID_PAIR)
        except Exception:
            pass

    if client is not None:
        await _do_send(client)
    else:
        _emit(f"Connecting to bracelet at {armband_address}...")
        async with BleakClient(armband_address) as new_client:
            await _do_send(new_client)

    ack = ack_holder["text"]
    if ack is None or not ack.startswith(ACK_OK_PREFIX):
        raise BlePairingError(f"Bracelet rejected the pairing packet: {ack}")

    _emit(f"Bracelet confirmed: {ack}")
    return ack


async def read_pair_ble(armband_address: str, client=None) -> bytes:
    """Read back the MAC currently stored in the bracelet's NVS. Returns
    empty bytes if nothing has been paired yet.

    `client`: reuse an already-connected BleakClient - see send_pair_ble()
    for why that matters here."""
    if client is not None:
        data = await client.read_gatt_char(CHAR_UUID_PAIR)
        return bytes(data)

    try:
        from bleak import BleakClient
    except ImportError:
        raise BlePairingError("bleak is not installed.")

    async with BleakClient(armband_address) as client:
        data = await client.read_gatt_char(CHAR_UUID_PAIR)
        return bytes(data)


async def clear_pair_ble(armband_address: str,
                         on_progress: Optional[ProgressCallback] = None,
                         timeout_s: float = 10.0,
                         client=None,
                         ) -> str:
    """Erase the MAC currently stored in the bracelet's NVS. Waits for the
    bracelet's ack notify (e.g. "OK:CLEAR") and returns it. Raises
    BlePairingError on timeout.

    `client`: reuse an already-connected BleakClient - see send_pair_ble()
    for why that matters here."""
    try:
        from bleak import BleakClient
    except ImportError:
        raise BlePairingError("bleak is not installed.")

    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    packet = build_clear_packet()

    ack_event = asyncio.Event()
    ack_holder = {"text": None}

    def on_notify(_sender, data):
        raw = bytes(data)
        text = raw.decode("utf-8", errors="replace").strip()
        _emit(f"  notify: {raw.hex(' ').upper()}  text={text!r}")
        if text.startswith(ACK_OK_PREFIX) or text.startswith("ERR:"):
            if ack_holder["text"] is not None:
                _emit(f"  (second ack-shaped notify arrived, ignoring - "
                      f"already had {ack_holder['text']!r})")
                return
            ack_holder["text"] = text
            ack_event.set()
        elif text:
            _emit(f"  (doesn't look like OK:/ERR:, not treating as the ack)")

    async def _do_clear(c) -> None:
        t0 = time.monotonic()
        await c.start_notify(CHAR_UUID_PAIR, on_notify)
        _emit(f"  notify subscribed in {time.monotonic() - t0:.2f}s")

        t_write = time.monotonic()
        _emit(f"Sending clear command - packet={packet.hex(' ').upper()}")
        await c.write_gatt_char(CHAR_UUID_PAIR, packet, response=True)
        _emit(f"  write (with response) took {time.monotonic() - t_write:.2f}s")

        t_ack = time.monotonic()
        _emit("Waiting for the bracelet to confirm...")
        try:
            await asyncio.wait_for(ack_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise BlePairingError(
                "Timed out waiting for the bracelet's confirmation.")
        _emit(f"  ack notify arrived {time.monotonic() - t_ack:.2f}s after write")

        try:
            await c.stop_notify(CHAR_UUID_PAIR)
        except Exception:
            pass

    if client is not None:
        await _do_clear(client)
    else:
        _emit(f"Connecting to bracelet at {armband_address}...")
        async with BleakClient(armband_address) as new_client:
            await _do_clear(new_client)

    ack = ack_holder["text"]
    if ack is None or not ack.startswith(ACK_OK_PREFIX):
        raise BlePairingError(f"Bracelet rejected the clear command: {ack}")

    _emit(f"Bracelet confirmed: {ack}")
    return ack
