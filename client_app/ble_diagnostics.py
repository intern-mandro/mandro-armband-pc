"""
ble_diagnostics.py
===================
Shared BLE scan/connect helpers with heavy diagnostic logging, used by every
worker thread that talks to the bracelet or the robotic hand (pairing,
weights transfer). Factored out of pair_hand_screen.py once the same
"scan finds it but connect hangs forever with no clue why" symptom showed
up in more than one place - see BleConnectError's callers for the story.

Every function takes a `progress_cb(str)` callback instead of assuming a Qt
signal, so it works the same from any QThread subclass.

Public API:
    class BleConnectError(Exception)
    async scan_for(name_needle, label, progress_cb, exact=False,
                    attempts=3, timeout=5.0) -> address
    async wait_visible(address, label, progress_cb,
                        attempts=6, timeout=5.0) -> None
    connected(address, label, timeout_s=20.0) -> async context manager -> BleakClient
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager


class BleConnectError(Exception):
    """Raised by scan_for/wait_visible/connected on failure. Callers
    typically catch this alongside their own domain error type (e.g.
    lib.ble_hand_pairing.BlePairingError, lib.ble_weights.BleWeightsError)."""


async def scan_for(name_needle, label, progress_cb,
                   exact=False, attempts=3, timeout=5.0):
    """Scan for a device by name, logging enough detail (per-attempt
    duration, device count, RSSI of the match or of the closest partial
    name match) to tell "target genuinely not in range" apart from "target
    is there but weak/borderline" - the two most common causes of
    intermittent scan failures seen in testing (the robotic hand's RSSI
    has been as weak as -80 to -83dBm, right at the edge of reliable
    discovery)."""
    from bleak import BleakScanner

    for attempt in range(1, attempts + 1):
        t_start = time.monotonic()
        progress_cb(
            f"Scanning for {label} ('{name_needle}')... "
            f"(attempt {attempt}/{attempts}, {timeout:.0f}s window)")
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
        elapsed = time.monotonic() - t_start

        for address, (dev, adv) in discovered.items():
            name = dev.name or (adv.local_name if adv else None) or ""
            matched = (name == name_needle) if exact \
                else name.lower().startswith(name_needle.lower())
            if matched:
                rssi = adv.rssi if adv else None
                progress_cb(
                    f"  -> found '{name}' ({address}) RSSI={rssi}dBm "
                    f"after {elapsed:.1f}s ({len(discovered)} devices seen this pass)")
                return address

        # Not matched this pass - report enough to tell "not in range" apart
        # from "in range but weak/name mismatch".
        candidates = [
            (dev.name or (adv.local_name if adv else None) or "", address,
             adv.rssi if adv else None)
            for address, (dev, adv) in discovered.items()
        ]
        partial = [c for c in candidates if name_needle.lower() in (c[0] or "").lower()]
        if partial:
            best = max(partial, key=lambda c: (c[2] if c[2] is not None else -999))
            progress_cb(
                f"  no exact match this pass ({elapsed:.1f}s, {len(discovered)} devices "
                f"seen) - closest partial name match: '{best[0]}' ({best[1]}) "
                f"RSSI={best[2]}dBm")
        else:
            progress_cb(
                f"  not seen this pass ({elapsed:.1f}s, {len(discovered)} devices seen, "
                f"none containing '{name_needle}')")

        if attempt < attempts:
            await asyncio.sleep(1.0)
    raise BleConnectError(f"{label} '{name_needle}' not found after {attempts} attempts")


async def wait_visible(address, label, progress_cb, attempts=6, timeout=5.0):
    """Confirm `address` is actually advertising right now before trying to
    connect to it. Needed for a pre-selected/remembered device - without
    this, connect() can get called right after the bracelet releases the
    hand (or right after any other disconnect), which races the target's
    own "notice the disconnect, resume advertising" delay: seen in testing
    as a connect attempt that just hangs for the full connect timeout
    instead of failing fast, because the target wasn't advertising yet
    when the attempt started. A plain scan_for() run right before
    connecting doesn't have this problem since its own retry loop already
    burns a few seconds first - this gives a pre-selected address the same
    buffer."""
    from bleak import BleakScanner

    for attempt in range(1, attempts + 1):
        t_start = time.monotonic()
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
        elapsed = time.monotonic() - t_start
        if address in discovered:
            dev, adv = discovered[address]
            rssi = adv.rssi if adv else None
            progress_cb(
                f"{label} {address} confirmed advertising (RSSI={rssi}dBm) "
                f"after {elapsed:.1f}s (attempt {attempt}/{attempts})")
            return
        progress_cb(
            f"{label} {address} not seen yet ({elapsed:.1f}s, {len(discovered)} "
            f"devices seen, attempt {attempt}/{attempts}) - waiting for it to "
            f"resume advertising...")
        if attempt < attempts:
            await asyncio.sleep(1.0)
    raise BleConnectError(
        f"{label} {address} never showed up as advertising after {attempts} attempts "
        f"- it may still be held by something else, out of range, or powered off.")


@asynccontextmanager
async def connected(address, label, timeout_s=20.0):
    """Connect to a BLE device with an explicit deadline instead of relying
    on bleak's own (backend-dependent, sometimes very long) default -
    plain `async with BleakClient(address)` was once seen to hang ~54s on
    a bad connection before failing with an exception whose str() was
    empty, which made the real cause invisible in the log. Raises
    BleConnectError with the exception type included either way, so a
    future failure is never silently blank."""
    from bleak import BleakClient

    client = BleakClient(address)
    try:
        async with asyncio.timeout(timeout_s):
            await client.connect()
    except TimeoutError:
        raise BleConnectError(
            f"Timed out ({timeout_s:.0f}s) connecting to the {label} at {address}.")
    except Exception as exc:
        raise BleConnectError(
            f"Failed to connect to the {label} at {address}: "
            f"{type(exc).__name__}: {exc}")

    if not client.is_connected:
        raise BleConnectError(f"Failed to connect to the {label} at {address}")

    try:
        yield client
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
