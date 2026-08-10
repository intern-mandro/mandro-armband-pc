"""
Firmware upload pipeline for the EMG bracelet.

This module is self-contained: it does NOT import any UI framework.
Higher-level code (e.g. Phase 3 in the Qt UI) wraps these functions
and forwards progress strings to the user through callbacks.

Public API:
    detect_bracelet_port()      -> str | None
    check_arduino_cli()         -> dict
    compile_sketch(...)         -> Path
    flash_firmware(...)         -> None
    upload_model(...)           -> None  (full pipeline)

Errors are raised as FirmwareUploaderError with human-readable messages.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Optional dependency: pyserial. Imported lazily so the module
# is importable even on machines that haven't installed it yet
# (e.g. during the early packaging tests).
try:
    import serial.tools.list_ports as _list_ports  # type: ignore
    _PYSERIAL_AVAILABLE = True
except ImportError:
    _list_ports = None
    _PYSERIAL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Espressif's USB vendor ID, registered with USB-IF.
# All ESP32 boards using the native USB-Serial/JTAG controller advertise this.
ESPRESSIF_VID = 0x303A

# Default ESP32-S3 FQBN for arduino-cli. Tested with esp32:esp32 3.3.9.
DEFAULT_FQBN = "esp32:esp32:esp32s3"

# Baud rate that has proven stable on the USB-Serial/JTAG interface.
# 460800 caused dropped bytes; 115200 is slower but reliable.
DEFAULT_BAUD = 115200

# arduino-cli is located via PATH first (shutil.which). These are per-OS
# fallback locations probed only if it isn't on PATH.
import sys as _sys


def _arduino_cli_fallback_paths() -> list:
    """Well-known arduino-cli locations to probe when it's not on PATH."""
    if _sys.platform == "darwin":
        return ["/opt/homebrew/bin/arduino-cli", "/usr/local/bin/arduino-cli"]
    if _sys.platform == "win32":
        out = []
        for var in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)", "USERPROFILE"):
            base = os.environ.get(var)
            if base:
                out.append(os.path.join(base, "Arduino CLI", "arduino-cli.exe"))
                out.append(os.path.join(base, "arduino-cli", "arduino-cli.exe"))
        return out
    return ["/usr/local/bin/arduino-cli", os.path.expanduser("~/bin/arduino-cli")]


def _arduino_cli_install_hint() -> str:
    """OS-appropriate install instruction for arduino-cli."""
    if _sys.platform == "darwin":
        return "Install it with: brew install arduino-cli"
    return ("Install it from "
            "https://arduino.github.io/arduino-cli/latest/installation/ "
            "and add it to your PATH")


# ─────────────────────────────────────────────────────────────────────
# Project layout helpers
# ─────────────────────────────────────────────────────────────────────

def project_root() -> Path:
    """Return the project root folder (the parent of /lib)."""
    # This file lives in <root>/lib/firmware_uploader.py
    return Path(__file__).resolve().parent.parent


def _resolve_path(p) -> Path:
    """Resolve a possibly relative path against project_root()."""
    p = Path(p)
    if not p.is_absolute():
        p = project_root() / p
    return p


def default_sketch_dir() -> Path:
    return project_root() / "firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid"


def default_export_script() -> Path:
    return project_root() / "scripts/export_teensy_headers.py"


# ─────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────

class FirmwareUploaderError(Exception):
    """Raised when any step of the upload pipeline fails."""


# ─────────────────────────────────────────────────────────────────────
# Tool / environment checks
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ToolCheck:
    """Result of probing the local environment for required tools."""
    arduino_cli_path: Optional[str]
    arduino_cli_version: Optional[str]
    has_esp32_core: bool
    pyserial_available: bool
    issues: list[str]


def _find_arduino_cli() -> Optional[str]:
    """Return the path to arduino-cli, searching PATH then per-OS fallbacks."""
    found = shutil.which("arduino-cli")
    if found:
        return found
    for candidate in _arduino_cli_fallback_paths():
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _arduino_config_file_candidates() -> list:
    """Well-known ASCII-path arduino-cli.yaml locations to probe.

    On accounts with a non-ASCII Windows username, arduino-cli's default
    config-file auto-discovery is unreliable when invoked from a Python
    subprocess (it can silently fall back to its built-in default data dir
    under %LOCALAPPDATA%\\<username>\\..., which breaks the ESP32 toolchain
    on non-ASCII paths - 'fatal error: ... Invalid argument' deep in
    newlib/lwip headers). Pinning an explicit --config-file avoids relying
    on that discovery at all. Same fix as the --config-file workaround
    already used for this board elsewhere (see project history).
    """
    if _sys.platform == "win32":
        return [r"C:\Arduino15\arduino-cli.yaml"]
    return []


def _find_arduino_config_file() -> Optional[str]:
    for candidate in _arduino_config_file_candidates():
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _arduino_cli_base_cmd() -> list:
    """[cli_path] or [cli_path, '--config-file', ...] if a known-good
    ASCII config file was found. Use this as the start of every arduino-cli
    command in this module instead of a bare [cli]."""
    cli = _find_arduino_cli()
    if not cli:
        raise FirmwareUploaderError(
            "arduino-cli is not installed. " + _arduino_cli_install_hint())
    cmd = [cli]
    config_file = _find_arduino_config_file()
    if config_file:
        cmd += ["--config-file", config_file]
    return cmd


def check_arduino_cli() -> ToolCheck:
    """Probe the local machine for arduino-cli, esp32 core and pyserial.

    Does NOT raise. Returns a structured result the caller can present
    to the user (e.g. with a 'fix this' button).
    """
    issues: list[str] = []

    cli = _find_arduino_cli()
    if not cli:
        issues.append(
            "arduino-cli is not installed. " + _arduino_cli_install_hint())
        return ToolCheck(
            arduino_cli_path=None, arduino_cli_version=None,
            has_esp32_core=False, pyserial_available=_PYSERIAL_AVAILABLE,
            issues=issues)

    base_cmd = _arduino_cli_base_cmd()

    # Version
    version_line = None
    try:
        out = subprocess.run(
            base_cmd + ["version"], capture_output=True, text=True, timeout=10)
        version_line = (out.stdout or "").strip().splitlines()[0] if out.stdout else None
    except Exception as exc:
        issues.append(f"arduino-cli runs but `version` failed: {exc}")

    # Is the esp32:esp32 core installed?
    has_esp32_core = False
    try:
        out = subprocess.run(
            base_cmd + ["core", "list"], capture_output=True, text=True, timeout=15)
        for line in (out.stdout or "").splitlines():
            if line.startswith("esp32:esp32"):
                has_esp32_core = True
                break
        if not has_esp32_core:
            issues.append(
                "esp32:esp32 core not installed. Install with:\n"
                "  arduino-cli core update-index\n"
                "  arduino-cli core install esp32:esp32")
    except Exception as exc:
        issues.append(f"could not list arduino-cli cores: {exc}")

    if not _PYSERIAL_AVAILABLE:
        issues.append(
            "pyserial is not installed. Install with: pip install pyserial")

    return ToolCheck(
        arduino_cli_path=cli,
        arduino_cli_version=version_line,
        has_esp32_core=has_esp32_core,
        pyserial_available=_PYSERIAL_AVAILABLE,
        issues=issues,
    )


# ─────────────────────────────────────────────────────────────────────
# Bracelet detection
# ─────────────────────────────────────────────────────────────────────

def detect_bracelet_port() -> Optional[str]:
    """Find the ESP32 bracelet's serial port.

    Returns the device path (e.g. '/dev/cu.usbmodem1101') if exactly one
    Espressif device is connected. Returns None if no bracelet is found.
    Raises FirmwareUploaderError if multiple bracelets are connected,
    so the caller can ask which one to use.
    """
    if not _PYSERIAL_AVAILABLE:
        raise FirmwareUploaderError(
            "pyserial is required to detect the bracelet. "
            "Install with: pip install pyserial")

    candidates = [
        p for p in _list_ports.comports()
        if p.vid == ESPRESSIF_VID
    ]
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0].device
    # Multiple bracelets — caller has to pick. Encode info into the message.
    devices = ", ".join(p.device for p in candidates)
    raise FirmwareUploaderError(
        f"Multiple Espressif devices found: {devices}. "
        "Disconnect the ones you do NOT want to flash.")


# ─────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str], None]


def _run_streaming(cmd: list[str], *, cwd: Optional[Path] = None,
                   on_progress: Optional[ProgressCallback] = None) -> str:
    """Run a subprocess, stream stdout line-by-line to on_progress,
    and return the full stdout text. Raises FirmwareUploaderError if the
    process returns non-zero.
    """
    _env = os.environ.copy()
    # Force UTF-8 so the child's Unicode prints (✅, ═, →) don't crash on a
    # Windows cp949/cp1252 console, and decode its output as UTF-8 here too.
    _env["PYTHONUTF8"] = "1"
    _env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        if on_progress and line:
            try:
                on_progress(line)
            except Exception:
                # Never let a UI callback kill the subprocess loop.
                pass
    proc.wait()
    output = "\n".join(lines)
    if proc.returncode != 0:
        raise FirmwareUploaderError(
            f"Command failed (exit {proc.returncode}):\n"
            f"  {' '.join(cmd)}\n\n{output}")
    return output


# ─────────────────────────────────────────────────────────────────────
# Header export (delegated to scripts/export_teensy_headers.py)
# ─────────────────────────────────────────────────────────────────────

def export_headers(keras_path: Path,
                   output_dir: Path,
                   on_progress: Optional[ProgressCallback] = None,
                   ) -> None:
    """Run the export script to produce MODEL.h / means.h / stds.h
    in `output_dir`. The scaler is auto-deduced from the model name."""
    script = default_export_script()
    if not script.exists():
        raise FirmwareUploaderError(f"Export script missing: {script}")
    keras_path = _resolve_path(keras_path)
    output_dir = _resolve_path(output_dir)
    if not keras_path.exists():
        raise FirmwareUploaderError(f"Model file not found: {keras_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(script),
        "--keras", str(keras_path),
        "--output-dir", str(output_dir),
    ]
    _run_streaming(cmd, cwd=project_root(), on_progress=on_progress)


def copy_headers_into_sketch(headers_dir: Path,
                             sketch_dir: Optional[Path] = None) -> None:
    """Copy MODEL.h / means.h / stds.h from `headers_dir` into the sketch folder."""
    if sketch_dir is None:
        sketch_dir = default_sketch_dir()
    if not sketch_dir.is_dir():
        raise FirmwareUploaderError(f"Sketch folder missing: {sketch_dir}")

    for name in ("MODEL.h", "means.h", "stds.h"):
        src = headers_dir / name
        dst = sketch_dir / name
        if not src.exists():
            raise FirmwareUploaderError(f"Generated header missing: {src}")
        shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────────────────────────────

def compile_sketch(sketch_dir: Optional[Path] = None,
                   build_dir: Optional[Path] = None,
                   fqbn: str = DEFAULT_FQBN,
                   on_progress: Optional[ProgressCallback] = None,
                   ) -> Path:
    """Compile the sketch via arduino-cli. Returns the build directory
    containing the .bin files. Raises FirmwareUploaderError on failure."""
    if sketch_dir is None:
        sketch_dir = default_sketch_dir()
    if build_dir is None:
        build_dir = project_root() / "_build_esp32"

    if not sketch_dir.is_dir():
        raise FirmwareUploaderError(f"Sketch folder missing: {sketch_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = _arduino_cli_base_cmd() + [
        "compile",
        "--fqbn", fqbn,
        "--output-dir", str(build_dir),
        # Without --build-path, arduino-cli puts its intermediate build
        # cache under %LOCALAPPDATA%\arduino\sketches\<hash> regardless of
        # where the sketch/project lives. On a Windows account with a
        # non-ASCII username that path breaks the linker (ld.exe can't
        # open its output file there) even though the project itself is
        # in an all-ASCII path. Pin it to our own (already-ASCII) build
        # dir instead - same fix as --config-file elsewhere in this repo.
        "--build-path", str(build_dir),
        # arduino-cli defaults to one cc1plus.exe per CPU core. On a
        # machine that's already low on free RAM, that many parallel C++
        # compiles (this sketch pulls in the BLE/lwip headers, which are
        # heavy) can exhaust memory and crash with
        # "cc1plus.exe: out of memory allocating ...". Compiling serially
        # is slower but avoids that failure mode entirely; this sketch is
        # small enough that the difference is a matter of seconds.
        "--jobs", "1",
        str(sketch_dir),
    ]
    _run_streaming(cmd, cwd=project_root(), on_progress=on_progress)
    return build_dir


# ─────────────────────────────────────────────────────────────────────
# Flash
# ─────────────────────────────────────────────────────────────────────

def flash_firmware(port: str,
                   build_dir: Path,
                   sketch_dir: Optional[Path] = None,
                   fqbn: str = DEFAULT_FQBN,
                   on_progress: Optional[ProgressCallback] = None,
                   ) -> None:
    """Flash the firmware located in `build_dir` to the ESP32 on `port`.
    Uses arduino-cli upload, which handles bootloader + partitions + app
    automatically."""
    if sketch_dir is None:
        sketch_dir = default_sketch_dir()
    if not build_dir.is_dir():
        raise FirmwareUploaderError(f"Build folder missing: {build_dir}")
    if not sketch_dir.is_dir():
        raise FirmwareUploaderError(f"Sketch folder missing: {sketch_dir}")

    cmd = _arduino_cli_base_cmd() + [
        "upload",
        "--fqbn", fqbn,
        "--port", port,
        "--input-dir", str(build_dir),
        str(sketch_dir),
    ]
    _run_streaming(cmd, cwd=project_root(), on_progress=on_progress)


# ─────────────────────────────────────────────────────────────────────
# High-level pipeline
# ─────────────────────────────────────────────────────────────────────

def upload_model(keras_path: Path,
                 port: Optional[str] = None,
                 on_progress: Optional[ProgressCallback] = None,
                 ) -> None:
    """End-to-end: export headers from a .keras model, compile the sketch,
    and flash it to the bracelet.

    If `port` is None, the bracelet is auto-detected. Raises
    FirmwareUploaderError with a human-readable message at the first
    failed step.

    on_progress receives status strings like 'Exporting headers...',
    'Compiling firmware...', etc., and one entry per line of subprocess
    stdout. Phase 3 wraps this into Qt signals."""
    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    keras_path = _resolve_path(keras_path)
    if not keras_path.exists():
        raise FirmwareUploaderError(f"Model file not found: {keras_path}")

    # ── Step 1: pick a port ────────────────────────────────────────────
    if port is None:
        _emit("Detecting bracelet on USB...")
        port = detect_bracelet_port()
        if port is None:
            raise FirmwareUploaderError(
                "No bracelet detected on USB.\n"
                "Plug the bracelet in and make sure it's powered on, then try again.")
        _emit(f"Bracelet found on port: {port}")

    # ── Step 2: export headers into firmware/teensy/ ───────────────────
    # (kept as 'teensy' for backwards compatibility — both Teensy and ESP32
    # use the same header format)
    headers_dir = project_root() / "firmware/teensy"
    _emit(f"Exporting MODEL.h from {keras_path.name}...")
    export_headers(keras_path, headers_dir, on_progress=_emit)

    # ── Step 3: copy headers into the sketch folder ────────────────────
    _emit("Copying headers into the sketch...")
    copy_headers_into_sketch(headers_dir)

    # ── Step 4: compile ────────────────────────────────────────────────
    build_dir = project_root() / "_build_esp32"
    _emit("Compiling the firmware... (this can take 1-2 min)")
    compile_sketch(build_dir=build_dir, on_progress=_emit)

    # ── Step 5: flash ──────────────────────────────────────────────────
    _emit(f"Flashing the bracelet on {port}...")
    flash_firmware(port=port, build_dir=build_dir, on_progress=_emit)

    _emit("Done. The bracelet has been re-flashed and rebooted.")



# ─────────────────────────────────────────────────────────────────────
# "Raw capture" firmware install (Phase 0)
#
# Historically a separate exo_armband_raw/ sketch (raw EMG streaming, no
# inference). Unified with the hybrid firmware: exo_armband_hybrid_6clf
# already streams raw EMG unconditionally (regardless of whether a model
# has been loaded via BLE - see NeuralNet::isLoaded() gating in the .ino),
# and its MODEL.h now only holds compile-time topology (no per-model
# weight injection needed before compiling, since weights load at runtime
# from LittleFS). So "install raw capture firmware" and "install on
# bracelet" (Phase 3) now flash the exact same sketch - there is only one
# firmware image for this board. The old exo_armband_raw/ sketch is kept
# in the repo for history but is no longer built by anything.
# ─────────────────────────────────────────────────────────────────────

def default_raw_sketch_dir() -> Path:
    """Sketch dir used by Phase 0's 'install capture firmware' - same as
    the hybrid sketch (see module docstring above)."""
    return default_sketch_dir()


def install_raw_firmware(
    port: str = None,
    sketch_dir: Path = None,
    build_dir: Path = None,
    on_progress=None,
) -> None:
    """Flash the bracelet with the (unified) hybrid firmware, with no
    model loaded yet. Used by Phase 0, before any model has been trained.

    Compiles and flashes exo_armband_hybrid_6clf as-is (no header export
    step - MODEL.h is topology-only). The bracelet boots streaming raw
    EMG with nn.isLoaded() == false, identical to the old raw-only sketch.
    Once a model exists, Phase 3's "Send weights over BLE" loads it
    without any further reflash.

    Parameters
    ----------
    port : str, optional
        Serial port (e.g. /dev/cu.usbmodem1101). Auto-detected if None.
    sketch_dir : Path, optional
        Path to the sketch folder. Default: default_sketch_dir() (hybrid).
    build_dir : Path, optional
        Path to the build output. Default: project_root()/_build_esp32
        (shared with upload_model(), since it's the same sketch).
    on_progress : callable, optional
        Receives progress messages as str. Pass to the GUI.
    """
    def _report(msg):
        if on_progress is not None:
            on_progress(msg)
        else:
            print(msg)

    # 1) Detect bracelet
    if port is None:
        _report("Detecting bracelet on USB...")
        port = detect_bracelet_port()
        if port is None:
            raise FirmwareUploaderError(
                "No bracelet detected. Plug it in and try again.")
        _report(f"Bracelet found on port: {port}")

    # 2) Resolve sketch and build dirs
    if sketch_dir is None:
        sketch_dir = default_raw_sketch_dir()
    else:
        sketch_dir = _resolve_path(sketch_dir)
    if build_dir is None:
        build_dir = project_root() / "_build_esp32"
    else:
        build_dir = _resolve_path(build_dir)

    if not sketch_dir.is_dir():
        raise FirmwareUploaderError(f"Sketch folder not found: {sketch_dir}")

    # 3) Compile
    _report("")
    _report("Compiling capture firmware...")
    _report("(This may take 1-2 minutes the first time, faster afterwards.)")
    compile_sketch(
        sketch_dir=sketch_dir,
        build_dir=build_dir,
        on_progress=on_progress,
    )

    # 4) Flash
    _report("")
    _report(f"Flashing bracelet on {port}...")
    flash_firmware(
        port=port,
        sketch_dir=sketch_dir,
        build_dir=build_dir,
        on_progress=on_progress,
    )

    _report("")
    _report("Done. The bracelet has been re-flashed with the raw capture firmware.")