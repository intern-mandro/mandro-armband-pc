"""
CLI smoke test for lib/firmware_uploader.py.

Lets you validate the upload pipeline step-by-step from the terminal,
before wiring it into the Phase 3 UI.

Usage:
    # 1. Probe the local environment
    python test_firmware_uploader.py --check-tools

    # 2. Detect the bracelet's USB port
    python test_firmware_uploader.py --detect-port

    # 3. Run the full upload for a specific model
    python test_firmware_uploader.py --full \
        models/trained/model_KOTA_6cl_20260528.keras

    # 4. Compile only, no flash (useful when bracelet is unplugged)
    python test_firmware_uploader.py --compile-only
"""

import argparse
import sys
from pathlib import Path

# Make `lib/firmware_uploader.py` importable when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import firmware_uploader as fu  # noqa: E402


def print_progress(msg: str) -> None:
    """Echo the uploader's progress strings to stdout."""
    print(f"  {msg}")


def cmd_check_tools() -> int:
    print("── Probing tools ──")
    res = fu.check_arduino_cli()
    print(f"  arduino-cli path     : {res.arduino_cli_path}")
    print(f"  arduino-cli version  : {res.arduino_cli_version}")
    print(f"  esp32:esp32 core     : {'yes' if res.has_esp32_core else 'NO'}")
    print(f"  pyserial available   : {res.pyserial_available}")
    if res.issues:
        print("\nIssues:")
        for line in res.issues:
            print(f"  - {line}")
        return 1
    print("\nAll tools OK.")
    return 0


def cmd_detect_port() -> int:
    print("── Detecting bracelet ──")
    try:
        port = fu.detect_bracelet_port()
    except fu.FirmwareUploaderError as exc:
        print(f"  ERROR: {exc}")
        return 1
    if port is None:
        print("  No bracelet found. Is it plugged in and powered on?")
        return 1
    print(f"  Bracelet on: {port}")
    return 0


def cmd_compile_only() -> int:
    print("── Compile sketch (no flash) ──")
    try:
        build_dir = fu.compile_sketch(on_progress=print_progress)
    except fu.FirmwareUploaderError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print(f"\nBuild artefacts: {build_dir}")
    for name in ("exo_armband_hybrid.ino.bin",
                 "exo_armband_hybrid.ino.bootloader.bin",
                 "exo_armband_hybrid.ino.partitions.bin"):
        p = build_dir / name
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        mark = "ok" if p.exists() else "MISSING"
        print(f"  [{mark}] {name} ({size_kb:.1f} KB)")
    return 0


def cmd_full(keras_path: Path) -> int:
    print(f"── Full upload pipeline for {keras_path} ──")
    try:
        fu.upload_model(keras_path, on_progress=print_progress)
    except fu.FirmwareUploaderError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("\nDone. The bracelet should now be running the new model.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test firmware_uploader.py")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-tools", action="store_true",
                   help="Probe arduino-cli, ESP32 core, pyserial")
    g.add_argument("--detect-port", action="store_true",
                   help="Find the bracelet's USB port")
    g.add_argument("--compile-only", action="store_true",
                   help="Compile the sketch without flashing")
    g.add_argument("--full", metavar="KERAS_PATH",
                   help="Run the full pipeline: export -> compile -> flash")
    args = parser.parse_args()

    if args.check_tools:
        return cmd_check_tools()
    if args.detect_port:
        return cmd_detect_port()
    if args.compile_only:
        return cmd_compile_only()
    if args.full:
        return cmd_full(Path(args.full))
    return 0


if __name__ == "__main__":
    sys.exit(main())

