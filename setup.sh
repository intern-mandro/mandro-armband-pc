#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup for the EMG bracelet app (macOS / Linux).
# Creates a virtual environment, installs Python dependencies, and (if
# arduino-cli is present) installs the ESP32 core.

cd "$(dirname "$0")"

# 1. Find a suitable Python (3.10-3.12 preferred; 3.13 ok on macOS/Linux).
PY=""
for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
        case "$v" in
            3.10|3.11|3.12) PY="$c"; break;;
            3.13) if [ -z "$PY" ]; then PY="$c"; fi;;
        esac
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: need Python 3.10-3.12 (3.12 recommended). Install it and re-run." >&2
    exit 1
fi
echo "Using $PY ($("$PY" --version))"

# 2. Create the venv.
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install dependencies.
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. ESP32 core (only if arduino-cli is installed).
if command -v arduino-cli >/dev/null 2>&1; then
    arduino-cli core update-index
    arduino-cli core install esp32:esp32 || true
else
    echo "NOTE: arduino-cli not found. To flash the bracelet, install it from"
    echo "      https://arduino.github.io/arduino-cli/latest/installation/"
fi

echo
echo "Done. Run the app with:"
echo "    source .venv/bin/activate"
echo "    python client_app/main.py"
