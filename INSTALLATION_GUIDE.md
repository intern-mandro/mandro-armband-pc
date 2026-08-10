# EMG Bracelet App — Installation Guide

How to set up the project from scratch on  **Windows** ,  **Ubuntu** , and  **macOS** .

The application has **two toolchains** that must both be installed:

1. **Python environment** — runs the app, training, and analysis.
2. **Firmware toolchain (`arduino-cli` + ESP32 core + Adafruit libraries)** — compiles and flashes the bracelet firmware in Phases 0 and 3.

Plus a **USB-serial driver / permission** so the computer can talk to the bracelet.

> Items marked `[TO CONFIRM]` depend on your repo and should be checked against it.

---

## Quickstart

Copy-paste, top to bottom, for your OS. The numbered sections below explain each step — go there only if something breaks.

<details><summary><b>Windows</b></summary>
Install first (once):  **Git** , **Python 3.12** (tick "Add to PATH"),  **MSVC Redistributable x64** , **arduino-cli** (zip → PATH), **USB-serial driver** (CP210x/CH340). See §2–§4.

```powershell
# Get the code into an all-ASCII path (important on Korean Windows)
md C:\emg; cd C:\emg
git clone https://github.com/ejayromero/sEMG_classifier.git                 # [TO CONFIRM: repo URL]
cd <project-folder>

# Python + dependencies
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# Firmware toolchain
arduino-cli config init
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "Adafruit NeoPixel" "Adafruit Unified Sensor" "Adafruit BNO055"

# Run
python client_app\main.py
```

**Korean Windows:** put the project under a **Latin-only path** (e.g. `C:\emg\...`). A path containing Korean characters triggers the cp949 encoding crash — an ASCII path is what fixes it.

</details>
<details><summary><b>Ubuntu</b></summary>
```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
sudo usermod -a -G dialout $USER          # then log out / log back in

git clone `https://github.com/ejayromero/sEMG_classifier.git`
cd `<project-folder>`

python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

arduino-cli config init
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "Adafruit NeoPixel" "Adafruit Unified Sensor" "Adafruit BNO055"

python client_app/main.py

```PowerShell


</details>
<details><summary><b>macOS</b></summary>
```bash
brew install python@3.12 arduino-cli git

git clone https://github.com/ejayromero/sEMG_classifier.git                       # [TO CONFIRM: repo URL]
cd <project-folder>

python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

arduino-cli config init
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install "Adafruit NeoPixel" "Adafruit Unified Sensor" "Adafruit BNO055"

python client_app/main.py
```

</details>
---

## 0. Prerequisites at a glance

| Component            | Why                    | Windows                      | Ubuntu                   | macOS                        |
| -------------------- | ---------------------- | ---------------------------- | ------------------------ | ---------------------------- |
| Git                  | clone the repo         | git-scm.com                  | `sudo apt install git` | `brew install git`         |
| Python 3.11 / 3.12   | run the app            | python.org                   | `apt`                  | python.org / brew            |
| arduino-cli          | compile/flash firmware | zip + PATH                   | install script           | `brew install arduino-cli` |
| ESP32 core           | board support          | `arduino-cli core install` | same                     | same                         |
| Adafruit libs        | firmware dependencies  | `arduino-cli lib install`  | same                     | same                         |
| USB-serial access    | see the COM/tty port   | CP210x/CH340 driver          | `dialout`group         | usually automatic            |
| MSVC Redistributable | TensorFlow needs it    | required                     | —                       | —                           |

**Use Python 3.11 or 3.12, not 3.13.** TensorFlow wheels for 3.13 on Windows are very recent/unreliable; 3.11–3.12 are the safe choice.

---

## 1. Get the code

```bash
git clone https://github.com/ejayromero/sEMG_classifier.git         # [TO CONFIRM: repo URL]
cd <project-folder>
```

Keep the project in an **all-ASCII path** (e.g. `C:\emg\...`). Korean/accented characters in the path break several tools.

---

## 2. Python environment

### Windows

1. Install **Python 3.12** (python.org installer, or `winget install Python.Python.3.12`). **Tick "Add python.exe to PATH"** during install.
2. Install the **Microsoft Visual C++ Redistributable (2015–2022, x64)** — TensorFlow fails to import without it (`winget install Microsoft.VCRedist.2015+.x64`, or download from Microsoft).
3. Create and activate the venv, then install dependencies:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1            # PowerShell
# .venv\Scripts\activate.bat          # cmd
python -m pip install -U pip
pip install -r requirements.txt        
```

If PowerShell refuses to run the activation script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

### Ubuntu

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip git
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt     
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

> **TensorFlow note:** on native Windows, GPU is not supported (CPU only) — fine for this project's export/inference. On all OSes, if `pip install -r requirements.txt` can't find TensorFlow, you are almost certainly on Python 3.13 → recreate the venv with 3.12.

---

## 3. Firmware toolchain (`arduino-cli` + ESP32 + libraries)

This is required to flash the bracelet (Phases 0 and 3). The same three steps apply on every OS once `arduino-cli` is installed.

### 3a. Install arduino-cli

**Windows** — download the Windows zip from the arduino-cli releases page, extract `arduino-cli.exe` into a folder (e.g. `C:\tools\arduino-cli\`), and add that folder to your  **PATH** . (Or, if you use a package manager: `winget install ArduinoSA.CLI`, `choco install arduino-cli`, or `scoop install arduino-cli`.)

**Ubuntu / macOS**

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
# macOS alternative: brew install arduino-cli
```

Verify: `arduino-cli version`.

### 3b. Install the ESP32 board core

```bash
arduino-cli config init
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
```

This downloads ~200–300 MB (compiler + toolchain) and takes a few minutes the first time.

To avoid repeating the `--additional-urls` flag, persist the URL once:

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
# older arduino-cli: use "config set" instead of "config add"
```

### 3c. Install the Adafruit libraries

```bash
arduino-cli lib install "Adafruit NeoPixel" "Adafruit Unified Sensor" "Adafruit BNO055"
```

(NeoPixel = the RGB status LED; BNO055 = the IMU; Unified Sensor = its dependency.)

### 3d. Verify

```bash
arduino-cli core list      # should list esp32:esp32
arduino-cli lib list       # should list the three Adafruit libraries
```

---

## 4. USB-serial access to the bracelet

### Windows

Plug the bracelet in and open  **Device Manager → Ports (COM & LPT)** :

* A `COMx` entry appears (e.g.  *Silicon Labs CP210x* ,  *USB Serial CH340* , or  *USB Serial Device* ) → you are ready.
* Nothing appears, or a device with a **yellow warning** → install the USB-serial driver for your board's chip: **CP210x** (Silicon Labs) or  **CH340** . Some ESP32-S3 boards use native USB and need no driver — replug and recheck first.

### Ubuntu

Add yourself to the `dialout` group so the app can open the serial port without `sudo`, then  **log out and back in** :

```bash
sudo usermod -a -G dialout $USER
```

The port appears as `/dev/ttyUSB0` or `/dev/ttyACM0`.

### macOS

Usually automatic; the port appears as `/dev/cu.usbmodem*` or `/dev/cu.usbserial-*`.

---

## 5. Korean / non-UTF-8 Windows

The blocker actually observed on Korean Windows was a  **non-ASCII project path** . With the default **cp949** code page, a path containing Korean characters makes Python crash with `UnicodeDecodeError` / `UnicodeEncodeError`.

* **The fix that works → use a Latin-only path.** Put the project under an all-ASCII path such as `C:\emg\...`. This is the reliable solution and the one that resolved it in practice.
* **Optional, and only for UTF-8 file *content* (not paths):** `setx PYTHONUTF8 1` (then reopen the terminal/IDE), or system-wide Settings → Time & Language → Administrative language settings → Change system locale → tick **"Beta: Use Unicode UTF-8 for worldwide language support"** → reboot. These do **not** fix a non-ASCII path — only moving to a Latin path does.

---

## 6. Run the application

From the **project root** (the folder containing `client_app/` and `lib/`), with the venv activated:

```bash
# Windows
python client_app\main.py

# Ubuntu / macOS
python client_app/main.py
```

---

## 7. Verification checklist

* [ ] `arduino-cli version` works
* [ ] `arduino-cli core list` shows `esp32:esp32`
* [ ] `arduino-cli lib list` shows the three Adafruit libraries
* [ ] venv activates and `python -c "import tensorflow"` runs without error
* [ ] the bracelet shows up as a COM/tty port when plugged in
* [ ] `python client_app/main.py` opens the app
* [ ] Phase 0 flashes the capture firmware successfully

If Phase 3 hangs on `Connecting…`, put the board in bootloader mode (hold  **BOOT** , press  **RESET** , release  **BOOT** ) and retry — this is a board state issue, not an install problem.
