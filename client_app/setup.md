# EMG Bracelet — Setup & test guide

Short handoff guide to install the client app and test it end-to-end with the bracelet.

## Requirements

* **Python 3.13** (the version the project runs on; other 3.x versions may not match the pinned dependencies).
* **The EMG bracelet** (ESP32-S3) + a USB cable.
* **For flashing firmware** (Phase 0 and Phase 3): `arduino-cli` available on your `PATH`, plus the `esp32:esp32` core installed.
* TensorFlow (for Phase 2 training / `.keras` export) is included in `requirements.txt` (`tensorflow==2.21.0`). It is large and platform-sensitive — see the install note if it causes trouble.

## Install

1. Open a terminal in the project root and create a fresh virtual environment  **with Python 3.13** :
   * Windows: `py -3.13 -m venv .venv` then `.venv\Scripts\activate`
   * macOS / Linux: `python3.13 -m venv .venv` then `source .venv/bin/activate`
2. Install the dependencies (use `python -m pip` so pip and python stay on the same interpreter):
   `python -m pip install --upgrade pip`
   `python -m pip install -r requirements.txt`
3. Sanity check that pip and python agree on this venv:
   `python -c "import sys; print(sys.executable)"` and `pip -V` should both point inside this project's `.venv` and report Python 3.13.
4. *(If `tensorflow==2.21.0` fails to install on your platform)* comment out that line in `requirements.txt`, install the rest, then install the closest available TensorFlow separately.
5. *(Only if you will flash firmware — Phases 0 / 3)* install `arduino-cli` and the ESP32 core:
   * Install `arduino-cli` (Windows: scoop / winget / the official installer; macOS: `brew install arduino-cli`), then reopen the terminal so it is on `PATH`.
   * `arduino-cli core update-index`
   * `arduino-cli core install esp32:esp32`

## Run

From the app folder:
`python main.py`

## What to test (you have the bracelet → full flow)

* **Phase 0** — install the raw capture firmware (or *Skip* if the bracelet already runs it).
* **Phase 1** — placement → calibration → capture session (the two-pane screen: gesture image on the left; connect/start, channel status, and the diagonal-vector view on the right).
* **Phase 2** — train a model from recorded batches. After a successful training, a new row should appear in `sessions.xlsx` (subject id, n_classes, accuracy, model path, data folder).
* **Phase 3** — flash the trained model onto the bracelet.
* **Phase 4** — live gesture verification over BLE.
* **Phase 5** — online scoring.
* **Menu screens** — *Subjects directory* (add / delete subjects, the *Trainings* and *Takes* columns) and  *Load an old session* .

## If something fails

* If the logging module (`applog.py`) is included, a full log with tracebacks is written to your system temp folder as **`emg_client.log`** (Windows: `%TEMP%\emg_client.log`; macOS/Linux: `$TMPDIR/emg_client.log` or `/tmp/emg_client.log`). Send this file when reporting an issue — it captures errors that do not appear in the UI.
* **Bracelet not detected for flashing** : check the USB cable / driver, and that `arduino-cli` is installed (the app shows an install hint). The bracelet's serial port is matched on the Espressif USB vendor id; a board using a CP210x/CH340 USB-UART bridge would not be detected.
* **"Session not saved" after training** : make sure `sessions.xlsx` is **not** open in Excel (the file gets locked and the save fails).
* **Live capture / verification finds no device** : the app scans for the BLE name `ESP32S3_FAST_BLE`; make sure the bracelet is powered on and advertising.

## Known limitations (please note when testing)

* **Not yet tested end-to-end on Windows** , especially firmware flashing and the TensorFlow export step. You may be the first to run it there — the log file will help diagnose any issue.
* A subject is linked to its `data/<profile>_BATCH*` folders by **name matching** (subject id / name / surname). Two subjects whose names normalize identically could therefore share counts.
* Outside Phase 2, some errors are currently only logged, not surfaced in the UI.
