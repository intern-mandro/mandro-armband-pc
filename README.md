# EMG Bracelet — Acquisition, Training & Live Classification

PyQt6 desktop app to record 8-channel surface-EMG from an ESP32-S3 bracelet,
train a personalized gesture classifier, and run live on-device inference.

## Hardware
- ESP32-S3 bracelet, 8 EMG channels, BLE (device name `ESP32S3_FAST_BLE`).
- Sampling 1200 Hz, 10-bit ADC.

## Requirements
- Python >= 3.10 (the code uses `X | None` typing; it crashes on 3.9).
- `arduino-cli` for flashing firmware: https://arduino.github.io/arduino-cli/
- See `WINDOWS_SETUP.md` for Windows steps (BLE pairing, toolchain).

## Install
    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    # if pip install fails with a version conflict do this first:
    pip install PyQt6 numpy openpyxl pyserial pyqtgraph bleak

Pin `scikit-learn==1.8.0` so saved scalers load without version warnings.

## Run
    python client_app/main.py

## Workflow (Phases 0 to 5)
- Phase 0 — Flash the capture firmware (live streaming). Same firmware as
  Phase 3 — see "Firmware" below. Flash it here or skip and flash it from
  Phase 3 instead; either works.
- Phase 1 — Baseline calibration: rest the arm to measure per-channel noise.
- Phase 2 — Record gestures and train a personalized model.
- Phase 3 — Install the trained model on the bracelet. Two ways, same screen:
  - **USB (full reflash)** — compiles and flashes the whole inference
    firmware over USB via arduino-cli. Required once (first flash of the
    BLE-capable firmware, or after any topology/firmware-logic change).
  - **BLE (weights only)** — pushes just the retrained weights over BLE to
    a bracelet already running that firmware; no recompile, no USB, a few
    seconds. See `firmware/esp32/exo_armband_hybrid_6clf/` and
    `lib/ble_weights.py`. Originally scoped in
    [ble_model_update_options.md](ble_model_update_options.md) (now
    implemented, kept for history).
- Phase 4 — Verify the installed model.
- Phase 5 — Live scoring (real-time classification).

Gestures (6-class): rest, flexion, extension, close, supination, pronation.

## Firmware
One sketch, `firmware/esp32/exo_armband_hybrid_6clf/`, handles everything:
it streams raw EMG unconditionally (used for Phase 0/1/2 capture) and runs
inference once a model has been loaded over BLE (used for Phase 4/5) —
`nn.isLoaded()` gates only the inference/prediction path, never the raw
stream. Phase 0 and Phase 3's "Install on bracelet" both flash this exact
sketch (`lib/firmware_uploader.py`'s `install_raw_firmware()` and
`upload_model()` share the same sketch/build dir); flashing from either
screen is equivalent.

There used to be a second sketch (`exo_armband_raw/`) dedicated to capture,
which had to be kept byte-identical to the inference sketch's signal path
(`analogReadResolution(...)`, the `tmp -= ...` offset, `sensorPin[]` order)
or classification would break silently — a mismatch here (8-bit vs 10-bit
ADC + swapped channels) once caused a hard-to-find bug. Unifying onto one
sketch removes that failure mode entirely (nothing to keep in sync anymore).
`exo_armband_raw/` is kept in the repo for history but nothing builds it.

A separate, unrelated variant, `exo_armband_hybrid_4clf_USBSERIAL/`, still
exists for 4-class/USB-serial use and was not part of this unification.

## Data format
CSV per take: `Time(ms), Raw_CH0..Raw_CH7, Amp_CH0..Amp_CH7, Label, Window`.
Training uses the band-pass-filtered `Raw_CH*`. Window size 128.

## Known limitations
- The Phase 2 trainer (`scripts/train_causal_concat.py`, launched by the app)
  shuffles windows before the train/val split -> validation accuracy is
  optimistic (leakage between adjacent windows). For an honest estimate use
  `scripts/train_multisession.py` (subject-grouped / LOSO).
- `DATA_TEST_RAW` has a default-subject fallback; set it explicitly to avoid
  silently evaluating on the wrong subject.

## Not in this repo
- `data/` — raw EMG recordings (large + biometric). Record your own.
- `models/` — trained models are subject-specific; train your own via Phase 2.

## Layout
- `client_app/` — the app (entry `main.py`), phases, dashboard, viewers.
- `lib/` — data loading, preprocessing, features, training, evaluation, firmware uploader.
- `scripts/` — offline training / benchmark / export tools.
- `firmware/esp32/` — the two Arduino sketches.
