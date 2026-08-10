# Developer Guide — EMG Bracelet Project

Handover documentation for whoever maintains or extends this project.
Pair it with `README.md` (quick start) and `WINDOWS_SETUP.md` (Windows specifics).

---

## 1. Overview & architecture

The system records 8-channel surface EMG from an ESP32-S3 bracelet over BLE,
trains a **per-subject** gesture classifier, and runs live on-device inference.
A PyQt6 desktop app guides the operator through a phased workflow (0 → 5).

Repository layout:

- `client_app/` — the PyQt6 application.
  - `main.py` — entry point.
  - `phase0_install_capture.py`, `phase2_training.py`, `phase3_install.py`,
    `phase4_verify.py`, `phase5_online_score.py` — the phase screens.
  - `dashboard_ui.py` — live signal dashboard.
  - `serial_worker.py` — BLE worker (async, ~1200 Hz stream).
  - `logger.py` — CSV recorder.
  - `app_window.py` — main window, protocol, gesture cues, quality guardrail.
  - `take_viewer.py`, `view_single_take.py` — post-acquisition viewers.
  - `subjects_registry.py`, `sessions_registry.py` — bookkeeping.
- `lib/` — the data-science core.
  - `data_loader.py`, `preprocessing.py`, `features/features.py`,
    `windowing.py`, `training.py`, `evaluation.py`, `models.py`,
    `pipeline.py`, `configs.py`, `utils.py`, `export.py`.
  - `firmware_uploader.py` — drives `arduino-cli` to compile/flash and to
    upload models; cross-platform serial-port detection.
- `scripts/` — offline tools.
  - `train_causal_concat.py` — the trainer the app calls in Phase 2.
  - `train_multisession.py` — honest LOSO / GroupKFold evaluation + final model.
  - `benchmark_offline.py`, `capture_live_predictions.py`,
    `export_teensy_headers.py`.
- `firmware/esp32/` — two Arduino sketches + the shared config header (see §4).
- `data/`, `models/` — recordings and trained artifacts (git-ignored).

---

## 2. Installation & running

Requirements:
- **Python ≥ 3.10** — the code uses `X | None` typing; it crashes on 3.9.
- `arduino-cli` with the `esp32:esp32` core installed.
- Python deps from `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python client_app/main.py
```

Pin `scikit-learn==1.8.0` so saved scalers load without version warnings.

Key identifiers:
- BLE device name: `ESP32S3_FAST_BLE`
- Board FQBN: `esp32:esp32:esp32s3`

---

## 3. Data pipeline

Constants (`lib/configs.py`): sampling **1200 Hz**, window **128 samples**
(~107 ms), **8 channels**, **6 gestures** (rest, flexion, extension, close,
supination, pronation).

CSV format — one file per *take*:

```
Time(ms), Raw_CH0..Raw_CH7, Amp_CH0..Amp_CH7, Label, Window
```

Training uses the band-pass-filtered `Raw_CH*` channels.

Flow:
1. **Load** — `data_loader.load_datasets_new_format`.
2. **Preprocess** — band-pass `LOWCUT`–`HIGHCUT`, etc. (`pipeline.preprocess_pipeline`).
3. **Window** — `windowing.get_windows`: **non-overlapping** 128-sample blocks;
   label = majority vote (`get_action_windows`).
4. **Features** — `pipeline.extract_features_pipeline`, mode `concat`
   (classic + TSD features).
5. **Scale** — scaler fit on the **train split only**.
6. **Model** — `models.build_model` (2 hidden layers, 64 units, lr 0.001).
7. **Train** — `training.train_model`.

---

## 4. Firmware (unified — one sketch)

`firmware/esp32/exo_armband_hybrid_6clf/` is the only sketch anything builds.
It streams raw EMG unconditionally (`pCharacteristic->notify()` in `loop()`
doesn't check model state) and runs inference only once `nn.isLoaded()` is
true (weights loaded from `/weights.bin` on LittleFS, sent over BLE — see
`lib/ble_weights.py`). Phase 0 ("install capture firmware") and Phase 3
("Install on bracelet") both flash this exact sketch via
`lib/firmware_uploader.py`'s `install_raw_firmware()` / `upload_model()` —
same sketch dir, same build dir. There is nothing to keep in sync anymore.

**History (no longer applies, kept for context):** there used to be a second
sketch, `exo_armband_raw/`, dedicated to capture. It had to be kept
byte-identical to the inference sketch's signal path (ADC resolution, the
`tmp -= OFFSET`, `sensorPin[]` order) or classification would break
**silently** — a historical 8-bit vs 10-bit ADC + swapped-channel mismatch
caused a hard-to-find bug this way. That two-sketch setup is gone; unifying
onto one sketch removes the failure mode structurally rather than requiring
discipline or tooling to enforce it.

*(Note: this section previously described a `firmware/sync_firmware_config.py`
/ `firmware/esp32/shared/emg_signal_config.h` mechanism for keeping the two
sketches in sync. That mechanism doesn't actually exist in this repo — it
looks like aspirational documentation that was never implemented. Moot now
that there's only one sketch, but flagging it in case it's referenced
elsewhere.)*

`exo_armband_raw/` is kept in the repo for history; nothing builds it anymore.
A separate, unrelated `exo_armband_hybrid_4clf_USBSERIAL/` variant still
exists for 4-class/USB-serial use and was not touched by this unification.

Compile check (no flashing):
```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3 \
  firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid
```

---

## 5. Training & honest evaluation

Phase 2 runs `scripts/train_causal_concat.py`: it loads BATCH1 (train + val)
and BATCH2 (held-out test), windows, extracts features, splits train/val
**grouped by take**, fits the scaler on train only, trains, and reports a
BATCH2 test accuracy.

Honest-metrics notes:
- The train/val split is **grouped by take** (`GroupShuffleSplit`), so the val
  number is leakage-free. (Earlier it used a window-level random shuffle that
  put near-identical neighbouring windows on both sides, inflating val
  accuracy. See §8.)
- The **BATCH2 test** (a different session) is the real generalization number —
  provided `DATA_TEST_RAW` points to the correct subject's BATCH2 (see §8).
- For a subject with only **one** batch, use `scripts/train_multisession.py`.
  It auto-detects the subject's sessions and reports:
  - **Leave-One-Session-Out** CV when ≥ 2 sessions (true inter-session number), or
  - **GroupKFold-by-take** when 1 session (intra-session estimate; the script
    warns it does NOT reflect live use).
  Force a single batch with `EMG_DATA_SESSIONS="data/<SUBJECT>_BATCH1"`, and set
  `EMG_MODEL_OUTPUT_NAME` to avoid overwriting a deployed model.

---

## 6. Models

Saved under `models/trained/*.keras` and `models/scalers/*.pkl`, named
`model_<SUBJECT>_<N>cl_<date>[...].keras`. Scalers are pickled scikit-learn
objects — pin `scikit-learn==1.8.0` to avoid version warnings/incompatibility.

---

## 7. Common tasks

- **Record + train a new subject** — run the app, Phases 0 → 3.
- **Re-train offline** — `python scripts/train_causal_concat.py` (with
  `DATA_RAW` / `DATA_TEST_RAW` set), or `train_multisession.py`.
- **Change firmware signal settings** — edit the constants directly in
  `firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid/exo_armband_hybrid.ino`
  (`analogReadResolution(...)`, the `tmp -= ...` offset, `sensorPin[]`), then
  reflash (one sketch now, so nothing else needs to stay in sync).
- **Add a new gesture class** — [TO CONFIRM: exact steps — likely edit the
  gesture list in `lib/configs.py` and `client_app/protocol_controller.py`,
  add an asset/frames folder, then re-record and retrain.]

---

## 8. Known limitations & backlog

- `DATA_TEST_RAW` has a **silent default-subject fallback** (S003): if unset,
  the "test" is silently measured on the wrong subject. Always set it
  explicitly. Recommended fix: raise an explicit error when it is unset.
- A shell alias `python=python3.9` (if present on the machine) breaks the app
  (`X | None` typing). Require Python ≥ 3.10.
- Inside each CV fold, the early-stopping val still uses a window-level shuffle
  (`split_train_val`). It does **not** inflate the reported metric, but is not
  perfectly clean — a future improvement.
- `scripts/benchmark_offline.py` also imports the window-level split — verify
  before trusting its numbers.
- Two separate "8-channel app" codebases exist on the shared repo (this one and
  a colleague's `client_app_esp32_8ch`). Coordinate before merging.

---

## 9. Troubleshooting

- **Live classification poor / random** — confirm the bracelet is running
  the current 10-bit unified firmware (not a stale flash), check electrode
  contact, confirm `nn.isLoaded()` (Serial log prints whether a model was
  loaded from `/weights.bin` at boot).
- **In-app quality warning "electrode contact"** — a channel is flat/dead;
  re-seat the bracelet.
- **In-app quality warning "weak / old firmware"** — data captured with the old
  8-bit firmware; reflash the corrected capture firmware.
- **Scaler version warning** — pin `scikit-learn==1.8.0`.
- **arduino-cli: core not installed** — `arduino-cli core install esp32:esp32`.
- [TO CONFIRM: common BLE pairing issues specific to your machines, esp. Windows.]
