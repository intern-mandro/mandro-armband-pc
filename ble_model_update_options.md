
# Updating the Bracelet Model over Bluetooth — Option B (Model-Only Update)

> **Implemented.** This scoping doc's recommendations (LittleFS, write-with-response,
> float32, one coherent means+stds+weights bundle, checksum) were built as scoped, with
> the exact wire protocol below. See `firmware/esp32/exo_armband_hybrid_6clf/` (the
> `nn.h`/`nn.cpp`/`preprocessor.h`/`preprocessor.cpp`/`.ino` changes) and
> `lib/ble_weights.py` on the PC side, wired into Phase 3's "Send weights over BLE"
> button. Kept below for history/rationale.

Handover / future-work note. Goal: retrain a model on the PC and push it onto the
bracelet **over BLE, without re-flashing over USB**. This document scopes the chosen
approach — **Option B: send only the model and keep it in a rewritable file**. It is a
scoping document; none of this is implemented yet.

Figures below (MTU, packet counts, storage limits) were checked against ESP32/BLE
documentation and community sources; they are realistic, not guesses. Where I am
unsure, it is stated explicitly.

---

## Current state (the starting point)

- The bracelet runs one firmware; the trained model is **compiled into the binary**
  as C arrays (`MODEL.h`, `means.h`, `stds.h`). Changing the model today means
  **recompiling and re-flashing over USB** (Phase 3).
- BLE is **notify-only**: the firmware only *sends* (raw EMG, predictions). There is
  **no write characteristic**, so nothing can currently be sent *to* the bracelet.
- MTU is set to 247 (~244 usable bytes/packet). BLE itself works well.

So "send the model over BLE in the current state" is **not possible** — two things
are missing: a way to *receive* data, and a way to *store* a model that isn't baked
into the compiled binary.

Sizes for reference:

- Fitted classifier weights only: **~51 KB** (13,062 float32 params).
- With means + stds + class list: **~52–55 KB**.
- Full compiled inference firmware: ~684 KB (for comparison — not what we transfer).

---

## The approach: send the model, keep it in a rewritable file

Send only the model data (~52–55 KB) over BLE. The firmware stores it in a
**rewritable file** and reloads it at boot, instead of having it baked into
`MODEL.h`. The program stays fixed; only the model file changes. This matches the
real need: it is the **model** that changes between subjects, not the BLE /
preprocessing / inference **code**.

At the current MTU, ~52 KB ≈ **~215 packets**; transfer takes seconds. No
flash-partition surgery, no brick risk (unlike a full-firmware OTA).

### Three building blocks

| Block                                                                          | Role                                                                                  | Status                           |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------------- |
| **BLE WRITE characteristic** (`PROPERTY_WRITE` + `onWrite` callback) | receive the bytes, packet by packet                                                   | to add (firmware is notify-only) |
| **LittleFS**                                                             | store the model as a file (`/model.bin`), survives power-off                        | to add                           |
| **Preferences (NVS)**                                                    | store small metadata (gesture set, n_classes, "model present" flag, version/checksum) | optional                         |

**Why LittleFS and not Preferences/NVS for the model:** NVS/EEPROM on ESP32 is tiny
(~2 KB effective) and meant for small key/values — it **cannot hold ~51 KB**. The
community consensus (and Espressif maintainers) is to use **LittleFS** for anything
file-sized. Preferences is still handy for the few bytes of metadata, but the model
itself must go to LittleFS. (You *could* keep metadata in a LittleFS file too and skip
Preferences entirely — one less library to learn.)

### What has to change in the firmware

- Add a `PROPERTY_WRITE` characteristic + `onWrite` callback that accumulates
  incoming packets into a buffer.
- Add LittleFS: write the reassembled bytes to `/model.bin`, verify the checksum.
- Refactor inference to **load the weights from `/model.bin` at boot** instead of from
  the compiled `MODEL.h`.
- Plus a PC/phone-side sender that chunks the file and pushes it over BLE.

---

## Decisions to make

1. **Write mode — with vs without response.**

   - *Write with response*: the bracelet ACKs every packet. Reliable, no lost
     packets, but slower. **Recommended** — simplicity over speed for a one-off model
     update.
   - *Write without response*: 4–6× faster (30–100 KB/s achievable), but no ACK — you
     must build your own reliability layer (sequence numbers, retransmit). Only worth
     it if update speed becomes a problem.
2. **Chunking protocol.** BLE sends small packets, so the model must be split and
   reassembled. Include a small header (total size, number of chunks), **numbered
   chunks**, and a **final checksum**. Without sequence numbers and a checksum, a
   silently corrupted transfer would degrade the model with no error shown.
3. **MTU.** The client (PC/phone) must **request** a high MTU on connect; otherwise
   BLE falls back to 20 usable bytes/packet. The firmware already sets 247.

---

## Three rules so the transferred model equals the deployed one

Get one wrong and the model silently degrades:

- **Keep float32.** Do not quantize to save space; that changes the outputs.
- **Same weight order/layout.** The firmware must read the weights back in exactly the
  layout it expects; any offset → garbage predictions.
- **Send means + stds + weights as one coherent bundle.** Normalization is paired with
  the model — a new model with old means/stds (or vice-versa) breaks inference. Version
  them together and verify the checksum before activating.

---

## Effort & honesty

Feasibility is not in question — BLE works, the MTU is set, and 52 KB transfers in
seconds. The real work is the **protocol** (chunking, checksum, reassembly, boot-time
loading) plus refactoring inference to read the model from a file. This is a genuine
firmware task, not a configuration change, and is scoped here as future work.

## Open questions / uncertainties

- Whether the inference firmware already exposes any write characteristic (unverified;
  the versions seen are notify-only). Check the hybrid sketch before starting.
- Exact LittleFS partition sizing vs the current ~53% flash usage — needs a custom
  partition table, to be validated on the actual board.
- Whether write-with-response speed is acceptable in practice, or write-without-response
  + a reliability layer is needed — decide after a first prototype.
