# User Guide — EMG Bracelet Gesture Recognition

How to use the EMG bracelet and app to record gestures, train a personalized
model, and recognize hand gestures in real time. No technical background needed.

---

## What it does

The bracelet reads the electrical activity of your forearm muscles through 8
sensors and learns to recognize your hand gestures. You record a few examples of
each gesture, the app trains a model that is personalized to you, and the
bracelet then recognizes those gestures live.

Recognized gestures: **rest, flexion, extension, close (fist), supination,
pronation**.

---

## Before you start

You need:

- the ESP32-S3 EMG bracelet,
- a computer with the app installed (see `README.md`, or `WINDOWS_SETUP.md` on
  Windows),
- Bluetooth enabled.

**Bracelet placement** (important — bad placement gives bad recognition):

- On the forearm, about **3 fingers below the elbow**.
- The marker facing **up, toward your palm**.
- **Snug** — it should not slide.

A placement photo is shown on the app's home screen.

[TO CONFIRM: how to power on / charge the bracelet.]

---

## Step by step

Start the app and follow the phases:

1. **Install capture firmware (Phase 0)** — prepares the bracelet to stream its
   raw signal. One click; wait for it to finish.
2. **Calibration (Phase 1)** — rest your arm for a few seconds so the app
   measures your baseline (your "resting" signal).
3. **Record gestures (Phase 2)** — for each gesture shown (with an animated
   demonstration), perform the movement while it records. Repeat for all
   gestures. The app then trains your personalized model automatically.
4. **Install the model (Phase 3)** — loads your trained model onto the bracelet.
5. **Verify (Phase 4)** — a quick check that the model is installed.
6. **Live use (Phase 5)** — the bracelet recognizes your gestures in real time.

---

## Tips for good recordings

- Make sure the sensors are in firm contact with the skin: snug bracelet, clean
  and dry skin.
- Watch the on-screen demonstration and mimic it; hold the gesture steadily
  while it records.
- Do the gestures the same way you intend to use them later.

If the app shows a **signal-quality warning**:

- **"electrode contact"** — a sensor isn't touching well. Re-seat the bracelet
  and try the take again.
- **"weak signal / old firmware"** — the bracelet is running outdated capture
  firmware. Ask your technical contact to reflash the corrected firmware.

---

## Troubleshooting & FAQ

- **The bracelet doesn't connect.** Check Bluetooth is on and the bracelet is
  powered. On Windows, pair the bracelet first (see `WINDOWS_SETUP.md`).
- **Recognition is poor during live use.** Re-check placement and skin contact,
  then re-record the gestures with a cleaner signal. The model is personal — if
  the bracelet is worn differently than during recording, accuracy drops.
- **A gesture is often confused with another.** Re-record that gesture with
  clearer, more distinct movements.
- [TO CONFIRM: support / contact information for clients.]
- [TO CONFIRM: battery life / charging instructions.]

---

*For installation, maintenance, and how the system works internally, see
`DEVELOPER_GUIDE.md`.*
