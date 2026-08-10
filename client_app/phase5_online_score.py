"""
phase5_online_score.py
======================
Phase 5 - Online score verification.

Guides the user through a sequence of gestures (takes x gestures) while the
bracelet streams its own predictions over BLE, then scores those predictions
against the expected gesture and saves a CSV.

This is the in-app, BLE equivalent of scripts/capture_live_predictions.py:
the reference script reads a Teensy over serial; here the bracelet (flashed
with the user's model) sends predictions over BLE on the prediction
characteristic, exactly like Phase 4. The protocol structure and the CSV
output format are kept identical to the script so the captures stay
compatible with the online_prediction_assessment notebook.

Output: <project_root>/benchmarks/live_<label>.csv
Columns: Timestamp, Prediction, Action, Take
"""

import asyncio
import csv
import os
import re
import time
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QFrame,
)
from PyQt6.QtCore import Qt, QThread, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImageReader

import sessions_registry

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    BleakScanner = None
    BleakClient = None


DEVICE_NAME = "ESP32S3_FAST_BLE"

import ble_selection
CHAR_UUID_PRED = "abcd1234-5678-1234-5678-abcdef123457"

GESTURES_4CL = ["rest", "flexion", "extension", "close"]
GESTURES_6CL = ["rest", "flexion", "extension", "close", "supination", "pronation"]

# Same mapping as scripts/capture_live_predictions.py so Prediction/Action ints align.
ACTION_TO_INT = {
    "rest": 0, "flexion": 1, "extension": 2,
    "close": 3, "supination": 4, "pronation": 5,
}
INT_TO_ACTION = {v: k for k, v in ACTION_TO_INT.items()}


class LivePredictionWorker(QThread):
    """Scans, connects to the bracelet over BLE and streams predicted gesture
    names. Timing/recording is handled on the UI side; this worker only feeds
    predictions so the interface stays responsive."""

    prediction_received = pyqtSignal(str)
    status = pyqtSignal(str)
    connected = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        if BleakScanner is None:
            self.failed.emit("bleak is not installed")
            return
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.failed.emit(str(exc))

    async def _run_async(self):
        selected = ble_selection.get_selected()
        if selected:
            target_address = selected
        else:
            target = None
            for attempt in range(1, 4):
                self.status.emit(f"Scanning for bracelet... (attempt {attempt}/3)")
                try:
                    devices = await BleakScanner.discover(timeout=5.0)
                except Exception as exc:
                    self.failed.emit(f"BLE scan failed: {exc}")
                    return
                target = next((d for d in devices if d.name == DEVICE_NAME), None)
                if target is not None:
                    break
                if attempt < 3:
                    await asyncio.sleep(1.0)   # let the ESP32 release the link
            if target is None:
                self.failed.emit(
                    f"Bracelet '{DEVICE_NAME}' not found after 3 attempts")
                return
            target_address = target.address

        self.status.emit(f"Connecting to {target_address}...")

        def on_notify(_sender, data):
            try:
                text = bytes(data).decode("utf-8", errors="replace")
            except Exception:
                return
            if text.strip():
                self.prediction_received.emit(text)

        try:
            async with BleakClient(target_address) as client:
                has_pred = any(
                    ch.uuid.lower() == CHAR_UUID_PRED.lower()
                    for s in client.services for ch in s.characteristics
                )
                if not has_pred:
                    self.failed.emit(
                        "The bracelet has no prediction output. It is running the "
                        "capture firmware - flash a trained model first (Phase 3), "
                        "then retry."
                    )
                    return
                await client.start_notify(CHAR_UUID_PRED, on_notify)
                self.connected.emit()
                self.status.emit("Connected - recording predictions")
                while not self._stop_requested:
                    await asyncio.sleep(0.05)
                try:
                    await client.stop_notify(CHAR_UUID_PRED)
                except Exception:
                    pass
        except Exception as exc:
            msg = str(exc)
            if "characteristic" in msg.lower() and "not found" in msg.lower():
                self.failed.emit(
                    "The bracelet has no prediction output - flash a trained "
                    "model first (Phase 3), then retry."
                )
            else:
                self.failed.emit(f"BLE connection failed: {exc}")


class Phase5OnlineScore(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.subject_id = None
        self.session_row_index = None  # Set by AppWindow from Phase 2

        # Protocol parameters (defaults mirror the reference script).
        self.n_classes = 6
        self.n_takes = 3
        self.duration_ms = 5000
        self.pause_ms = 2000
        self.ready_ms = 3000

        # Runtime state.
        self._steps = []
        self._step_idx = -1
        self._step_start = 0.0
        self._cur_step = None
        self._recording = False
        self._records = []
        self._session_start = 0.0
        self._win_count = 0
        self._win_correct = 0

        # Gesture set resolved at _start (from the session, else the combo).
        self._gesture_set = None
        self._gestures = GESTURES_6CL
        self._action_to_int = dict(ACTION_TO_INT)
        self._int_to_action = dict(INT_TO_ACTION)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()

    def set_subject_id(self, subject_id):
        """Set the subject ID from AppWindow after registration."""
        self.subject_id = subject_id
        print(f"Phase 5 received subject_id: {subject_id}")
    
    def set_session_row_index(self, row_index):
        """Set the session row index from AppWindow (passed from Phase 2)."""
        self.session_row_index = row_index
        print(f"Phase 5 received session_row_index: {row_index}")
        if hasattr(self, "set_display"):
            self._refresh_set_display()

    def _refresh_set_display(self):
        """Show the resolved scoring set and hide the now-redundant 6/4 combo
        when a model/session is linked."""
        gset = None
        if self.session_row_index is not None:
            try:
                gset = sessions_registry.get_gesture_set(self.session_row_index)
            except Exception:
                gset = None
        if gset:
            self.set_display.setText(
                f"Scoring set: {gset.upper()}  (from the loaded model)")
            self.set_display.show()
            self._classes_wrap.hide()
        else:
            self.set_display.hide()
            self._classes_wrap.show()

    # ── Gesture-set resolution (set follows the model, not the env) ──────
    def _cfg(self):
        try:
            from lib import configs as c
            return c
        except Exception:
            try:
                import configs as c
                return c
            except Exception:
                return None

    def _resolve_gesture_set(self):
        gset = None
        if self.session_row_index is not None:
            try:
                gset = sessions_registry.get_gesture_set(self.session_row_index)
            except Exception:
                gset = None
        if not gset:
            gset = "6cl" if self.classes_combo.currentIndex() == 0 else "4cl"
        return gset

    def _gestures_for(self, gset):
        c = self._cfg()
        if c is not None:
            try:
                return list(c.gestures_for(gset))
            except Exception:
                pass
        return {"4cl": list(GESTURES_4CL), "6cl": list(GESTURES_6CL),
                "rps": ["idle", "rock", "paper", "scissors"]}.get(
                    gset, list(GESTURES_6CL))

    def _resolve_subject(self):
        if self.subject_id:
            return re.sub(r"\s+", "_", str(self.subject_id).strip())
        if self.session_row_index is not None:
            try:
                sid = sessions_registry.get_subject_id(self.session_row_index)
                if sid:
                    return re.sub(r"\s+", "_", str(sid).strip())
            except Exception:
                pass
        lbl = re.sub(r"\s+", "_", self.label_edit.text().strip())
        return lbl or None

    @staticmethod
    def _macro_f1(records):
        """Macro-averaged F1 over the target classes (real F1, not accuracy)."""
        classes = sorted({r["Action"] for r in records if r["Action"] is not None})
        if not classes:
            return 0.0
        scores = []
        for c in classes:
            tp = sum(1 for r in records if r["Action"] == c and r["Prediction"] == c)
            fp = sum(1 for r in records if r["Action"] != c and r["Prediction"] == c)
            fn = sum(1 for r in records if r["Action"] == c and r["Prediction"] != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return sum(scores) / len(scores)

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(18)

        title = QLabel("Check your online score")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        info = QLabel(
            "Put the bracelet on and turn it on. We'll guide you through each\n"
            "gesture; the bracelet predicts live and we score it against what\n"
            "you were asked to do.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; padding:18px; color:#c5cce0;"
            " font-size:13px; line-height:1.6; }")
        outer.addWidget(info)

        self.set_display = QLabel("")
        self.set_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_display.setStyleSheet(
            "QLabel { color:#2ed573; font-size:13px; font-weight:700;"
            " border:none; }")
        self.set_display.hide()
        outer.addWidget(self.set_display)

        outer.addWidget(self._build_settings_card())

        self.choose_button = QPushButton("Choose bracelet")
        self.choose_button.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-weight:700;"
            " border:1px solid #2a3550; border-radius:8px; padding:8px; }"
            "QPushButton:hover { border:1px solid #3d4d75; }")
        self.choose_button.clicked.connect(self._choose_bracelet)
        outer.addWidget(self.choose_button)

        self.toggle_button = QPushButton("Start verification")
        self.toggle_button.setMinimumHeight(50)
        self.toggle_button.setStyleSheet(self._btn_style(False))
        self.toggle_button.clicked.connect(self._on_toggle)
        outer.addWidget(self.toggle_button)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color:#9fb3ff; font-size:32px; font-weight:800; border:none;")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.gesture_anim = QLabel("")
        self.gesture_anim.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_anim.setStyleSheet("border:none;")
        outer.addWidget(self.gesture_anim, stretch=1)

        self.gesture_label = QLabel("--")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_label.setStyleSheet(self._gesture_style("#2ed573"))
        outer.addWidget(self.gesture_label)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "QLabel { background:#16203a; border-left:3px solid #2ed573;"
            " color:#e6e9f2; font-size:13px; padding:14px 18px; }")
        self.result_label.hide()
        outer.addWidget(self.result_label)

        self._refresh_set_display()

    def _build_settings_card(self):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; }")
        row = QHBoxLayout(card)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(16)

        self.classes_combo = QComboBox()
        self.classes_combo.addItems(["6 gestures", "4 gestures"])
        self._classes_wrap = self._labeled("Classes", self.classes_combo)
        row.addWidget(self._classes_wrap)

        self.takes_spin = QSpinBox()
        self.takes_spin.setRange(1, 20)
        self.takes_spin.setValue(self.n_takes)
        row.addWidget(self._labeled("Takes", self.takes_spin))

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(self.duration_ms // 1000)
        self.duration_spin.setSuffix(" s")
        row.addWidget(self._labeled("Per gesture", self.duration_spin))

        self.pause_spin = QSpinBox()
        self.pause_spin.setRange(0, 30)
        self.pause_spin.setValue(self.pause_ms // 1000)
        self.pause_spin.setSuffix(" s")
        row.addWidget(self._labeled("Pause", self.pause_spin))

        self.label_edit = QLineEdit("session")
        row.addWidget(self._labeled("Label", self.label_edit))

        return card

    def _labeled(self, text, widget):
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(text)
        cap.setStyleSheet("color:#9fb3ff; font-size:11px; border:none;")
        lay.addWidget(cap)
        lay.addWidget(widget)
        return wrap

    def _btn_style(self, running):
        color = "#ff6b6b" if running else "#2ed573"
        fg = "#1a0a0a" if running else "#0a1020"
        return (f"QPushButton {{ background:{color}; color:{fg};"
                " font-size:15px; font-weight:800; border:none;"
                " border-radius:10px; padding:12px; }")

    def _gesture_style(self, color):
        return ("QLabel { background:#1a2236; border:1px solid #2a3550;"
                f" border-radius:12px; padding:36px; color:{color};"
                " font-size:44px; font-weight:800; }")

    def _set_settings_enabled(self, enabled):
        for w in (self.classes_combo, self.takes_spin,
                  self.duration_spin, self.pause_spin, self.label_edit):
            w.setEnabled(enabled)

    # ── Start / stop ────────────────────────────────────────────────────

    def _choose_bracelet(self):
        from ble_selection import BraceletSelectorDialog
        BraceletSelectorDialog(self).exec()

    def _on_toggle(self):
        if self.worker is None:
            self._start()
        else:
            self._abort()

    def _start(self):
        self._gesture_set = self._resolve_gesture_set()
        self._gestures = self._gestures_for(self._gesture_set)
        self.n_classes = len(self._gestures)
        self._action_to_int = {g: i for i, g in enumerate(self._gestures)}
        self._int_to_action = {i: g for i, g in enumerate(self._gestures)}
        self.n_takes = self.takes_spin.value()
        self.duration_ms = self.duration_spin.value() * 1000
        self.pause_ms = self.pause_spin.value() * 1000

        self._records = []
        self._step_idx = -1
        self._recording = False
        self.result_label.hide()
        self._set_settings_enabled(False)

        self.status_label.setText("Starting BLE scan...")
        self.gesture_label.setText("--")
        self.gesture_label.setStyleSheet(self._gesture_style("#9fb3ff"))

        self.worker = LivePredictionWorker()
        self.worker.prediction_received.connect(self._on_prediction)
        self.worker.status.connect(self.status_label.setText)
        self.worker.connected.connect(self._begin_protocol)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self.toggle_button.setText("Stop")
        self.toggle_button.setStyleSheet(self._btn_style(True))

    def _abort(self):
        self._timer.stop()
        self._recording = False
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self._reset_button()
        self._set_settings_enabled(True)
        self.status_label.setText("Stopped")
        self.gesture_label.setText("--")
        self.gesture_label.setStyleSheet(self._gesture_style("#9fb3ff"))

    def _reset_button(self):
        self._stop_gesture_frames()
        self.toggle_button.setText("Start verification")
        self.toggle_button.setStyleSheet(self._btn_style(False))

    def _on_failed(self, message):
        self._timer.stop()
        self._recording = False
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self._reset_button()
        self._set_settings_enabled(True)
        self.status_label.setText(f"Failed: {message}")
        self.gesture_label.setText("--")
        self.gesture_label.setStyleSheet(self._gesture_style("#ff6b6b"))

    # ── Protocol state machine ──────────────────────────────────────────

    def _begin_protocol(self):
        self._steps = self._build_steps()
        self._session_start = time.time()
        self._step_idx = 0
        self._enter_step()
        self._timer.start()

    def _build_steps(self):
        gestures = self._gestures
        a2i = self._action_to_int
        steps = []
        for take in range(1, self.n_takes + 1):
            for g in gestures:
                steps.append({"kind": "ready", "gesture": g,
                              "action": a2i[g], "ms": self.ready_ms, "take": take})
                steps.append({"kind": "record", "gesture": g,
                              "action": a2i[g], "ms": self.duration_ms, "take": take})
                is_last = (take == self.n_takes and g == gestures[-1])
                if not is_last:
                    steps.append({"kind": "pause", "gesture": None,
                                  "action": None, "ms": self.pause_ms, "take": take})
        return steps

    def _enter_step(self):
        step = self._steps[self._step_idx]
        self._cur_step = step
        self._step_start = time.perf_counter()
        self._recording = (step["kind"] == "record")
        if self._recording:
            self._win_count = 0
            self._win_correct = 0
        self._render_step(step, step["ms"] // 1000)

    def _on_tick(self):
        if self._cur_step is None:
            return
        step = self._cur_step
        elapsed_ms = (time.perf_counter() - self._step_start) * 1000
        remaining_s = max(0, int((step["ms"] - elapsed_ms + 999) // 1000))
        self._render_step(step, remaining_s)

        if elapsed_ms >= step["ms"]:
            self._step_idx += 1
            if self._step_idx >= len(self._steps):
                self._finish()
            else:
                self._enter_step()

    def _load_gesture_frames(self, label):
        cache = getattr(self, "_frames_cache", None)
        if cache is None:
            cache = {}
            self._frames_cache = cache
        if label in cache:
            return cache[label]
        base = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base, "assets", "gestures", "frames", label)
        frames = []
        if os.path.isdir(folder):
            names = sorted(f for f in os.listdir(folder)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            MAX_FRAMES = 24
            if len(names) > MAX_FRAMES:
                stride = len(names) / MAX_FRAMES
                names = [names[int(i * stride)] for i in range(MAX_FRAMES)]
            TARGET_H = 240
            for nm in names:
                reader = QImageReader(os.path.join(folder, nm))
                sz = reader.size()
                if sz.isValid() and sz.height() > TARGET_H:
                    w = max(1, int(sz.width() * TARGET_H / sz.height()))
                    reader.setScaledSize(QSize(w, TARGET_H))
                img = reader.read()
                if not img.isNull():
                    frames.append(QPixmap.fromImage(img))
        cache[label] = frames
        return frames

    def _play_gesture_frames(self, label):
        label = str(label).lower().strip()
        if getattr(self, "_anim_label", None) == label and \
                getattr(self, "_anim_timer", None) is not None and \
                self._anim_timer.isActive():
            return
        frames = self._load_gesture_frames(label)
        if not frames:
            self._stop_gesture_frames()
            self.gesture_anim.setPixmap(QPixmap())
            return
        self._anim_frames = frames
        self._anim_idx = 0
        self._anim_label = label
        self.gesture_anim.setPixmap(frames[0].scaledToHeight(
            220, Qt.TransformationMode.SmoothTransformation))
        if getattr(self, "_anim_timer", None) is None:
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._advance_gesture_frame)
        self._anim_timer.start(83)

    def _advance_gesture_frame(self):
        frames = getattr(self, "_anim_frames", None)
        if not frames:
            return
        self._anim_idx = (self._anim_idx + 1) % len(frames)
        self.gesture_anim.setPixmap(frames[self._anim_idx].scaledToHeight(
            220, Qt.TransformationMode.SmoothTransformation))

    def _stop_gesture_frames(self):
        t = getattr(self, "_anim_timer", None)
        if t is not None:
            t.stop()
        self._anim_label = None
        self.gesture_anim.setPixmap(QPixmap())

    def _render_step(self, step, remaining_s):
        kind = step["kind"]
        take = step["take"]
        if kind == "ready":
            self.gesture_label.setText(step["gesture"].upper())
            self.gesture_label.setStyleSheet(self._gesture_style("#ffdd59"))
            self._play_gesture_frames(step["gesture"])
            self.status_label.setText(
                f"Take {take}/{self.n_takes}  -  get ready  -  {remaining_s}s")
        elif kind == "record":
            self.gesture_label.setText(step["gesture"].upper())
            self.gesture_label.setStyleSheet(self._gesture_style("#2ed573"))
            self._play_gesture_frames(step["gesture"])
            acc = (self._win_correct / self._win_count * 100) if self._win_count else 0.0
            self.status_label.setText(
                f"Take {take}/{self.n_takes}  -  RECORDING  -  {remaining_s}s"
                f"  -  {self._win_count} preds ({acc:.0f}% match)")
        else:  # pause
            self.gesture_label.setText("REST")
            self.gesture_label.setStyleSheet(self._gesture_style("#9fb3ff"))
            self._play_gesture_frames("rest")
            self.status_label.setText(f"Pause  -  {remaining_s}s")

    # ── Prediction intake ───────────────────────────────────────────────

    def _on_prediction(self, text):
        if not self._recording or self._cur_step is None:
            return
        parts = text.split("|")
        raw = parts[0].strip() if parts else ""
        pred_int = -1
        # Prefer the model's own logits (argmax over the active set) over the
        # gesture name baked into the firmware (which is a fixed 6cl table).
        try:
            logits = [float(x) for x in parts[1:] if x.strip() != ""]
            logits = logits[:len(self._gestures)]
            if logits:
                pred_int = max(range(len(logits)), key=lambda k: logits[k])
        except Exception:
            pred_int = -1
        if pred_int < 0:
            pred_int = self._action_to_int.get(raw.lower(), -1)
            if pred_int < 0:
                pred_int = ACTION_TO_INT.get(raw.lower(), -1)
        record = {
            "Timestamp": round(time.time() - self._session_start, 4),
            "Prediction": pred_int,
            "Action": self._cur_step["action"],
            "Take": self._cur_step["take"],
        }
        self._records.append(record)
        self._win_count += 1
        if pred_int == self._cur_step["action"]:
            self._win_correct += 1

    # ── Finish ──────────────────────────────────────────────────────────

    def _finish(self):
        self._timer.stop()
        self._recording = False
        self._cur_step = None
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self._reset_button()
        self._set_settings_enabled(True)
        self.gesture_label.setText("Done")
        self.gesture_label.setStyleSheet(self._gesture_style("#2ed573"))

        out_path = self._save_csv()
        
        # Calculate and save online metrics to Excel
        if self.session_row_index is not None:
            self._save_online_metrics()
        
        self._show_results(out_path)
    
    def _save_online_metrics(self):
        """Calculate accuracy and save to sessions.xlsx."""
        if not self._records:
            print("No records to save")
            return
        
        try:
            total = len(self._records)
            correct = sum(1 for r in self._records if r["Prediction"] == r["Action"])
            acc_online = correct / total if total > 0 else 0.0
            
            f1_online = self._macro_f1(self._records)
            
            # Update the session record with online metrics using row_index
            sessions_registry.update_online_metrics(
                self.session_row_index,
                acc_online=acc_online,
                f1_online=f1_online
            )
            print(f"Updated online metrics at row {self.session_row_index}: accuracy={acc_online:.3f}")
        except Exception as e:
            print(f"Warning: Could not save online metrics: {e}")

    def _save_csv(self):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        out_dir = Path(project_root) / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)

        subj = self._resolve_subject() or "unknown"
        gset = getattr(self, "_gesture_set", None) or "set"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"live_{subj}_{gset}_{stamp}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Timestamp", "Prediction", "Action", "Take"])
            writer.writeheader()
            writer.writerows(self._records)
        return out_path

    def _show_results(self, out_path):
        if not self._records:
            self.status_label.setText("Finished - no predictions received")
            self.result_label.setStyleSheet(
                "QLabel { background:#2a1a1a; border-left:3px solid #ff6b6b;"
                " color:#ffd6d6; font-size:13px; padding:14px 18px; }")
            self.result_label.setText(
                "No predictions were received over BLE during the recording "
                "windows. Check that the bracelet is flashed and advertising.")
            self.result_label.show()
            return

        total = len(self._records)
        correct = sum(1 for r in self._records if r["Prediction"] == r["Action"])
        overall = correct / total * 100

        # Per-gesture accuracy.
        per = {}
        for r in self._records:
            a = r["Action"]
            per.setdefault(a, [0, 0])
            per[a][1] += 1
            if r["Prediction"] == a:
                per[a][0] += 1

        lines = [f"Overall live accuracy (raw, no smoothing): {overall:.1f}%",
                 f"{correct}/{total} predictions matched",
                 f"Macro-F1: {self._macro_f1(self._records):.3f}",
                 f"Gesture set: {getattr(self, '_gesture_set', '?')}", ""]
        for a in sorted(per):
            c, n = per[a]
            name = self._int_to_action.get(a, str(a))
            pct = c / n * 100 if n else 0.0
            lines.append(f"  {name:<11}: {pct:5.1f}%   ({c}/{n})")
        lines.append("")
        lines.append(f"Saved: {out_path}")

        self.status_label.setText("Finished")
        self.result_label.setStyleSheet(
            "QLabel { background:#16203a; border-left:3px solid #2ed573;"
            " color:#e6e9f2; font-family:monospace; font-size:13px;"
            " padding:14px 18px; }")
        self.result_label.setText("\n".join(lines))
        self.result_label.show()