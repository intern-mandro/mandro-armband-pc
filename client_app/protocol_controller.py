"""Protocol controller for the EMG recorder.

Generates a randomized sequence of gestures, each separated by a pause step.
Drives the logger label via set_label() and emits a Qt signal when the displayed
label should change in the UI.

Example session (4 classes):
    Shuffle [rest, flexion, extension, close] -> [flexion, rest, close, extension]

    Sequence emitted (10s gesture, 5s pause):
        ("flexion",   10000)
        ("pause",      5000)
        ("rest",      10000)
        ("pause",      5000)
        ("close",     10000)
        ("pause",      5000)
        ("extension", 10000)
"""

import random
import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


# Legacy fallback for int callers. The dashboard passes an explicit gesture
# list (resolved from lib.configs at runtime), so the active set is the source
# of truth; this dict only covers callers that still pass n_classes as an int.
GESTURES_BY_N_CLASSES = {
    4: ["rest", "flexion", "extension", "close"],
    6: ["rest", "flexion", "extension", "close", "supination", "pronation"],
}

GESTURE_DURATION_MS = 10_000   # 10 s per gesture
PAUSE_DURATION_MS = 5_000      # 5 s pause between gestures
TICK_INTERVAL_MS = 200         # how often to check progress (5 Hz, smooth countdown)


class ProtocolController(QObject):
    """Drives a randomized gesture sequence and updates the logger label.

    Signals:
        sig_step_changed(label, remaining_ms_in_step, step_idx, total_steps)
        sig_session_done()
        sig_session_aborted()
    """

    sig_step_changed = pyqtSignal(str, int, int, int)
    sig_session_done = pyqtSignal()
    sig_session_aborted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sequence = []          # list of (label, duration_ms)
        self.step_idx = -1
        self.step_start_ts = None
        self.logger = None
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_INTERVAL_MS)
        self.timer.timeout.connect(self._on_tick)
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────

    def build_sequence(self, gestures):
        """Generate a randomized [gesture, pause, gesture, pause, ...] sequence.

        `gestures` may be an explicit list of gesture names, or an int
        (legacy: number of classes, resolved via GESTURES_BY_N_CLASSES).
        """
        if isinstance(gestures, int):
            if gestures not in GESTURES_BY_N_CLASSES:
                raise ValueError(f"Unsupported n_classes={gestures}")
            gestures = GESTURES_BY_N_CLASSES[gestures]
        gestures = list(gestures)
        random.shuffle(gestures)
        seq = [("pause", PAUSE_DURATION_MS)]  # initial pause to let user prepare
        for i, g in enumerate(gestures):
            seq.append((g, GESTURE_DURATION_MS))
            if i < len(gestures) - 1:
                seq.append(("pause", PAUSE_DURATION_MS))
        self.sequence = seq
        return seq

    def start(self, logger, gestures):
        """Start a new protocol session.

        `gestures` may be a list of gesture names or an int (legacy n_classes).
        """
        self.logger = logger
        self.build_sequence(gestures)
        self.step_idx = 0
        self.step_start_ts = time.perf_counter()
        self._running = True

        first_label, first_dur = self.sequence[0]
        if self.logger is not None:
            self.logger.set_label(first_label)
        self.sig_step_changed.emit(first_label, first_dur, 0, len(self.sequence))
        self.timer.start()

    def stop(self):
        """Abort the session early."""
        if not self._running:
            return
        self._running = False
        self.timer.stop()
        if self.logger is not None:
            self.logger.set_label("none")
        self.sig_session_aborted.emit()

    def is_running(self):
        return self._running

    def total_duration_ms(self):
        return sum(d for _, d in self.sequence)

    # ── Internal ──────────────────────────────────────────────────────

    def _on_tick(self):
        if not self._running or self.step_idx >= len(self.sequence):
            return

        label, duration = self.sequence[self.step_idx]
        elapsed_ms = (time.perf_counter() - self.step_start_ts) * 1000
        remaining_ms = max(0, int(duration - elapsed_ms))

        # Still in current step
        if elapsed_ms < duration:
            self.sig_step_changed.emit(label, remaining_ms, self.step_idx, len(self.sequence))
            return

        # Advance to next step
        self.step_idx += 1
        if self.step_idx >= len(self.sequence):
            # Session finished
            self._running = False
            self.timer.stop()
            if self.logger is not None:
                self.logger.set_label("none")
            self.sig_session_done.emit()
            return

        self.step_start_ts = time.perf_counter()
        new_label, new_duration = self.sequence[self.step_idx]
        if self.logger is not None:
            self.logger.set_label(new_label)
        self.sig_step_changed.emit(new_label, new_duration, self.step_idx, len(self.sequence))
