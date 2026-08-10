"""
app_window.py
=============
Main window of the client application.

Hosts the 5 phase frames in a stack. Phase 1 embeds the existing dashboard
(EMGDashboard from dashboard_ui.py) and now lets the user pick the recording
profile/batch before capturing. Phases 2-5 follow.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QScrollArea,
    QComboBox, QLineEdit, QSpinBox, QMessageBox,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import os
import re

import numpy as np

import config
import subjects_registry
import sessions_registry
from dashboard_ui import EMGDashboard
from phase0_install_capture import Phase0InstallCapture
from phase2_training import Phase2Training
from phase3_install import Phase3Install
from phase4_verify import Phase4Verify
from phase5_online_score import Phase5OnlineScore
from take_viewer import TakeViewerDialog


PHASE_NAMES = [
    "0 - Install capture firmware",
    "1 - Capture",
    "2 - Training",
    "3 - Installation",
    "4 - Verification",
    "5 - Online score",
]

PHASE_SHORT = ["Firmware", "Capture", "Training", "Install", "Verify", "Score"]
PHASE_INTROS = [
    "Flash the bracelet with the capture firmware so it can stream raw EMG.",
    "Wear the bracelet and record several takes of each gesture.",
    "Train a model on the recorded data and check its offline accuracy.",
    "Export the trained model and flash it onto the bracelet.",
    "Wear the bracelet and watch your gestures recognised live, wirelessly.",
    "Run a scored protocol to measure how well the model works online.",
]

# A channel whose recent peak-to-peak stays below this (raw units) for
# SENSOR_DISCONNECT_TICKS consecutive checks (~0.5 s each) is flagged
# "not connected" (flat / no signal).
SENSOR_DISCONNECT_P2P = 2.0
SENSOR_DISCONNECT_TICKS = 3

# Live noise heuristic (tune these on real data).
NOISE_WINDOW_SAMPLES = 600   # recent samples inspected for clipping
CLIP_NEAR = 2                # within this of a rail (0 / 255) = clipped
CLIP_WARN_FRAC = 0.02
CLIP_BAD_FRAC = 0.05
REST_P2P_WARN = 12.0         # p2p during a pause above this = noisy
REST_P2P_BAD = 25.0
NOISE_BAD_TICKS = 2          # consecutive checks before showing red


def make_placeholder(title, subtitle):
    w = QWidget()
    layout = QVBoxLayout(w)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    t = QLabel(title)
    t.setAlignment(Qt.AlignmentFlag.AlignCenter)
    s = QLabel(subtitle)
    s.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(t)
    layout.addWidget(s)
    return w


class AppWindow(QMainWindow):

    NEW_PROFILE_LABEL = "+ New profile..."

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Bracelet - Build your model")
        self.resize(1200, 800)
        self._sensor_low_ticks = [0] * 8
        self._noise_bad_ticks = 0
        self.current_subject_id = None      # set when subject is registered in Phase 1
        self.current_subject_name = None    # for display purposes
        self.current_n_classes = 6          # default to 6 classes
        self.current_session_row_index = None  # set by Phase 2, used by Phase 5

        # Calibration baseline (populated after calibration screen)
        self.baseline_mean = None   # shape (N_CH,)
        self.baseline_std  = None   # shape (N_CH,)
        self._calibration_ok = False

        # Session take loop state (10-takes workflow)
        self.session_n_takes = 10
        self.session_current_take = 0
        self.session_completed_csvs = []
        self._session_active = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── Header: stepper + context banner + per-phase intro ────────────
        header_box = QVBoxLayout()
        header_box.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self._step_chips = []
        for i, name in enumerate(PHASE_SHORT):
            chip = QLabel(f"{i} {name}")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._step_chips.append(chip)
            top_row.addWidget(chip)
            if i < len(PHASE_SHORT) - 1:
                sep = QLabel("›")
                sep.setStyleSheet("color:#3d4d75; border:none;")
                top_row.addWidget(sep)
        top_row.addStretch(1)
        self.ble_label = QLabel()
        self.ble_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.ble_label.setTextFormat(Qt.TextFormat.RichText)
        top_row.addWidget(self.ble_label)
        header_box.addLayout(top_row)

        self.context_label = QLabel("")
        self.context_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.context_label.setStyleSheet(
            "color:#9fb3ff; font-size:12px; font-weight:700; border:none;")
        header_box.addWidget(self.context_label)

        self.intro_label = QLabel("")
        self.intro_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.intro_label.setWordWrap(True)
        self.intro_label.setStyleSheet(
            "color:#c5cce0; font-size:13px; border:none;")
        header_box.addWidget(self.intro_label)

        root.addLayout(header_box)

        # ── Phase stack ───────────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.phase0 = Phase0InstallCapture()
        self.stack.addWidget(self.phase0)
        self.dashboard = EMGDashboard()
        self.dashboard.panel_settings.hide()
        self.dashboard.panel_raw.hide()
        self.dashboard.panel_pwr.hide()
        # The hidden raw/pwr column still reserved 2/3 of the dashboard width
        # via its layout stretch. Hand all the width to the left column (the
        # diagonal-vector panel) so it fills the available space.
        try:
            _dash_main = self.dashboard.centralWidget().layout()
            if _dash_main is not None and _dash_main.count() >= 2:
                _dash_main.setStretch(0, 1)
                _dash_main.setStretch(1, 0)
        except Exception:
            pass
        self.dashboard.protocol.sig_session_done.connect(self._on_recording_finished)
        self.phase1 = self._build_phase1(self.dashboard)
        self.stack.addWidget(self.phase1)
        self.phase2 = Phase2Training()
        self.phase3 = Phase3Install()
        self.phase4 = Phase4Verify()
        self.phase5 = Phase5OnlineScore()
        self.stack.addWidget(self.phase2)
        self.stack.addWidget(self.phase3)
        self.stack.addWidget(self.phase4)
        self.stack.addWidget(self.phase5)
        # Scrollable phase area: if a screen is taller than the window, it
        # scrolls instead of pushing the Next/Previous bar off-screen.
        self._phase_scroll = QScrollArea()
        self._phase_scroll.setWidgetResizable(True)
        self._phase_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._phase_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._phase_scroll.setWidget(self.stack)
        root.addWidget(self._phase_scroll, stretch=1)

        # ── Navigation buttons ────────────────────────────────────────────
        nav = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.go_previous)
        self.next_button.clicked.connect(self.go_next)
        nav.addWidget(self.prev_button)
        nav.addStretch(1)
        nav.addWidget(self.next_button)
        root.addLayout(nav)

        self.update_navigation()

        # ── BLE status polling timer ──────────────────────────────────────
        self._refresh_ble_status()
        self._ble_timer = QTimer(self)
        self._ble_timer.timeout.connect(self._refresh_ble_status)
        self._ble_timer.start(500)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #121826;
                color: #e6e9f2;
                font-family: -apple-system, "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QLabel { color: #e6e9f2; }
            QPushButton {
                background: #2a3550;
                color: #ffffff;
                padding: 10px 18px;
                border-radius: 8px;
                font-weight: 700;
            }
            QPushButton:hover { background: #3d4d75; }
            QPushButton:disabled {
                background: #1c2538;
                color: #4d5d7e;
            }
            QComboBox, QLineEdit, QSpinBox {
                background: #121826;
                color: #e6e9f2;
                border: 1px solid #2a3550;
                border-radius: 6px;
                padding: 6px 8px;
            }
        """)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1 sub-stack
    #   screen 0 = welcome / placement instructions + recording profile
    #   screen 1 = calibration (baseline noise measurement)
    #   screen 2 = capture (start button, protocol label, EMG dashboard)
    # ──────────────────────────────────────────────────────────────────────

    def _build_phase1(self, dashboard):
        self.phase1_stack = QStackedWidget()
        self.phase1_stack.addWidget(self._build_welcome_screen())       # 0
        self.phase1_stack.addWidget(self._build_calibration_screen())   # 1
        self.phase1_stack.addWidget(self._build_capture_screen(dashboard))  # 2
        return self.phase1_stack

    def _image_placeholder(self, text):
        frame = QLabel(text)
        frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setMinimumHeight(120)
        frame.setStyleSheet("color:#888; border:1px dashed #888;")
        return frame

    def _card(self, title, body_widget):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)
        head = QLabel(title)
        head.setStyleSheet(
            "font-size:15px; font-weight:800; color:#9fb3ff; border:none;")
        lay.addWidget(head)
        lay.addWidget(body_widget)
        return card

    # ── Screen 0: Welcome ─────────────────────────────────────────────────

    def _build_welcome_screen(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(20)

        title = QLabel("Let's build your personalized model")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(20)


        # Gestures card
        gestures_inner = QWidget()
        gi = QVBoxLayout(gestures_inner)
        gi.setContentsMargins(0, 0, 0, 0)
        gi.setSpacing(12)

        hint = QLabel("Tap a gesture to see how to perform it:")
        hint.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        gi.addWidget(hint)

        buttons_row = QWidget()
        br = QHBoxLayout(buttons_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(6)
        self._gesture_buttons = {}
        self._gesture_btn_row = br
        gi.addWidget(buttons_row)

        self.gesture_image = QLabel()
        self.gesture_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_image.setMinimumHeight(140)
        self.gesture_image.setStyleSheet(
            "color:#5f6b8a; border:1px dashed #3d4d75; border-radius:8px;")
        gi.addWidget(self.gesture_image)

        columns.addWidget(self._card("Gestures to record", gestures_inner))
        placement_inner = QWidget()
        pl = QVBoxLayout(placement_inner)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(8)
        placement_hint = QLabel("Put the bracelet on as shown:")
        placement_hint.setStyleSheet(
            "color:#c5cce0; font-size:13px; border:none;")
        pl.addWidget(placement_hint)
        placement_img = QLabel()
        placement_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placement_img.setStyleSheet(
            "border:1px dashed #3d4d75; border-radius:8px;")
        _pp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "placement.png")
        if os.path.exists(_pp):
            placement_img.setPixmap(QPixmap(_pp).scaledToHeight(
                260, Qt.TransformationMode.SmoothTransformation))
        else:
            placement_img.setText("[ placement.png missing ]")
        pl.addWidget(placement_img)
        columns.addWidget(self._card("Bracelet placement", placement_inner))
        self._rebuild_gesture_buttons(
            ["rest", "flexion", "extension", "close", "supination", "pronation"])

        outer.addLayout(columns)

        tips = QLabel(
            "Tips for a good capture:   relax fully between gestures  -  "
            "keep each repetition identical  -  steady medium intensity")
        tips.setWordWrap(True)
        tips.setStyleSheet(
            "background:#16203a; color:#c5cce0; font-size:13px;"
            " border-left:3px solid #2ed573; padding:14px 18px;")
        outer.addWidget(tips)

        outer.addWidget(self._card("Recording profile", self._build_profile_inner()))

        ready_button = QPushButton("I'm ready - continue")
        ready_button.setMinimumHeight(50)
        ready_button.setStyleSheet(
            "background:#2ed573; color:#0a1020; font-size:15px;"
            " font-weight:800; border-radius:10px; padding:12px 40px;")
        ready_button.clicked.connect(self._on_ready_clicked)
        outer.addWidget(ready_button)

        return page

    # ── Recording profile selector ────────────────────────────────────────

    def _build_profile_inner(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        row.addWidget(self._labeled_field("Profile", self.profile_combo), stretch=2)

        self.new_profile_edit = QLineEdit()
        self.new_profile_edit.setPlaceholderText("New profile name")
        self.new_profile_edit.textChanged.connect(self._update_profile_preview)
        self._new_profile_field = self._labeled_field("New name", self.new_profile_edit)
        row.addWidget(self._new_profile_field, stretch=2)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 99)
        self.batch_spin.valueChanged.connect(self._update_profile_preview)
        row.addWidget(self._labeled_field("Batch", self.batch_spin), stretch=1)

        lay.addLayout(row)

        # Gesture set: which gestures the capture protocol will record
        set_row = QHBoxLayout()
        set_row.setSpacing(12)
        set_row.addWidget(QLabel("Gesture set:"))
        self.rb_set_6cl = QRadioButton("6 gestures")
        self.rb_set_4cl = QRadioButton("4 gestures")
        self.rb_set_rps = QRadioButton("Rock-Paper-Scissors")
        self.rb_set_6cl.setChecked(True)
        self._gesture_set_group = QButtonGroup(self)
        self._gesture_set_group.addButton(self.rb_set_6cl)
        self._gesture_set_group.addButton(self.rb_set_4cl)
        self._gesture_set_group.addButton(self.rb_set_rps)
        self._gesture_set_group.buttonClicked.connect(self._on_gesture_set_changed)
        set_row.addWidget(self.rb_set_6cl)
        set_row.addWidget(self.rb_set_4cl)
        set_row.addWidget(self.rb_set_rps)
        set_row.addStretch()
        lay.addLayout(set_row)

        # Subject name/surname collection removed for anonymization.
        # Stub attributes kept so external references stay valid.
        self._subject_fields_wrapper = QWidget()
        self._subject_fields_wrapper.setVisible(False)
        self.subject_name_edit = QLineEdit()
        self.subject_surname_edit = QLineEdit()
        lay.addWidget(self._subject_fields_wrapper)

        self.profile_preview = QLabel()
        self.profile_preview.setStyleSheet(
            "color:#5f6b8a; font-size:12px; border:none;")
        lay.addWidget(self.profile_preview)

        self._refresh_profiles()
        return inner

    def _labeled_field(self, caption, widget):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        cap = QLabel(caption)
        cap.setStyleSheet("color:#9fb3ff; font-size:11px; border:none;")
        v.addWidget(cap)
        v.addWidget(widget)
        return wrap

    def _data_dir(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "data")

    def _scan_profiles(self):
        """Read registered subjects from subjects.xlsx instead of scanning data/ folders."""
        try:
            import pandas as pd
        except ImportError:
            return []
        subjects_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "subjects.xlsx")
        if not os.path.exists(subjects_file):
            return []
        try:
            df = pd.read_excel(subjects_file, dtype={"subject_id": str}).fillna("")
            subjects = []
            for _, row in df.iterrows():
                sid = str(row.get("subject_id", "")).strip()
                name = str(row.get("name", "")).strip()
                surname = str(row.get("surname", "")).strip()
                if sid:
                    display = sid
                    subjects.append(display)
            return sorted(subjects)
        except Exception as e:
            print(f"Error reading subjects.xlsx: {e}")
            return []

    def _refresh_profiles(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        try:
            import pandas as pd
            subjects_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "subjects.xlsx")
            if os.path.exists(subjects_file):
                df = pd.read_excel(subjects_file, dtype={"subject_id": str}).fillna("")
                for _, row in df.iterrows():
                    sid = str(row.get("subject_id", "")).strip()
                    name = str(row.get("name", "")).strip()
                    surname = str(row.get("surname", "")).strip()
                    if sid:
                        display = sid
                        self.profile_combo.addItem(display, sid)
        except Exception as e:
            print(f"Warning: Could not load subjects: {e}")
        self.profile_combo.addItem(self.NEW_PROFILE_LABEL, None)
        self.profile_combo.setCurrentIndex(0)
        self.profile_combo.blockSignals(False)
        self._on_profile_combo_changed()

    def _suggest_batch(self, profile):
        """Return the next available batch number for a given profile."""
        data_dir = self._data_dir()
        max_b = 0
        if profile and os.path.isdir(data_dir):
            for name in os.listdir(data_dir):
                m = re.match(rf"^{re.escape(profile)}_BATCH(\d+)$", name)
                if m:
                    max_b = max(max_b, int(m.group(1)))
        return min(99, max_b + 1) if max_b else 1

    def _on_profile_combo_changed(self, *_):
        is_new = self.profile_combo.currentData() is None
        self._new_profile_field.setVisible(is_new)
        self._subject_fields_wrapper.setVisible(is_new)
        if not is_new:
            self.batch_spin.blockSignals(True)
            self.batch_spin.setValue(self._suggest_batch(self.profile_combo.currentData()))
            self.batch_spin.blockSignals(False)
        else:
            self.batch_spin.blockSignals(True)
            self.batch_spin.setValue(1)
            self.batch_spin.blockSignals(False)
        self._update_profile_preview()

    def _resolve_capture_profile(self):
        """Return (profile_name, batch) or (None, None) if not set."""
        if self.profile_combo.currentData() is None:
            name = self.new_profile_edit.text()
        else:
            name = self.profile_combo.currentData()
        name = re.sub(r"\s+", "_", (name or "").strip())
        if not name:
            return None, None
        return name, self.batch_spin.value()

    def _update_profile_preview(self, *_):
        name, batch = self._resolve_capture_profile()
        if not name:
            self.profile_preview.setText(
                "Choose an existing subject or enter a new one before recording.")
            self.profile_preview.setStyleSheet(
                "color:#5f6b8a; font-size:12px; border:none;")
            return

        is_new = self.profile_combo.currentData() is None
        subject_info = (
            f"NEW: {name}" if is_new
            else f"EXISTING: {self.profile_combo.currentText()}"
        )
        target_rel = f"data/{name}_BATCH{batch}/"
        target_abs = os.path.join(self._data_dir(), f"{name}_BATCH{batch}")

        # Check whether the target folder exists and how many takes it
        # already holds. Show a warning when the user is about to write
        # into a populated folder.
        line1 = f"{subject_info}  ->  {target_rel}"
        warning = ""
        warning_color = "#5f6b8a"  # neutral by default

        if os.path.isdir(target_abs):
            try:
                csv_files = [n for n in os.listdir(target_abs)
                             if n.lower().endswith(".csv")]
            except OSError:
                csv_files = []
            n_csv = len(csv_files)
            if n_csv > 0:
                warning = (
                    f"⚠ WARNING: this folder already exists "
                    f"({n_csv} take{'s' if n_csv != 1 else ''} inside) "
                    f"— new recordings will be added next to them."
                )
                warning_color = "#ffa726"   # orange
            else:
                warning = "Folder already exists but is empty."
                warning_color = "#9fb3ff"   # info blue

        if warning:
            self.profile_preview.setText(f"{line1}\n{warning}")
            self.profile_preview.setStyleSheet(
                f"color:{warning_color}; font-size:12px; border:none;")
        else:
            self.profile_preview.setText(line1)
            self.profile_preview.setStyleSheet(
                "color:#5f6b8a; font-size:12px; border:none;")

    def _on_ready_clicked(self):
        """Validate profile and subject info, then move to the calibration screen."""
        name, batch = self._resolve_capture_profile()
        if not name:
            QMessageBox.warning(
                self, "Profile required",
                "Choose an existing subject or type a new name before continuing.")
            return

        is_new_subject = self.profile_combo.currentData() is None

        if is_new_subject:
            try:
                self.current_subject_id = subjects_registry.add_subject()
                self.current_subject_name = self.current_subject_id
                print(f"Registered NEW subject: {self.current_subject_id}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving subject",
                    f"Could not register subject: {e}")
                return
        else:
            self.current_subject_id = self.profile_combo.currentData()
            self.current_subject_name = self.profile_combo.currentText()
            print(f"Using existing subject: {self.current_subject_id} - {self.current_subject_name}")

        # Forward subject id to downstream phases
        self.phase2.set_subject_id(self.current_subject_id)
        self.phase5.set_subject_id(self.current_subject_id)

        self.dashboard.set_capture_profile(self.current_subject_id, batch)
        self.phase1_stack.setCurrentIndex(1)   # → calibration screen

    def _gesture_btn_style(self, active):
        if active:
            return ("QPushButton { background:#e6e9f2; color:#121826;"
                    " font-size:12px; font-weight:800;"
                    " border:1px solid #e6e9f2; border-radius:15px;"
                    " padding:7px 18px; }")
        return ("QPushButton { background:#2a3550; color:#e6e9f2;"
                " font-size:12px; font-weight:600;"
                " border:1px solid #2a3550; border-radius:15px;"
                " padding:7px 18px; }")

    def _rebuild_gesture_buttons(self, gestures):
        """Rebuild the welcome-screen gesture preview to show only `gestures`."""
        row = getattr(self, "_gesture_btn_row", None)
        if row is None:
            return
        while row.count():
            w = row.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self._gesture_buttons = {}
        for g in gestures:
            btn = QPushButton(g.capitalize())
            btn.setCheckable(True)
            btn.setStyleSheet(self._gesture_btn_style(False))
            btn.clicked.connect(lambda _, name=g: self._show_gesture(name))
            row.addWidget(btn)
            self._gesture_buttons[g] = btn
        if gestures:
            self._show_gesture(gestures[0])

    def _on_gesture_set_changed(self, *_):
        """When the gesture-set radio changes, preview only that set's gestures."""
        gset = self._selected_gesture_set()
        try:
            from lib.configs import GESTURE_SETS
        except Exception:
            try:
                from configs import GESTURE_SETS
            except Exception:
                GESTURE_SETS = {
                    "4cl": ["rest", "flexion", "extension", "close"],
                    "6cl": ["rest", "flexion", "extension", "close",
                            "supination", "pronation"],
                    "rps": ["idle", "rock", "paper", "scissors"]}
        self._rebuild_gesture_buttons(list(GESTURE_SETS.get(gset, [])))

    def _show_gesture(self, name):
        for g, btn in self._gesture_buttons.items():
            is_active = (g == name)
            btn.setChecked(is_active)
            btn.setStyleSheet(self._gesture_btn_style(is_active))
        if self._anim_play(self.gesture_image, name, 130):
            return
        base = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base, "assets", "gestures", name + ".png")
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.gesture_image.setPixmap(
                pix.scaledToHeight(130, Qt.TransformationMode.SmoothTransformation))
            self.gesture_image.setText("")
        else:
            self.gesture_image.setPixmap(QPixmap())
            self.gesture_image.setText(f"[ image: {name} - to add ]")

    # ── Screen 1: Calibration ─────────────────────────────────────────────

    def _build_calibration_screen(self):
        """Build the baseline noise calibration screen.

        The user rests their arm for 5 s while raw samples are collected.
        Per-channel mean and std are stored as baseline_mean / baseline_std
        on AppWindow for use by downstream phases.
        Noisy channels are flagged using an adaptive threshold (> 2 × median std).
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 28, 40, 28)
        lay.setSpacing(16)

        title = QLabel("Calibration — resting signal baseline")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:20px; font-weight:800; color:#ffffff; border:none;")
        lay.addWidget(title)

        instructions = QLabel(
            "Rest your arm on the table and relax your hand completely.\n"
            "Press 'Start calibration' and stay still for 5 seconds.")
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        lay.addWidget(instructions)

        # Sensor contact pills
        pills_frame = QFrame()
        pills_frame.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:10px; }")
        pf_lay = QVBoxLayout(pills_frame)
        pf_lay.setContentsMargins(16, 12, 16, 12)
        pf_lay.setSpacing(8)
        pf_head = QLabel("Sensor contact")
        pf_head.setStyleSheet(
            "font-size:13px; font-weight:800; color:#9fb3ff; border:none;")
        pf_lay.addWidget(pf_head)
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)
        self.calib_pills = []
        for i in range(8):
            pill = QLabel(f"CH{i}")
            pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pill.setStyleSheet(self._pill_style("neutral", config.get_ch_color(i)))
            pills_row.addWidget(pill)
            self.calib_pills.append(pill)
        pills_row.addStretch(1)
        pf_lay.addLayout(pills_row)
        lay.addWidget(pills_frame)

        # Progress label shown during measurement
        self.calib_progress_label = QLabel("")
        self.calib_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calib_progress_label.setStyleSheet(
            "color:#9fb3ff; font-size:13px; border:none;")
        lay.addWidget(self.calib_progress_label)

        # Per-channel results (hidden until calibration completes)
        self.calib_results_frame = QFrame()
        self.calib_results_frame.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:10px; }")
        cr_lay = QVBoxLayout(self.calib_results_frame)
        cr_lay.setContentsMargins(16, 12, 16, 12)
        cr_head = QLabel("Measured baseline noise per channel")
        cr_head.setStyleSheet(
            "font-size:13px; font-weight:800; color:#9fb3ff; border:none;")
        cr_lay.addWidget(cr_head)
        self.calib_channel_labels = []
        for i in range(8):
            lbl = QLabel(f"CH{i}: —")
            lbl.setStyleSheet("color:#5f6b8a; font-size:12px; border:none;")
            cr_lay.addWidget(lbl)
            self.calib_channel_labels.append(lbl)
        self.calib_results_frame.setVisible(False)
        lay.addWidget(self.calib_results_frame)

        # Global status banner
        self.calib_status_banner = QLabel(
            "Connect the bracelet then start calibration.")
        self.calib_status_banner.setWordWrap(True)
        self.calib_status_banner.setStyleSheet(self._noise_style("neutral"))
        lay.addWidget(self.calib_status_banner)

        lay.addStretch(1)

        # Connect / disconnect button
        self.calib_connect_button = QPushButton("Connect bracelet")
        self.calib_connect_button.setMinimumHeight(42)
        self.calib_connect_button.setStyleSheet(self._connect_btn_style(False))
        self.calib_connect_button.clicked.connect(self._calib_toggle_connect)
        lay.addWidget(self.calib_connect_button)

        # Action buttons row
        btn_row = QHBoxLayout()

        self.calib_back_button = QPushButton("← Back")
        self.calib_back_button.clicked.connect(
            lambda: self.phase1_stack.setCurrentIndex(0))
        btn_row.addWidget(self.calib_back_button)
        btn_row.addStretch(1)

        self.calib_start_button = QPushButton("Start calibration")
        self.calib_start_button.setMinimumHeight(48)
        self.calib_start_button.setStyleSheet(
            "QPushButton { background:#1e90ff; color:#fff; font-size:15px;"
            " font-weight:800; border-radius:10px; padding:12px 40px; }"
            "QPushButton:hover { background:#3aa0ff; }"
            "QPushButton:disabled { background:#1c2538; color:#4d5d7e; }")
        self.calib_start_button.clicked.connect(self._start_calibration)
        btn_row.addWidget(self.calib_start_button)
        btn_row.addStretch(1)

        self.calib_next_button = QPushButton("Continue to capture →")
        self.calib_next_button.setMinimumHeight(48)
        self.calib_next_button.setEnabled(False)
        self.calib_next_button.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020; font-size:15px;"
            " font-weight:800; border-radius:10px; padding:12px 40px; }"
            "QPushButton:disabled { background:#1c2538; color:#4d5d7e; }")
        self.calib_next_button.clicked.connect(
            lambda: self.phase1_stack.setCurrentIndex(2))
        btn_row.addWidget(self.calib_next_button)

        lay.addLayout(btn_row)

        # Internal timer: fires every 500 ms to collect raw buffer snapshots
        self._calib_timer = QTimer(self)
        self._calib_timer.timeout.connect(self._calib_tick)
        self._calib_samples = []    # list of np.ndarray (N_CH, k)
        self._calib_ticks = 0
        self._calib_total_ticks = 10   # 10 × 500 ms = 5 s

        return page

    # ── Calibration logic ─────────────────────────────────────────────────

    def _start_calibration(self):
        """Start the 5-second baseline measurement."""
        dash = self.dashboard
        worker = getattr(dash, "worker", None)
        if not (worker and worker.isRunning()):
            QMessageBox.warning(
                self, "Bracelet not connected",
                "Connect the bracelet before running calibration.")
            return

        self._calib_samples = []
        self._calib_ticks = 0
        self._calibration_ok = False

        # Session take loop state (10-takes workflow)
        self.session_n_takes = 10
        self.session_current_take = 0
        self.session_completed_csvs = []
        self._session_active = False
        self.calib_results_frame.setVisible(False)
        self.calib_next_button.setEnabled(False)
        self.calib_start_button.setEnabled(False)
        self.calib_status_banner.setText("Calibration in progress — stay still…")
        self.calib_status_banner.setStyleSheet(self._noise_style("neutral"))
        self._calib_timer.start(500)

    def _calib_tick(self):
        """Called every 500 ms: snapshot the raw buffer and update UI."""
        # Collect the last 50 raw samples from the ring buffer (N_CH, k)
        win = self._recent_raw_window(50)
        if win is not None and win.shape[1] > 0:
            self._calib_samples.append(win.copy())

        self._calib_ticks += 1
        remaining = max(0, self._calib_total_ticks * 500 - self._calib_ticks * 500)
        self.calib_progress_label.setText(
            f"Measuring… {remaining // 1000 + 1} s remaining")

        # Refresh sensor contact pills
        self._refresh_calib_pills()

        if self._calib_ticks >= self._calib_total_ticks:
            self._calib_timer.stop()
            self._finish_calibration()

    def _refresh_calib_pills(self):
        """Update the per-channel contact pills on the calibration screen."""
        dash = self.dashboard
        amp = getattr(dash, "last_amp", None)
        n = config.N_CH
        for i, pill in enumerate(self.calib_pills):
            if i >= n:
                pill.setVisible(False)
                continue
            pill.setVisible(True)
            val = float(amp[i]) if amp is not None and i < len(amp) else 0.0
            if val < SENSOR_DISCONNECT_P2P:
                self._sensor_low_ticks[i] = min(
                    self._sensor_low_ticks[i] + 1, SENSOR_DISCONNECT_TICKS + 1)
            else:
                self._sensor_low_ticks[i] = 0
            bad = self._sensor_low_ticks[i] >= SENSOR_DISCONNECT_TICKS
            pill.setStyleSheet(self._pill_style("bad" if bad else "ok", config.get_ch_color(i)))

    def _finish_calibration(self):
        """Compute baseline statistics and display per-channel verdict."""
        self.calib_start_button.setEnabled(True)
        self.calib_progress_label.setText("")

        if not self._calib_samples:
            self.calib_status_banner.setText(
                "No data received. Check the connection and try again.")
            self.calib_status_banner.setStyleSheet(self._noise_style("bad"))
            return

        # Concatenate all (N_CH, k) snapshots → (N_CH, T_total)
        data = np.concatenate(self._calib_samples, axis=1)
        self.baseline_mean = data.mean(axis=1)   # (N_CH,)
        self.baseline_std  = data.std(axis=1)    # (N_CH,)

        # Adaptive threshold: a channel is noisy if its std > 2 × median std
        # This adjusts automatically to each person's signal level.
        median_std = np.median(self.baseline_std)
        noisy = self.baseline_std > 2 * median_std
        n_noisy = int(noisy.sum())

        # Show per-channel results
        self.calib_results_frame.setVisible(True)
        for i, lbl in enumerate(self.calib_channel_labels):
            if i >= config.N_CH:
                lbl.setVisible(False)
                continue
            m = self.baseline_mean[i]
            s = self.baseline_std[i]
            flag = " ⚠" if noisy[i] else " ✓"
            color = "#ff6b6b" if noisy[i] else "#2ed573"
            lbl.setText(f"CH{i}: mean={m:.1f}  noise(σ)={s:.2f}{flag}")
            lbl.setStyleSheet(f"color:{color}; font-size:12px; border:none;")

        if n_noisy == 0:
            self.calib_status_banner.setText(
                "✓ Calibration successful — clean signal on all channels.")
            self.calib_status_banner.setStyleSheet(self._noise_style("ok"))
            self._calibration_ok = True
            self.calib_next_button.setEnabled(True)
            self.calib_start_button.setText("Re-calibrate")
        elif n_noisy <= 2:
            self.calib_status_banner.setText(
                f"{n_noisy} channel(s) slightly noisy — you can continue "
                "or readjust the bracelet and re-calibrate.")
            self.calib_status_banner.setStyleSheet(self._noise_style("warn"))
            self._calibration_ok = True
            self.calib_next_button.setEnabled(True)
            self.calib_start_button.setText("Re-calibrate")
        else:
            self.calib_status_banner.setText(
                f"{n_noisy} noisy channels — readjust the bracelet and re-calibrate.")
            self.calib_status_banner.setStyleSheet(self._noise_style("bad"))
            self.calib_start_button.setText("Re-calibrate")

    def _calib_toggle_connect(self):
        """Connect or disconnect the bracelet from the calibration screen."""
        dash = self.dashboard
        worker = getattr(dash, "worker", None)
        if worker and worker.isRunning():
            dash.stop_serial()
        else:
            dash.start_preview()
        self._sync_calib_connect_button()

    def _sync_calib_connect_button(self):
        """Sync the calibration connect button label and colour."""
        if not hasattr(self, "calib_connect_button"):
            return
        worker = getattr(self.dashboard, "worker", None)
        connected = bool(worker and worker.isRunning())
        self.calib_connect_button.setText(
            "Disconnect bracelet" if connected else "Connect bracelet")
        self.calib_connect_button.setStyleSheet(self._connect_btn_style(connected))

    # ── Screen 2: Capture ─────────────────────────────────────────────────

    def _build_capture_screen(self, dashboard):
        page = QWidget()
        root = QVBoxLayout(page)

        # ── Top status strip (take counter, target hint, protocol label,
        #    noise banner). These are not part of the two-pane body but are
        #    kept here because they carry essential live status.
        self.session_take_label = QLabel("Take 0 / 10")
        self.session_take_label.setStyleSheet(
            "font-size:18px; font-weight:800; color:#ffffff;"
            " background:#1a2236; border:1px solid #2a3550;"
            " border-radius:8px; padding:8px 16px;")
        self.capture_target_label = QLabel("")
        self.capture_target_label.setStyleSheet(
            "color:#9fb3ff; font-size:12px; border:none;")

        status_top = QHBoxLayout()
        status_top.setSpacing(12)
        status_top.addWidget(self.session_take_label)
        status_top.addWidget(self.capture_target_label, stretch=1)
        root.addLayout(status_top)
        root.addWidget(self._build_noise_banner())

        # ── Two-pane body ──────────────────────────────────────────────
        split = QHBoxLayout()
        split.setSpacing(16)

        # Left pane: top-left header (current + time, next), image below
        left_pane = QVBoxLayout()
        left_pane.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        self.gesture_name_label = QLabel("\u2014")
        self.gesture_name_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.gesture_name_label.setStyleSheet(
            "font-size:40px; font-weight:900; color:#ffffff;"
            " letter-spacing:1px; border:none; padding:2px;")
        self.gesture_time_label = QLabel("")
        self.gesture_time_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.gesture_time_label.setStyleSheet(
            "font-size:22px; font-weight:800; color:#2ed573;"
            " border:none; padding:2px 8px;")
        top_row.addWidget(self.gesture_name_label)
        top_row.addWidget(self.gesture_time_label)
        top_row.addStretch(1)
        left_pane.addLayout(top_row)

        self.gesture_next_label = QLabel("")
        self.gesture_next_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.gesture_next_label.setStyleSheet(
            "font-size:18px; font-weight:700; color:#9fb3ff;"
            " border:none; padding:2px;")
        left_pane.addWidget(self.gesture_next_label)

        self.gesture_panel = QLabel("\u2014")
        self.gesture_panel.setMinimumWidth(220)
        self.gesture_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_panel.setStyleSheet("border:1px dashed #3d4d75;")
        left_pane.addWidget(self.gesture_panel, stretch=1)

        split.addLayout(left_pane, stretch=1)

        # Right pane: three stacked levels
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # Level 1: disconnect (connect toggle) next to start recording
        self.connect_button = QPushButton("Connect bracelet")
        self.connect_button.setMinimumHeight(46)
        self.connect_button.setStyleSheet(self._connect_btn_style(False))
        self.connect_button.clicked.connect(self._toggle_connect)

        self.start_recording_button = QPushButton("Start session (10 takes)")
        self.start_recording_button.setMinimumHeight(46)
        self.start_recording_button.setStyleSheet(self._recording_btn_style(False))
        self.start_recording_button.clicked.connect(self._start_recording)

        level1 = QHBoxLayout()
        level1.setSpacing(12)
        level1.addWidget(self.connect_button, stretch=1)
        level1.addWidget(self.start_recording_button, stretch=1)
        right_col.addLayout(level1)

        self.abort_take_button = QPushButton("ABORT TAKE")
        self.abort_take_button.setMinimumHeight(38)
        self.abort_take_button.setStyleSheet(
            "QPushButton { background:#2a1620; color:#ff8a9f; font-size:13px;"
            " font-weight:700; border:1px solid #502a35; border-radius:8px;"
            " padding:8px; } QPushButton:hover { border:1px solid #753d4d; }")
        self.abort_take_button.clicked.connect(self._abort_take)
        right_col.addWidget(self.abort_take_button)

        self.see_previous_button = QPushButton("SEE PREVIOUS TAKES")
        self.see_previous_button.setMinimumHeight(38)
        self.see_previous_button.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-size:13px;"
            " font-weight:700; border:1px solid #2a3550; border-radius:8px;"
            " padding:8px; } QPushButton:hover { border:1px solid #3d4d75; }")
        self.see_previous_button.clicked.connect(self.dashboard.see_previous_takes)
        right_col.addWidget(self.see_previous_button)

        # Level 2: channel (sensor contact) status
        right_col.addWidget(self._build_sensor_status())

        # Level 3: signal view in diagonal-vector format. The dashboard shows
        # only its diagonal-vector panel because the settings/raw/pwr panels
        # were hidden in __init__.
        right_col.addWidget(dashboard, stretch=1)

        split.addLayout(right_col, stretch=1)
        root.addLayout(split, stretch=1)

        self._capture_protocol = dashboard.protocol
        from PyQt6.QtCore import QTimer
        self._remaining_ms = 0
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(100)
        self._capture_timer.timeout.connect(self._tick_capture_time)
        # Hook the protocol step signal to update the gesture image
        for sig_name in ("sig_step_changed", "sig_protocol_step",
                         "sig_step", "sig_label_changed"):
            sig = getattr(dashboard.protocol, sig_name, None)
            if sig is not None:
                try:
                    sig.connect(self._on_protocol_step_capture)
                    break
                except Exception:
                    pass

        return page


    def _tick_capture_time(self):
        self._remaining_ms = max(0, self._remaining_ms - 100)
        if hasattr(self, "gesture_time_label"):
            self.gesture_time_label.setText(f"{self._remaining_ms/1000:.1f}s")
        if self._remaining_ms <= 0 and hasattr(self, "_capture_timer"):
            self._capture_timer.stop()

    def _load_gesture_frames(self, label):
        cache = getattr(self, "_gesture_frames_cache", None)
        if cache is None:
            cache = {}
            self._gesture_frames_cache = cache
        if label in cache:
            return cache[label]
        base = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base, "assets", "gestures", "frames", label)
        frames = []
        if os.path.isdir(folder):
            names = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
            # Cap the frame count and decode at a reduced height so this
            # synchronous load does not freeze the GUI thread at session start.
            MAX_FRAMES = 24
            if len(names) > MAX_FRAMES:
                stride = len(names) / MAX_FRAMES
                names = [names[int(i * stride)] for i in range(MAX_FRAMES)]
            from PyQt6.QtGui import QImageReader
            from PyQt6.QtCore import QSize
            TARGET_H = 240
            for n in names:
                reader = QImageReader(os.path.join(folder, n))
                sz = reader.size()
                if sz.isValid() and sz.height() > TARGET_H:
                    w = max(1, int(sz.width() * TARGET_H / sz.height()))
                    reader.setScaledSize(QSize(w, TARGET_H))
                img = reader.read()
                if not img.isNull():
                    frames.append(QPixmap.fromImage(img))
        cache[label] = frames
        return frames

    def _play_gesture_frames(self, label, height):
        label = str(label).lower().strip()
        frames = self._load_gesture_frames(label)
        if not frames:
            return False
        self._gesture_anim_height = int(height)
        if (getattr(self, "_gesture_anim_label", None) == label
                and getattr(self, "_gesture_anim_timer", None) is not None
                and self._gesture_anim_timer.isActive()):
            return True
        self._gesture_anim_frames = frames
        self._gesture_anim_idx = 0
        self._gesture_anim_label = label
        self.gesture_panel.setText("")
        self.gesture_panel.setStyleSheet("border:1px dashed #3d4d75;")
        self.gesture_panel.setPixmap(frames[0].scaledToHeight(
            self._gesture_anim_height,
            Qt.TransformationMode.SmoothTransformation))
        if getattr(self, "_gesture_anim_timer", None) is None:
            self._gesture_anim_timer = QTimer(self)
            self._gesture_anim_timer.timeout.connect(
                self._advance_gesture_frame)
        self._gesture_anim_timer.start(83)
        return True

    def _advance_gesture_frame(self):
        frames = getattr(self, "_gesture_anim_frames", None)
        if not frames:
            return
        self._gesture_anim_idx = (self._gesture_anim_idx + 1) % len(frames)
        self.gesture_panel.setPixmap(
            frames[self._gesture_anim_idx].scaledToHeight(
                getattr(self, "_gesture_anim_height", 220),
                Qt.TransformationMode.SmoothTransformation))

    def _stop_gesture_frames(self):
        t = getattr(self, "_gesture_anim_timer", None)
        if t is not None:
            t.stop()
        self._gesture_anim_label = None

    def _anim_play(self, widget, label, height, interval=83):
        label = str(label).lower().strip()
        frames = self._load_gesture_frames(label)
        if not frames:
            self._anim_stop(widget)
            return False
        store = getattr(self, "_anims", None)
        if store is None:
            store = {}
            self._anims = store
        key = id(widget)
        st = store.get(key)
        if st is not None and st["label"] == label and st["timer"].isActive():
            st["height"] = int(height)
            return True
        if st is None:
            timer = QTimer(widget)
            st = {"timer": timer, "frames": frames, "idx": 0,
                  "label": label, "height": int(height)}
            store[key] = st
            timer.timeout.connect(lambda w=widget: self._anim_tick(w))
        st["frames"] = frames
        st["idx"] = 0
        st["label"] = label
        st["height"] = int(height)
        widget.setText("")
        widget.setPixmap(frames[0].scaledToHeight(
            int(height), Qt.TransformationMode.SmoothTransformation))
        st["timer"].start(interval)
        return True

    def _anim_tick(self, widget):
        st = getattr(self, "_anims", {}).get(id(widget))
        if not st or not st["frames"]:
            return
        st["idx"] = (st["idx"] + 1) % len(st["frames"])
        widget.setPixmap(st["frames"][st["idx"]].scaledToHeight(
            st["height"], Qt.TransformationMode.SmoothTransformation))

    def _anim_stop(self, widget):
        st = getattr(self, "_anims", {}).get(id(widget))
        if st:
            st["timer"].stop()
            st["label"] = None

    def _on_protocol_step_capture(self, *args):
        """Update self.gesture_panel with the current gesture image.

        Called by the protocol's sig_step_changed signal. The first
        positional argument is the gesture label (e.g. "rest",
        "flexion", "pronation", or "pause" between gestures).
        """
        label = (args[0] if args else "")
        label = str(label).lower().strip()
        if not label:
            return

        remaining_ms = args[1] if len(args) > 1 else None
        if remaining_ms is not None and hasattr(self, "gesture_time_label"):
            try:
                self._remaining_ms = int(remaining_ms)
                self.gesture_time_label.setText(f"{self._remaining_ms/1000:.1f}s")
                if hasattr(self, "_capture_timer"):
                    self._capture_timer.start()
            except (TypeError, ValueError):
                pass

        step_idx = args[2] if len(args) > 2 else None
        nxt = None
        seq = getattr(getattr(self, "_capture_protocol", None), "sequence", None)
        if seq is not None and step_idx is not None and step_idx + 1 < len(seq):
            cand = str(seq[step_idx + 1][0]).lower().strip()
            if cand:
                nxt = cand
        if hasattr(self, "gesture_next_label"):
            self.gesture_next_label.setText(
                f"\u279c next: {nxt.upper()}" if nxt else "\u2014 last \u2014")
        # Pause: preview the upcoming gesture (animation, else static PNG)
        if label == "pause":
            self.gesture_name_label.setText(
                f"get ready \u279c {nxt.upper()}" if nxt else "pause")
            if nxt:
                if self._play_gesture_frames(nxt, 220):
                    return
                base = os.path.dirname(os.path.abspath(__file__))
                img_path = os.path.join(base, "assets", "gestures", nxt + ".png")
                if os.path.exists(img_path):
                    self._stop_gesture_frames()
                    self.gesture_panel.setPixmap(QPixmap(img_path).scaledToHeight(
                        220, Qt.TransformationMode.SmoothTransformation))
                    self.gesture_panel.setText("")
                    self.gesture_panel.setStyleSheet("border:1px dashed #3d4d75;")
                    return
            self._stop_gesture_frames()
            self.gesture_panel.setPixmap(QPixmap())
            self.gesture_panel.setText("— pause —")
            self.gesture_panel.setStyleSheet(
                "color:#9fb3ff; font-size:14px; font-weight:800;"
                " border:1px dashed #3d4d75;")
            return
        self.gesture_name_label.setText(label.upper())
        if self._play_gesture_frames(label, 220):
            return
        base = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base, "assets", "gestures", label + ".png")
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.gesture_panel.setPixmap(
                pix.scaledToHeight(220,
                    Qt.TransformationMode.SmoothTransformation))
            self.gesture_panel.setText("")
        else:
            self.gesture_panel.setPixmap(QPixmap())
            self.gesture_panel.setText(f"[ {label}.png missing ]")
            self.gesture_panel.setStyleSheet(
                "color:#ff6b6b; font-size:13px; border:1px dashed #3d4d75;")

    # ── Sensor contact widget (capture screen) ────────────────────────────

    def _build_sensor_status(self):
        wrap = QFrame()
        wrap.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:10px; }")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)

        head = QLabel("Sensor contact")
        head.setStyleSheet(
            "font-size:13px; font-weight:800; color:#9fb3ff; border:none;")
        lay.addWidget(head)

        pills = QHBoxLayout()
        pills.setSpacing(6)
        self.sensor_pills = []
        for i in range(8):
            pill = QLabel(f"CH{i}")
            pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pill.setStyleSheet(self._pill_style("neutral", config.get_ch_color(i)))
            pills.addWidget(pill)
            self.sensor_pills.append(pill)
        pills.addStretch(1)
        lay.addLayout(pills)

        self.sensor_status_msg = QLabel("Start recording to check sensor contact.")
        self.sensor_status_msg.setStyleSheet(self._msg_style("neutral"))
        lay.addWidget(self.sensor_status_msg)
        return wrap

    # ── Shared style helpers ──────────────────────────────────────────────

    def _pill_style(self, state, ch_color=None):
        colors = {
            "neutral": ("#222c44", "#5f6b8a"),
            "ok":      ("#16321f", "#2ed573"),
            "bad":     ("#3a1a1a", "#ff6b6b"),
        }
        bg, fg = colors[state]
        border = ch_color if ch_color is not None else fg
        bw = 2 if ch_color is not None else 1
        return (f"QLabel {{ background:{bg}; color:{fg}; border:{bw}px solid {border};"
                " border-radius:11px; padding:4px 10px; font-size:12px;"
                " font-weight:800; }")

    def _msg_style(self, state):
        color = {"neutral": "#9fb3ff", "ok": "#2ed573", "bad": "#ff6b6b"}[state]
        return f"color:{color}; font-size:12px; border:none;"

    def _noise_style(self, level):
        palette = {
            "neutral": ("#16203a", "#9fb3ff", "#2a3550"),
            "ok":      ("#16321f", "#2ed573", "#2ed573"),
            "warn":    ("#332a16", "#ffdd59", "#ffdd59"),
            "bad":     ("#3a1a1a", "#ff6b6b", "#ff6b6b"),
        }
        bg, fg, border = palette[level]
        return (f"QLabel {{ background:{bg}; color:{fg};"
                f" border-left:3px solid {border}; border-radius:8px;"
                " padding:10px 14px; font-size:13px; font-weight:700; }")

    def _recording_btn_style(self, recording):
        if recording:
            return ("QPushButton { background:#ff6b6b; color:#1a0a0a;"
                    " font-size:15px; font-weight:800; border:none;"
                    " border-radius:10px; padding:12px; }")
        return ("QPushButton { background:#2ed573; color:#0a1020;"
                " font-size:15px; font-weight:800; border:none;"
                " border-radius:10px; padding:12px; }")

    def _connect_btn_style(self, connected):
        color = "#ff6b6b" if connected else "#1e90ff"
        fg    = "#1a0a0a" if connected else "#ffffff"
        return (f"QPushButton {{ background:{color}; color:{fg};"
                " font-size:14px; font-weight:800; border:none;"
                " border-radius:10px; padding:10px; }")

    # ── Sensor / noise status helpers ─────────────────────────────────────

    def _refresh_sensor_status(self):
        if not hasattr(self, "sensor_pills"):
            return
        dash = self.dashboard
        n = config.N_CH
        worker = getattr(dash, "worker", None)
        streaming = (bool(worker and worker.isRunning())
                     and getattr(dash, "is_running", False))

        if not streaming:
            for i, pill in enumerate(self.sensor_pills):
                pill.setVisible(i < n)
                pill.setText(f"CH{i}")
                pill.setStyleSheet(self._pill_style("neutral", config.get_ch_color(i)))
            self.sensor_status_msg.setText(
                "Start recording to check sensor contact.")
            self.sensor_status_msg.setStyleSheet(self._msg_style("neutral"))
            self._sensor_low_ticks = [0] * 8
            return

        amp = getattr(dash, "last_amp", None)
        disconnected = []
        for i in range(8):
            pill = self.sensor_pills[i]
            if i >= n:
                pill.setVisible(False)
                continue
            pill.setVisible(True)
            val = float(amp[i]) if amp is not None and i < len(amp) else 0.0
            if val < SENSOR_DISCONNECT_P2P:
                self._sensor_low_ticks[i] += 1
            else:
                self._sensor_low_ticks[i] = 0
            bad = self._sensor_low_ticks[i] >= SENSOR_DISCONNECT_TICKS
            pill.setText(f"CH{i}")
            pill.setStyleSheet(self._pill_style("bad" if bad else "ok", config.get_ch_color(i)))
            if bad:
                disconnected.append(i)

        if disconnected:
            names = ", ".join(f"CH{i}" for i in disconnected)
            self.sensor_status_msg.setText(
                f"Not connected: {names} - check electrode contact.")
            self.sensor_status_msg.setStyleSheet(self._msg_style("bad"))
        else:
            self.sensor_status_msg.setText("All sensors connected.")
            self.sensor_status_msg.setStyleSheet(self._msg_style("ok"))

    def _build_noise_banner(self):
        self.noise_banner = QLabel(
            "Signal quality will appear here during recording.")
        self.noise_banner.setWordWrap(True)
        self.noise_banner.setStyleSheet(self._noise_style("neutral"))
        return self.noise_banner

    def _set_noise_banner(self, level, text):
        self.noise_banner.setText(text)
        self.noise_banner.setStyleSheet(self._noise_style(level))

    def _recent_raw_window(self, k):
        """Return the last k columns of the raw ring buffer, or None."""
        dash = self.dashboard
        buf  = getattr(dash, "raw_np_buf", None)
        maxd = getattr(dash, "max_display", 0)
        ptr  = getattr(dash, "ptr", 0)
        if buf is None or maxd <= 0:
            return None
        k = min(k, maxd)
        if getattr(dash, "is_buf_full", False):
            idx = (ptr - k + np.arange(k)) % maxd
            return buf[:, idx]
        if ptr <= 0:
            return None
        return buf[:, max(0, ptr - k):ptr]

    def _refresh_noise_status(self):
        if not hasattr(self, "noise_banner"):
            return
        dash = self.dashboard
        worker = getattr(dash, "worker", None)
        streaming = (bool(worker and worker.isRunning())
                     and getattr(dash, "is_running", False))
        if not streaming:
            self._set_noise_banner(
                "neutral", "Signal quality will appear here during recording.")
            self._noise_bad_ticks = 0
            return

        n = config.N_CH
        issues = []
        level = "ok"

        win = self._recent_raw_window(NOISE_WINDOW_SAMPLES)
        clip_frac = 0.0
        if win is not None and win.size:
            active  = win[:n]
            clipped = (active <= CLIP_NEAR) | (active >= 255 - CLIP_NEAR)
            clip_frac = float(clipped.mean())

        amp      = getattr(dash, "last_amp", None)
        rest_p2p = float(np.max(amp[:n])) if amp is not None and len(amp) >= n else 0.0
        logger   = getattr(dash, "csv_logger", None)
        label    = getattr(logger, "current_label", None) if logger else None
        in_rest  = label in ("pause", "rest")

        if clip_frac > CLIP_BAD_FRAC:
            level = "bad"
            issues.append("saturation")
        elif clip_frac > CLIP_WARN_FRAC:
            level = "warn"
            issues.append("some clipping")

        if in_rest and rest_p2p > REST_P2P_BAD:
            level = "bad"
            issues.append("high noise at rest")
        elif in_rest and rest_p2p > REST_P2P_WARN and level == "ok":
            level = "warn"
            issues.append("noise at rest")

        if level == "bad":
            self._noise_bad_ticks += 1
        else:
            self._noise_bad_ticks = 0
        # Require the bad condition to persist for NOISE_BAD_TICKS before turning red
        if level == "bad" and self._noise_bad_ticks < NOISE_BAD_TICKS:
            level = "warn"

        if level == "ok":
            self._set_noise_banner("ok", "Signal quality OK.")
        elif level == "warn":
            self._set_noise_banner("warn", "Noise detected: " + ", ".join(issues) + ".")
        else:
            self._set_noise_banner(
                "bad",
                "Heavy noise (" + ", ".join(issues)
                + ") - consider stopping and redoing this session.")

    # ── BLE connect / recording helpers ──────────────────────────────────

    def _toggle_connect(self):
        dash = self.dashboard
        worker = getattr(dash, "worker", None)
        if worker and worker.isRunning():
            dash.stop_serial()
        else:
            dash.start_preview()
        self._sync_connect_button()

    def _sync_connect_button(self):
        if not hasattr(self, "connect_button"):
            return
        worker = getattr(self.dashboard, "worker", None)
        connected = bool(worker and worker.isRunning())
        self.connect_button.setText(
            "Disconnect bracelet" if connected else "Connect bracelet")
        self.connect_button.setStyleSheet(self._connect_btn_style(connected))

    def _on_recording_finished(self):
        """Called by the dashboard when one take is done."""
        if self.dashboard.btn_protocol.isChecked():
            self.dashboard.btn_protocol.setChecked(False)
        QTimer.singleShot(300, self._show_last_take_with_choice)

    def _take_signal_problem(self, csv_path, min_ratio=1.5, min_gesture_abs=50.0):
        """Detect a bad take. Returns None if fine, else a dict with dead
        gestures (ratio < min_ratio) and a weak flag (mean gesture amplitude
        < min_gesture_abs = poor electrode coupling)."""
        try:
            import pandas as pd
            from lib.configs import LABELS, GESTURES_6CL
            df = pd.read_csv(csv_path)
            lc = None
            for c in df.columns:
                vals = {str(v).strip().lower() for v in df[c].dropna().unique()[:50]}
                if vals & set(GESTURES_6CL):
                    lc = c
                    break
            if lc is None:
                return None
            ch = df[LABELS].values.astype(float)
            lab = df[lc].astype(str).str.strip().str.lower().values
            def act(g):
                m = lab == g
                return float(ch[m].std(0).sum()) if m.any() else None
            rest = act("rest")
            if not rest:
                return None
            dead, gabs = [], []
            for g in GESTURES_6CL:
                if g == "rest":
                    continue
                a = act(g)
                if a is None:
                    continue
                gabs.append(a)
                if a / rest < min_ratio:
                    dead.append(g)
            mean_abs = sum(gabs) / len(gabs) if gabs else 0.0
            weak = mean_abs < min_gesture_abs
            if not dead and not weak:
                return None
            return {"dead": dead, "weak": weak, "mean_abs": mean_abs}
        except Exception:
            return None
    def _show_last_take_with_choice(self):
        """Open the take viewer; act on Continue or Redo."""
        path = getattr(self.dashboard, "last_csv_path", None)
        if path and os.path.exists(path) and self._session_active:
            self.session_completed_csvs.append(path)

        if not path or not os.path.exists(path):
            # Recover gracefully
            self._after_take_decision(continue_take=True, last_csv=None)
            return

        _prob = self._take_signal_problem(path)
        if _prob:
            _parts = []
            if _prob["dead"]:
                _parts.append(
                    "Almost no muscle activation detected for:\n"
                    f"   {', '.join(_prob['dead'])}\n"
                    "\u2192 Likely poor electrode contact on those channels "
                    "(skin prep / gel / bracelet position).")
            if _prob["weak"]:
                _parts.append(
                    "Overall signal amplitude is too low "
                    f"(mean {_prob['mean_abs']:.0f}, expected \u2265 50).\n"
                    "\u2192 The bracelet is most likely running the OLD 8-bit "
                    "capture firmware. Reflash the corrected (10-bit) firmware. "
                    "If the firmware is correct, check electrode contact.")
            QMessageBox.warning(
                self, "\u26a0 Signal quality on this take",
                "\n\n".join(_parts) + "\n\n"
                "Fix the cause and redo this take before continuing \u2014 "
                "otherwise the whole batch will be unusable.")
        dlg = TakeViewerDialog(path, self, baseline_std=self.baseline_std)
        result = dlg.exec()
        redo = (result == getattr(dlg, "RESULT_REDO",
                                  TakeViewerDialog.RESULT_REDO))
        self._after_take_decision(continue_take=not redo, last_csv=path)

    def _after_take_decision(self, continue_take, last_csv):
        """Apply the user's Continue/Redo choice."""
        if not self._session_active:
            # Take was outside a session loop (legacy single-take mode)
            self.start_recording_button.setEnabled(True)
            self.start_recording_button.setText("Start session (10 takes)")
            return

        if not continue_take:
            # Redo: delete the bad CSV, replay the same take number — but WAIT
            # for the user to press the button (do not relaunch automatically,
            # so the recording never starts by surprise).
            if last_csv and os.path.exists(last_csv):
                try:
                    os.remove(last_csv)
                except OSError as exc:
                    print(f"[warn] could not remove {last_csv}: {exc}")
            if last_csv in self.session_completed_csvs:
                self.session_completed_csvs.remove(last_csv)
            self.session_current_take -= 1
            self.start_recording_button.setEnabled(True)
            self.start_recording_button.setText(
                f"Next take ({self.session_current_take + 1}/"
                f"{self.session_n_takes})")
            return

        # Continue: are we at the end?
        if self.session_current_take >= self.session_n_takes:
            self._show_end_of_session()
            return

        # Wait for user to press the button for the next take
        self.start_recording_button.setEnabled(True)
        self.start_recording_button.setText(
            f"Next take ({self.session_current_take + 1}/"
            f"{self.session_n_takes})")

    def _abort_take(self):
        """Stop the current take immediately (mistake mid-recording), discard
        its partial CSV, and wait for the user to relaunch the same take number
        with the Next take button."""
        if not self.dashboard.protocol.is_running():
            return
        self.dashboard.protocol.stop()
        path = getattr(self.dashboard, "last_csv_path", None)
        self._after_take_decision(continue_take=False, last_csv=path)

    def _show_end_of_session(self):
        """All 10 takes done — congrats screen."""
        self._session_active = False
        name  = getattr(self.dashboard, "capture_profile", "?")
        batch = getattr(self.dashboard, "capture_batch", "?")
        n = len(self.session_completed_csvs)
        QMessageBox.information(
            self, "End of session",
            f"\u2713 {n} take(s) saved to:\n\n"
            f"data/{name}_BATCH{batch}/\n\n"
            "You can now move to Phase 2 to train your model.")
        self.session_current_take = 0
        self.session_take_label.setText("Take 0 / 10")
        self.start_recording_button.setEnabled(True)
        self.start_recording_button.setText("Start a new session")


    def _selected_gesture_set(self):
        """Gesture set chosen in the capture screen: '6cl', '4cl', or 'rps'."""
        if self.rb_set_rps.isChecked():
            return "rps"
        if self.rb_set_4cl.isChecked():
            return "4cl"
        return "6cl"

    def _start_recording(self):
        """Start a new 10-takes session or launch the next take within one."""
        if self._session_active:
            # In session: this click means 'launch the next take'
            self._launch_next_take()
            return
        name, batch = self._resolve_capture_profile()
        if not name:
            QMessageBox.warning(
                self, "Profile required",
                "Go back and choose or create a profile before recording.")
            return
        subject_id = getattr(self, "current_subject_id", None) or name
        self.dashboard.set_capture_profile(subject_id, batch)
        self.session_current_take = 0
        self.session_completed_csvs = []
        self._session_active = True
        self._session_gesture_set = self._selected_gesture_set()
        # Pre-warm preview frames for the whole set BEFORE the protocol/worker
        # start, so frame decoding never competes with the first gesture's
        # animation (which otherwise looks frozen at session start).
        try:
            from lib.configs import GESTURE_SETS
            for _g in GESTURE_SETS.get(self._session_gesture_set, []):
                self._load_gesture_frames(str(_g).lower().strip())
        except Exception:
            pass
        self._launch_next_take()

    def _launch_next_take(self):
        """Launch take N (increments the counter first)."""
        self.session_current_take += 1
        self.session_take_label.setText(
            f"Take {self.session_current_take} / {self.session_n_takes}")
        self.start_recording_button.setText(
            f"Take {self.session_current_take}/{self.session_n_takes} "
            "in progress\u2026")
        self.start_recording_button.setEnabled(False)
        gset = getattr(self, "_session_gesture_set", "6cl")
        if gset == "rps":
            self.dashboard.rb_rps.setChecked(True)
        elif gset == "4cl":
            self.dashboard.rb_4cl.setChecked(True)
        else:
            self.dashboard.rb_6cl.setChecked(True)
        if not self.dashboard.btn_protocol.isChecked():
            self.dashboard.btn_protocol.setChecked(True)
            self.dashboard.toggle_protocol()

    def _cancel_session(self):
        """User pressed the button while a session was active."""
        self._session_active = False
        if self.dashboard.btn_protocol.isChecked():
            self.dashboard.btn_protocol.setChecked(False)
            self.dashboard.toggle_protocol()
        self.session_current_take = 0
        self.session_take_label.setText("Take 0 / 10")
        self.start_recording_button.setText("Start session (10 takes)")
        self.start_recording_button.setEnabled(True)


    def _sync_recording_button(self):
        # While a 10-takes session is running, the session loop
        # owns the button text and state. Don't overwrite it.
        if getattr(self, "_session_active", False):
            return
        recording = self.dashboard.btn_protocol.isChecked()
        self.start_recording_button.setText(
            "Stop recording" if recording else "Start recording")
        self.start_recording_button.setStyleSheet(
            self._recording_btn_style(recording))
        if hasattr(self, "capture_target_label"):
            name  = getattr(self.dashboard, "capture_profile", None)
            batch = getattr(self.dashboard, "capture_batch", 1)
            if name:
                self.capture_target_label.setText(
                    f"Recording into  data/{name}_BATCH{batch}/")
            else:
                self.capture_target_label.setText("")

    # ── Phase navigation ──────────────────────────────────────────────────

    def go_previous(self):
        index = self.stack.currentIndex()
        if index > 0:
            self.stack.setCurrentIndex(index - 1)
            self.update_navigation()

    def go_next(self):
        index = self.stack.currentIndex()
        if index < self.stack.count() - 1:
            # When leaving Phase 2, capture the session_row_index for Phase 5
            if index == 2:
                self.current_session_row_index = self.phase2.session_row_index
                if self.current_session_row_index:
                    self.phase5.set_session_row_index(self.current_session_row_index)
                    self.phase4.set_session_row_index(self.current_session_row_index)
                if getattr(self.phase2, "last_model_path", None):
                    self.phase3.set_model_path(self.phase2.last_model_path)
                    print(f"[app_window] Phase 3 will flash: {self.phase2.last_model_path}")
            self.stack.setCurrentIndex(index + 1)
            self.update_navigation()

    def update_navigation(self):
        index = self.stack.currentIndex()
        last  = self.stack.count() - 1
        self._update_stepper(index)
        self.context_label.setText(self._context_text())
        self.intro_label.setText(
            PHASE_INTROS[index] if 0 <= index < len(PHASE_INTROS) else "")
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < last)

    def _update_stepper(self, current):
        for i, chip in enumerate(self._step_chips):
            if i == current:
                chip.setStyleSheet(
                    "QLabel { background:#2ed573; color:#0a1020; font-weight:800;"
                    " border-radius:8px; padding:4px 10px; font-size:12px; }")
            elif i < current:
                chip.setStyleSheet(
                    "QLabel { color:#2ed573; font-weight:700; padding:4px 10px;"
                    " font-size:12px; border:none; }")
            else:
                chip.setStyleSheet(
                    "QLabel { color:#5f6b8a; padding:4px 10px; font-size:12px;"
                    " border:none; }")

    def _context_text(self):
        parts = []
        sid = getattr(self, "current_subject_id", None)
        if sid:
            parts.append(f"Subject: {sid}")
        gset = getattr(self, "_session_gesture_set", None)
        if not gset and getattr(self, "current_session_row_index", None) is not None:
            try:
                import sessions_registry
                gset = sessions_registry.get_gesture_set(self.current_session_row_index)
            except Exception:
                gset = None
        if gset:
            parts.append(f"Set: {str(gset).upper()}")
        return "   ·   ".join(parts) if parts else "No subject selected yet"

    # ── BLE status polling (every 500 ms) ─────────────────────────────────

    def _refresh_ble_status(self):
        if hasattr(self, "start_recording_button"):
            self._sync_recording_button()
        self._sync_connect_button()
        self._sync_calib_connect_button()
        worker = getattr(self.dashboard, "worker", None)
        connected = bool(worker and worker.isRunning())
        if connected:
            self.ble_label.setText(
                "<span style='color:#2ed573;'>&#9679;</span> "
                "Bracelet: connected")
        else:
            self.ble_label.setText(
                "<span style='color:#ff4757;'>&#9679;</span> "
                "Bracelet: disconnected")
        self._refresh_sensor_status()
        self._refresh_noise_status()
    
    def _on_start_session(self):
        self.current_take = 1
        self.session_active = True
        self._start_take()

    def _start_take(self):
        self.take_active = True

        self.take_label.setText(f"Take {self.current_take} / {self.total_takes}")

        self.dashboard.set_capture_profile(...)
        self.dashboard.start_protocol()
    
    def _end_take(self):
        self.take_active = False
        self.dashboard.stop_protocol()

        self.take_label.setText("Take finished")
        self.next_button.setText("Next take")
    
    def _on_next_take(self):
        if self.current_take < self.total_takes:
            self.current_take += 1
            self._start_take()
        else:
            self._end_session()
    
    def _end_session(self):
        self._stop_gesture_frames()
        self.session_active = False

        self.take_label.setText("End of session")

        self.gesture_panel.setText(
            "All recordings saved successfully\n"
            f"Folder: data/{name}_BATCH{batch}/"
        )
    
    def _update_gesture_cue(self, gesture_name):
        if self._play_gesture_frames(str(gesture_name).lower().strip(), 140):
            return
        base = os.path.dirname(os.path.abspath(__file__))

        img_path = os.path.join(
            base,
            "assets",
            "gestures",
            f"{gesture_name}.png"
        )

        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.gesture_panel.setPixmap(
                pix.scaled(
                    140,
                    140,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.gesture_panel.setText("")
        else:
            self.gesture_panel.setPixmap(QPixmap())
            self.gesture_panel.setText(f"No image for: {gesture_name}")