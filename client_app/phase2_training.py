"""
phase2_training.py
==================
Phase 2 - Train your model.

Reads the current subject from set_subject_id() and offers two
dropdowns: which batch to use as training data, which one as the
holdout test set. The dropdowns are dynamic: they list every
data/<subject>_BATCH*/ folder that actually exists on disk.

When "Create my model" is clicked, the worker sets two environment
variables (EMG_DATA_RAW, EMG_DATA_TEST_RAW) before launching
scripts/train_causal_concat.py. lib/configs.py reads those env
vars and falls back to its defaults otherwise, so the CLI workflow
keeps working.
"""

import os
import re
import subprocess
import sys
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QFrame, QMessageBox, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import sessions_registry
import subjects_registry

CV_SENTINEL = "__CV_SINGLE_SESSION__"


# ─────────────────────────────────────────────────────────────────────
# Worker — runs train_causal_concat.py with EMG_DATA_* env vars
# ─────────────────────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """Run train_causal_concat.py as a subprocess and report progress."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(float, str)
    failed = pyqtSignal(str)

    EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
    ACC_RE = re.compile(r"BATCH2 test accuracy:\s*([0-9.]+)")

    def __init__(self, data_raw_path=None, data_test_path=None,
                 model_output_name=None, single_cv=False, gesture_set=None, parent=None):
        super().__init__(parent)
        self.data_raw_path     = data_raw_path
        self.data_test_path    = data_test_path
        self.model_output_name = model_output_name
        self.single_cv         = single_cv
        self.gesture_set       = gesture_set
        self.mode_label        = ""

    def run(self):
        try:
            client_app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(client_app_dir)
            if self.single_cv:
                script_name = "train_multisession.py"
                self.mode_label = "CV intra-session (optimiste)"
            else:
                script_name = "train_causal_concat.py"
                self.mode_label = "Test inter-session (BATCH2)"
            script_path = os.path.join(
                project_root, "scripts", script_name)
            if not os.path.exists(script_path):
                self.failed.emit(
                    f"Training script not found: {script_path}")
                return

            if not self.single_cv and (not self.data_test_path or not os.path.isdir(self.data_test_path)):
                self.failed.emit(
                    "No test batch (BATCH2) for this subject: "
                    f"{self.data_test_path!r}. Record a BATCH2 before training. "
                    "Refusing to fall back to another subject's data "
                    "(that would record an invalid offline score)."
                )
                return
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            # Force UTF-8 in the training subprocess so its box-drawing prints
            # (═, →, ✓) don't crash on a Windows cp1252 console.
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            # The gesture set is auto-detected from the data, never set by hand.
            if self.gesture_set:
                env["EMG_GESTURE_SET"] = self.gesture_set

            # Inject our chosen train/test folders. configs.py reads these.
            if self.data_raw_path:
                env["EMG_DATA_RAW"] = self.data_raw_path
            if self.single_cv and self.data_raw_path:
                env["EMG_DATA_SESSIONS"] = self.data_raw_path  # pin CV au batch choisi
            if self.data_test_path and not self.single_cv:
                env["EMG_DATA_TEST_RAW"] = self.data_test_path
            if self.model_output_name:
                env["EMG_MODEL_OUTPUT_NAME"] = self.model_output_name

            import tempfile; log_path = os.path.join(tempfile.gettempdir(), "phase2_training.log")
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, "-u", script_path],
                cwd=project_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            proc.wait()
            log_file.close()
            # Re-read the log to extract accuracy
            with open(log_path, encoding="utf-8", errors="replace") as f:
                log_content = f.read()
            import re as _re
            accuracy = None
            for m in _re.finditer(r'BATCH2 test accuracy:\s*([0-9.]+)', log_content):
                accuracy = float(m.group(1)) * 100.0
            if proc.returncode != 0:
                tail = log_content.split('\n')[-30:]
                self.failed.emit(
                    f"Training failed (code {proc.returncode}). Full log: {log_path}\n\n"
                    + '\n'.join(tail))
                return
            if accuracy is None:
                self.failed.emit(f"No accuracy reported. Check {log_path}")
                return
            self.finished_ok.emit(accuracy, self.mode_label)
            return  # Skip the old loop below

            # OLD STREAMING CODE BELOW (now unreachable, kept for ref)
            proc = subprocess.Popen(
                [sys.executable, "-u", script_path],
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            accuracy = None
            for line in proc.stdout:
                m = self.EPOCH_RE.search(line)
                if m:
                    epoch = int(m.group(1))
                    total = int(m.group(2))
                    if total > 0:
                        pct = min(100, int(epoch * 100 / total))
                        self.progress.emit(pct)
                    continue
                m = self.ACC_RE.search(line)
                if m:
                    try:
                        accuracy = float(m.group(1)) * 100.0
                    except ValueError:
                        pass

            proc.wait()
            if proc.returncode != 0:
                self.failed.emit(
                    f"Training script exited with code {proc.returncode}")
                return
            if accuracy is None:
                self.failed.emit(
                    "Training script did not report a final accuracy")
                return

            self.finished_ok.emit(accuracy)

        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Phase 2 widget
# ─────────────────────────────────────────────────────────────────────

N_CLASSES = 6
CLASS_NAMES = ["rest", "flexion", "extension", "close", "supination", "pronation"]


class Phase2Training(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.subject_id = None
        self.session_row_index = None
        self.last_model_path = None
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────
    def set_subject_id(self, subject_id):
        self.subject_id = subject_id
        print(f"Phase 2 received subject_id: {subject_id}")
        self._refresh_batches()
        self._update_summary()

    def showEvent(self, event):
        # Re-scan when the screen becomes visible (in case new takes
        # were recorded since the last visit)
        super().showEvent(event)
        self._refresh_batches()
        self._update_summary()

    # ── Gesture-set resolution (set is a property of the data) ─────
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

    def _detect_set(self, folder):
        """Detect the gesture set of a batch folder from its CSV labels (cached)."""
        if not folder:
            return None
        cache = getattr(self, "_set_cache", None)
        if cache is None:
            cache = self._set_cache = {}
        if folder in cache:
            return cache[folder]
        c = self._cfg()
        s = None
        if c is not None:
            try:
                s = c.gesture_set_of_folder(folder)
            except Exception:
                s = None
        cache[folder] = s
        return s

    def _gestures_for(self, gset):
        c = self._cfg()
        if c is not None:
            try:
                return c.gestures_for(gset)
            except Exception:
                pass
        return {"4cl": ["rest", "flexion", "extension", "close"],
                "6cl": ["rest", "flexion", "extension", "close",
                        "supination", "pronation"],
                "rps": ["idle", "rock", "paper", "scissors"]}.get(gset, [])

    # ── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(14)

        title = QLabel("Phase 2 - Train your model")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        intro = QLabel(
            "Review the training summary below, then click Create my model.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet("color:#c5cce0; font-size:13px; border:none;")
        outer.addWidget(intro)

        # ── Summary card ──
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:8px; }")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(20, 16, 20, 16)
        card_lay.setSpacing(10)

        card_title = QLabel("Training summary")
        card_title.setStyleSheet(
            "color:#e5ebff; font-size:15px; font-weight:700; border:none;")
        card_lay.addWidget(card_title)

        # Subject row
        self.lbl_subject = self._info_row("Subject", "(none yet)")
        card_lay.addLayout(self.lbl_subject["layout"])

        # Train batch row
        train_row = QHBoxLayout()
        train_row.setSpacing(8)
        train_label = QLabel("Train batch:")
        train_label.setStyleSheet(
            "color:#9aa6c8; font-size:13px; border:none; min-width:120px;")
        train_row.addWidget(train_label)
        self.train_combo = QComboBox()
        self.train_combo.setStyleSheet(self._combo_style())
        self.train_combo.currentIndexChanged.connect(self._update_summary)
        train_row.addWidget(self.train_combo)
        self.lbl_train_count = QLabel("")
        self.lbl_train_count.setStyleSheet(
            "color:#9aa6c8; font-size:12px; border:none;")
        train_row.addWidget(self.lbl_train_count)
        train_row.addStretch(1)
        card_lay.addLayout(train_row)

        # Test batch row
        test_row = QHBoxLayout()
        test_row.setSpacing(8)
        test_label = QLabel("Test batch:")
        test_label.setStyleSheet(
            "color:#9aa6c8; font-size:13px; border:none; min-width:120px;")
        test_row.addWidget(test_label)
        self.test_combo = QComboBox()
        self.test_combo.setStyleSheet(self._combo_style())
        self.test_combo.currentIndexChanged.connect(self._update_summary)
        test_row.addWidget(self.test_combo)
        self.lbl_test_count = QLabel("")
        self.lbl_test_count.setStyleSheet(
            "color:#9aa6c8; font-size:12px; border:none;")
        test_row.addWidget(self.lbl_test_count)
        test_row.addStretch(1)
        card_lay.addLayout(test_row)

        # Classes
        classes_str = f"{N_CLASSES} ({', '.join(CLASS_NAMES)})"
        self.lbl_classes = self._info_row("Classes", classes_str)
        card_lay.addLayout(self.lbl_classes["layout"])

        # Output
        self.lbl_output = self._info_row("Output model", "(predicted name)")
        card_lay.addLayout(self.lbl_output["layout"])

        outer.addWidget(card)

        # ── Start button ──
        self.start_button = QPushButton("\u25b6  Create my model")
        self.start_button.setMinimumHeight(48)
        self.start_button.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020;"
            " font-size:15px; font-weight:800; border:none;"
            " border-radius:8px; padding:10px 28px; }"
            "QPushButton:hover { background:#3ee685; }"
            "QPushButton:disabled { background:#3a4570; color:#7c87a8; }")
        self.start_button.clicked.connect(self._start_training)
        outer.addWidget(self.start_button)

        # ── Status & progress ──
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color:#c5cce0; font-size:13px; border:none;")
        outer.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:6px; height:14px; text-align:center;"
            " color:#e5ebff; font-size:11px; }"
            "QProgressBar::chunk { background:#2ed573; border-radius:5px; }")
        self.progress.hide()
        outer.addWidget(self.progress)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "color:#e5ebff; font-size:14px; border:none;"
            " padding:8px 0;")
        outer.addWidget(self.result_label)

        outer.addStretch(1)
        self._update_summary()

    def _combo_style(self):
        return (
            "QComboBox { background:#0e1422; color:#e5ebff;"
            " border:1px solid #2a3550; border-radius:4px;"
            " padding:4px 8px; font-size:13px; min-width:180px; }"
            "QComboBox::drop-down { border:none; }"
        )

    def _info_row(self, label, value):
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(f"{label}:")
        lbl.setStyleSheet(
            "color:#9aa6c8; font-size:13px; border:none; min-width:120px;")
        row.addWidget(lbl)
        val = QLabel(value)
        val.setStyleSheet("color:#e5ebff; font-size:13px; border:none;")
        val.setWordWrap(True)
        row.addWidget(val, stretch=1)
        return {"layout": row, "label": lbl, "value": val}

    # ── Filesystem scan ────────────────────────────────────────────
    def _short_subject(self):
        if not self.subject_id:
            return None
        return self.subject_id.split("-")[0].strip() if "-" in self.subject_id else self.subject_id

    def _project_root(self):
        client_app_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(client_app_dir)

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", "_", str(s or "").strip()).lower()

    def _candidate_profiles(self):
        """Normalized folder-prefix candidates for the current subject.

        A recording folder is data/<profile>_BATCH<N>, where <profile> is
        the subject_id when an existing subject was picked, but the free
        profile text (often the subject's name) for a new one. So accept a
        match on the subject's id, name, surname or 'name surname'.
        """
        keys = set()
        sid = self._short_subject()
        if sid:
            keys.add(self._norm(sid))
            try:
                subj = subjects_registry.find_subject(sid)
            except Exception:
                subj = None
            if subj:
                name = subj.get("name", "")
                surname = subj.get("surname", "")
                for k in (name, surname, f"{name} {surname}"):
                    keys.add(self._norm(k))
        keys.discard("")
        return keys

    def _list_batches(self):
        """List all BATCH folders for the current subject.

        Matches data/<profile>_BATCH<suffix> folders whose <profile> is
        one of the subject's id/name/surname candidates. Returns a dict
        {folder_name: (folder_name, full_path)} (keyed by folder name so
        two profiles sharing a batch number do not clobber each other).
        """
        candidates = self._candidate_profiles()
        if not candidates:
            return {}
        data_dir = os.path.join(self._project_root(), "data")
        if not os.path.isdir(data_dir):
            return {}
        pat = re.compile(r"^(.+)_BATCH(.+)$")
        batches = {}
        for name in os.listdir(data_dir):
            m = pat.match(name)
            if not m:
                continue
            full = os.path.join(data_dir, name)
            if not os.path.isdir(full):
                continue
            if self._norm(m.group(1)) in candidates:
                batches[name] = (name, full)
        return batches

    def _count_csvs(self, folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return 0
        try:
            return len([f for f in os.listdir(folder_path) if f.endswith(".csv")])
        except Exception:
            return 0

    def _refresh_batches(self):
        """Repopulate the train/test dropdowns based on current filesystem."""
        # Save current selections to restore after rebuild
        prev_train = self.train_combo.currentData()
        prev_test  = self.test_combo.currentData()

        # Block signals to avoid spurious _update_summary calls
        self.train_combo.blockSignals(True)
        self.test_combo.blockSignals(True)
        self.train_combo.clear()
        self.test_combo.clear()
        # Cross-validation is always the first option in the test dropdown.
        self.test_combo.addItem("(cross-validation - single session)", CV_SENTINEL)
        batches = self._list_batches()
        if not batches:
            # No subject or no batches yet
            self.train_combo.addItem("(no data — record first)", None)
        else:
            # Sort: numeric first ascending, then named ones
            sortable = sorted(
                batches.items(),
                key=lambda kv: (isinstance(kv[0], str), kv[0]))
            for key, (folder_name, full_path) in sortable:
                display = folder_name  # e.g. "S001_BATCH1" or "S003_BATCH1_PRONATION"
                self.train_combo.addItem(display, full_path)
                self.test_combo.addItem(display, full_path)
            # Default: train = last (most recent) batch, test = cross-validation
            self.train_combo.setCurrentIndex(self.train_combo.count() - 1)
            self.test_combo.setCurrentIndex(0)

        # Restore previous selection if possible
        if prev_train is not None:
            idx = self.train_combo.findData(prev_train)
            if idx >= 0:
                self.train_combo.setCurrentIndex(idx)
        if prev_test is not None:
            idx = self.test_combo.findData(prev_test)
            if idx >= 0:
                self.test_combo.setCurrentIndex(idx)

        self.train_combo.blockSignals(False)
        self.test_combo.blockSignals(False)

    # ── Summary update ─────────────────────────────────────────────
    def _update_summary(self):
        # Subject
        if self.subject_id:
            self.lbl_subject["value"].setText(self.subject_id)
            self.lbl_subject["value"].setStyleSheet(
                "color:#2ed573; font-size:13px; font-weight:700; border:none;")
        else:
            self.lbl_subject["value"].setText("(none yet \u2014 enter a subject first)")
            self.lbl_subject["value"].setStyleSheet(
                "color:#ff9f43; font-size:13px; border:none;")

        # Train batch count
        train_path = self.train_combo.currentData()
        n_train = self._count_csvs(train_path)
        self._set_count_label(self.lbl_train_count, n_train)

        # Test batch count
        test_path = self.test_combo.currentData()
        n_test = self._count_csvs(test_path)
        self._set_count_label(self.lbl_test_count, n_test)

        # Same batch chosen for train and test? warn
        if train_path and test_path and train_path == test_path:
            self.status_label.setText(
                "\u26a0 Train and Test point to the same batch. "
                "This will overestimate accuracy.")
            self.status_label.setStyleSheet(
                "color:#ff9f43; font-size:12px; border:none;")
        else:
            self.status_label.setText("")

        # Gesture set detected from the selected train batch
        gset = self._detect_set(train_path) or "6cl"
        gestures = self._gestures_for(gset)
        if gestures:
            self.lbl_classes["value"].setText(
                f"{len(gestures)} — {gset} ({', '.join(gestures)})")

        # Predicted output name
        if self.subject_id and train_path:
            short = self._short_subject()
            today = datetime.now().strftime("%Y%m%d")
            predicted = f"model_{short}_{gset}_{today}.keras"
            self.lbl_output["value"].setText(predicted)
            self.lbl_output["value"].setStyleSheet(
                "color:#e5ebff; font-family: 'Menlo'; font-size:12px; border:none;")
        else:
            self.lbl_output["value"].setText("(predicted once a subject is set)")
            self.lbl_output["value"].setStyleSheet(
                "color:#9aa6c8; font-size:13px; border:none;")

        # Enable button only if both batches selected
        self.start_button.setEnabled(
            bool(self.subject_id) and bool(train_path) and bool(test_path))

    def _set_count_label(self, label, n):
        if n == 0:
            label.setText("(empty)")
            label.setStyleSheet("color:#ff6b6b; font-size:12px; border:none;")
        elif n < 10:
            label.setText(f"{n}/10 \u26a0")
            label.setStyleSheet("color:#ff9f43; font-size:12px; font-weight:700; border:none;")
        else:
            label.setText(f"{n}/10 \u2713")
            label.setStyleSheet("color:#2ed573; font-size:12px; font-weight:700; border:none;")

    # ── Training ───────────────────────────────────────────────────
    def _start_training(self):
        train_path = self.train_combo.currentData()
        test_path  = self.test_combo.currentData()
        single_cv  = (test_path == CV_SENTINEL)

        if not train_path:
            QMessageBox.warning(
                self, "Missing data",
                "A training batch must be selected.")
            return
        if not single_cv and not test_path:
            QMessageBox.warning(
                self, "Missing test batch",
                "Select a test batch, or choose cross-validation.")
            return

        if self._count_csvs(train_path) == 0:
            QMessageBox.warning(
                self, "Empty train batch",
                f"Train folder is empty:\n{train_path}")
            return

        if not single_cv and self._count_csvs(test_path) == 0:
            QMessageBox.warning(
                self, "Empty test batch",
                f"Test folder is empty:\n{test_path}")
            return

        # Final confirm if same batch
        if not single_cv and train_path == test_path:
            reply = QMessageBox.question(
                self, "Same batch for train and test",
                "Train and Test point to the same folder.\n"
                "Accuracy estimates will be optimistic.\n\n"
                "Continue anyway?")
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.start_button.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText(
            f"Training on {os.path.basename(train_path)} / "
            f"testing on {os.path.basename(test_path)}\u2026")
        self.result_label.setText("")

        from datetime import datetime as _dt
        short = self._short_subject() or "unknown"
        today = _dt.now().strftime("%Y%m%d")
        gset = self._detect_set(train_path) or "6cl"
        self._gesture_set = gset
        model_output_name = f"model_{short}_{gset}_{today}.keras"
        self.last_model_path = f"models/trained/{model_output_name}"

        self.worker = TrainingWorker(
            data_raw_path=train_path,
            data_test_path=None if single_cv else test_path,
            model_output_name=model_output_name,
            single_cv=single_cv,
            gesture_set=gset,
        )
        self._train_path = train_path
        self._test_path = None if single_cv else test_path
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, accuracy, mode_label=""):
        self.progress.setValue(100)
        self.status_label.setText("Training complete")
        _tag = (mode_label + "\n") if mode_label else ""
        self.result_label.setText(
            f"{_tag}Accuracy: {accuracy:.2f}%\n"
            f"Click Continue to install the model on the bracelet.")
        self.start_button.setText("Train again")
        self.start_button.setEnabled(True)

        if self.subject_id:
            try:
                gset = getattr(self, "_gesture_set", "6cl")
                short = self._short_subject()
                today = datetime.now().strftime("%Y%m%d")
                model_name = f"model_{short}_{gset}_{today}.keras"
                n_classes = len(self._gestures_for(gset)) or 6
                acc_offline = accuracy / 100.0

                test_path = getattr(self, "_test_path", None)
                if test_path:
                    try:
                        data_folder = os.path.relpath(test_path, self._project_root())
                    except ValueError:
                        data_folder = test_path
                else:
                    _tp = getattr(self, "_train_path", None)
                    if _tp:
                        try:
                            data_folder = os.path.relpath(_tp, self._project_root())
                        except ValueError:
                            data_folder = _tp
                    else:
                        data_folder = None

                self.session_row_index = sessions_registry.add_session(
                    self.subject_id,
                    n_classes,
                    acc_offline=acc_offline,
                    f1_offline=acc_offline,
                    model_path=getattr(self, "last_model_path", None) or
                               f"models/trained/{model_name}",
                    data_folder=data_folder,
                    gesture_set=gset,
                )
                print(
                    f"Saved session for {self.subject_id}: "
                    f"accuracy={acc_offline:.3f}, row_index={self.session_row_index}")
            except Exception as exc:
                print(f"Could not save session: {exc}")

    def _on_failed(self, error):
        self.progress.hide()
        self.status_label.setText("Training failed")
        self.result_label.setText(f"Error: {error}")
        self.start_button.setText("Retry")
        self.start_button.setEnabled(True)