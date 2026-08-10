"""
browse_takes.py
===============
Standalone screen: browse recorded data by subject and date batch, then
review the takes — reusing the same take viewer as "Load an old session".

Unlike LoadSessionScreen (which is session/model-centric and reads
sessions.xlsx), this screen scans the data/ folders directly, so it shows
every recorded batch, including ones never used to train a model.
"""

import os
import re
import glob

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QListWidget, QListWidgetItem, QDialog, QMessageBox,
)
from PyQt6.QtCore import Qt


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir():
    return os.path.join(_project_root(), "data")


# Matches folders like S001_BATCH11 or S003_BATCH1_PRONATION
_BATCH_RE = re.compile(r"^(?P<sid>[^_]+)_BATCH(?P<batch>\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_")


def _batch_date(csvs):
    """Derive YYYY-MM-DD from the first take filename (e.g. 20260624_133304_emg.csv)."""
    for c in csvs:
        m = _DATE_RE.match(os.path.basename(c))
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return "\u2014"


_SET_TAG = {"6cl": "6cl", "4cl": "4cl", "rps": "RPS"}


def _detect_set(folder):
    """Gesture set of a batch folder. Uses the canonical configs helper if
    available (lib.configs / configs), else a local CSV-label fallback."""
    for imp in ("lib.configs", "configs"):
        try:
            mod = __import__(imp, fromlist=["gesture_set_of_folder"])
            fn = getattr(mod, "gesture_set_of_folder", None)
            if fn:
                return fn(folder)
        except Exception:
            continue
    try:
        import csv
        labels = set()
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "Label" not in reader.fieldnames:
                    continue
                for r in reader:
                    v = r.get("Label")
                    if not v:
                        continue
                    labels.add(str(v).strip().lower())
                    if labels & {"rock", "paper", "scissors", "idle"}:
                        return "rps"
                    if labels & {"supination", "pronation"}:
                        return "6cl"
    except Exception:
        return None
    # 4cl only after the full scan (flexion is shared with 6cl).
    if labels & {"flexion", "extension", "close"}:
        return "4cl"
    return None


def _scan_batches():
    """Return {subject_id: [(folder_name, abs_path, date_str, n_takes), ...]}."""
    root = _data_dir()
    out = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = _BATCH_RE.match(name)
        if not m:
            continue
        csvs = sorted(glob.glob(os.path.join(p, "*.csv")))
        out.setdefault(m.group("sid"), []).append(
            (name, p, _batch_date(csvs), len(csvs)))
    return out


class BrowseTakesScreen(QWidget):
    def __init__(self):
        super().__init__()
        self._batches = {}
        self._build_ui()
        self.reload_data()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(14)

        title = QLabel("Browse takes")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)
        sub = QLabel("Pick a subject and a date batch, then review its recorded takes.")
        sub.setStyleSheet("font-size:13px; color:#9fb3ff; border:none;")
        outer.addWidget(sub)

        row = QHBoxLayout()
        self.subject_combo = QComboBox()
        self.subject_combo.setMinimumWidth(240)
        self.subject_combo.setStyleSheet(
            "QComboBox { background:#1c2538; color:#fff; border:1px solid #3d4d75;"
            " border-radius:6px; padding:6px; }")
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        row.addWidget(self._labeled("Subject", self.subject_combo))
        row.addStretch(1)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_data)
        row.addWidget(reload_btn)
        outer.addLayout(row)

        outer.addWidget(self._labeled_caption("Batches (most recent first)"))
        self.batch_list = QListWidget()
        self.batch_list.setStyleSheet(
            "QListWidget { background:#0e1422; color:#e6e9f2; border:1px solid #2a3550;"
            " border-radius:8px; padding:4px; }"
            "QListWidget::item { padding:8px; }"
            "QListWidget::item:selected { background:#2a3550; }")
        self.batch_list.itemSelectionChanged.connect(self._on_batch_selected)
        self.batch_list.itemDoubleClicked.connect(lambda *_: self._open_review())
        outer.addWidget(self.batch_list, 1)

        self.view_btn = QPushButton("View takes")
        self.view_btn.setMinimumHeight(42)
        self.view_btn.setEnabled(False)
        self.view_btn.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020; font-size:15px;"
            " font-weight:800; border-radius:8px; }"
            "QPushButton:disabled { background:#1c2538; color:#4d5d7e; }")
        self.view_btn.clicked.connect(self._open_review)
        outer.addWidget(self.view_btn)

        self.info = QLabel("")
        self.info.setStyleSheet("color:#ff6b6b; border:none;")
        outer.addWidget(self.info)

    def _labeled(self, caption, widget):
        box = QWidget()
        l = QVBoxLayout(box)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        l.addWidget(self._labeled_caption(caption))
        l.addWidget(widget)
        return box

    def _labeled_caption(self, text):
        c = QLabel(text)
        c.setStyleSheet("font-size:12px; color:#9fb3ff; border:none;")
        return c

    # ── Data ────────────────────────────────────────────────────────────
    def reload_data(self):
        self._batches = _scan_batches()
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        for sid in sorted(self._batches.keys()):
            n = len(self._batches[sid])
            self.subject_combo.addItem(f"{sid}   ({n} batch{'es' if n != 1 else ''})", sid)
        self.subject_combo.blockSignals(False)
        if self.subject_combo.count():
            self.subject_combo.setCurrentIndex(0)
            self._on_subject_changed()
        else:
            self.batch_list.clear()
            self.view_btn.setEnabled(False)
            self.info.setText(f"No batches found in {_data_dir()}")

    def _set_for(self, folder):
        cache = getattr(self, "_set_cache", None)
        if cache is None:
            cache = self._set_cache = {}
        if folder not in cache:
            cache[folder] = _detect_set(folder)
        return cache[folder]

    def _on_subject_changed(self, *_):
        self.batch_list.clear()
        self.view_btn.setEnabled(False)
        self.info.setText("")
        sid = self.subject_combo.currentData()
        if not sid:
            return
        batches = sorted(self._batches.get(sid, []), key=lambda x: x[2], reverse=True)
        for name, path, date_str, n in batches:
            tag = self._set_for(path)
            suffix = f"    \u2014    {_SET_TAG.get(tag, '?')}" if tag else ""
            item = QListWidgetItem(
                f"{date_str}    {name}    \u2014    {n} take(s){suffix}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.batch_list.addItem(item)

    def _on_batch_selected(self):
        self.view_btn.setEnabled(self.batch_list.currentItem() is not None)

    def _current_folder(self):
        it = self.batch_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    # ── Take review (reuses take_viewer.TakeViewerDialog) ───────────────
    def _open_review(self):
        folder = self._current_folder()
        if not folder:
            return
        csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not csvs:
            self.info.setText("No takes in this batch.")
            return
        self.info.setText("")
        self._review_dialog(folder, csvs)

    def _review_dialog(self, folder, csvs):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Takes - {os.path.basename(folder)}")
        dlg.setStyleSheet("QDialog { background:#0e1422; }")
        outer = QVBoxLayout(dlg)
        outer.setSpacing(8)
        title = QLabel("")
        title.setStyleSheet("color:#ffffff; font-size:14px; font-weight:800; border:none;")
        outer.addWidget(title)
        holder = QVBoxLayout()
        holder.setSpacing(6)
        outer.addLayout(holder)

        def _clear(layout):
            while layout.count():
                w = layout.takeAt(0).widget()
                if w is not None:
                    w.setParent(None)

        def _rebuild():
            _clear(holder)
            items = sorted(glob.glob(os.path.join(folder, "*.csv")))
            title.setText(f"{len(items)} take(s) - click to review, Delete to remove")
            for i, c in enumerate(items, 1):
                roww = QWidget()
                rl = QHBoxLayout(roww)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(6)
                b = QPushButton(f"Take {i}    {os.path.basename(c)}")
                b.setStyleSheet(
                    "QPushButton { background:#1a2236; color:#e6e9f2; text-align:left;"
                    " border:1px solid #2a3550; border-radius:6px; padding:8px 12px; }"
                    "QPushButton:hover { border:1px solid #3d4d75; }")
                b.clicked.connect(lambda _=False, path=c: self._open_one_take(path))
                del_b = QPushButton("Delete")
                del_b.setFixedWidth(74)
                del_b.setStyleSheet(
                    "QPushButton { background:#2a1820; color:#ff6b6b;"
                    " border:1px solid #5a2a2a; border-radius:6px; padding:8px; }"
                    "QPushButton:hover { background:#3a2028; }")
                del_b.clicked.connect(
                    lambda _=False, path=c: self._delete_take(path, _rebuild))
                rl.addWidget(b, stretch=1)
                rl.addWidget(del_b)
                holder.addWidget(roww)
            if not items:
                empty = QLabel("No takes left in this batch.")
                empty.setStyleSheet("color:#9fb3ff; border:none;")
                holder.addWidget(empty)

        _rebuild()
        close = QPushButton("Close")
        close.setStyleSheet(
            "QPushButton { background:#2a3550; color:#e6e9f2; border:none;"
            " border-radius:6px; padding:8px; }")
        close.clicked.connect(dlg.accept)
        outer.addWidget(close)
        dlg.resize(460, 480)
        dlg.exec()
        # Refresh the batch list (take counts may have changed)
        self.reload_data()

    def _delete_take(self, path, refresh_cb):
        resp = QMessageBox.question(
            self, "Delete take",
            f"Delete this take permanently?\n\n{os.path.basename(path)}\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(path)
        except OSError as exc:
            QMessageBox.warning(self, "Delete take", f"Could not delete:\n{exc}")
            return
        refresh_cb()

    def _open_one_take(self, path):
        try:
            from take_viewer import TakeViewerDialog
        except ImportError:
            from client_app.take_viewer import TakeViewerDialog
        try:
            TakeViewerDialog(path, self, baseline_std=None).exec()
        except Exception as exc:
            self.info.setText(f"Could not open take: {exc}")