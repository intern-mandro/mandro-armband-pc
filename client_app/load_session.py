"""
load_session.py
===============
'Load an old session' screen.

Reads two Excel files and lets the user pick a past session, then shows its
offline/online metrics and offers to load its model.

  - subjects.xlsx : subject_id, name, surname
  - sessions.xlsx : subject_id, date, hour, acc_offline, f1_offline,
                    acc_online, f1_online, model_path

UX: a Subject dropdown (shown as "S001 - Sheryl Ann" via the join) filters a
Session/date dropdown; the two together pinpoint one session row. Its metrics
appear below, then a "Load this model" button (flashing wired up later).

Column names live in COLS so they are trivial to change if your headers differ.
The Excel file paths default to the project root (one level above the app dir).
"""

import os
import tempfile
import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSizePolicy, QDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

import subjects_registry
import sessions_registry

try:
    import pandas as pd
except ImportError:
    pd = None


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


GESTURE_SETS = {
    "4cl": ["rest", "flexion", "extension", "close"],
    "6cl": ["rest", "flexion", "extension", "close", "supination", "pronation"],
    "rps": ["idle", "rock", "paper", "scissors"],
}


def _names_for_set(gset, n):
    g = GESTURE_SETS.get(gset)
    if g and len(g) >= n:
        return g[:n]
    return [str(i) for i in range(n)]


def _online_matrix_from_csv(path):
    """Return (matrix, n_classes, total, correct) from a benchmarks/live_*.csv."""
    import csv as _csv
    pairs = []
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            try:
                pairs.append((int(r["Action"]), int(r["Prediction"])))
            except (ValueError, TypeError, KeyError):
                continue
    if not pairs:
        return None, 0, 0, 0
    n = max(max(t, p) for t, p in pairs) + 1
    M = [[0] * n for _ in range(n)]
    correct = 0
    for t, p in pairs:
        M[t][p] += 1
        if t == p:
            correct += 1
    return M, n, len(pairs), correct


def _render_confusion_png(matrix, names, title, outpath):
    """Render a confusion matrix to a PNG via the Agg backend (no pyplot, so it
    does not interfere with the running Qt event loop)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    n = len(names)
    fig = Figure(figsize=(1.25 * n + 1.8, 1.25 * n + 1.4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    rowsum = [max(1, sum(r)) for r in matrix]
    norm = [[matrix[i][j] / rowsum[i] for j in range(n)] for i in range(n)]
    im = ax.imshow(norm, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if norm[i][j] > 0.5 else "#222", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)


SUBJECTS_XLSX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subjects.xlsx")
SESSIONS_XLSX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.xlsx")

COLS = {
    "subject_id": "subject_id",
    "date": "date",
    "hour": "hour",
    "acc_offline": "acc_offline",
    "f1_offline": "f1_offline",
    "acc_online": "acc_online",
    "f1_online": "f1_online",
    "model_path": "model_path",
    "name": "name",
    "surname": "surname",
}


class LoadSessionScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.sessions = None
        self.subjects = None
        self._current_model_path = ""
        self._load_handler = None  # optional callback(model_path) wired later
        self._row_for_table = []
        self._build_ui()
        self.reload_data()

    def set_load_handler(self, fn):
        """Plug the real model-loading/flashing logic later without touching
        this file. fn receives the resolved model_path string."""
        self._load_handler = fn

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(18)

        title = QLabel("Load an old session")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        # Selectors
        selectors = QHBoxLayout()
        selectors.setSpacing(16)
        self.subject_combo = QComboBox()
        self.subject_combo.setStyleSheet(
            'QComboBox { background:#0e1422; color:#e6e9f2; border:1px solid #2a3550; border-radius:6px; padding:6px 10px; font-size:13px; }QComboBox:hover { border:1px solid #3d4d75; }QComboBox::drop-down { border:none; width:24px; }QComboBox QAbstractItemView { background:#1a2236; color:#e6e9f2; border:1px solid #2a3550; selection-background-color:#2a3550; }')
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        selectors.addWidget(self._labeled("By subject ID", self.subject_combo), stretch=1)
        outer.addWidget(self._card("Select a subject", self._wrap_layout(selectors)))

        self.session_table = QTableWidget(0, 3)
        self.session_table.setHorizontalHeaderLabels(["\u2605", "Date", "Hour"])
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.setMinimumHeight(180)
        self.session_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.session_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.session_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        _hh = self.session_table.horizontalHeader()
        _hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        _hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        _hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.session_table.setStyleSheet(
            "QTableWidget { background:#0e1422; color:#e6e9f2;"
            " border:1px solid #2a3550; border-radius:8px;"
            " gridline-color:#1c2538; font-size:13px; }"
            "QHeaderView::section { background:#1a2236; color:#9fb3ff;"
            " border:none; padding:6px; font-weight:700; }"
            "QTableWidget::item:selected { background:#2a3550; }")
        self.session_table.cellClicked.connect(self._on_table_cell_clicked)
        self.session_table.itemSelectionChanged.connect(self._on_table_selection)
        # Sessions (left) and Performance (right) side by side
        mid = QHBoxLayout()
        mid.setSpacing(18)
        sessions_card = self._card(
            "Sessions (tap \u2605 to favourite)", self.session_table)
        metrics_card = self._build_metrics_card()
        metrics_card.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        mid.addWidget(sessions_card, stretch=3)
        mid.addWidget(metrics_card, stretch=2, alignment=Qt.AlignmentFlag.AlignTop)
        outer.addLayout(mid)

        # Model + load
        self.model_label = QLabel("-")
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet(
            "color:#9fb3ff; font-size:12px; border:none; background:transparent;")
        self.load_button = QPushButton("Load this model")
        self.load_button.setMinimumHeight(46)
        self.load_button.setEnabled(False)
        self.load_button.setStyleSheet(
            "QPushButton { background:#2ed573; color:#0a1020; font-size:15px;"
            " font-weight:800; border:none; border-radius:10px; padding:12px; }"
            "QPushButton:disabled { background:#1c2538; color:#4d5d7e; }")
        self.load_button.clicked.connect(self._on_load_clicked)

        model_inner = QWidget()
        ml = QVBoxLayout(model_inner)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(10)
        ml.addWidget(self.model_label)
        ml.addWidget(self.load_button)
        self.view_takes_button = QPushButton("View takes")
        self.view_takes_button.setMinimumHeight(40)
        self.view_takes_button.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-size:13px;"
            " font-weight:700; border:1px solid #2a3550; border-radius:8px;"
            " padding:8px; } QPushButton:hover { border:1px solid #3d4d75; }")
        self.view_takes_button.clicked.connect(self._on_view_takes)
        ml.addWidget(self.view_takes_button)
        self.delete_model_button = QPushButton("Delete model")
        self.delete_model_button.setMinimumHeight(40)
        self.delete_model_button.setStyleSheet(
            "QPushButton { background:#2a1620; color:#ff8a9f; font-size:13px;"
            " font-weight:700; border:1px solid #502a35; border-radius:8px;"
            " padding:8px; } QPushButton:hover { border:1px solid #753d4d; }")
        self.delete_model_button.clicked.connect(self._on_delete_model)
        ml.addWidget(self.delete_model_button)
        self.confusion_button = QPushButton("Display confusion matrix")
        self.confusion_button.setMinimumHeight(40)
        self.confusion_button.setStyleSheet(
            "QPushButton { background:#1a2236; color:#9fb3ff; font-size:13px;"
            " font-weight:700; border:1px solid #2a3550; border-radius:8px;"
            " padding:8px; } QPushButton:hover { border:1px solid #3d4d75; }")
        self.confusion_button.clicked.connect(self._on_show_confusion)
        ml.addWidget(self.confusion_button)
        outer.addWidget(self._card("Model", model_inner))

        # Footer: reload + error
        footer = QHBoxLayout()
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#ff6b6b; font-size:12px; border:none;")
        reload_btn = QPushButton("Reload files")
        reload_btn.clicked.connect(self.reload_data)
        footer.addWidget(self.error_label, stretch=1)
        footer.addWidget(reload_btn)
        outer.addLayout(footer)

        outer.addStretch(1)

    def _build_metrics_card(self):
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(10)

        def col_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-size:13px; font-weight:800; color:#9fb3ff;"
                " border:none; background:transparent;")
            return lbl

        def metric_name(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color:#c5cce0; font-size:13px; border:none; background:transparent;")
            return lbl

        def value():
            lbl = QLabel("-")
            lbl.setStyleSheet(
                "color:#ffffff; font-size:18px; font-weight:800;"
                " border:none; background:transparent;")
            return lbl

        grid.addWidget(col_header("Offline"), 0, 1)
        grid.addWidget(col_header("Online"), 0, 2)

        grid.addWidget(metric_name("Accuracy"), 1, 0)
        grid.addWidget(metric_name("F1 score"), 2, 0)

        self.val_acc_off = value()
        self.val_f1_off = value()
        self.val_acc_on = value()
        self.val_f1_on = value()
        grid.addWidget(self.val_acc_off, 1, 1)
        grid.addWidget(self.val_acc_on, 1, 2)
        grid.addWidget(self.val_f1_off, 2, 1)
        grid.addWidget(self.val_f1_on, 2, 2)

        grid.addWidget(metric_name("Train batch"), 3, 0)
        self.batch_label = value()
        grid.addWidget(self.batch_label, 3, 1, 1, 2)

        grid.addWidget(metric_name("Gesture set"), 4, 0)
        self.set_label = value()
        grid.addWidget(self.set_label, 4, 1, 1, 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return self._card("Performance", inner)

    def _labeled(self, caption, widget):
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        cap = QLabel(caption)
        cap.setStyleSheet("color:#9fb3ff; font-size:11px; border:none;")
        v.addWidget(cap)
        v.addWidget(widget)
        return wrap

    def _wrap_layout(self, layout):
        w = QWidget()
        w.setLayout(layout)
        return w

    def _card(self, title, body_widget):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            " border-radius:12px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)
        head = QLabel(title)
        head.setStyleSheet(
            "font-size:13px; font-weight:800; color:#9fb3ff;"
            " border:none; background:transparent;")
        lay.addWidget(head)
        lay.addWidget(body_widget)
        return card

    # ── Data ────────────────────────────────────────────────────────────

    def reload_data(self):
        self.error_label.setText("")
        if pd is None:
            self._fail("pandas is required to read the Excel files "
                       "(pip install pandas openpyxl).")
            return

        sid = COLS["subject_id"]
        try:
            self.subjects = pd.read_excel(SUBJECTS_XLSX, dtype={sid: str})
            self.subjects[sid] = self.subjects[sid].astype(str).str.strip()
        except Exception:
            self.subjects = None  # names are optional; fall back to IDs

        try:
            from sessions_registry import ensure_favorite_column
            ensure_favorite_column()
        except Exception:
            pass
        try:
            self.sessions = pd.read_excel(SESSIONS_XLSX, dtype={sid: str})
            self.sessions[sid] = self.sessions[sid].astype(str).str.strip()
        except Exception as exc:
            self.sessions = None
            self._fail(f"Could not read sessions file:\n{SESSIONS_XLSX}\n{exc}")
            return

        self._populate_subjects()

    def _subject_label(self, sid):
        return sid

    def _populate_subjects(self):
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        ids = sorted(self.sessions[COLS["subject_id"]].dropna().unique().tolist())
        for s in ids:
            self.subject_combo.addItem(self._subject_label(s), s)
        self.subject_combo.blockSignals(False)
        if ids:
            self.subject_combo.setCurrentIndex(0)
            self._on_subject_changed()
        else:
            self._fail("No sessions found in the file.")
            self._clear_metrics()

    def _on_subject_changed(self, *_):
        self._populate_session_table(self.subject_combo.currentData())

    @staticmethod
    def _is_fav(row):
        try:
            v = row.get("favorite")
        except AttributeError:
            return False
        if v is None or (pd is not None and isinstance(v, float) and pd.isna(v)):
            return False
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "x")
        return bool(v)

    def _populate_session_table(self, sid):
        self.session_table.blockSignals(True)
        self.session_table.setRowCount(0)
        self._row_for_table = []
        if sid is not None and self.sessions is not None:
            rows = self.sessions[self.sessions[COLS["subject_id"]] == sid]
            items = []
            for idx, row in rows.iterrows():
                key = f"{row[COLS['date']]} {row[COLS['hour']]}"
                items.append((self._is_fav(row), key, int(idx), row))
            items.sort(key=lambda t: (t[0], t[1]), reverse=True)
            for fav, key, idx, row in items:
                r = self.session_table.rowCount()
                self.session_table.insertRow(r)
                star = QTableWidgetItem("\u2605" if fav else "\u2606")
                star.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.session_table.setItem(r, 0, star)
                self.session_table.setItem(r, 1, QTableWidgetItem(str(row[COLS['date']])))
                self.session_table.setItem(r, 2, QTableWidgetItem(str(row[COLS['hour']])))
                self._row_for_table.append(idx)
        self.session_table.blockSignals(False)
        if self.session_table.rowCount():
            self.session_table.selectRow(0)
            self._on_table_selection()
        else:
            self._clear_metrics()

    def _on_table_selection(self, *_):
        r = self.session_table.currentRow()
        if r < 0 or r >= len(self._row_for_table):
            self._clear_metrics()
            return
        self._show_metrics(self.sessions.loc[self._row_for_table[r]])

    def _on_table_cell_clicked(self, r, c):
        if c != 0 or r < 0 or r >= len(self._row_for_table):
            return
        idx = self._row_for_table[r]
        new_fav = not self._is_fav(self.sessions.loc[idx])
        try:
            try:
                from sessions_registry import set_favorite
            except ImportError:
                from client_app.sessions_registry import set_favorite
            set_favorite(int(idx) + 2, new_fav)
        except Exception as exc:
            self.error_label.setText(f"Could not save favourite: {exc}")
        if "favorite" not in self.sessions.columns:
            self.sessions["favorite"] = False
        self.sessions.loc[idx, "favorite"] = new_fav
        self._populate_session_table(self.subject_combo.currentData())

    def _on_session_changed(self, *_):
        idx = self.session_combo.currentData()
        if idx is None or self.sessions is None:
            self._clear_metrics()
            return
        self._show_metrics(self.sessions.loc[idx])

    # ── Metrics rendering ───────────────────────────────────────────────

    @staticmethod
    def _is_missing(v):
        return v is None or (isinstance(v, float) and pd is not None and pd.isna(v))

    def _fmt_pct(self, v):
        if self._is_missing(v):
            return "-"
        try:
            return f"{float(v):.1f}%"
        except (TypeError, ValueError):
            return "-"

    def _fmt_f1(self, v):
        if self._is_missing(v):
            return "-"
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return "-"

    _SET_DISPLAY = {
        "6cl": "6 gestures (6cl)",
        "4cl": "4 gestures (4cl)",
        "rps": "Rock-Paper-Scissors (rps)",
    }

    def _resolve_set(self, row):
        """Gesture set of a session: stored column, else model name, else n_classes."""
        v = row.get("gesture_set")
        if not self._is_missing(v) and str(v).strip():
            return str(v).strip()
        mp = row.get(COLS["model_path"])
        base = "" if self._is_missing(mp) else os.path.basename(str(mp)).lower()
        for t in ("rps", "6cl", "4cl"):
            if f"_{t}_" in base or f"_{t}." in base:
                return t
        nc = row.get("n_classes")
        try:
            nc = int(nc)
        except (TypeError, ValueError):
            nc = None
        return {6: "6cl", 4: "4cl"}.get(nc)

    def _batch_from_folder(self, folder):
        folder = "" if self._is_missing(folder) else str(folder).strip()
        if not folder:
            return "not recorded"
        base = os.path.basename(folder.rstrip("/\\"))
        m = re.search(r"BATCH\d+", base, re.IGNORECASE)
        return m.group(0).upper() if m else base

    def _show_metrics(self, row):
        self.val_acc_off.setText(self._fmt_pct(row.get(COLS["acc_offline"])))
        self.val_f1_off.setText(self._fmt_f1(row.get(COLS["f1_offline"])))
        self.val_acc_on.setText(self._fmt_pct(row.get(COLS["acc_online"])))
        self.val_f1_on.setText(self._fmt_f1(row.get(COLS["f1_online"])))
        self.batch_label.setText(self._batch_from_folder(row.get("data_folder")))
        _s = self._resolve_set(row)
        self.set_label.setText(self._SET_DISPLAY.get(_s, "-") if _s else "-")

        mp = row.get(COLS["model_path"])
        mp = "" if self._is_missing(mp) else str(mp).strip()
        self._current_model_path = mp
        self.model_label.setText(mp if mp else "(no model_path for this session)")
        self.load_button.setEnabled(bool(mp))

    def _clear_metrics(self):
        for lbl in (self.val_acc_off, self.val_f1_off, self.val_acc_on, self.val_f1_on):
            lbl.setText("-")
        self.batch_label.setText("-")
        self.set_label.setText("-")
        self._current_model_path = ""
        self.model_label.setText("-")
        self.load_button.setEnabled(False)

    # ── Actions ─────────────────────────────────────────────────────────

    def _current_session_row(self):
        r = self.session_table.currentRow()
        if r < 0 or r >= len(self._row_for_table):
            return None
        return self.sessions.loc[self._row_for_table[r]]

    def _resolve_set(self, row):
        v = row.get("gesture_set")
        if not self._is_missing(v):
            return str(v).strip().lower()
        mp = row.get("model_path")
        base = "" if self._is_missing(mp) else os.path.basename(str(mp)).lower()
        for s in ("rps", "6cl", "4cl"):
            if f"_{s}_" in base or f"_{s}." in base:
                return s
        try:
            return {6: "6cl", 4: "4cl"}.get(int(row.get("n_classes")))
        except (ValueError, TypeError):
            return None

    def _offline_confusion_png(self, row):
        mp = row.get("model_path")
        if self._is_missing(mp):
            return None, "No model recorded for this session."
        base = os.path.basename(str(mp))
        stem = base[:-6] if base.endswith(".keras") else base
        csvp = os.path.join(_project_root(), "models", "trained",
                            f"{stem}.confusion_offline.csv")
        if not os.path.isfile(csvp):
            return None, ("Offline confusion not available for this model.\n"
                          "Retrain to generate it.")
        try:
            import csv as _csv
            with open(csvp, newline="") as f:
                rows = list(_csv.reader(f))
            names = rows[0][1:]
            M = [[int(x) for x in r[1:]] for r in rows[1:]]
            out = os.path.join(tempfile.gettempdir(),
                               f"conf_off_{os.getpid()}.png")
            _render_confusion_png(M, names, "Offline confusion", out)
            return out, None
        except Exception as exc:
            return None, f"Could not read offline confusion: {exc}"

    def _online_confusion_png(self, row):
        sid = row.get("subject_id")
        if self._is_missing(sid):
            return None, "No subject id for this session."
        sid = str(sid).strip()
        gset = self._resolve_set(row)
        bench = os.path.join(_project_root(), "benchmarks")
        if not os.path.isdir(bench):
            return None, ("No benchmarks folder.\nRun an online score "
                          "(Phase 5) first.")
        import glob as _glob
        cands = _glob.glob(os.path.join(bench, f"live_{sid}_*.csv"))
        if gset:
            filtered = [c for c in cands
                        if f"_{gset}_" in os.path.basename(c).lower()]
            cands = filtered or cands
        if not cands:
            return None, ("No online score found for this subject/set.\n"
                          "Run Phase 5 first.")
        path = max(cands, key=os.path.getmtime)
        try:
            M, n, total, correct = _online_matrix_from_csv(path)
            if not total:
                return None, "Online file has no usable predictions."
            names = _names_for_set(gset, n)
            out = os.path.join(tempfile.gettempdir(),
                               f"conf_on_{os.getpid()}.png")
            _render_confusion_png(
                M, names, f"Online confusion (acc {correct/total*100:.1f}%)", out)
            return out, None
        except Exception as exc:
            return None, f"Could not build online confusion: {exc}"

    def _on_show_confusion(self):
        row = self._current_session_row()
        if row is None:
            self.error_label.setText("Select a session first.")
            return
        self.error_label.setText("")
        offline = self._offline_confusion_png(row)
        online = self._online_confusion_png(row)
        self._open_confusion_dialog(offline, online)

    def _open_confusion_dialog(self, offline, online):
        from PyQt6.QtWidgets import (
            QDialog, QLabel, QPushButton,
            QVBoxLayout, QHBoxLayout, QWidget
        )
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtCore import Qt

        class ConfusionDialog(QDialog):

            def __init__(self, parent, offline, online):
                super().__init__(parent)

                self.setWindowTitle("Confusion matrices")
                self.resize(1200, 700)
                self.setStyleSheet("QDialog { background:#0e1422; }")

                self.offline_pix = QPixmap(offline[0]) if offline[0] else None
                self.online_pix = QPixmap(online[0]) if online[0] else None

                layout = QVBoxLayout(self)

                title = QLabel(
                    "Confusion matrices - Offline (left) / Online (right)"
                )
                title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                title.setStyleSheet(
                    "color:white;font-size:15px;font-weight:bold;"
                )
                layout.addWidget(title)

                row = QHBoxLayout()

                self.off_label = QLabel()
                self.off_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self.on_label = QLabel()
                self.on_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                row.addWidget(self.off_label)
                row.addWidget(self.on_label)

                layout.addLayout(row)

                if not self.offline_pix:
                    self.off_label.setText(offline[1] or "Unavailable")

                if not self.online_pix:
                    self.on_label.setText(online[1] or "Unavailable")

                close = QPushButton("Close")
                close.clicked.connect(self.accept)
                layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)

                self.update_pixmaps()

            def resizeEvent(self, event):
                super().resizeEvent(event)
                self.update_pixmaps()

            def update_pixmaps(self):
                w = self.width() // 2 - 40
                h = self.height() - 120

                if self.offline_pix:
                    self.off_label.setPixmap(
                        self.offline_pix.scaled(
                            w,
                            h,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    )

                if self.online_pix:
                    self.on_label.setPixmap(
                        self.online_pix.scaled(
                            w,
                            h,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    )

        dlg = ConfusionDialog(self, offline, online)
        dlg.exec()

    def _on_delete_model(self):
        row = self._current_session_row()
        if row is None:
            self.error_label.setText("Select a session first.")
            return
        subject = str(row.get("subject_id") or "").strip()
        date    = str(row.get("date") or "").strip()
        hour    = str(row.get("hour") or "").strip()
        mp      = row.get("model_path")
        mp      = "" if self._is_missing(mp) else str(mp).strip()

        msg = [f"Subject {subject}  -  {date} {hour}", ""]
        msg.append(f"Model: {mp}" if mp else "(no model file recorded)")
        msg += ["",
                "Removes the session row from sessions.xlsx, and the model",
                "file(s) if no other session still uses them.",
                "This cannot be undone. Continue?"]
        reply = QMessageBox.question(
            self, "Delete model", "\n".join(msg),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            import sessions_registry
            n = sessions_registry.delete_session_by_match(subject, date, hour, mp)
        except Exception as exc:
            self.error_label.setText(f"Registry not updated: {exc}")
            return

        removed, kept = [], []
        if mp:
            try:
                still_used = any(
                    str(x.get("model_path") or "").strip() == mp
                    for x in sessions_registry.list_all_sessions())
            except Exception:
                still_used = True
            mp_abs = mp if os.path.isabs(mp) else os.path.join(_project_root(), mp)
            paths = [("model", mp_abs)]
            base = os.path.basename(mp)
            if base.startswith("model_"):
                sc = base.replace("model_", "scaler_", 1)
                sc = sc[:-6] if sc.endswith(".keras") else sc
                paths.append(("scaler",
                    os.path.join(_project_root(), "models", "scalers", sc + ".pkl")))
            for label, p in paths:
                if still_used:
                    kept.append(label); continue
                try:
                    if os.path.isfile(p):
                        os.remove(p); removed.append(label)
                except OSError as exc:
                    self.error_label.setText(f"Could not delete {label}: {exc}")

        parts = [f"{n} registry row removed"]
        if removed:
            parts.append("files deleted: " + ", ".join(removed))
        if kept:
            parts.append("files kept (still used): " + ", ".join(kept))
        self.error_label.setText(". ".join(parts) + ".")
        self.reload_data()
        self._populate_session_table(subject)

    def _on_view_takes(self):
        row = self._current_session_row()
        if row is None:
            self.error_label.setText("Select a session first.")
            return
        folder = row.get("data_folder")
        folder = "" if self._is_missing(folder) else str(folder).strip()
        if not folder:
            self.error_label.setText("No data_folder recorded for this session.")
            return
        if not os.path.isabs(folder):
            folder = os.path.join(_project_root(), folder)
        if not os.path.isdir(folder):
            self.error_label.setText(f"Takes folder not found: {folder}")
            return
        import glob as _glob
        csvs = sorted(_glob.glob(os.path.join(folder, "*.csv")))
        if not csvs:
            self.error_label.setText(f"No takes found in {folder}")
            return
        self.error_label.setText("")
        self._open_takes_review(folder)

    def _open_takes_review(self, folder):
        from PyQt6.QtWidgets import QDialog
        import glob as _glob
        dlg = QDialog(self)
        dlg.setWindowTitle("Session takes")
        dlg.setStyleSheet("QDialog { background:#0e1422; }")
        outer = QVBoxLayout(dlg)
        outer.setSpacing(8)
        title = QLabel("")
        title.setStyleSheet(
            "color:#ffffff; font-size:14px; font-weight:800; border:none;")
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
            csvs = sorted(_glob.glob(os.path.join(folder, "*.csv")))
            title.setText(f"{len(csvs)} take(s) - click to review, Delete to remove")
            for i, c in enumerate(csvs, 1):
                roww = QWidget()
                rl = QHBoxLayout(roww)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(6)
                view_b = QPushButton(f"Take {i}    {os.path.basename(c)}")
                view_b.setStyleSheet(
                    "QPushButton { background:#1a2236; color:#e6e9f2;"
                    " text-align:left; border:1px solid #2a3550;"
                    " border-radius:6px; padding:8px 12px; }"
                    "QPushButton:hover { border:1px solid #3d4d75; }")
                view_b.clicked.connect(
                    lambda _=False, path=c: self._open_one_take(path))
                del_b = QPushButton("Delete")
                del_b.setFixedWidth(74)
                del_b.setStyleSheet(
                    "QPushButton { background:#2a1820; color:#ff6b6b;"
                    " border:1px solid #5a2a2a; border-radius:6px; padding:8px; }"
                    "QPushButton:hover { background:#3a2028; }")
                del_b.clicked.connect(
                    lambda _=False, path=c: self._delete_take(path, _rebuild))
                rl.addWidget(view_b, stretch=1)
                rl.addWidget(del_b)
                holder.addWidget(roww)
            if not csvs:
                empty = QLabel("No takes left in this session.")
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

    def _delete_take(self, path, refresh_cb):
        from PyQt6.QtWidgets import QMessageBox
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
            self.error_label.setText(f"Could not open take: {exc}")

    def _on_load_clicked(self):
        mp = self._current_model_path
        if not mp:
            return
        if self._load_handler is not None:
            self._load_handler(mp)
        else:
            QMessageBox.information(
                self, "Load this model",
                f"Model to load:\n{mp}\n\n"
                "Flashing to the bracelet (bypassing Arduino IDE) will be "
                "wired up next.")

    def _fail(self, msg):
        self.error_label.setText(msg)


class SubjectsScreen(QWidget):
    """Read-only directory: which subject_id maps to which name/surname.

    Reads subjects.xlsx (same path as LoadSessionScreen)."""

    def __init__(self):
        super().__init__()
        self._build_ui()
        self.reload_data()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 28, 40, 28)
        outer.setSpacing(16)

        title = QLabel("Subjects")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:24px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)

        subtitle = QLabel("Recorded subjects (anonymized).")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color:#9fb3ff; font-size:13px; border:none;")
        outer.addWidget(subtitle)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search by subject ID")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_filter)
        self.search_edit.setStyleSheet(
            "QLineEdit { background:#121826; color:#e6e9f2;"
            " border:1px solid #2a3550; border-radius:8px; padding:8px 12px; }")
        outer.addWidget(self.search_edit)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Subject ID", "Trainings", "Takes"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet(
            "QTableWidget { background:#1a2236; color:#e6e9f2;"
            " gridline-color:#2a3550; border:1px solid #2a3550;"
            " border-radius:8px; }"
            "QHeaderView::section { background:#222c44; color:#9fb3ff;"
            " font-weight:800; border:none; padding:8px; }"
            "QTableWidget::item { padding:6px; }"
            "QTableWidget::item:selected { background:#2a3550; color:#ffffff; }")
        outer.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#ff6b6b; font-size:12px; border:none;")
        self.delete_btn = QPushButton("Delete subject(s)")
        self.delete_btn.setStyleSheet(
            "QPushButton { background:#ff6b6b; color:#1a0a0a; font-weight:800;"
            " border:none; border-radius:8px; padding:8px 16px; }"
            "QPushButton:hover { background:#ff8585; }")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        reload_btn = QPushButton("Reload files")
        reload_btn.clicked.connect(self.reload_data)
        footer.addWidget(self.error_label, stretch=1)
        footer.addWidget(self.delete_btn)
        footer.addWidget(reload_btn)
        outer.addLayout(footer)

    def _on_delete_clicked(self):
        self.error_label.setText("")
        rows = sorted(
            r for r in {idx.row() for idx in self.table.selectionModel().selectedRows()}
            if not self.table.isRowHidden(r)
        )
        sids = []
        for r in rows:
            item = self.table.item(r, 0)
            if item and item.text().strip():
                sids.append(item.text().strip())
        if not sids:
            self.error_label.setText("Select at least one subject to delete.")
            return

        try:
            counts = {sid: sessions_registry.count_sessions_for_subject(sid)
                      for sid in sids}
        except Exception as exc:
            self.error_label.setText(f"Could not check sessions: {exc}")
            return

        blocked = {sid: n for sid, n in counts.items() if n > 0}
        safe = [sid for sid in sids if counts[sid] == 0]

        if not blocked:
            preview = ", ".join(sids[:6]) + ("..." if len(sids) > 6 else "")
            resp = QMessageBox.question(
                self, "Delete subject(s)",
                f"Permanently delete {len(sids)} subject(s)?\n\n{preview}\n\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp == QMessageBox.StandardButton.Yes:
                self._do_delete(subjects=sids, sessions_of=[])
            return

        total_sessions = sum(blocked.values())
        detail = "\n".join(f"  - {sid}: {n} session(s)" for sid, n in blocked.items())
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Subjects still have sessions")
        box.setText(
            f"{len(blocked)} of the selected subject(s) still have recorded "
            "sessions in sessions.xlsx:\n\n" + detail +
            "\n\nThey were not deleted to avoid leaving orphan sessions."
        )
        btn_force = box.addButton(
            f"Delete all + {total_sessions} session(s)",
            QMessageBox.ButtonRole.DestructiveRole)
        btn_skip = None
        if safe:
            btn_skip = box.addButton(
                f"Delete only the {len(safe)} without sessions",
                QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(btn_cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked is btn_cancel:
            return
        if btn_skip is not None and clicked is btn_skip:
            self._do_delete(subjects=safe, sessions_of=[])
        elif clicked is btn_force:
            self._do_delete(subjects=sids, sessions_of=list(blocked.keys()))

    def _do_delete(self, subjects, sessions_of):
        try:
            n_sess = (sessions_registry.delete_sessions_for_subjects(sessions_of)
                      if sessions_of else 0)
            n_subj = subjects_registry.delete_subjects(subjects) if subjects else 0
        except Exception as exc:
            self.error_label.setText(f"Delete failed: {exc}")
            return
        self.reload_data()
        msg = f"Deleted {n_subj} subject(s)"
        if n_sess:
            msg += f" and {n_sess} session(s)"
        self.error_label.setText(msg + ".")

    def reload_data(self):
        self.error_label.setText("")
        if pd is None:
            self.table.setRowCount(0)
            self.error_label.setText(
                "pandas is required (pip install pandas openpyxl).")
            return

        sid = COLS["subject_id"]
        try:
            df = pd.read_excel(SUBJECTS_XLSX, dtype={sid: str}).fillna("")
        except Exception as exc:
            self.table.setRowCount(0)
            self.error_label.setText(
                f"Could not read {SUBJECTS_XLSX}\n{exc}")
            return

        summary = self._build_summary(df)

        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            sid_disp = str(row.get(COLS["subject_id"], ""))
            n_sessions, n_takes = summary.get(sid_disp.strip(), (0, 0))
            self.table.setItem(r, 0, QTableWidgetItem(sid_disp))
            for c, value in ((1, n_sessions), (2, n_takes)):
                cell = QTableWidgetItem(str(value))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, cell)
        self._apply_filter()

    def _build_summary(self, subjects_df):
        """Return {subject_id: (n_sessions, n_takes)} for the directory.

        n_sessions = number of session rows recorded for the subject in
                     sessions.xlsx.
        n_takes    = number of *.csv recordings on disk, found by scanning
                     data/<profile>_BATCH<N> folders and attributing each
                     folder to a subject when its profile prefix matches
                     the subject's id, name, surname or "name surname"
                     (case- and whitespace-insensitive). This does NOT rely
                     on sessions.xlsx.data_folder, which is often empty.
        """
        counts = self._session_counts()
        folders = self._scan_data_folders()
        summary = {}
        sid_col = COLS["subject_id"]
        for _, row in subjects_df.iterrows():
            sid = str(row.get(sid_col, "")).strip()
            if not sid:
                continue
            name = str(row.get(COLS["name"], ""))
            surname = str(row.get(COLS["surname"], ""))
            summary[sid] = (counts.get(sid, 0),
                            self._takes_for(sid, name, surname, folders))
        return summary

    def _session_counts(self):
        """Return {subject_id: n_session_rows} from sessions.xlsx."""
        counts = {}
        if pd is None:
            return counts
        sid_col = COLS["subject_id"]
        try:
            sdf = pd.read_excel(SESSIONS_XLSX, dtype={sid_col: str})
        except Exception:
            return counts
        if sid_col not in sdf.columns:
            return counts
        sdf = sdf[sdf[sid_col].notna()]
        for sid_key, grp in sdf.groupby(sdf[sid_col].astype(str).str.strip()):
            counts[sid_key] = len(grp)
        return counts

    def _scan_data_folders(self):
        """Scan data/ for <profile>_BATCH<N> dirs.

        Returns a list of (profile_norm, abs_path, csv_count).
        """
        out = []
        data_dir = os.path.join(_project_root(), "data")
        try:
            names = os.listdir(data_dir)
        except OSError:
            return out
        pat = re.compile(r"^(.+)_BATCH(.+)$")
        for nm in names:
            m = pat.match(nm)
            if not m:
                continue
            path = os.path.join(data_dir, nm)
            if not os.path.isdir(path):
                continue
            try:
                csvs = sum(1 for f in os.listdir(path) if f.lower().endswith(".csv"))
            except OSError:
                csvs = 0
            out.append((self._norm(m.group(1)), path, csvs))
        return out

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", "_", str(s or "").strip()).lower()

    def _takes_for(self, sid, name, surname, folders):
        keys = {self._norm(sid), self._norm(name), self._norm(surname),
                self._norm(f"{name} {surname}")}
        keys.discard("")
        return sum(csvs for prof, _path, csvs in folders if prof in keys)

    def _apply_filter(self, *_):
        query = self.search_edit.text().strip().lower()
        for r in range(self.table.rowCount()):
            if not query:
                self.table.setRowHidden(r, False)
                continue
            haystack = " ".join(
                (self.table.item(r, c).text() if self.table.item(r, c) else "")
                for c in range(self.table.columnCount())
            ).lower()
            self.table.setRowHidden(r, query not in haystack)