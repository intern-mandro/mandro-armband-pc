"""
start_menu.py
=============
Landing menu shown before anything else. Entry points:
  - New subject           -> phases 1 to 5 workflow
  - Load an old session   -> workflow to be built later
  - Browse takes          -> review recorded takes
  - Subjects directory    -> per-subject session/take counts
  - Pair robotic hand     -> BLE MAC pairing (bracelet <-> hand), standalone
  - EMG sensors debugging -> live visualization only (no recording)

The menu is purely a router: it takes zero-arg callbacks and calls the
matching one when a card is clicked. RootWindow (main.py) wires them.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt


class _MenuCard(QFrame):
    """A clickable card with a title and a short description."""

    def __init__(self, title, desc, accent, callback):
        super().__init__()
        self._callback = callback
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame { background:#1a2236; border:1px solid #2a3550;"
            f" border-left:4px solid {accent}; border-radius:12px; }}"
            "QFrame:hover { background:#222c44; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(4)

        t = QLabel(title)
        t.setStyleSheet(
            "font-size:18px; font-weight:800; color:#ffffff;"
            " border:none; background:transparent;")
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet(
            "font-size:13px; color:#9fb3ff;"
            " border:none; background:transparent;")
        lay.addWidget(t)
        lay.addWidget(d)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self._callback()
        super().mouseReleaseEvent(event)


class StartMenu(QWidget):
    def __init__(self, on_new, on_load, on_debug, on_subjects, on_browse, on_pair_hand):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(80, 40, 80, 40)
        outer.setSpacing(14)

        title = QLabel("EMG Bracelet")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:30px; font-weight:800; color:#ffffff; border:none;")
        outer.addWidget(title)


        outer.addStretch(1)
        outer.addWidget(_MenuCard(
            "New subject",
            "Record data and build a personalized model (phases 1 to 5).",
            "#2ed573", on_new))
        outer.addWidget(_MenuCard(
            "Load an old session",
            "Open a previous capture to review or continue.",
            "#9fb3ff", on_load))
        outer.addWidget(_MenuCard(
            "Browse takes",
            "Pick a subject and a date batch to review its recorded takes.",
            "#26d0ce", on_browse))
        outer.addWidget(_MenuCard(
            "Subjects directory",
            "Show the number of sessions and takes per subject.",
            "#a55eea", on_subjects))
        outer.addWidget(_MenuCard(
            "Pair robotic hand",
            "Send the robotic hand's BLE MAC to the bracelet so it can "
            "auto-reconnect once the PC disconnects.",
            "#ff6b6b", on_pair_hand))
        outer.addWidget(_MenuCard(
            "EMG sensors debugging",
            "Live signal visualization only - no recording.",
            "#ff9f43", on_debug))
        outer.addStretch(2)