"""
main.py
=======
Entry point for the EMG client application.

Shows a landing menu first, then routes to one of three destinations:
  - New subject           -> the phases 1-5 workflow (AppWindow)
  - Load an old session    -> placeholder (to be built together)
  - EMG sensors debugging  -> EMGDashboard in view-only mode (logging off)

Screens are created lazily on first use so unused ones never spin up BLE
workers or render timers.
"""

import os
import sys
import tempfile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt

# Window size we ask for, shrunk to whatever the display actually offers.
PREFERRED_W = 1200
PREFERRED_H = 800
# Slack left around the window so the frame and taskbar never clip it.
SCREEN_MARGIN = 80

from start_menu import StartMenu
from app_window import AppWindow
from dashboard_ui import EMGDashboard
from load_session import LoadSessionScreen, SubjectsScreen
from browse_takes import BrowseTakesScreen


class RootWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Bracelet")
        self._fit_to_screen()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar with a "back to menu" button (hidden on the menu itself).
        self.topbar = QWidget()
        tb = QHBoxLayout(self.topbar)
        tb.setContentsMargins(12, 8, 12, 8)
        self.back_btn = QPushButton("\u2190 Menu")
        self.back_btn.clicked.connect(self.show_menu)
        tb.addWidget(self.back_btn)
        tb.addStretch(1)
        root.addWidget(self.topbar)

        self.stack = QStackedWidget()
        # screen widget -> its QScrollArea container in the stack
        self._pages = {}
        root.addWidget(self.stack, 1)

        self.menu = StartMenu(self.show_workflow, self.show_load,
                              self.show_debug, self.show_subjects,
                              self.show_browse)
        self._add_page(self.menu)

        # Lazily created screens.
        self.workflow = None
        self.load_screen = None
        self.subjects_screen = None
        self.debug = None
        self.browse_screen = None

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
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 700;
            }
            QPushButton:hover { background: #3d4d75; }
        """)

        self.show_menu()

    # ── Screen plumbing ─────────────────────────────────────────────────

    def _fit_to_screen(self):
        """Size the window to the preferred size, capped by the space the
        display actually has. A QStackedWidget reports the *tallest* of its
        children as its minimum, so without the per-page scroll areas below
        this cap would simply be overridden by the layout."""
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = min(PREFERRED_W, avail.width() - SCREEN_MARGIN)
        h = min(PREFERRED_H, avail.height() - SCREEN_MARGIN)
        self.resize(w, h)
        self.move(avail.left() + (avail.width() - w) // 2,
                  avail.top() + (avail.height() - h) // 2)

    def _add_page(self, widget):
        """Put a screen in the stack behind its own scroll area, so a screen
        taller than the window scrolls instead of forcing the window past the
        edge of the display."""
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setWidget(widget)
        self.stack.addWidget(page)
        self._pages[widget] = page

    def _show_page(self, widget):
        self.stack.setCurrentWidget(self._pages[widget])

    # ── Routing ─────────────────────────────────────────────────────────

    def show_menu(self):
        self.topbar.hide()
        self._show_page(self.menu)

    def show_workflow(self):
        if self.workflow is None:
            self.workflow = AppWindow()
            self._add_page(self.workflow)
        self.topbar.show()
        self._show_page(self.workflow)

    def show_load(self):
        if self.load_screen is None:
            self.load_screen = LoadSessionScreen()
            self.load_screen.set_load_handler(self._on_session_picked)
            self._add_page(self.load_screen)
        else:
            # Re-read the Excel files in case they were edited externally.
            self.load_screen.reload_data()
        self.topbar.show()
        self._show_page(self.load_screen)

    def show_browse(self):
        if self.browse_screen is None:
            self.browse_screen = BrowseTakesScreen()
            self._add_page(self.browse_screen)
        else:
            self.browse_screen.reload_data()
        self.topbar.show()
        self._show_page(self.browse_screen)

    def _on_session_picked(self, model_path):
        """Called by LoadSessionScreen when the user picks a session.

        Opens (or reuses) the AppWindow and jumps it straight to Phase 3
        with the chosen model pre-selected for header export.
        """
        if not model_path:
            return
        # Make sure the workflow window exists
        if self.workflow is None:
            self.workflow = AppWindow()
            self._add_page(self.workflow)
        # Tell Phase 3 which .keras to export
        if hasattr(self.workflow, "phase3"):
            self.workflow.phase3.set_model_path(model_path)
        # Link this loaded session to Phase 5 so it can resolve the gesture set
        # and write online metrics back to the correct sessions.xlsx row.
        try:
            import sessions_registry
            ri = sessions_registry.find_row_by_model_path(model_path)
            if ri:
                if hasattr(self.workflow, "phase5"):
                    self.workflow.phase5.set_session_row_index(ri)
                if hasattr(self.workflow, "phase4"):
                    self.workflow.phase4.set_session_row_index(ri)
        except Exception as exc:
            print(f"[warn] could not link session row to Phase 5: {exc}")
        # Jump straight to Phase 3 (index 3 in the phase stack, after Phase 0 insertion)
        try:
            self.workflow.stack.setCurrentIndex(3)
            self.workflow.update_navigation()
        except Exception as exc:
            print(f"[warn] could not jump to Phase 3: {exc}")
        # Bring the workflow window to the front
        self.topbar.show()
        self._show_page(self.workflow)

    def show_subjects(self):
        if self.subjects_screen is None:
            self.subjects_screen = SubjectsScreen()
            self._add_page(self.subjects_screen)
        else:
            self.subjects_screen.reload_data()
        self.topbar.show()
        self._show_page(self.subjects_screen)

    def show_debug(self):
        if self.debug is None:
            # Full monitoring dashboard, no permanent recording. Each START/STOP
            # is still logged to a throwaway temp file so the last take can be
            # reviewed via "View last take" without touching data/.
            self.debug = EMGDashboard()
            self.debug.enable_logging = False
            self.debug.temp_log_dir = self._debug_take_dir()
            self._strip_recording_controls(self.debug)
            self._add_page(self.debug)
        self.topbar.show()
        self._show_page(self.debug)

    def _debug_take_dir(self):
        path = os.path.join(tempfile.gettempdir(), "emg_debug_takes")
        os.makedirs(path, exist_ok=True)
        return path

    def _strip_recording_controls(self, dash):
        """Debug view is visualization-only: hide the protocol controls.
        'View last take' stays visible and works on the temp take."""
        for attr in ("btn_protocol", "rb_4cl", "rb_6cl", "lbl_protocol"):
            widget = getattr(dash, attr, None)
            if widget is not None:
                widget.hide()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _placeholder(self, title, subtitle):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(12)
        t = QLabel(title)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size:24px; font-weight:800; color:#ffffff;")
        s = QLabel(subtitle)
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setStyleSheet("font-size:14px; color:#9fb3ff;")
        lay.addWidget(t)
        lay.addWidget(s)
        return w

    def closeEvent(self, event):
        # Stop any running BLE/serial workers cleanly.
        if self.workflow is not None and hasattr(self.workflow, "dashboard"):
            try:
                self.workflow.dashboard.stop_serial()
            except Exception:
                pass
        if isinstance(self.debug, EMGDashboard):
            try:
                self.debug.stop_serial()
            except Exception:
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = RootWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()