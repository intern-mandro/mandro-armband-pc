import time
import numpy as np
import serial.tools.list_ports
import pyqtgraph as pg
import os

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QFrame, QSpinBox, QMessageBox,
    QRadioButton, QButtonGroup, QStackedWidget,
)

import config
from emg_scale import EMGScaleManager
from logger import CSVLogger
from config import (
    FPS, PLOT_SEC,
    N_MULT_DEFAULT,
    ENABLE_CSV_LOGGING,
    get_ch_color, SUM_BAR_COLOR,
    CH_OFFSET,
    RAW_LINE_WIDTH,
    COLOR_BG, COLOR_CARD_BORDER,
    COLOR_STATUS_CONNECTED, COLOR_STATUS_DISCONNECTED,
    MIN_BUF, MAX_BUF, RATE_UPDATE_INTERVAL, BUF_RESIZE_THRESHOLD,
    FIRST_RESIZE_AFTER_SEC,
    RAW_SAMPLE_RATE_DEFAULT,
    FFT_MAX_HZ,
)
from serial_worker import SerialWorker
from protocol_controller import ProtocolController
#from view_single_take import view_last_take
from graph_render import render as render_impl


class EMGDashboard(QMainWindow):
    # global sig_sample
    # global sig_status
    # global sig_error
    # global sig_channel_detected

    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True, useOpenGL=False)
        self.setWindowTitle("EMG Dashboard (Real-time Monitoring)")
        self.resize(1600, 800)

        self.scale_manager = EMGScaleManager(n_channels=config.N_CH)
        self.n_mult = int(N_MULT_DEFAULT)
        self.view_mode = "raw"
        # ── Protocol controller state ─────────────────────────
        self.protocol = ProtocolController(self)
        self.protocol.sig_step_changed.connect(self._on_protocol_step)
        self.protocol.sig_session_done.connect(self._on_protocol_done)
        self.protocol.sig_session_aborted.connect(self._on_protocol_done)
        self.last_csv_path = None  # for View Last Take
        # RAW 버퍼: 초기값 = 예상 rate × PLOT_SEC (5초 분량), START 후 실제 수신 속도로 동적 조정
        self.max_display = max(MIN_BUF, min(MAX_BUF, int(round(RAW_SAMPLE_RATE_DEFAULT * PLOT_SEC))))
        self.raw_np_buf = np.zeros((config.N_CH, self.max_display))
        self.x_axis = np.linspace(0, PLOT_SEC * 1000, self.max_display)
        self._last_rate_update_time = 0.0

        self.cursor_colors = ["#ffffff"] * config.N_CH
        self.cursor_rects = []

        self.init_ui()

        for i in range(config.N_CH):
            rect = pg.ScatterPlotItem(size=8, symbol="s", brush=self.cursor_colors[i])
            self.raw_plot.addItem(rect)
            self.cursor_rects.append(rect)

        self.ptr = 0           # 링 버퍼 쓰기 인덱스
        self.is_buf_full = False
        self.sample_count = 0
        self.last_amp = np.zeros(config.N_CH, dtype=float)
        self.is_running = False

        self.csv_logger = None
        self.capture_profile = None
        self.capture_batch = 1
        self.enable_logging = True
        self.temp_log_dir = None
        self.worker = SerialWorker()
        self.worker.sig_sample.connect(self.on_sample)
        self.worker.sig_status.connect(self.set_status)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_channel_detected.connect(self.on_channel_detected)

        self.set_running_ui(False)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self.render)
        self.timer.start()
        #self.refresh_ports()

        self.height_buf = np.zeros((config.N_CH, self.max_display))
        self.height_buf.fill(1.5)

    def set_running_ui(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        #self.cb_port.setEnabled(not running)
        #self.btn_refresh.setEnabled(not running)
        self.sp_nmult.setEnabled(not running)
        self.is_running = running

    def _apply_raw_line(self):
        """Line 선택 → Raw 라인 뷰로 바로 전환"""
        if getattr(self, "view_mode", None) == "raw" and getattr(self, "rb_line", None) and self.rb_line.isChecked():
            return
        self.view_mode = "raw"
        if hasattr(self, "stacked_plots"):
            self.stacked_plots.setCurrentIndex(0)
        if hasattr(self, "rb_line") and hasattr(self, "rb_fill") and hasattr(self, "rb_fft"):
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(True)
            self.rb_line.setChecked(True)
            self.rb_fill.setChecked(False)
            self.rb_fft.setChecked(False)
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(False)
        if self.graph_panel_title_label:
            self.graph_panel_title_label.setText("RAW GRAPH (Dynamic Auto-Scaling)")

    def _apply_raw_fill(self):
        """Fill 선택 → Raw Bar 뷰로 바로 전환"""
        if getattr(self, "view_mode", None) == "raw" and getattr(self, "rb_fill", None) and self.rb_fill.isChecked():
            return
        self.view_mode = "raw"
        if hasattr(self, "stacked_plots"):
            self.stacked_plots.setCurrentIndex(0)
        if hasattr(self, "rb_line") and hasattr(self, "rb_fill") and hasattr(self, "rb_fft"):
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(True)
            self.rb_line.setChecked(False)
            self.rb_fill.setChecked(True)
            self.rb_fft.setChecked(False)
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(False)
        if self.graph_panel_title_label:
            self.graph_panel_title_label.setText("RAW GRAPH (Dynamic Auto-Scaling)")

    def _apply_fft(self):
        """FFT 선택 → 주파수 시각화로 바로 전환"""
        if getattr(self, "view_mode", None) == "fft":
            return
        self.view_mode = "fft"
        if hasattr(self, "stacked_plots"):
            self.stacked_plots.setCurrentIndex(1)
        if hasattr(self, "rb_line") and hasattr(self, "rb_fill") and hasattr(self, "rb_fft"):
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(True)
            self.rb_line.setChecked(False)
            self.rb_fill.setChecked(False)
            self.rb_fft.setChecked(True)
            for rb in (self.rb_line, self.rb_fill, self.rb_fft):
                rb.blockSignals(False)
        if self.graph_panel_title_label:
            self.graph_panel_title_label.setText("FFT (Frequency)")

    # 센서가 줄 단위로 보낸 개수(4 또는 6)로 채널 수 자동 감지
    def on_channel_detected(self, n: int):
        config.N_CH = n
        self.reinit_channel_mode()

    # 채널 모드 재설정 (N_CH 변경 시 전체 UI / 버퍼 초기화, 워커는 유지)
    def reinit_channel_mode(self):

        n = config.N_CH

        # 내부 데이터 버퍼 초기화 
        self.scale_manager = EMGScaleManager(n_channels=n)
        self.raw_np_buf = np.zeros((n, self.max_display))
        self.height_buf = np.zeros((n, self.max_display))
        self.height_buf.fill(1.5)
        self.last_amp = np.zeros(n, dtype=float)
        self.cursor_colors = ["#ffffff"] * n
        self.ptr = 0
        self.is_buf_full = False
        self.sample_count = 0

        # RAW Plot 재생성 
        for r in self.cursor_rects:
            self.raw_plot.removeItem(r)
        self.cursor_rects.clear()

        for i in range(n):
            rect = pg.ScatterPlotItem(size=8, symbol="s", brush=self.cursor_colors[i])
            self.raw_plot.addItem(rect)
            self.cursor_rects.append(rect)

        # 기존 라인, 바 제거 
        for i in range(len(self.past_lines)):
            self.raw_plot.removeItem(self.past_lines[i])
            self.raw_plot.removeItem(self.raw_lines[i])
            self.raw_plot.removeItem(self.bar_items[i])
        self.past_lines.clear()
        self.raw_lines.clear()
        self.bar_items.clear()

        # 채널 수에 맞게 라인, 바 재생성 
        for i in range(n):
            past_line = self.raw_plot.plot(
                [], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH, alpha=0.6)
            )
            self.past_lines.append(past_line)
            line = self.raw_plot.plot([], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH))
            self.raw_lines.append(line)
            bar = pg.BarGraphItem(
                x=[], height=[], width=20.0,
                brush=pg.mkBrush(get_ch_color(i)), pen=None,
            )
            self.raw_plot.addItem(bar)
            self.bar_items.append(bar)
            bar.setVisible(False)

        # Raw y축 범위 재설정 
        y_max = n * CH_OFFSET
        self.raw_plot.setYRange(0, y_max, padding=0)
        vb = self.raw_plot.getViewBox()
        vb.setLimits(yMin=0, yMax=y_max, minYRange=50, maxYRange=y_max)
        vb.setRange(yRange=(0, y_max))
        QTimer.singleShot(10, lambda: (self.raw_plot.setYRange(0, y_max, padding=0),
                                       self.raw_plot.getViewBox().setRange(yRange=(0, y_max))))

        # Diagonal Plot 재생성 
        for line in self.diag_lines:
            self.diag_plot.removeItem(line)
        self.diag_lines.clear()

        for i in range(n):
            line = pg.PlotCurveItem()
            self.diag_lines.append(line)
            self.diag_plot.addItem(line)

        # FFT Plot 라인 및 Y축 범위 재생성 (채널 수에 맞게)
        for line in self.fft_lines:
            self.fft_plot.removeItem(line)
        self.fft_lines.clear()
        y_max_fft = n * CH_OFFSET
        self.fft_plot.setYRange(0, y_max_fft, padding=0)
        self.fft_plot.getViewBox().setLimits(yMin=0, yMax=y_max_fft, minYRange=50, maxYRange=y_max_fft)
        for i in range(n):
            line = self.fft_plot.plot([], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH))
            self.fft_lines.append(line)

        # PWR Plot 재생성 
        self.pwr_plot.removeItem(self.bar_item)
        
        x_axis = self.pwr_plot.getAxis("bottom")
        ticks = [(i, f"CH{i}") for i in range(n)] + [(n, "AVG")]
        x_axis.setTicks([ticks])

        self.bar_item = pg.BarGraphItem(
            x=np.arange(n + 1),
            height=[0] * (n + 1),
            width=0.6,
            brushes=[pg.mkBrush(get_ch_color(i)) for i in range(n)] + [pg.mkBrush(SUM_BAR_COLOR)],
        )
        self.pwr_plot.addItem(self.bar_item)

    # 수신 속도에 맞춰 RAW 링 버퍼를 new_len으로 조정. 최근 데이터만 복사
    def _resize_raw_buffers(self, new_len: int):
        n_ch = config.N_CH
        old_len = self.max_display
        if self.is_buf_full:
            seq = [(self.ptr + i) % old_len for i in range(old_len)]
        else:
            seq = list(range(self.ptr))
        take = min(new_len, len(seq))
        new_raw = np.zeros((n_ch, new_len))
        new_height = np.ones((n_ch, new_len)) * 1.5
        if take > 0:
            indices = seq[-take:]
            for ch in range(n_ch):
                new_raw[ch, :take] = self.raw_np_buf[ch, indices]
                new_height[ch, :take] = self.height_buf[ch, indices]
        self.raw_np_buf = new_raw
        self.height_buf = new_height
        self.max_display = new_len
        self.x_axis = np.linspace(0, PLOT_SEC * 1000, new_len)
        if take >= new_len:
            self.ptr = 0
            self.is_buf_full = True
        else:
            self.ptr = take
            self.is_buf_full = False

    def card(self, title: str):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame { background: #121826; border: 1px solid #2a3550; border-radius: 12px; }
            QLabel { color: #e6e9f2; border: none; }
            QPushButton { background: #2a3550; color: white; padding: 6px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background: #3d4d75; }
            QPushButton:disabled { background: #1c2538; color: #4d5d7e; border: 1px solid #2a3550; }
            QComboBox { background: #1c2538; color: white; border: 1px solid #3d4d75; border-radius: 4px; padding: 4px; }
            QSpinBox { background: #1c2538; color: white; border: 1px solid #3d4d75; border-radius: 4px; padding-right: 2px; }
        """)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(15, 15, 15, 15)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 14px; font-weight: 800; color: #9fb3ff; margin-bottom: 5px;")
        lay.addWidget(lbl)
        return frame, lay

    # 패널 배치 
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        self.panel_settings = self.build_settings_panel()
        self.panel_raw = self.build_raw_plot_panel()
        self.panel_diag = self.build_diag_panel()
        self.panel_pwr = self.build_pwr_panel()

        left_container = QVBoxLayout()
        left_container.addWidget(self.panel_settings, 0)
        left_container.addWidget(self.panel_diag, 1)

        right_container = QVBoxLayout()
        right_container.addWidget(self.panel_raw, 5)
        right_container.addWidget(self.panel_pwr, 3)

        main_layout.addLayout(left_container, 1)
        main_layout.addLayout(right_container, 2)

    def build_settings_panel(self):

        frame, lay = self.card("SETTINGS & MODE")
        lay.setSpacing(15)

        # Port 선택 
        row_port = QHBoxLayout()

        #self.cb_port = QComboBox()
        #self.cb_port.setMinimumWidth(160)

        #self.btn_refresh = QPushButton("Refresh")
        #self.btn_refresh.setFixedWidth(85)

        #self.btn_refresh.clicked.connect(self.refresh_ports)

        #row_port.addWidget(QLabel("Port:"))
        #row_port.addWidget(self.cb_port, 1)
        #row_port.addWidget(self.btn_refresh)

        lay.addLayout(row_port)

        # Start/Stop 버튼 영역 
        row_btn = QHBoxLayout()

        self.btn_start = QPushButton("START")

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_serial)
        self.btn_stop.clicked.connect(self.stop_serial)

        self.btn_choose = QPushButton("Choose bracelet")
        self.btn_choose.clicked.connect(self._choose_bracelet)
        row_btn.addWidget(self.btn_choose)
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_stop)

        # Protocol Mode + View Last Take row

        row_proto = QHBoxLayout()

        self.btn_protocol = QPushButton("PROTOCOL MODE")

        self.btn_protocol.setCheckable(True)

        self.btn_protocol.clicked.connect(self.toggle_protocol)

        self.btn_view_last = QPushButton("VIEW LAST TAKE")

        self.btn_view_last.clicked.connect(self.view_last)

        self.btn_view_last.setEnabled(False)

        row_proto.addWidget(self.btn_protocol)

        row_proto.addWidget(self.btn_view_last)

        # Class selector (4 or 6)

        row_classes = QHBoxLayout()

        row_classes.addWidget(QLabel("Classes:"))

        self.rb_4cl = QRadioButton("4")

        self.rb_6cl = QRadioButton("6")

        self.rb_rps = QRadioButton("RPS")

        self.rb_4cl.setChecked(True)

        self.bg_classes = QButtonGroup(self)

        self.bg_classes.addButton(self.rb_4cl)

        self.bg_classes.addButton(self.rb_6cl)

        self.bg_classes.addButton(self.rb_rps)

        row_classes.addWidget(self.rb_4cl)

        row_classes.addWidget(self.rb_6cl)

        row_classes.addWidget(self.rb_rps)

        row_classes.addStretch()

        # Protocol status label

        self.lbl_protocol = QLabel("Protocol idle.")

        self.lbl_protocol.setStyleSheet(

            "font-size: 16px; font-weight: 700; color: #9fb3ff; padding: 6px;"

        )
        lay.addLayout(row_btn)
        lay.addLayout(row_proto)
        lay.addLayout(row_classes)
        lay.addWidget(self.lbl_protocol)

        # Window Size + View 모드 (시작 전에만 변경 가능)
        row_form = QHBoxLayout()
        row_form.addWidget(QLabel("Window Size:"))
        self.sp_nmult = QSpinBox()
        self.sp_nmult.setRange(1, 100)
        self.sp_nmult.setValue(self.n_mult)
        row_form.addWidget(self.sp_nmult)
        row_form.addStretch()
        lay.addLayout(row_form)

        # 연결 상태 표시 라벨 
        self.lbl_status = QLabel("● DISCONNECTED")
        self.lbl_status.setStyleSheet(f"color:{COLOR_STATUS_DISCONNECTED}; font-weight:800;")
        lay.addWidget(self.lbl_status)
        return frame

    def build_raw_plot_panel(self):
        frame, lay = self.card("RAW GRAPH (Dynamic Auto-Scaling)")

        self.raw_lines = []
        self.bar_items = []

        # 헤더 영역 (제목 + 모드 선택). 제목은 Raw/FFT 전환 시 문구만 바꿈
        header_layout = QHBoxLayout()
        self.graph_panel_title_label = None
        if lay.count() > 0:
            header_label = lay.itemAt(0).widget()
            if header_label:
                self.graph_panel_title_label = header_label
                header_layout.addWidget(header_label)

        header_layout.addStretch()  # 오른쪽으로 밀기

        # Line / Fill / FFT 중 하나 선택 시 해당 뷰로 바로 전환
        self.rb_line = QRadioButton("Line")
        self.rb_fill = QRadioButton("Fill")
        self.rb_fft = QRadioButton("FFT")
        self.rb_line.setChecked(True)
        for rb in (self.rb_line, self.rb_fill, self.rb_fft):
            rb.setStyleSheet("color: white; font-weight: bold;")
        self.bg_display = QButtonGroup(self)
        self.bg_display.addButton(self.rb_line)
        self.bg_display.addButton(self.rb_fill)
        self.bg_display.addButton(self.rb_fft)
        self.rb_line.toggled.connect(lambda checked: self._apply_raw_line() if checked else None)
        self.rb_fill.toggled.connect(lambda checked: self._apply_raw_fill() if checked else None)
        self.rb_fft.toggled.connect(lambda checked: self._apply_fft() if checked else None)
        header_layout.addWidget(self.rb_line)
        header_layout.addWidget(self.rb_fill)
        header_layout.addWidget(self.rb_fft)

        lay.addLayout(header_layout)
        
        # PlotWidget 생성 및 기본 설정 
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setBackground(COLOR_BG)
        self.raw_plot.hideButtons()
        self.raw_plot.getAxis("left").setStyle(showValues=False)
        self.raw_plot.getAxis("bottom").enableAutoSIPrefix(False)

        # x축, y축 범위 설정 
        x_max_ms = PLOT_SEC * 1000
        y_max = config.N_CH * CH_OFFSET

        self.raw_plot.setXRange(0, x_max_ms, padding=0)
        self.raw_plot.setLabel("bottom", "Time", units="ms")

        self.raw_plot.setYRange(0, y_max, padding=0)

        vb = self.raw_plot.getViewBox()
        vb.setLimits(
            xMin=0, xMax=x_max_ms,
            minXRange=200, maxXRange=x_max_ms,   # 최소, 최대 확대 범위 
            yMin=0, yMax=y_max,
            minYRange=50, maxYRange=y_max,
        )
        self.raw_plot.setMouseEnabled(x=True, y=True)

        # 채널별 라인/바 아이템 생성 
        self.past_lines = []

        for i in range(config.N_CH):
            
            # 과거 라인 
            past_line = self.raw_plot.plot(
                [], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH, alpha=0.6)
            )
            self.past_lines.append(past_line)

            # 현재 라인 
            line = self.raw_plot.plot([], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH))
            self.raw_lines.append(line)

            # bar 모드용 막대그래프 
            bar = pg.BarGraphItem(
                x=[], height=[],
                width=20.0,
                brush=pg.mkBrush(get_ch_color(i)),
                pen=None,
            )
            self.raw_plot.addItem(bar)
            self.bar_items.append(bar)
            bar.setVisible(False)

        # Raw / FFT 전환용 스택
        self.stacked_plots = QStackedWidget()
        self.stacked_plots.addWidget(self.raw_plot)

        # FFT 플롯 (RAW처럼 채널별 Y 오프셋으로 분리 표시)
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setBackground(COLOR_BG)
        self.fft_plot.hideButtons()
        self.fft_plot.getAxis("left").setStyle(showValues=False)
        self.fft_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        y_max_fft = config.N_CH * CH_OFFSET
        self.fft_plot.setXRange(0, FFT_MAX_HZ, padding=0)
        self.fft_plot.setYRange(0, y_max_fft, padding=0)
        vb_fft = self.fft_plot.getViewBox()
        vb_fft.setLimits(xMin=0, xMax=FFT_MAX_HZ, yMin=0, yMax=y_max_fft, minYRange=50, maxYRange=y_max_fft)
        self.fft_plot.setMouseEnabled(x=True, y=True)
        self.fft_lines = []
        for i in range(config.N_CH):
            line = self.fft_plot.plot([], [], pen=pg.mkPen(color=get_ch_color(i), width=RAW_LINE_WIDTH))
            self.fft_lines.append(line)
        self.stacked_plots.addWidget(self.fft_plot)
        # 초기: Line 선택 상태에 맞춰 Raw 뷰·제목 동기화
        self._apply_raw_line()

        lay.addWidget(self.stacked_plots, 1)
        return frame


    def build_diag_panel(self):

        frame, lay = self.card("Diagonal Vector")

        self.diag_plot = pg.PlotWidget()
        self.diag_plot.setBackground(COLOR_BG)
        self.diag_plot_limit = 50

        self.diag_plot.setXRange(-self.diag_plot_limit, self.diag_plot_limit)
        self.diag_plot.setYRange(-self.diag_plot_limit, self.diag_plot_limit)

        self.diag_plot.getAxis("left").hide()
        self.diag_plot.getAxis("bottom").hide()

        self.diag_plot.setAspectLocked(True)
        self.diag_plot.setMouseEnabled(x=False, y=False)

        # 기준 가이드라인 
        pen_guide = pg.mkPen(color=COLOR_CARD_BORDER, width=1, style=Qt.PenStyle.DashLine)
        self.diag_plot.addLine(x=0, pen=pen_guide)
        self.diag_plot.addLine(y=0, pen=pen_guide)

        # 채널별 벡터 라인 생성 
        self.diag_lines = []

        for i in range(config.N_CH):
            line = pg.PlotCurveItem()
            self.diag_lines.append(line)
            self.diag_plot.addItem(line)

        lay.addWidget(self.diag_plot, 1)

        return frame

    def build_pwr_panel(self):

        frame, lay = self.card("PWR BARS")
        self.pwr_plot = pg.PlotWidget()
        self.pwr_plot.setBackground(COLOR_BG)

        # y축 범위 고정 
        self.pwr_plot.setYRange(0, 110, padding=0)
        self.pwr_plot.enableAutoRange(axis="y", enable=False)

        # x축 라벨 설정 
        x_axis = self.pwr_plot.getAxis("bottom")
        ticks = [(i, f"CH{i}") for i in range(config.N_CH)] + [(config.N_CH, "AVG")]
        x_axis.setTicks([ticks])

        # 막대 그래프 생성 
        self.bar_item = pg.BarGraphItem(
            x=np.arange(config.N_CH + 1),
            height=[0] * (config.N_CH + 1),
            width=0.6,
            brushes=[pg.mkBrush(get_ch_color(i)) for i in range(config.N_CH)] + [pg.mkBrush(SUM_BAR_COLOR)],
        )
        self.pwr_plot.addItem(self.bar_item)
        lay.addWidget(self.pwr_plot, 1)

        return frame

    # QTimer 콜백하면 graph_render.render(self) 호출
    def render(self):
        render_impl(self)

    def on_sample(self, raw_vals, amp_vals):
        #print("ON SAMPLE:")
        #print(raw_vals)
        
        if not self.is_running:
            return
        if len(raw_vals) != config.N_CH:
            return
        self.last_amp = amp_vals
        self.sample_count += 1

        # 현재 타임스탬프(ms): START 시점 기준
        curr_ts_ms = (time.time() - self.start_time_ref) * 1000

        # raw 데이터 링버퍼에 저장
        self.raw_np_buf[:, self.ptr] = raw_vals

        # 동적 오토스케일 계산 — throttled (every 4th sample) to lighten
        # the GUI thread so the gesture animation does not stall at session start.
        if self.sample_count % 4 == 0:
            for i in range(config.N_CH):
                self.scale_manager.scalers[i].update(raw_vals[i])
        
        # 링버퍼 포인터 이동 및 순환
        self.ptr += 1
        if self.ptr >= self.max_display:
            self.ptr = 0
            self.is_buf_full = True

        # 수신 속도 기반 버퍼 크기: START 후 FIRST_RESIZE_AFTER_SEC(2초) 시점에 한 번만 리사이즈
        # 첫 리사이즈는 임계값 없이 항상 적용 → 한 화면이 정확히 5초가 되도록
        elapsed = time.time() - self.start_time_ref
        if not self._has_resized_once and elapsed >= FIRST_RESIZE_AFTER_SEC and elapsed > 0:
            self._has_resized_once = True
            self._last_rate_update_time = time.time()
            rate = self.sample_count / elapsed
            new_len = int(round(rate * PLOT_SEC))
            new_len = max(MIN_BUF, min(MAX_BUF, new_len))
            if new_len != self.max_display:
                self._resize_raw_buffers(new_len)
        
        # csv 로깅 처리 
        if self.csv_logger:
            self.csv_logger.write_row(raw_vals, amp_vals, timestamp=curr_ts_ms)


    #def refresh_ports(self):
    #    self.cb_port.clear()
    #    self.cb_port.addItems([p.device for p in serial.tools.list_ports.comports()])

    def _choose_bracelet(self):
        from ble_selection import BraceletSelectorDialog
        BraceletSelectorDialog(self).exec()

    def start_serial(self):
        #port = self.cb_port.currentText().strip()
        #if not port:
        #    return
        self.raw_np_buf.fill(0)
        self.height_buf.fill(1.5)
        self.ptr = 0
        self.is_buf_full = False
        self.sample_count = 0
        self.start_time_ref = time.time()
        self._last_rate_update_time = self.start_time_ref
        self._has_resized_once = False  # START당 리사이즈 1회만

        # RAW 뷰를 채널 수에 맞게 0~y_max로 설정 (4ch/6ch 모두 전체 채널 보이도록)
        y_max = config.N_CH * CH_OFFSET
        self.raw_plot.setYRange(0, y_max, padding=0)
        self.raw_plot.getViewBox().setRange(yRange=(0, y_max))

        # 스케일 정보 초기화
        self.scale_manager.reset()

        self.csv_logger = None
        if ENABLE_CSV_LOGGING and (self.enable_logging or self.temp_log_dir):
            try:
                if self.enable_logging:
                    self.csv_logger = CSVLogger(
                        buffer_size=500,
                        profile=self.capture_profile,
                        batch=self.capture_batch,
                        gesture_set=self._current_gesture_set(),
                    )
                else:
                    self.csv_logger = CSVLogger(
                        buffer_size=500,
                        directory=self.temp_log_dir,
                    )
            except Exception as e:
                QMessageBox.critical(self, "Logger Error", str(e))
                return

        self.n_mult = self.sp_nmult.value()
        #self.worker.configure(port, 115200, self.n_mult)
        self.worker.start()
        self.set_running_ui(True)


    def stop_serial(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(500)
        if self.csv_logger:
            self.last_csv_path = self.csv_logger.filename
            self.csv_logger.close()
            self.csv_logger = None
            self.btn_view_last.setEnabled(True)
        self.set_running_ui(False)

    def set_status(self, txt):
        self.lbl_status.setText(f"● {txt}")
        self.lbl_status.setStyleSheet(
            f"color:{COLOR_STATUS_CONNECTED if 'CONNECTED' in txt else COLOR_STATUS_DISCONNECTED}; font-weight:800;"
        )

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.stop_serial()
    def toggle_protocol(self):
        """Start or stop the protocol mode."""
        if self.btn_protocol.isChecked():
            if not self.worker.isRunning():
                self.start_serial()
            else:
                self._start_recording_logger()
            gesture_set = self._current_gesture_set()
            from lib.configs import GESTURE_SETS
            self.protocol.start(self.csv_logger, list(GESTURE_SETS[gesture_set]))
            ...
            self.btn_protocol.setText("STOP PROTOCOL")
            self.rb_4cl.setEnabled(False)
            self.rb_6cl.setEnabled(False)
            self.rb_rps.setEnabled(False)
        else:
            self.protocol.stop()
            self.btn_protocol.setText("PROTOCOL MODE")

    def _on_protocol_step(self, label, remaining_ms, step_idx, total_steps):
        remaining_s = remaining_ms / 1000.0
        if label == "pause":
            color = "#888888"
            text = f">>> PAUSE ({remaining_s:.1f}s) <<<"
        else:
            color = "#2ed573"
            text = f">>> {label.upper()} ({remaining_s:.1f}s) <<<"

        # Look up the next step (if any)
        next_label = None
        if step_idx + 1 < len(self.protocol.sequence):
            next_label = self.protocol.sequence[step_idx + 1][0]

        # Compose multi-line label: current + next + step counter
        if next_label is not None:
            full_text = (
                f"{text}\n"
                f"<span style='font-size:14px; color:#9fb3ff;'>"
                f"Next: {next_label.upper()}</span>\n"
                f"<span style='font-size:11px; color:#999;'>"
                f"step {step_idx+1}/{total_steps}</span>"
            )
        else:
            full_text = (
                f"{text}\n"
                f"<span style='font-size:14px; color:#9fb3ff;'>End of session</span>\n"
                f"<span style='font-size:11px; color:#999;'>"
                f"step {step_idx+1}/{total_steps}</span>"
            )

        self.lbl_protocol.setText(full_text)
        self.lbl_protocol.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_protocol.setStyleSheet(
            f"font-size: 20px; font-weight: 900; color: {color}; padding: 8px;"
        )

    def _on_protocol_done(self):
        self.btn_protocol.setChecked(False)
        self.btn_protocol.setText("PROTOCOL MODE")
        self.rb_4cl.setEnabled(True)
        self.rb_6cl.setEnabled(True)
        self.rb_rps.setEnabled(True)
        self.lbl_protocol.setText("Protocol idle.")
        self.lbl_protocol.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_protocol.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #9fb3ff; padding: 6px;"
        )
        if self.worker.isRunning():
            self.stop_serial()

    def view_last(self):
        if not self.last_csv_path:
            return
        try:
            import subprocess
            import sys
            # Launch view_single_take.py in a separate Python process
            # to avoid matplotlib/Qt event-loop conflict.
            subprocess.Popen(
                [sys.executable, "view_single_take.py", self.last_csv_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        except Exception as e:
            QMessageBox.warning(self, "View Error", str(e))



    def _current_batch_dir(self):
        """Folder holding the current batch's takes: data/<profile>_BATCH<n>/.
        Mirrors CSVLogger's own path logic so both always agree."""
        if not self.capture_profile:
            return None
        import re
        safe = re.sub(r"\s+", "_", str(self.capture_profile).strip())
        try:
            n = int(self.capture_batch)
        except (TypeError, ValueError):
            n = 1
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "data", f"{safe}_BATCH{n}")

    def see_previous_takes(self):
        folder = self._current_batch_dir()
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(
                self, "Previous takes",
                "No takes recorded yet for this batch.")
            return
        import glob
        from PyQt6.QtWidgets import QDialog, QScrollArea
        dlg = QDialog(self)
        dlg.setWindowTitle("Previous takes")
        dlg.setStyleSheet("QDialog { background:#0e1422; }")
        dlg.resize(520, 500)
        outer = QVBoxLayout(dlg)
        title = QLabel("")
        title.setStyleSheet(
            "color:#ffffff; font-size:14px; font-weight:800; border:none;")
        outer.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        holder_w = QWidget()
        holder = QVBoxLayout(holder_w)
        holder.setSpacing(6)
        scroll.setWidget(holder_w)
        outer.addWidget(scroll, 1)

        def _clear():
            while holder.count():
                w = holder.takeAt(0).widget()
                if w is not None:
                    w.setParent(None)

        def _rebuild():
            _clear()
            csvs = sorted(glob.glob(os.path.join(folder, "*_emg.csv")))
            title.setText(
                f"{len(csvs)} take(s) in {os.path.basename(folder)} "
                "- view or delete before training")
            for i, c in enumerate(csvs, 1):
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(6)
                view_b = QPushButton(f"Take {i}    {os.path.basename(c)}")
                view_b.setStyleSheet(
                    "QPushButton { background:#1a2236; color:#e6e9f2;"
                    " text-align:left; border:1px solid #2a3550;"
                    " border-radius:6px; padding:8px 12px; }"
                    "QPushButton:hover { border:1px solid #3d4d75; }")
                view_b.clicked.connect(
                    lambda _=False, path=c: self._view_take_file(path))
                del_b = QPushButton("Delete")
                del_b.setFixedWidth(74)
                del_b.setStyleSheet(
                    "QPushButton { background:#2a1820; color:#ff6b6b;"
                    " border:1px solid #5a2a2a; border-radius:6px; padding:8px; }"
                    "QPushButton:hover { background:#3a2028; }")
                del_b.clicked.connect(
                    lambda _=False, path=c: self._delete_take_file(path, _rebuild))
                rl.addWidget(view_b, 1)
                rl.addWidget(del_b)
                holder.addWidget(row)
            if not csvs:
                empty = QLabel("No takes in this batch yet.")
                empty.setStyleSheet("color:#9fb3ff; border:none;")
                holder.addWidget(empty)

        _rebuild()
        close = QPushButton("Close")
        close.setStyleSheet(
            "QPushButton { background:#2a3550; color:#e6e9f2; border:none;"
            " border-radius:6px; padding:8px; }")
        close.clicked.connect(dlg.accept)
        outer.addWidget(close)
        dlg.exec()

    def _view_take_file(self, path):
        try:
            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "view_single_take.py", path],
                cwd=os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            QMessageBox.warning(self, "View Error", str(e))

    def _delete_take_file(self, path, refresh_cb):
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
        except OSError as e:
            QMessageBox.warning(self, "Delete take", f"Could not delete:\n{e}")
            return
        refresh_cb()

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()

    def _current_gesture_set(self):
        """The gesture set selected for this capture (driven by the radios,
        which AppWindow sets from the chosen profile)."""
        if self.rb_rps.isChecked():
            return "rps"
        if self.rb_4cl.isChecked():
            return "4cl"
        return "6cl"

    def set_capture_profile(self, profile, batch):
        """Set by AppWindow before recording so the CSV lands in
        data/<profile>_BATCH<batch>/."""
        self.capture_profile = profile
        self.capture_batch = batch
    
    def start_preview(self):
        """Connect and stream live WITHOUT recording or protocol, so the user
        can check sensor contact / signal quality before a session."""
        if self.worker.isRunning():
            return
        self.raw_np_buf.fill(0)
        self.height_buf.fill(1.5)
        self.ptr = 0
        self.is_buf_full = False
        self.sample_count = 0
        self.start_time_ref = time.time()
        self._last_rate_update_time = self.start_time_ref
        self._has_resized_once = False
        y_max = config.N_CH * CH_OFFSET
        self.raw_plot.setYRange(0, y_max, padding=0)
        self.raw_plot.getViewBox().setRange(yRange=(0, y_max))
        self.scale_manager.reset()
        self.csv_logger = None          # preview = pas d'enregistrement
        self.n_mult = self.sp_nmult.value()
        self.worker.start()
        self.set_running_ui(True)

    def _start_recording_logger(self):
        """Worker déjà en stream (preview) : repart sur une timeline propre et
        crée le logger CSV, sans reconnecter le BLE."""
        self.raw_np_buf.fill(0)
        self.height_buf.fill(1.5)
        self.ptr = 0
        self.is_buf_full = False
        self.sample_count = 0
        self.start_time_ref = time.time()
        self._has_resized_once = False
        self.scale_manager.reset()
        self.csv_logger = None
        if ENABLE_CSV_LOGGING and self.enable_logging:
            try:
                self.csv_logger = CSVLogger(
                    buffer_size=500,
                    profile=self.capture_profile,
                    batch=self.capture_batch,
                    gesture_set=self._current_gesture_set(),
                )
            except Exception as e:
                QMessageBox.critical(self, "Logger Error", str(e))