import time
import numpy as np
import pyqtgraph as pg

import config
from config import (
    get_ch_color, get_diag_directions, CH_OFFSET, NO_SIGNAL_VARIATION_RAW,
    FFT_WINDOW_SEC, FFT_SAMPLE_RATE_DEFAULT, FFT_MAX_HZ,
    FFT_APPLY_FILTER, FFT_FILTER_OUT_RANGES,
    PLOT_SEC, BAR_INTERVAL_MS,
    FFT_Y_GAIN,
)

try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def render(win):
    view_mode = getattr(win, "view_mode", "raw")
    if view_mode == "fft":
        if hasattr(win, "stacked_plots"):
            win.stacked_plots.setCurrentIndex(1)
        update_fft_graph(win)
        if not win.is_running or win.sample_count == 0:
            return
        update_diag_vector(win)
        update_power_info(win)
        return
    # Raw 모드
    if not win.is_running or win.sample_count == 0:
        return
    is_fill_mode = win.rb_fill.isChecked()
    unified_x_ms = win.x_axis[win.ptr % win.max_display]
    update_raw_graph(win, is_fill_mode, unified_x_ms)
    update_diag_vector(win)
    update_power_info(win)

LINE_HEIGHT_PX = 1.5  # Bar 모드 신호 없을 때 막대 기본 높이

# 필터 차수 (시간 영역 선필터용)
_FFT_FILTER_ORDER = 2


def _apply_time_domain_filter(samples: np.ndarray, fs: float, filter_ranges: list) -> np.ndarray:
    """시간 영역에서 지정 구간 제거. samples는 복사본으로 넣어도 됨 (in-place 수정)."""
    if not _HAS_SCIPY or not filter_ranges or len(samples) < 4:
        return samples
    x = samples.astype(float)
    nyq = fs / 2.0
    for (low_hz, high_hz) in filter_ranges:
        if high_hz <= 0 or high_hz >= nyq:
            continue
        try:
            if low_hz <= 0 or low_hz < 1.0:
                # 0~high_hz 제거 → 고역통과 (high_hz 위만 통과)
                b, a = butter(_FFT_FILTER_ORDER, high_hz, btype="high", fs=fs)
            else:
                # low_hz~high_hz 제거 → 밴드스탑
                b, a = butter(_FFT_FILTER_ORDER, [low_hz, high_hz], btype="bandstop", fs=fs)
            x = filtfilt(b, a, x)
        except Exception:
            continue
    return x


def update_fft_graph(win):
    # STOP 상태면 마지막 프레임 유지, 뷰(줌/팬)만 사용 가능하도록 아무것도 갱신하지 않음
    if not getattr(win, "is_running", True):
        return

    # 초기값 vs 실제 수신속도: 측정 가능하면 실제 속도로 주파수 축 결정
    if win.sample_count > 0 and hasattr(win, "start_time_ref"):
        elapsed = time.time() - win.start_time_ref
        if elapsed >= 0.1:
            fs = win.sample_count / elapsed  # 실제 수신속도 (Hz)
        else:
            fs = FFT_SAMPLE_RATE_DEFAULT  # 측정 전까지 초기값
    else:
        fs = FFT_SAMPLE_RATE_DEFAULT  # 미수신 시 초기값

    # 나이퀴스트: 표시 상한 = min(측정 fs/2, FFT_MAX_HZ)
    max_freq = min(fs / 2.0, FFT_MAX_HZ)
    max_freq = max(max_freq, 1.0)  # 0이 되지 않도록

    win.fft_plot.setXRange(0, max_freq, padding=0)
    # 0번 채널 위쪽 여유를 넉넉히 둬서 피크가 잘리지 않도록
    y_max_fft = config.N_CH * CH_OFFSET + (1.5 * CH_OFFSET)
    win.fft_plot.setYRange(0, y_max_fft, padding=0)
    if hasattr(win, "graph_panel_title_label") and win.graph_panel_title_label:
        win.graph_panel_title_label.setText(
            f"FFT (Frequency)  fs≈{fs:.0f}Hz · 0–{max_freq:.0f}Hz"
        )
    win.fft_plot.setTitle("")
    vb = win.fft_plot.getViewBox()
    vb.setLimits(xMin=0, xMax=max_freq, yMin=0, yMax=y_max_fft, minYRange=50, maxYRange=y_max_fft)

    # 0.1초 분량 샘플 수 → 2의 거듭제곱으로 올림(ceil_pow2), 단 max_display 초과 시 캡
    n_fft = int(fs * FFT_WINDOW_SEC)
    n_fft = min(win.max_display, max(8, n_fft))
    n_fft = 1 << int(np.ceil(np.log2(n_fft)))  # ceil to power of 2
    n_fft = min(n_fft, win.max_display)

    for i in range(config.N_CH):
        # 수신 전이거나 데이터 부족 시 해당 채널만 빈 라인 (is_running은 위에서 이미 체크됨)
        if win.ptr < 2:
            win.fft_lines[i].setData([], [])
            continue
        if win.is_buf_full:
            idx = (win.ptr - n_fft + np.arange(n_fft)) % win.max_display
            samples = win.raw_np_buf[i, idx].astype(float)
        else:
            take = min(win.ptr, n_fft)
            take = int(2 ** int(np.log2(max(2, take))))
            samples = win.raw_np_buf[i, win.ptr - take : win.ptr].astype(float)

        if len(samples) < 2:
            win.fft_lines[i].setData([], [])
            continue

        # config에서 켜 둔 경우: 주파수 계산 전에 시간 영역 필터 적용 (0~5Hz, 50~60Hz 등)
        if FFT_APPLY_FILTER:
            samples = _apply_time_domain_filter(samples.copy(), fs, FFT_FILTER_OUT_RANGES)

        # DC 제거 + Hann 윈도우 후 FFT
        samples = samples - np.mean(samples)
        samples = samples * np.hanning(len(samples))

        spectrum = np.fft.rfft(samples)
        mag = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(samples), 1.0 / fs)

        mask = freqs <= max_freq
        freqs = freqs[mask]
        mag = mag[mask]
        # scipy 없이 필터 켠 경우만 주파수 영역에서 구간 0으로 (폴백)
        if FFT_APPLY_FILTER and not _HAS_SCIPY:
            for low_hz, high_hz in FFT_FILTER_OUT_RANGES:
                kill = (freqs >= low_hz) & (freqs <= high_hz)
                mag[kill] = 0.0
        if len(mag) == 0:
            win.fft_lines[i].setData([], [])
            continue

        # 선형 스케일: 샘플 수로만 나눈 뒤, FFT_Y_GAIN으로 전체 크기만 조정
        n_samp = len(samples)
        mag_scaled = (mag / (n_samp + 1e-12)) * FFT_Y_GAIN

        # 채널별 밴드(center = base_offset)를 기준으로 매핑 (클리핑 없이 그대로 표시)
        base_offset = (config.N_CH - 1 - i) * CH_OFFSET + (CH_OFFSET / 2)
        band_half = CH_OFFSET * 0.45
        y_fft = base_offset - band_half + mag_scaled * (2 * band_half)
        win.fft_lines[i].setData(freqs, y_fft)


CHUNK_SIZE = 30  # Bar 높이 계산에 쓸 샘플 수 (30개씩)
gap_range = 5


def update_raw_graph(win, is_fill_mode, unified_x_ms):
    for i in range(config.N_CH):

        # 현재 Raw 스케일 범위 계산 
        data_range, allowed_half_height = win.scale_manager._data_range_and_half_height()
        max_bar_pixels = 2.0 * allowed_half_height
        half_range = max(data_range / 2.0, 1.0)

        # 구간별(CHUNK_SIZE) 진폭 계산 (height_buf는 line 쪽에서 안 쓰이면 fill 전용으로만 씀)
        for chunk_start in range(0, win.max_display, CHUNK_SIZE):
            chunk = win.raw_np_buf[i, chunk_start : chunk_start + CHUNK_SIZE]

            if chunk.size > 0:
                ch_max = np.nanmax(chunk)
                ch_min = np.nanmin(chunk)
                if (ch_max - ch_min) < NO_SIGNAL_VARIATION_RAW:
                    new_h = LINE_HEIGHT_PX
                else:
                    bar_height_raw = ch_max - ch_min
                    ratio = min(bar_height_raw / half_range, 1.0)
                    new_h = max(ratio * max_bar_pixels, LINE_HEIGHT_PX)
                win.height_buf[i, chunk_start : chunk_start + CHUNK_SIZE] = new_h

        base_offset = (config.N_CH - 1 - i) * CH_OFFSET + (CH_OFFSET / 2)

        # fill 모드: 바 간격을 시간(ms) 기준으로 고정 → max_display가 바뀌어도 간격 일정
        if is_fill_mode:
            win.past_lines[i].setVisible(False)
            win.raw_lines[i].setVisible(False)

            x_max_ms = PLOT_SEC * 1000
            num_bars = int(x_max_ms / BAR_INTERVAL_MS)
            bar_x = np.arange(num_bars, dtype=float) * BAR_INTERVAL_MS
            display_heights = np.zeros(num_bars)

            for b in range(num_bars):
                center_idx = int(b * win.max_display / num_bars)
                idx = (center_idx - CHUNK_SIZE // 2 + np.arange(CHUNK_SIZE)) % win.max_display
                if idx.size < CHUNK_SIZE:
                    continue
                chunk = win.raw_np_buf[i, idx]
                ch_max = np.nanmax(chunk)
                ch_min = np.nanmin(chunk)
                if (ch_max - ch_min) < NO_SIGNAL_VARIATION_RAW:
                    h = LINE_HEIGHT_PX
                else:
                    bar_height_raw = ch_max - ch_min
                    ratio = min(bar_height_raw / half_range, 1.0)
                    h = max(ratio * max_bar_pixels, LINE_HEIGHT_PX)
                display_heights[b] = h
                # gap: 링 버퍼 경계 또는 미충전 시 ptr 이후
                if not win.is_buf_full:
                    t_bar = b * BAR_INTERVAL_MS
                    t_gap = (win.ptr / win.max_display) * x_max_ms
                    if t_bar >= t_gap:
                        display_heights[b] = 0

            # 버퍼가 찬 경우: 경계(ptr) 기준 앞쪽 5개 바만 비움
            if win.is_buf_full:
                gap_bar = int(win.ptr * num_bars / win.max_display)
                for k in range(gap_range):
                    display_heights[(gap_bar + k) % num_bars] = 0

            win.bar_items[i].setOpts(
                x=bar_x,
                height=display_heights,
                y0=base_offset - (display_heights / 2),
                width=min(20, BAR_INTERVAL_MS * 0.6),
            )
            win.bar_items[i].setVisible(True)

            # 커서 x: 마지막 샘플 위치(ptr 기준)
            last_x = win.x_axis[(win.ptr - 1) % win.max_display] if win.sample_count > 0 else unified_x_ms
            win.cursor_rects[i].setData(pos=[(last_x, base_offset)])


        # line 모드 
        else:
            win.bar_items[i].setVisible(False)

            # 링 버퍼가 안 찼을 때 
            if not win.is_buf_full:
                win.past_lines[i].setData([], [])
                win.past_lines[i].setVisible(False)

                if win.ptr <= 0:
                    x, y = np.array([]), np.array([])

                # 0 ~ ptr 구간 좌표
                else:
                    x = win.x_axis[: win.ptr]
                    y = win.scale_manager.get_scaled_array(i, win.raw_np_buf[i, : win.ptr])

            # 링 버퍼 한 바퀴 이상: past(ptr~끝) + current(0~ptr)
            else:
                # ptr 이후 구간은 과거 데이터 
                x_past = win.x_axis[win.ptr :]
                y_past = win.scale_manager.get_scaled_array(
                    i, win.raw_np_buf[i, win.ptr :]
                )
                win.past_lines[i].setData(x_past, y_past)
                win.past_lines[i].setVisible(True)

                # 0 ~ ptr 구간은 현재 구간 
                if win.ptr <= 0:
                    x, y = np.array([]), np.array([])
                else:
                    x = win.x_axis[: win.ptr]
                    y = win.scale_manager.get_scaled_array(
                        i, win.raw_np_buf[i, : win.ptr]
                    )
            # 현재 구간 라인 표시 
            win.raw_lines[i].setData(x, y)
            win.raw_lines[i].setVisible(True)

            
            # 커서 표시 (x, y 모두 마지막 샘플 = ptr 기준)
            if win.sample_count > 0:
                prev_idx = (win.ptr - 1) % win.max_display
                last_x = win.x_axis[prev_idx]
                y_cursor = win.scale_manager.get_scaled_array(
                    i, np.array([win.raw_np_buf[i, prev_idx]])
                )[0]
                win.cursor_rects[i].setData(pos=[(last_x, y_cursor)])
            else:
                win.cursor_rects[i].setData(pos=[])


def update_diag_vector(win):

    directions = get_diag_directions()
    DATA_LEN = 100

    for i in range(config.N_CH):
        if win.ptr >= DATA_LEN:
            diag_raw = win.raw_np_buf[i, win.ptr - DATA_LEN : win.ptr]
        
        # ptr이 DATA_LEN보다 작을 때 
        else:
            diag_raw = (
                # 버퍼가 한 바퀴 돈 경우에는 '뒤+앞'을 이어 붙임 
                np.concatenate(
                    [win.raw_np_buf[i, -DATA_LEN + win.ptr :], win.raw_np_buf[i, : win.ptr]]
                )
                if win.is_buf_full
                # 버퍼가 안찼으면 현재 데이터까지만 사용 
                else win.raw_np_buf[i, : win.ptr]
            )

        if diag_raw.size < 2:
            continue

        # ratio(강도) 및 파형 정규화 
        scaler = win.scale_manager.scalers[i]

        # 채널별 진폭을 0~1로 변환 
        ratio = win.scale_manager.get_vector_intensity(i, win.last_amp[i])
        
        # 신호 크기(정규화된 파형)
        denom = max(scaler.current_max - scaler.baseline, 30)
        wave = (diag_raw - scaler.baseline) / denom


        # 벡터 길이 val 계산 
        val = ratio * 1.4 * win.diag_plot_limit

        # 너무 짧으면 안그림 
        if val < 2.0:
            win.diag_lines[i].setData([], [])
            continue

        # 방향 벡터 & 수직 벡터 계산 
        dx, dy = directions[i]
        perp_x, perp_y = -dy, dx

        # 진행 거리 t (대각선 방향 진행 거리)
        t = np.linspace(0, val, len(diag_raw))

        # 물결의 흔들림 크기 
        wave_amp = 15 * ratio * (t / (val + 1e-6))
        
        # 최종 x, y 좌표 계산 (뻗는 방향 + 수직 방향)
        res_x = (t * dx) + (perp_x * wave * wave_amp)
        res_y = (t * dy) + (perp_y * wave * wave_amp)

        # 시작점 (원점)
        res_x[0], res_y[0] = 0, 0

        # 선 데이터 적용 
        win.diag_lines[i].setData(res_x, res_y)

        # 투명도 
        color = pg.mkColor(get_ch_color(i))
        color.setAlpha(int(np.clip(80 + (ratio * 175), 80, 255)))

        # 선 굵기 
        win.diag_lines[i].setPen(pg.mkPen(color=color, width=1.5 + (ratio * 3.0)))


# PWR 그래프 막대 높이 갱신 
def update_power_info(win):

    ratios = np.array([
        win.scale_manager.get_vector_intensity(i, win.last_amp[i])
        for i in range(config.N_CH)
    ])
    height_pct = np.clip(ratios * 100.0, 0, 100)
    avg_pct = float(np.clip(np.mean(ratios) * 100.0, 0, 100))
    win.bar_item.setOpts(height=list(height_pct) + [avg_pct])
