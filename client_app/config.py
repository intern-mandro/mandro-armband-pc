import math

N_CH = 8
FPS = 60
PLOT_SEC = 5.0

MIN_BUF = 1000
MAX_BUF = 100000
RATE_UPDATE_INTERVAL = 1.0
BUF_RESIZE_THRESHOLD = 0.15
# 첫 리사이즈 전 대기 시간(초). 5초 뒤 한 번만 리사이즈
FIRST_RESIZE_AFTER_SEC = 5.0
# 수신 속도 측정 전 RAW 버퍼 초기 크기 = 이 값 × PLOT_SEC (5초 분량 가정)
RAW_SAMPLE_RATE_DEFAULT = 1000  # Hz

# [시리얼/신호 처리]
ENABLE_CSV_LOGGING = True
BASE_SAMPLES = 5
N_MULT_DEFAULT = 10

# [RAW 그래프 스케일]
CH_OFFSET = 100
BAR_INTERVAL_MS = 30
RAW_Y_MIN_INIT = 55
RAW_Y_MAX_INIT = 100
RAW_ZERO_REF = 100
RAW_ZERO_THRESHOLD = 1.0
NO_SIGNAL_VARIATION_RAW = 1.0

# [디자인 설정]
RAW_LINE_WIDTH = 1.6
COLOR_BG = "#121826"
COLOR_CARD_BORDER = "#2a3550"
COLOR_STATUS_CONNECTED = "#2ed573"
COLOR_STATUS_DISCONNECTED = "#ff6b6b"

CH_COLORS = [
    "#ff4757",   # 빨강
    "#ff9f43",   # 주황
    "#ffdd59",   # 노랑
    "#2ed573",   # 초록
    "#1e90ff",   # 파랑
    "#a55eea",   # 보라
    "#fe5dde",   # 보라
    "#fe1a89",   # 보라
]
CH_4_INDICES = (0, 1, 3, 4)


# 채널별 인덱스 색상 반환 
def get_ch_color(ch_idx):
    idx = CH_4_INDICES[ch_idx] if N_CH == 4 else ch_idx
    return CH_COLORS[idx]



def get_diag_directions():
    if N_CH == 4:
        angles_deg = [225, 135, 45, 315] 
    elif N_CH == 6:
        angles_deg = [225, 180, 135, 45, 0, 315]
    else:
        angles_deg = [240, 200, 160, 120, 60, 20, 340, 300]
    return [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angles_deg]


SUM_BAR_COLOR = "#9fb3ff"

# [FFT 시각화]
FFT_WINDOW_SEC = 0.8  # FFT에 쓸 구간 길이 (초)
FFT_SAMPLE_RATE_DEFAULT = 1000  # 초기값: 수신속도 측정 전에만 사용 (Hz)
FFT_MAX_HZ = 500
FFT_APPLY_FILTER = True
FFT_FILTER_OUT_RANGES = [(0, 5), (50, 60)]  # (low_hz, high_hz) 리스트
# FFT 전체 진폭 배율 (1.0 = 현재 값 그대로, 0.5 = 절반 크기, 2.0 = 두 배)
FFT_Y_GAIN = 0.3