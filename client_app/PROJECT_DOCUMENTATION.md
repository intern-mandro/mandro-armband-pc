# EMG Dashboard — 프로젝트 상세 문서

> **대상: 개발자**  
> 설정값·아키텍처·알고리즘·용어를 자세히 참조할 때 사용합니다.  
> 사용 방법은 [README.md](README.md), 모듈 구조·데이터 흐름은 [MODULES.md](MODULES.md)를 먼저 보면 됩니다.

이 문서는 개발 환경, 아키텍처, 핵심 로직, UI, **설정값·용어·수식** 등을 한곳에 모은 **참조용** 정리입니다.  

---

## 목차 (찾기 쉽게)

| 번호 | 섹션 | 내용 한 줄 |
|------|------|------------|
| 1 | 개발 환경 요구사항 | Python·라이브러리·실행 방법 |
| 2 | 아키텍처 개요 | MVC 유사 구조, 클래스·위젯 역할 |
| 3 | 핵심 로직 상세 | 파일별 메서드, 동작 순서, 스레드, 프로토콜 |
| 4 | UI 구성 요소 | 디자인·레이아웃 |
| 5 | 주요 기능 및 알고리즘 | 데이터 구조, 스케일링·동적 버퍼 요약 |
| 6 | 설정값·상수 일람 | config·버퍼·스케일 상수 표 |
| 7 | 용어 정의 | RAW, AMP, baseline, ptr 등 |
| 8 | 시그널·슬롯·이벤트 | Qt 시그널·버튼·타이머 |
| 9 | RAW 링 버퍼·구간 분리 | ptr, is_buf_full, 과거/현재 구간 |
| 10 | ChannelScaler·baseline | min/max/baseline 갱신 로직 |
| 11 | Diagonal Vector 수식 | 방향·파형·좌표 계산 |
| 12 | CSV 로거 상세 | 파일명·헤더·write_row |
| 13 | 에러 처리·종료 | 시리얼·CSV·closeEvent |

---

## 1. 개발 환경 요구사항

### 1.1 소프트웨어

| 구분 | 요구 사항 | 용도 |
|------|-----------|------|
| **언어** | Python 3.x | 전역 |
| **GUI** | PyQt6 | 메인 윈도우, 위젯(버튼·콤보박스·스핀박스·라디오·레이아웃), 스타일시트 |
| **그래프** | pyqtgraph | RAW 시계열, Diagonal Vector, PWR 막대 (PlotWidget, PlotCurveItem, BarGraphItem, ScatterPlotItem) |
| **수치** | NumPy | 버퍼·배열, min/max/clip, 스케일·비율 계산 |
| **시리얼** | pyserial | 포트 열기/읽기, 줄 단위 또는 바이트 단위 파싱 |
| **표준 라이브러리** | `sys`, `re`, `time`, `csv`, `os`, `datetime`, `collections.deque` | 진입점, 정규식 파싱, 타이밍, 로깅, deque 버퍼 |

- **실행**: [README.md](README.md) 참고. 시리얼 포트 선택 후 START로 수신 시작.

### 1.2 설치 및 실행 절차

설치·실행 절차는 [README.md](README.md)를 참고하세요.

### 1.3 하드웨어

- **시리얼 장치**: EMG 센서(또는 한 줄로 N_CH개 값을 전송하는 장치). 프로토콜 상세는 §3.4 참고.

### 1.4 그래픽·렌더 설정

- **pyqtgraph 전역 옵션** (`dashboard_ui.__init__`):
  - `antialias=True`: 선·곡선 안티앨리어싱.
  - `useOpenGL=True`: GPU 가속으로 고속 렌더.
- **렌더 주기**: `FPS=30` → `QTimer` 간격 `1000/30` ms. `is_running`·`sample_count == 0`이면 그리기 생략.
- **RAW 플롯**: Y축 숫자 비표시(`showValues=False`), X축 단위 ms, ViewBox 범위·줌 제한 설정.

---

## 2. 아키텍처 개요

### 2.1 역할 분리 (MVC 유사)

| 역할 | 담당 모듈 | 설명 |
|------|-----------|------|
| **View + Controller** | `dashboard_ui.py` | 메인 윈도우, 패널·위젯 구성, 사용자 입력(START/STOP/포트/Window Size), 시그널 슬롯 연결, 타이머로 주기적 렌더 호출 |
| **Model (데이터·스케일)** | `emg_scale.py`, 버퍼(`raw_np_buf`, `height_buf`, `last_amp` 등) | 채널별 min/max/baseline, 공통 data_range, 스케일/비율 계산; RAW·AMP 데이터 보관 |
| **데이터 수집·파싱** | `serial_worker.py` | 시리얼 수신, 줄/프레임 파싱, 진폭 계산, 시그널로 UI에 전달 |
| **시각화** | `graph_render.py` | RAW 그래프(Line/Bar), FFT(주파수 스펙트럼), Diagonal Vector, PWR 막대의 좌표·아이템 갱신 (모델은 건드리지 않고 win 참조만) |
| **설정** | `config.py` | 채널 수, FPS, 색상, 오프셋, 스케일 관련 상수 |
| **부가 I/O** | `logger.py` | CSV 로깅(버퍼 후 일괄 flush) |

- **진입점**: `main.py` — `QApplication` 생성, `EMGDashboard` 생성·표시, `app.exec()`.

### 2.2 핵심 클래스·객체

| 클래스/객체 | 파일 | 역할 |
|-------------|------|------|
| `EMGDashboard` | dashboard_ui | QMainWindow 서브클래스; 전체 UI, 버퍼, 타이머, 시리얼 워커·시그널 연결 |
| `SerialWorker` | serial_worker | QThread; 시리얼 루프, 파싱, 진폭 계산, `sig_sample`/`sig_status`/`sig_error` |
| `EMGScaleManager` | emg_scale | 채널별 `ChannelScaler` 리스트, `get_scaled_array`, `get_vector_intensity`, `_data_range_and_half_height()` |
| `ChannelScaler` | emg_scale | 채널당 min/max/baseline 유지, `update(raw_value)` |
| `CSVLogger` | logger | 세션당 CSV 파일, 버퍼·flush·close |

### 2.3 주요 위젯·아이템

| 위젯/아이템 | 소유 | 역할 |
|-------------|------|------|
| `raw_plot` | EMGDashboard | pyqtgraph PlotWidget; RAW 시계열(Line 또는 Bar) |
| `fft_plot` | EMGDashboard | pyqtgraph PlotWidget; 채널별 FFT 주파수 스펙트럼 |
| `stacked_plots` | EMGDashboard | RAW/FFT PlotWidget을 전환하는 QStackedWidget (Line/Fill/FFT 라디오와 연동) |
| `past_lines`, `raw_lines`, `bar_items`, `cursor_rects` | EMGDashboard | 채널별 라인·막대·커서(ScatterPlotItem) |
| `fft_lines` | EMGDashboard | FFT 플롯에서 채널별 스펙트럼 PlotCurveItem 리스트 |
| `diag_plot`, `diag_lines` | EMGDashboard | Diagonal Vector 플롯 및 채널별 PlotCurveItem |
| `pwr_plot`, `bar_item` | EMGDashboard | PWR 막대 1개 BarGraphItem (N_CH+1개: CH0~CH(N-1), AVG) |
| `cb_port`, `btn_start`, `btn_stop`, `sp_nmult`, `lbl_status` | EMGDashboard | 설정 패널 컨트롤 |
| `rb_line`, `rb_fill`, `rb_fft` | EMGDashboard | RAW Line / Fill(Bar) / FFT 모드 선택 라디오 버튼 |

---

## 3. 핵심 로직 상세

### 3.1 파일별 주요 메서드·함수

파일마다 누가 무엇을 하는지를 보여줍니다. 

**main.py**

| 메서드 | 역할 |
|--------|------|
| `main` | QApplication 생성, EMGDashboard 생성·표시, 이벤트 루프 실행 |

**dashboard_ui.py**

| 메서드 | 역할 |
|--------|------|
| `__init__` | 버퍼·스케일러·UI 초기화, 워커·타이머·시그널 연결 |
| `init_ui` | 좌/우 레이아웃, 설정·RAW·Diagonal·PWR 패널 배치 |
| `build_settings_panel` | 포트, Refresh, START/STOP, Window Size, 상태 라벨 |
| `build_raw_plot_panel` | RAW/FFT 플롯 패널 구성. Line/Fill/FFT 라디오, RAW PlotWidget·FFT PlotWidget, 채널별 라인·막대·커서, QStackedWidget 전환 |
| `build_diag_panel` | Diagonal Vector 플롯, 가이드 라인, diag_lines N_CH개 |
| `build_pwr_panel` | PWR PlotWidget, BarGraphItem(N_CH+1), CH0~AVG 틱 |
| `render` | graph_render.render(win) 호출 |
| `on_sample` | raw/amp 수신 → 버퍼·스케일러 갱신, 동적 버퍼 조정, CSV 기록 |
| `start_serial` | 버퍼 초기화, scale_manager.reset(), CSVLogger(옵션), worker 시작 |
| `stop_serial` | worker 중지·대기, csv_logger 종료 |
| `set_running_ui` | START/STOP/포트/Refresh/Window Size 활성·비활성 |

**serial_worker.py**

| 메서드 | 역할 |
|--------|------|
| `parse_line(line)` | 한 줄에서 숫자 추출. 4개 또는 6개일 때만 (값 리스트, 개수) 반환 |
| `compute_amp_from_samples(sample_buf)` | deque → 채널별 (max−min) 진폭 배열 |
| `run` | 시리얼 열기, 줄 단위 읽기·파싱, AMP 계산, sig_sample·sig_channel_detected 발송 |
| `update_params(n_mult)` | n_samples·sample_buf 재설정 |

**graph_render.py**

| 메서드 | 역할 |
|--------|------|
| `render(win)` | is_running·sample_count 확인 후 `view_mode`에 따라 RAW 또는 FFT 그래프를 갱신하고, 공통으로 Diagonal/PWR 갱신 |
| `update_raw_graph(win, ...)` | height_buf 계산, Line/Bar 분기, Y 좌표·커서 (RAW 모드에서만 호출) |
| `update_fft_graph(win)` | RAW 버퍼에서 최근 샘플을 추출해 FFT 수행, 주파수·진폭 스펙트럼을 FFT 플롯에 채널별로 표시 |
| `update_diag_vector(win)` | 채널별 최근 100샘플, 방향 벡터, diag_lines setData |
| `update_power_info(win)` | get_vector_intensity → 0~100% 막대 높이·AVG |

**emg_scale.py**

| 메서드 | 역할 |
|--------|------|
| `_data_range_and_half_height()` | 전 채널 global_min/max → data_range, allowed_half_height |
| `get_scaled_array(ch_idx, raw_array)` | raw → 비율 → Y 좌표 (공통 data_range 기준) |
| `get_vector_intensity(ch_idx, amp_value)` | 진폭 → 0~1 강도 (Diagonal·PWR용) |

**logger.py**

| 메서드 | 역할 |
|--------|------|
| `write_row(...)` | 한 행 버퍼 추가, buffer_size 도달 시 flush |
| `flush`, `close` | writerows·파일 닫기 |

---

### 3.2 동작 흐름 (요약)

앱이 켜진 뒤부터 데이터가 화면에 나올 때까지의 순서입니다.

| 단계 | 내용 |
|------|------|
| **1. 기동** | main → EMGDashboard 생성 → init_ui → 타이머 start, refresh_ports |
| **2. START** | start_serial → 버퍼 초기화, worker.configure·start → run() 진입 |
| **3. 수신 루프** | SerialWorker: 시리얼 읽기 → 줄 단위 split → parse_line → sample_buf에 누적, n_samples마다 진폭 계산 → sig_sample.emit(raw_vals, last_amp) |
| **4. UI 수신** | on_sample: raw_np_buf 기록, scale_manager 갱신, ptr·링 버퍼 처리, START 후 FIRST_RESIZE_AFTER_SEC(5초) 시점에 한 번만 수신 속도로 버퍼 길이 재계산·리사이즈, CSV 기록 |
| **5. 렌더** | QTimer → render(win) → `view_mode`가 `raw`이면 update_raw_graph, `fft`이면 update_fft_graph 호출 후, 공통으로 update_diag_vector·update_power_info 실행 (is_running·sample_count 확인 후) |
| **6. STOP** | stop_serial → worker.stop·wait, csv_logger.close, set_running_ui(False) |

---

### 3.3 스레드 안정성

어떤 일이 어느 스레드에서 일어나는지를 보여줍니다. 

| 담당 | 스레드 | 설명 |
|------|--------|------|
| 시리얼 수신·파싱·진폭 계산 | SerialWorker (QThread) | UI는 시그널로만 데이터 수신 |
| 버퍼·스케일러 갱신 | 메인 스레드 | on_sample에서만 raw_np_buf·ptr·scale_manager 수정 |
| 렌더 | 메인 스레드 | 타이머가 render 호출, win 버퍼는 읽기 전용 → 레이스 없음 |

---

### 3.4 시리얼 프로토콜

**데이터 형식 (텍스트, 구현됨)**

- 한 줄: 공백으로 구분된 숫자 + 줄바꿈(`\n`).
- 숫자 개수가 **4개**면 4채널, **6개**면 6채널로 자동 인식. 그 외는 무시.

**파싱 (parse_line)**

- 정규식으로 숫자만 추출.
- 개수가 4 또는 6일 때만 `([float, ...], 개수)` 반환, 아니면 `None`.
- 채널 수는 사용자 설정이 아니라 **줄당 값 개수**로만 결정.

**채널 자동 감지**

- SerialWorker가 첫 유효 줄에서 4 또는 6 감지 → `sig_channel_detected(n)` 발송.
- 대시보드가 N_CH·UI 재구성. STOP 후 다시 START하면 첫 줄부터 다시 감지.

**이진 23 bytes (문서화만, 미구현)**

- 헤더 2 + 데이터 20 + 체크섬 1 (XOR).

---

### 3.5 설정·실행 순서

1. **config 로드** — import 시 상수 로드.
2. **EMGDashboard 생성** — scale_manager, raw_np_buf, x_axis, height_buf, cursor_rects, last_amp, worker, 타이머.
3. **init_ui** — 패널 순서: settings → raw → diag → pwr. 좌측(settings, diag), 우측(raw, pwr).
4. **START 시** — 버퍼 0·height_buf 1.5, ptr=0, is_buf_full=False, sample_count=0, start_time_ref, CSVLogger(옵션), worker.configure·start.

---

## 4. UI 구성 요소

### 4.1 디자인 시스템

- **배경**: `#121826` (COLOR_BG). 카드 테두리: `#2a3550` (COLOR_CARD_BORDER).
- **카드**: QFrame, 배경 #121826, 테두리 1px solid #2a3550, border-radius 12px. 내부 여백 15px.
- **버튼**: 배경 #2a3550, hover #3d4d75, disabled #1c2538, border-radius 6px.
- **콤보/스핀**: 배경 #1c2538, 테두리 #3d4d75.
- **채널 색상**: CH_COLORS — 빨강·주황·초록·파랑 (`#ff4757`, `#ff9f43`, `#2ed573`, `#1e90ff`). AVG 막대: SUM_BAR_COLOR `#9fb3ff`.
- **상태**: CONNECTED `#2ed573`, DISCONNECTED `#ff6b6b`. 카드 제목·강조: `#9fb3ff`, 라벨 `#e6e9f2`.

### 4.2 레이아웃 구조

- **최상위**: QMainWindow → central QWidget → QHBoxLayout(main_layout).
- **좌측 (left_container, QVBoxLayout)**: panel_settings(0), panel_diag(1).
- **우측 (right_container, QVBoxLayout)**: panel_raw(5), panel_pwr(3).
- **main_layout**: left_container(1), right_container(2). Margins 15, spacing 15.
- **RAW/FFT 패널**: 카드 내부 — 제목+Line/Fill/FFT 라디오(header_layout), RAW·FFT PlotWidget을 담는 stacked_plots(1).
- **PWR 패널**: 카드 내부 — pwr_plot(1), BarGraphItem N_CH+1개(CH0~CH(N-1), AVG).

---

## 5. 주요 기능 및 알고리즘

### 5.1 데이터 구조

| 이름 | 타입 | 크기/형태 | 용도 |
|------|------|-----------|------|
| raw_np_buf | np.ndarray | (N_CH, max_display), float | RAW 시계열 링 버퍼. ptr로 기록 위치 관리. max_display는 수신 속도에 따라 동적 조정 |
| x_axis | np.ndarray | (max_display,) | 0 ~ PLOT_SEC*1000 ms 균등 분할. 버퍼 리사이즈 시 새 길이로 재생성 |
| height_buf | np.ndarray | (N_CH, max_display) | Bar 모드 구간별 막대 높이 캐시 |
| last_amp | np.ndarray | (N_CH,) | 최근 진폭(채널별 max−min) |
| sample_buf | deque | maxlen=n_samples | SerialWorker 내부, 최근 N샘플 (진폭 계산용) |
| CSVLogger.buffer | list | 최대 buffer_size(600) | 로그 행 누적 후 writerows 일괄 기록 |

### 5.2 색상·채널

- CH_COLORS: 4채널 순서대로 적용 (RAW 라인·막대·Diagonal·PWR). AVG는 SUM_BAR_COLOR.
- cursor_rects: 흰색(#ffffff) 사각형 ScatterPlotItem.

### 5.3 주요 기능 요약

- **RAW Line**: 링 버퍼 past/current 구간 분리, get_scaled_array로 Y 계산, 0 근처는 RAW_ZERO_REF(100) 위치에 표시.
- **RAW Bar**: step=30 구간별 max−min → data_range/2 기준 비율(Line과 동일) → height_buf, BarGraphItem setOpts.
- **FFT (Frequency)**: RAW 버퍼에서 최근 N샘플(window_sec 기준)을 추출해 FFT 수행, 주파수(Hz) vs 샘플 수로 정규화된 magnitude를 채널별 밴드에 표시. X축 상한은 min(fs/2, FFT_MAX_HZ).
- **Diagonal Vector**: 4ch는 4방향, 6ch는 6방향(각도 360°/N). 최근 100샘플, get_vector_intensity로 길이·펜 두께·알파.
- **PWR**: get_vector_intensity → 0~100% 높이, AVG는 N_CH개 채널 비율 평균.

### 5.4 최적화

- **진폭 계산**: n_samples개 샘플이 들어올 때마다 한 번 compute_amp_from_samples(sample_buf) 호출. (예: 윈도우 50이면 50줄마다 계산.)
- **Bar**: 구간(step) 단위로 height_buf 채운 뒤 sampled_indices로 setOpts 한 번에 전달.
- **렌더**: is_running·sample_count == 0이면 return. FPS 30으로 주기 제한.
- **CSV**: 버퍼가 buffer_size만큼 차면 writerows+flush. 시리얼은 in_waiting만큼 한 번에 read, 루프 끝 sleep(0.001).

### 5.5 스케일링 기법 (상세)

- **공통 data_range (RAW Line)**  
  - `_data_range_and_half_height`: has_data인 채널 중 current_min > 0, current_max > 0인 것만 모아 global_min/max 계산. 없으면 RAW_Y_MIN_INIT·RAW_Y_MAX_INIT 사용. data_range = max(global_max − global_min, 20).  
  - 같은 raw 변동폭이면 모든 채널에서 같은 세로 폭. baseline만 채널별.
- **get_scaled_array**  
  - effective_raw: raw_array ≤ RAW_ZERO_THRESHOLD(1.0)이면 RAW_ZERO_REF(100)으로 치환.  
  - ratios = (effective_raw − baseline) / (data_range/2), clip −1~1.  
  - Y = base_offset + ratios * allowed_half_height. base_offset = (N_CH−1−ch_idx)*CH_OFFSET + CH_OFFSET/2.
- **get_vector_intensity (Diagonal·PWR)**  
  - dynamic_half_range = current_max − baseline (최소 20).  
  - intensity = (amp_value / dynamic_half_range) * gains[ch_idx] * 1.3, clip 0~1.  
  - 채널별 “자기 관측 범위 대비” 비율.
- **RAW Bar**: 구간별 (ch_max−ch_min) / (data_range/2) 비율로 막대 높이. Line과 동일한 공통 data_range 사용 → 채널 간 비교 일치.


---

## 6. 설정값·상수 일람

| 구분 | 이름 | 값 | 위치 | 설명 |
|------|------|-----|------|------|
| **공통** | N_CH | 4 | config | 채널 수 초기값. 실제는 START 시 첫 줄에서 4 또는 6 자동 감지 |
| | FPS | 30 | config | 렌더 주기(Hz), 타이머 간격 = 1000/FPS ms |
| | PLOT_SEC | 5.0 | config | RAW/FFT에 표시할 시간(초). 목표는 “한 화면 ≈ PLOT_SEC초” |
| | max_display | 초기: RAW_SAMPLE_RATE_DEFAULT×PLOT_SEC | dashboard_ui | RAW 링 버퍼·x_axis 샘플 수. START 직후 예상 샘플 레이트로 초기화 후, FIRST_RESIZE_AFTER_SEC 시점에 실제 rate×PLOT_SEC로 한 번만 리사이즈 |
| | MIN_BUF, MAX_BUF | 100, 100000 | config | RAW 링 버퍼 길이 하한·상한 |
| | RATE_UPDATE_INTERVAL | 1.0 | config | (과거용) 수신 속도 재계산 주기(초). 현재는 FIRST_RESIZE_AFTER_SEC를 사용해 1회만 리사이즈 |
| | BUF_RESIZE_THRESHOLD | 0.15 | config | (과거용) 비율 차이 임계값. 현재는 초기 1회 리사이즈에는 적용하지 않음 |
| | FIRST_RESIZE_AFTER_SEC | 5.0 | config | START 후 이 시간(초)이 지난 시점에 실제 수신 속도로 버퍼 길이를 한 번만 재계산·리사이즈 |
| | RAW_SAMPLE_RATE_DEFAULT | 500 | config | 수신 속도 측정 전 RAW 버퍼 초기 크기를 잡을 때 사용하는 예상 샘플 레이트(Hz) |
| **시리얼/신호** | BASE_SAMPLES | 5 | config | 진폭 윈도우 기본 샘플 수 |
| | N_MULT_DEFAULT | 10 | config | n_samples = BASE_SAMPLES * n_mult |
| | (진폭 계산 주기) | n_samples | serial_worker | n_samples개 들어올 때마다 진폭 재계산 |
| **RAW 스케일** | RAW_Y_MIN_INIT, RAW_Y_MAX_INIT | 55, 100 | config | ChannelScaler 초기 min/max |
| | CH_OFFSET | 100 | config | 채널 밴드 세로 간격(px) |
| | RAW_ZERO_REF | 100 | config | 신호 없음일 때 표시할 raw 기준값 |
| | RAW_ZERO_THRESHOLD | 1.0 | config | 이 값 이하면 "신호 없음"으로 치환 |
| | safe_margin_factor | 0.85 | emg_scale | allowed_half_height 계산 |
| **ChannelScaler** | baseline_alpha | 0.05 | emg_scale | baseline 스무딩 계수 |
| | decay_alpha | 0.00001 | emg_scale | min/max 감쇠 계수 |
| **RAW Bar** | step | 30 | graph_render | 구간 크기, Bar 샘플 간격 |
| | gap_range | 5 | graph_render | Bar 갭 구간 수 |
| | NO_SIGNAL_VARIATION_RAW | 1.0 | config | Bar 모드: 구간 변동폭 < 이 값이면 최소 높이 |
| | LINE_HEIGHT_PX | 1.5 | graph_render | 신호 없음 구간 막대 최소 높이 |
| **FFT** | FFT_WINDOW_SEC | 0.8 | config | FFT에 사용할 시간 창 길이(초) |
| | FFT_SAMPLE_RATE_DEFAULT | 1000 | config | 수신 속도 측정 전에만 사용하는 FFT 샘플 레이트(Hz) |
| | FFT_MAX_HZ | 500 | config | FFT X축 최대 표시 주파수(Hz). 실제는 min(fs/2, FFT_MAX_HZ)를 사용 |
| | FFT_Y_GAIN | 0.3 | config | FFT 진폭 전체 배율 (mag/n_samp × FFT_Y_GAIN) |
| **Diagonal** | DATA_LEN | 100 | graph_render | 채널당 최근 100샘플 |
| | diag_plot_limit | 50 | dashboard_ui | diag_plot X/Y 범위 ±50 |
| | boost_gain | 1.3 | emg_scale | get_vector_intensity 부스트 |
| **CSV** | buffer_size | 500 | dashboard_ui | CSVLogger 생성 시 전달 |
| | directory | "data" | logger | CSV 기본 저장 폴더 |
| **Window Size** | sp_nmult | 1 ~ 100 (SpinBox setRange) | dashboard_ui | START 시 worker.configure(port, 115200, n_mult)에 전달 |

---

## 7. 용어 정의

| 용어 | 설명 |
|------|------|
| **RAW** | 시리얼에서 수신한 원시 값. 채널당 하나의 스칼라 per 샘플. |
| **AMP (진폭)** | 최근 N샘플 구간에서 채널별 (max − min). |
| **baseline** | ChannelScaler의 "중앙" 추정값. (min+max)/2 방향으로 스무딩. |
| **data_range** | 전 채널 공통. global_max − global_min (최소 20). |
| **allowed_half_height** | 채널 밴드 중심에서 위/아래로 쓸 수 있는 반높이(px). |
| **dynamic_half_range** | 채널별 (current_max − baseline), 최소 20. |
| **effective_raw** | raw ≤ RAW_ZERO_THRESHOLD이면 RAW_ZERO_REF(100)으로 치환한 배열. |
| **ptr** | raw_np_buf에 다음으로 쓸 인덱스. 0 ~ max_display−1 순환. |
| **is_buf_full** | 링 버퍼가 한 바퀴 이상 채워졌는지. |
| **n_samples** | 진폭 계산에 쓰는 샘플 수. BASE_SAMPLES * n_mult. |

---

## 8. 시그널·슬롯·이벤트

| 시그널/이벤트 | 소스 | 슬롯/동작 |
|---------------|------|-----------|
| sig_sample | SerialWorker | on_sample → 버퍼·ptr·스케일러·동적 버퍼 조정·CSV 갱신 |
| sig_channel_detected | SerialWorker | on_channel_detected → N_CH 갱신·reinit_channel_mode (UI만 n채널로 재구성) |
| sig_status | SerialWorker | set_status → lbl_status 텍스트·색상 |
| sig_error | SerialWorker | on_error → QMessageBox.critical, stop_serial |
| QTimer.timeout | QTimer | render → graph_render.render(win) |
| btn_start.clicked | - | start_serial |
| btn_stop.clicked | - | stop_serial |
| btn_refresh.clicked | - | refresh_ports |
| closeEvent | QMainWindow | stop_serial 후 event.accept() |

---

## 9. RAW 링 버퍼·구간 분리

- **버퍼**: raw_np_buf (N_CH, max_display). 매 샘플마다 raw_np_buf[:, ptr] = raw_vals, ptr += 1. ptr가 max_display에 도달하면 ptr=0, is_buf_full=True.
- **과거/현재 분리 (Line 모드)**: is_buf_full == True일 때 과거 = x_axis[ptr:], raw_np_buf[i, ptr:] → past_lines. 현재 = x_axis[:ptr], raw_np_buf[i, :ptr] → raw_lines. is_buf_full == False일 때 past_lines는 빈 데이터, 현재만 raw_lines에.
- **Y 좌표**: 두 구간 모두 get_scaled_array(ch_idx, raw_slice)로 변환 후 setData. 커서는 (ptr−1) 인덱스의 x, get_scaled_array로 구한 y 한 점.
- **채널 인덱스와 Y 방향**: base_offset = (N_CH−1−ch_idx)*CH_OFFSET + CH_OFFSET/2. ch_idx=0일 때 Y가 가장 크고(화면 상단), ch_idx=N_CH−1일 때 Y가 가장 작음(화면 하단).

---

## 10. ChannelScaler·baseline 로직 상세

- **초기값**: current_min = RAW_Y_MIN_INIT(55), current_max = RAW_Y_MAX_INIT(100), baseline = (min+max)/2. has_data = False.
- **update(raw_value)**: raw_value ≤ RAW_ZERO_THRESHOLD(1.0)이면 return(0은 min/max 갱신에서 제외). has_data = True. raw_value > current_max → current_max 갱신. raw_value < current_min → current_min 갱신. 그 외 → decay로 current_max/current_min 수축. target_baseline = (current_max + current_min)/2, baseline 스무딩(baseline_alpha).
- **의미**: 신호가 벗어나면 범위 확장, 안에 있으면 감쇠. baseline은 (min+max)/2 쪽으로 스무딩. 0은 센서 미인식 등으로 간주하고 min/max에 반영하지 않음.

---

## 11. Diagonal Vector 수식

- **방향 (4채널)**: (-0.707,-0.707), (-0.707,0.707), (0.707,0.707), (0.707,-0.707).
- **최근 100샘플**: ptr ≥ DATA_LEN(100)이면 diag_raw = raw_np_buf[i, ptr−100:ptr]. ptr < 100이면: is_buf_full이 True일 때 raw_np_buf[i, −100+ptr:] 와 raw_np_buf[i, :ptr]를 concatenate해 100개로 채움. is_buf_full이 False일 때는 diag_raw = raw_np_buf[i, :ptr]만 사용(길이 < 100). diag_raw.size < 2이면 해당 채널은 스킵(continue).
- **파형**: denom = max(current_max − baseline, 30), wave = (diag_raw − baseline) / denom.
- **벡터 길이**: val = ratio * 1.4 * diag_plot_limit. ratio = get_vector_intensity(i, last_amp[i]).
- **좌표**: t = linspace(0, val, len(diag_raw)). perp = (−dy, dx). res_x = t*dx + perp_x * wave * wave_amp, res_y = t*dy + perp_y * wave * wave_amp. wave_amp = 15*ratio*(t/(val+1e-6)). 시작점 (0,0) 고정.
- **스타일**: 펜 두께 1.5 + ratio*3, 알파 80 + ratio*175.

---

## 12. CSV 로거 상세

- **생성**: START 시 ENABLE_CSV_LOGGING이 True면 CSVLogger(buffer_size=500). directory 기본 "data".
- **파일명**: data/YYYYMMDD_HHMMSS_emg.csv.
- **헤더**: Time(ms), Raw_CH0~Raw_CH(N-1), Amp_CH0~Amp_CH(N-1). config.N_CH 기준. 한 번만 기록.
- **write_row**: timestamp는 start_serial에서 설정한 start_time_ref 기준 (time.time() - start_time_ref)*1000. 전달받은 timestamp가 있으면 그대로 사용. processed_raw = [int(float(v)) for v in raw_vals], processed_amp = [int(round(float(v))) for v in amp_vals]. 버퍼에 [relative_time_ms, raw0~N-1, amp0~N-1] 추가. len(buffer) >= buffer_size면 flush. ValueError/TypeError 발생 시 print만 하고 해당 행은 버퍼에 넣지 않음.
- **종료**: STOP 시 close() → flush 후 파일 닫기. flush 시 writerows + file.flush + buffer.clear. close 시 file.closed 확인 후 닫고 "Saved: ..." 출력.

---

## 13. 에러 처리·종료

- **시리얼 열기 실패**: run() 내 try에서 실패 시 sig_error.emit, return. on_error에서 QMessageBox.critical, stop_serial.
- **파싱 실패**: 줄 단위 try/except, 개별 라인 실패 시 무시.
- **루프 예외·종료**: run() 상위 try/except에서 예외 시 sig_error. run() 종료 시(정상/예외 무관) **finally 블록에서 항상 cleanup() 호출**. cleanup()에서 _ser 닫기, _ser = None, sig_status.emit("DISCONNECTED"). 대시보드에서 STOP 후 refresh_ports()로 포트 목록 갱신.
- **CSVLogger 생성 실패**: start_serial 내에서 예외 시 QMessageBox.critical, return.
- **closeEvent**: 창 닫을 때 stop_serial() 후 event.accept().

---

*이 문서는 EMG Dashboard 기준으로 작성되었으며, 4ch/6ch는 시리얼 한 줄의 숫자 개수(4 또는 6)로 자동 감지한다. 프로토콜·채널 동작 변경 시 MODULES.md와 serial_worker.py를 참고하면 된다.*
