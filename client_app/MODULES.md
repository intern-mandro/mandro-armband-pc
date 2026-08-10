# EMG Dashboard — 모듈 구조와 데이터 흐름

> **대상: 개발자**  
> 사용 방법·핵심 기능은 [README.md](README.md)를 보면 됩니다.  
> 이 문서는 **파일별 역할**과 **데이터 흐름**을 보고 싶을 때 읽습니다.

---

## 목차

1. [한눈에 보기: 파일별 역할](#1-한눈에-보기-파일별-역할)
2. [데이터가 도는 순서](#2-데이터가-도는-순서)
3. [모듈별 상세](#3-모듈별-상세)
4. [동적 버퍼·동적 스케일 요약](#4-동적-버퍼동적-스케일-요약)

---

## 1. 한눈에 보기: 파일별 역할

| 파일 | 역할 (한 줄) |
|------|----------------|
| **main.py** | 앱 실행 진입점. 창 띄우고 이벤트 루프 실행 |
| **config.py** | 채널 수 초기값, FPS, 색상, 버퍼 한계 등 전역 상수 |
| **dashboard_ui.py** | 메인 창·패널 UI, 시리얼 연결/해제, 수신 데이터 처리·버퍼·렌더 타이머 |
| **serial_worker.py** | 시리얼 수신, 한 줄 파싱·채널 수 자동 감지, 진폭 계산, UI로 시그널 전달 |
| **graph_render.py** | RAW / Diagonal Vector / PWR 그래프 그리기 (데이터 읽기만) |
| **emg_scale.py** | 채널별 min·max·baseline, Y축·진폭 비율 계산 |
| **logger.py** | Raw·진폭·시간(ms) CSV 저장 |

---

## 2. 데이터가 도는 순서

```
[센서] → 시리얼로 한 줄씩 전송 (예: "123 156 119 72\n")
    ↓
serial_worker.run()
    · 줄 단위로 읽어서 parse_line() → 숫자 4개 또는 6개면 (값 리스트, 개수) 반환
    · 첫 유효 줄에서 채널 수(4 또는 6) 감지 → sig_channel_detected(n) 발송
    · 이후 같은 세션에서는 그 개수만 파싱, sample_buf에 누적
    · n_samples개마다 진폭 계산 → sig_sample.emit(raw_vals, last_amp)
    ↓
dashboard_ui.on_sample(raw_vals, amp_vals)
    · 채널 수가 아직 안 맞추어졌으면 무시 (sig_channel_detected 먼저 처리됨)
    · raw_np_buf에 저장, scale_manager 갱신, ptr 이동/링 버퍼 순환
    · 1초마다 수신 속도(rate) 계산 → 필요 시 버퍼 크기 조정(최근 5초 분량)
    · CSV 로거에 한 행 기록
    ↓
QTimer (약 30 FPS)
    · render() → graph_render.render(win)
    · RAW / Diagonal Vector / PWR 그래프만 갱신 (win의 버퍼/스케일 읽기)
    ↓
[화면에 표시]
```

- 채널 자동 감지·동적 버퍼 상세는 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) §3.4·§5 참고.

---

## 3. 모듈별 상세

각 파일의 역할 요약입니다. **메서드·함수별 역할 표**는 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) §3.1 참고.

| 파일 | 요약 |
|------|------|
| **main.py** | QApplication·EMGDashboard 생성·표시, 이벤트 루프 실행. |
| **config.py** | N_CH, PLOT_SEC, FPS, 버퍼 한계(MIN_BUF·MAX_BUF 등), 색상·스케일 상수. 전체 목록은 PROJECT_DOCUMENTATION §6. |
| **dashboard_ui.py** | 메인 창·패널, 시리얼 연결/해제, on_sample(버퍼·스케일러·동적 버퍼·CSV), on_channel_detected, render. |
| **serial_worker.py** | 시리얼 수신, parse_line(4/6개만 유효), 첫 줄 채널 감지·sig_channel_detected, 진폭 계산·sig_sample. |
| **graph_render.py** | render, update_raw_graph, update_diag_vector, update_power_info — win 버퍼·스케일 읽기만. |
| **emg_scale.py** | ChannelScaler(min/max/baseline), EMGScaleManager(data_range, get_scaled_array, get_vector_intensity). |
| **logger.py** | CSVLogger, write_row·flush·close. ENABLE_CSV_LOGGING일 때만 사용. 상세는 PROJECT_DOCUMENTATION §12. |

---

## 4. 동적 버퍼·동적 스케일 요약

목적·방법 상세는 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) §5 참고.

- **동적 버퍼**: 수신 속도(rate)에 따라 "최근 5초" 분량으로 버퍼 길이 조정. 1초마다 재계산, 15% 이상 차이 시 리사이즈.
- **동적 스케일**: RAW Line/Bar는 공통 data_range, Diagonal·PWR은 채널별 dynamic_half_range.

---

**참고:** 프로토콜이나 채널 수 동작을 바꿀 때는 `serial_worker.py`의 `parse_line`·첫 줄 감지·시그널, 그리고 `config.py`·`dashboard_ui.py`의 채널 관련 부분을 함께 보면 됩니다.
