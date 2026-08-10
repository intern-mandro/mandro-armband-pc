import config
import numpy as np


# 채널별 EMG 신호의 동적 범위를 추적하고 baseline을 스무딩하는 스케일 관리 클래스
class ChannelScaler:

    def __init__(self):
        self.current_min = config.RAW_Y_MIN_INIT
        self.current_max = config.RAW_Y_MAX_INIT
        self.baseline = (self.current_min + self.current_max) / 2
        self.has_data = False

        self.baseline_alpha = 0.05
        self.decay_alpha = 0.000001

    # 스케일 정보를 초기 상태로 리셋
    def reset(self):
        self.current_min = config.RAW_Y_MIN_INIT
        self.current_max = config.RAW_Y_MAX_INIT
        self.baseline = (self.current_min + self.current_max) / 2
        self.has_data = False

    # 새 raw 값으로 current_min/max/baseline 갱신
    def update(self, raw_value):
        if raw_value <= config.RAW_ZERO_THRESHOLD:
            return

        self.has_data = True

        if raw_value > self.current_max:
            self.current_max = raw_value
        elif raw_value < self.current_min:
            self.current_min = raw_value
        else:
            # 범위 안이면 decay로 점진적 수축
            self.current_max -= (self.current_max - raw_value) * self.decay_alpha
            self.current_min += (raw_value - self.current_min) * self.decay_alpha

        target_baseline = (self.current_max + self.current_min) / 2
        self.baseline += (target_baseline - self.baseline) * self.baseline_alpha


class EMGScaleManager:

    def __init__(self, n_channels=config.N_CH):
        self.scalers = [ChannelScaler() for _ in range(n_channels)]
        self.gains = [1.0] * n_channels  # 채널별 민감도 가중치 (1.0 기본)

    def reset(self):
        for scaler in self.scalers:
            scaler.reset()

    def _data_range_and_half_height(self):

        valid_mins = [s.current_min for s in self.scalers if s.has_data and s.current_min > 0]
        valid_maxs = [s.current_max for s in self.scalers if s.has_data and s.current_max > 0]
        
        # 데이터가 없으면 기본값 사용 
        if not valid_mins or not valid_maxs:
            global_min = config.RAW_Y_MIN_INIT
            global_max = config.RAW_Y_MAX_INIT
        else:
            global_min = min(valid_mins)
            global_max = max(valid_maxs)
        
        data_range = max(global_max - global_min, 20)
        safe_margin_factor = 0.85
        allowed_half_height = (config.CH_OFFSET / 2) * safe_margin_factor
        return data_range, allowed_half_height

    # EMG raw 신호를 전 채널 공통 스케일 기준으로 정규화해 화면 Y좌표로 변환하는 함수
    def get_scaled_array(self, ch_idx, raw_array):
        scaler = self.scalers[ch_idx]
        data_range, allowed_half_height = self._data_range_and_half_height()
        
        # 중앙점 좌표 계산 
        base_offset = (config.N_CH - 1 - ch_idx) * config.CH_OFFSET + (config.CH_OFFSET / 2)

        # 신호 없음(0 근처)인 경우 
        effective_raw = np.where(
            raw_array <= config.RAW_ZERO_THRESHOLD,
            config.RAW_ZERO_REF,
            raw_array,
        )

        # 정규화 및 범위 제한 
        ratios = (effective_raw - scaler.baseline) / (data_range / 2)
        ratios = np.clip(ratios, -1.0, 1.0)
        
        return base_offset + (ratios * allowed_half_height)   # 최종 Y좌표 
    
    
    # 진폭(amp)을 동적 스케일링 (0~1)
    def get_vector_intensity(self, ch_idx, amp_value):
        scaler = self.scalers[ch_idx]
        dynamic_half_range = (scaler.current_max - scaler.baseline)   # 현재 진폭이 전체 유효 범위 중에서 몇 %인지 계산 
        
        if dynamic_half_range < 20:
            dynamic_half_range = 20
        boost_gain = 1.3
        intensity = (amp_value / dynamic_half_range) * self.gains[ch_idx] * boost_gain
        return np.clip(intensity, 0.0, 1.0)