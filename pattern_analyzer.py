import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    def __init__(self):
        self.window_size = {
            'default': 20,
            'long': 30,
            'extra_long': 50
        }
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """모든 차트 패턴 분석"""
        try:
            patterns = []
            current_price = df['close'].iloc[-1]
            
            # 패턴 체크 리스트
            pattern_checks = [
                (self._check_double_bottom, '이중 바닥', 'bullish', 1.15),
                (self._check_double_top, '이중 천장', 'bearish', 0.85),
                (self._check_ascending_triangle, '상승 삼각형', 'bullish', 1.10),
                (self._check_descending_triangle, '하락 삼각형', 'bearish', 0.90),
                (self._check_head_and_shoulders, '헤드앤숄더', 'bearish', 0.85),
                (self._check_inverse_head_and_shoulders, '역헤드앤숄더', 'bullish', 1.15),
                (self._check_cup_and_handle, '컵앤핸들', 'bullish', 1.12)
            ]
            
            for check_func, name, pattern_type, target_mult in pattern_checks:
                if check_func(df):
                    patterns.append({
                        'name': name,
                        'pattern_type': pattern_type,
                        'reliability': 'high',
                        'target': current_price * target_mult
                    })
            
            return {
                'patterns': patterns,
                'current_price': current_price,
                'analysis_time': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"패턴 분석 중 오류 발생: {e}")
            return {'patterns': [], 'error': str(e)}

    def _check_double_bottom(self, df: pd.DataFrame) -> bool:
        """이중 바닥 패턴 확인"""
        try:
            window = self.window_size['default']
            recent_df = df.tail(window).copy()
            prices = recent_df['low']
            
            # 저점 찾기
            bottoms = []
            for i in range(1, len(prices)-1):
                if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                    bottoms.append((i, prices.iloc[i]))
            
            if len(bottoms) < 2:
                return False
            
            # 마지막 두 저점 분석
            last_bottoms = bottoms[-2:]
            price_diff = abs(last_bottoms[0][1] - last_bottoms[1][1])
            price_threshold = last_bottoms[0][1] * 0.02
            time_diff = last_bottoms[1][0] - last_bottoms[0][0]
            
            return price_diff <= price_threshold and 3 <= time_diff <= 15
            
        except Exception as e:
            logger.error(f"이중 바닥 패턴 확인 중 오류: {e}")
            return False

    def _check_double_top(self, df: pd.DataFrame) -> bool:
        """이중 천장 패턴 확인"""
        try:
            window = self.window_size['default']
            recent_df = df.tail(window).copy()
            prices = recent_df['high']
            
            # 고점 찾기
            tops = []
            for i in range(1, len(prices)-1):
                if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                    tops.append((i, prices.iloc[i]))
            
            if len(tops) < 2:
                return False
            
            # 마지막 두 고점 분석
            last_tops = tops[-2:]
            price_diff = abs(last_tops[0][1] - last_tops[1][1])
            price_threshold = last_tops[0][1] * 0.02
            time_diff = last_tops[1][0] - last_tops[0][0]
            
            return price_diff <= price_threshold and 3 <= time_diff <= 15
            
        except Exception as e:
            logger.error(f"이중 천장 패턴 확인 중 오류: {e}")
            return False

    def _check_ascending_triangle(self, df: pd.DataFrame) -> bool:
        """상승 삼각형 패턴 확인"""
        try:
            window = self.window_size['default']
            recent_df = df.tail(window).copy()
            highs = recent_df['high']
            lows = recent_df['low']
            
            # 저점들의 기울기 계산
            low_points = []
            for i in range(1, len(lows)-1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    low_points.append((i, lows.iloc[i]))
            
            if len(low_points) < 2:
                return False
            
            # 저점들의 기울기가 양수인지 확인
            low_slope = np.polyfit([x[0] for x in low_points], [x[1] for x in low_points], 1)[0]
            
            # 고점들이 비슷한 수준인지 확인
            high_points = []
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    high_points.append((i, highs.iloc[i]))
            
            if len(high_points) < 2:
                return False
            
            high_values = [x[1] for x in high_points]
            high_std = np.std(high_values)
            high_mean = np.mean(high_values)
            
            return low_slope > 0 and high_std/high_mean < 0.02
            
        except Exception as e:
            logger.error(f"상승 삼각형 패턴 확인 중 오류: {e}")
            return False

    def _check_descending_triangle(self, df: pd.DataFrame) -> bool:
        """하락 삼각형 패턴 확인"""
        try:
            window = self.window_size['default']
            recent_df = df.tail(window).copy()
            highs = recent_df['high']
            lows = recent_df['low']
            
            # 고점들의 기울기 계산
            high_points = []
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    high_points.append((i, highs.iloc[i]))
            
            if len(high_points) < 2:
                return False
            
            # 고점들의 기울기가 음수인지 확인
            high_slope = np.polyfit([x[0] for x in high_points], [x[1] for x in high_points], 1)[0]
            
            # 저점들이 비슷한 수준인지 확인
            low_points = []
            for i in range(1, len(lows)-1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    low_points.append((i, lows.iloc[i]))
            
            if len(low_points) < 2:
                return False
            
            low_values = [x[1] for x in low_points]
            low_std = np.std(low_values)
            low_mean = np.mean(low_values)
            
            return high_slope < 0 and low_std/low_mean < 0.02
            
        except Exception as e:
            logger.error(f"하락 삼각형 패턴 확인 중 오류: {e}")
            return False

    def _check_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """헤드앤숄더 패턴 확인"""
        try:
            window = self.window_size['long']
            recent_df = df.tail(window).copy()
            highs = recent_df['high']
            
            # 고점 찾기
            peaks = []
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    peaks.append((i, highs.iloc[i]))
            
            if len(peaks) < 3:
                return False
            
            # 마지막 세 개의 고점 분석
            last_peaks = peaks[-3:]
            left_shoulder = last_peaks[0][1]
            head = last_peaks[1][1]
            right_shoulder = last_peaks[2][1]
            
            # 패턴 조건 확인
            head_higher = head > left_shoulder and head > right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / left_shoulder < 0.03
            
            return head_higher and shoulders_similar
            
        except Exception as e:
            logger.error(f"헤드앤숄더 패턴 확인 중 오류: {e}")
            return False

    def _check_inverse_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """역헤드앤숄더 패턴 확인"""
        try:
            window = self.window_size['long']
            recent_df = df.tail(window).copy()
            lows = recent_df['low']
            
            # 저점 찾기
            troughs = []
            for i in range(1, len(lows)-1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    troughs.append((i, lows.iloc[i]))
            
            if len(troughs) < 3:
                return False
            
            # 마지막 세 개의 저점 분석
            last_troughs = troughs[-3:]
            left_shoulder = last_troughs[0][1]
            head = last_troughs[1][1]
            right_shoulder = last_troughs[2][1]
            
            # 패턴 조건 확인
            head_lower = head < left_shoulder and head < right_shoulder
            shoulders_similar = abs(left_shoulder - right_shoulder) / left_shoulder < 0.03
            
            return head_lower and shoulders_similar
            
        except Exception as e:
            logger.error(f"역헤드앤숄더 패턴 확인 중 오류: {e}")
            return False

    def _check_cup_and_handle(self, df: pd.DataFrame) -> bool:
        """컵앤핸들 패턴 확인"""
        try:
            window = self.window_size['extra_long']
            if len(df) < window:
                return False
            
            recent_df = df.tail(window).copy()
            recent_df = recent_df.reset_index(drop=True)  # 인덱스 리셋
            prices = recent_df['close']
            
            # 데이터를 3등분하여 분석
            first_third = len(prices) // 3
            second_third = 2 * first_third
            
            # U자 모양 찾기
            mid_section = prices[first_third:second_third]
            if len(mid_section) == 0:
                return False
            
            min_idx = mid_section.idxmin()
            if min_idx is None:
                return False
            
            left_section = prices[:min_idx]
            right_section = prices[min_idx:]
            
            if len(left_section) == 0 or len(right_section) == 0:
                return False
            
            left_max = left_section.max()
            right_max = right_section.max()
            
            # 패턴 조건 확인
            if left_max == 0:  # 0으로 나누기 방지
                return False
            
            price_similar = abs(left_max - right_max) / left_max < 0.02
            cup_depth = (left_max - prices[min_idx]) / left_max
            valid_depth = 0.1 < cup_depth < 0.3
            
            return price_similar and valid_depth
            
        except Exception as e:
            logger.error(f"컵앤핸들 패턴 확인 중 오류: {e}")
            return False

    def find_support_resistance_levels(self, df: pd.DataFrame, current_price: float) -> Dict[str, List[float]]:
        """지지/저항 레벨 찾기"""
        try:
            # 지지선 찾기
            lows = df['low'].values
            supports = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    if lows[i] < current_price:
                        supports.append(lows[i])
            
            # 저항선 찾기
            highs = df['high'].values
            resistances = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    if highs[i] > current_price:
                        resistances.append(highs[i])
            
            # 레벨 필터링 및 정렬
            supports = self._filter_price_levels(supports, current_price, 'support')
            resistances = self._filter_price_levels(resistances, current_price, 'resistance')
            
            return {
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3]
            }
            
        except Exception as e:
            logger.error(f"지지/저항 레벨 분석 중 오류: {e}")
            return {'support_levels': [], 'resistance_levels': []}

    def _filter_price_levels(self, levels: List[float], current_price: float, level_type: str) -> List[float]:
        """가격 레벨 필터링"""
        unique_levels = []
        sorted_levels = sorted(levels, reverse=(level_type == 'support'))
        
        for level in sorted_levels:
            if not unique_levels or abs(level/unique_levels[-1] - 1) > 0.02:
                if level_type == 'support' and 0.7 * current_price <= level <= current_price:
                    unique_levels.append(level)
                elif level_type == 'resistance' and current_price <= level <= 1.3 * current_price:
                    unique_levels.append(level)
        
        return unique_levels 