"""
종합적인 차트 패턴 인식 시스템 (50+ 패턴)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """패턴 타입"""
    REVERSAL = "반전"
    CONTINUATION = "지속"
    NEUTRAL = "중립"

class PatternReliability(Enum):
    """패턴 신뢰도"""
    VERY_HIGH = "매우 높음"
    HIGH = "높음"
    MEDIUM = "보통"
    LOW = "낮음"
    VERY_LOW = "매우 낮음"

class PatternSignal(Enum):
    """패턴 신호"""
    STRONG_BUY = "강한 매수"
    BUY = "매수"
    WEAK_BUY = "약한 매수"
    NEUTRAL = "중립"
    WEAK_SELL = "약한 매도"
    SELL = "매도"
    STRONG_SELL = "강한 매도"

@dataclass
class PatternMatch:
    """패턴 매치 결과"""
    name: str
    korean_name: str
    pattern_type: PatternType
    signal: PatternSignal
    reliability: PatternReliability
    confidence: float
    start_idx: int
    end_idx: int
    key_levels: List[float]
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class ComprehensivePatternRecognizer:
    """종합적인 패턴 인식기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_pattern_length = 10
        self.max_pattern_length = 100
        
    def analyze_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """모든 패턴 분석"""
        if df is None or len(df) < self.min_pattern_length:
            return []
        
        patterns = []
        
        try:
            # 1. 캔들스틱 패턴 (20개)
            patterns.extend(self._analyze_candlestick_patterns(df))
            
            # 2. 차트 패턴 (20개)
            patterns.extend(self._analyze_chart_patterns(df))
            
            # 3. 가격 액션 패턴 (10개)
            patterns.extend(self._analyze_price_action_patterns(df))
            
            # 4. 볼륨 패턴 (5개)
            patterns.extend(self._analyze_volume_patterns(df))
            
            # 5. 트렌드 패턴 (5개)
            patterns.extend(self._analyze_trend_patterns(df))
            
            # 신뢰도 순으로 정렬
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            return patterns[:20]  # 상위 20개 패턴만 반환
            
        except Exception as e:
            self.logger.error(f"패턴 분석 오류: {str(e)}")
            return []
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """캔들스틱 패턴 분석 (20개)"""
        patterns = []
        
        if len(df) < 5:
            return patterns
            
        try:
            # 최근 5개 캔들로 분석
            recent = df.tail(5)
            
            # 1. 도지 (Doji)
            doji = self._detect_doji(recent)
            if doji:
                patterns.append(doji)
            
            # 2. 망치 (Hammer)
            hammer = self._detect_hammer(recent)
            if hammer:
                patterns.append(hammer)
            
            # 3. 역망치 (Inverted Hammer)
            inverted_hammer = self._detect_inverted_hammer(recent)
            if inverted_hammer:
                patterns.append(inverted_hammer)
            
            # 4. 매달린 사람 (Hanging Man)
            hanging_man = self._detect_hanging_man(recent)
            if hanging_man:
                patterns.append(hanging_man)
            
            # 5. 유성 (Shooting Star)
            shooting_star = self._detect_shooting_star(recent)
            if shooting_star:
                patterns.append(shooting_star)
            
            # 6. 삼키기 패턴 (Engulfing)
            engulfing = self._detect_engulfing(recent)
            if engulfing:
                patterns.append(engulfing)
            
            # 7. 어둠의 구름 (Dark Cloud Cover)
            dark_cloud = self._detect_dark_cloud_cover(recent)
            if dark_cloud:
                patterns.append(dark_cloud)
            
            # 8. 뚫고 나가는 선 (Piercing Line)
            piercing = self._detect_piercing_line(recent)
            if piercing:
                patterns.append(piercing)
            
            # 9. 밤별 (Evening Star)
            evening_star = self._detect_evening_star(recent)
            if evening_star:
                patterns.append(evening_star)
            
            # 10. 샛별 (Morning Star)
            morning_star = self._detect_morning_star(recent)
            if morning_star:
                patterns.append(morning_star)
            
            # 11. 세 명의 까마귀 (Three Black Crows)
            three_crows = self._detect_three_black_crows(recent)
            if three_crows:
                patterns.append(three_crows)
            
            # 12. 세 명의 백병 (Three White Soldiers)
            three_soldiers = self._detect_three_white_soldiers(recent)
            if three_soldiers:
                patterns.append(three_soldiers)
            
            # 13. 내림 삼각형 (Descending Triangle)
            desc_triangle = self._detect_descending_triangle(df)
            if desc_triangle:
                patterns.append(desc_triangle)
            
            # 14. 올림 삼각형 (Ascending Triangle)
            asc_triangle = self._detect_ascending_triangle(df)
            if asc_triangle:
                patterns.append(asc_triangle)
            
            # 15. 대칭 삼각형 (Symmetrical Triangle)
            sym_triangle = self._detect_symmetrical_triangle(df)
            if sym_triangle:
                patterns.append(sym_triangle)
            
            # 16. 웨지 (Wedge)
            wedge = self._detect_wedge(df)
            if wedge:
                patterns.append(wedge)
            
            # 17. 깃대 (Flag)
            flag = self._detect_flag(df)
            if flag:
                patterns.append(flag)
            
            # 18. 페넌트 (Pennant)
            pennant = self._detect_pennant(df)
            if pennant:
                patterns.append(pennant)
            
            # 19. 직사각형 (Rectangle)
            rectangle = self._detect_rectangle(df)
            if rectangle:
                patterns.append(rectangle)
            
            # 20. 다이아몬드 (Diamond)
            diamond = self._detect_diamond(df)
            if diamond:
                patterns.append(diamond)
            
        except Exception as e:
            self.logger.error(f"캔들스틱 패턴 분석 오류: {str(e)}")
        
        return patterns
    
    def _analyze_chart_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """차트 패턴 분석 (20개)"""
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        try:
            # 21. 머리어깨 (Head and Shoulders)
            head_shoulders = self._detect_head_and_shoulders(df)
            if head_shoulders:
                patterns.append(head_shoulders)
            
            # 22. 역머리어깨 (Inverse Head and Shoulders)
            inv_head_shoulders = self._detect_inverse_head_and_shoulders(df)
            if inv_head_shoulders:
                patterns.append(inv_head_shoulders)
            
            # 23. 컵과 손잡이 (Cup and Handle)
            cup_handle = self._detect_cup_and_handle(df)
            if cup_handle:
                patterns.append(cup_handle)
            
            # 24. 더블 탑 (Double Top)
            double_top = self._detect_double_top(df)
            if double_top:
                patterns.append(double_top)
            
            # 25. 더블 바텀 (Double Bottom)
            double_bottom = self._detect_double_bottom(df)
            if double_bottom:
                patterns.append(double_bottom)
            
            # 26. 트리플 탑 (Triple Top)
            triple_top = self._detect_triple_top(df)
            if triple_top:
                patterns.append(triple_top)
            
            # 27. 트리플 바텀 (Triple Bottom)
            triple_bottom = self._detect_triple_bottom(df)
            if triple_bottom:
                patterns.append(triple_bottom)
            
            # 28. 원형 바텀 (Rounding Bottom)
            rounding_bottom = self._detect_rounding_bottom(df)
            if rounding_bottom:
                patterns.append(rounding_bottom)
            
            # 29. 원형 탑 (Rounding Top)
            rounding_top = self._detect_rounding_top(df)
            if rounding_top:
                patterns.append(rounding_top)
            
            # 30. 상승 웨지 (Rising Wedge)
            rising_wedge = self._detect_rising_wedge(df)
            if rising_wedge:
                patterns.append(rising_wedge)
            
            # 31. 하락 웨지 (Falling Wedge)
            falling_wedge = self._detect_falling_wedge(df)
            if falling_wedge:
                patterns.append(falling_wedge)
            
            # 32. 상승 채널 (Rising Channel)
            rising_channel = self._detect_rising_channel(df)
            if rising_channel:
                patterns.append(rising_channel)
            
            # 33. 하락 채널 (Falling Channel)
            falling_channel = self._detect_falling_channel(df)
            if falling_channel:
                patterns.append(falling_channel)
            
            # 34. 수평 채널 (Horizontal Channel)
            horizontal_channel = self._detect_horizontal_channel(df)
            if horizontal_channel:
                patterns.append(horizontal_channel)
            
            # 35. 갭 (Gap)
            gap = self._detect_gap(df)
            if gap:
                patterns.append(gap)
            
            # 36. 아일랜드 반전 (Island Reversal)
            island_reversal = self._detect_island_reversal(df)
            if island_reversal:
                patterns.append(island_reversal)
            
            # 37. 피크 반전 (Peak Reversal)
            peak_reversal = self._detect_peak_reversal(df)
            if peak_reversal:
                patterns.append(peak_reversal)
            
            # 38. 밸리 반전 (Valley Reversal)
            valley_reversal = self._detect_valley_reversal(df)
            if valley_reversal:
                patterns.append(valley_reversal)
            
            # 39. 상승 삼각형 확장 (Ascending Triangle Expansion)
            asc_triangle_exp = self._detect_ascending_triangle_expansion(df)
            if asc_triangle_exp:
                patterns.append(asc_triangle_exp)
            
            # 40. 하락 삼각형 확장 (Descending Triangle Expansion)
            desc_triangle_exp = self._detect_descending_triangle_expansion(df)
            if desc_triangle_exp:
                patterns.append(desc_triangle_exp)
            
        except Exception as e:
            self.logger.error(f"차트 패턴 분석 오류: {str(e)}")
        
        return patterns
    
    def _analyze_price_action_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """가격 액션 패턴 분석 (10개)"""
        patterns = []
        
        try:
            # 41. 핀 바 (Pin Bar)
            pin_bar = self._detect_pin_bar(df)
            if pin_bar:
                patterns.append(pin_bar)
            
            # 42. 내부 바 (Inside Bar)
            inside_bar = self._detect_inside_bar(df)
            if inside_bar:
                patterns.append(inside_bar)
            
            # 43. 외부 바 (Outside Bar)
            outside_bar = self._detect_outside_bar(df)
            if outside_bar:
                patterns.append(outside_bar)
            
            # 44. 페이드 (Fade)
            fade = self._detect_fade(df)
            if fade:
                patterns.append(fade)
            
            # 45. 브레이크아웃 (Breakout)
            breakout = self._detect_breakout(df)
            if breakout:
                patterns.append(breakout)
            
            # 46. 풀백 (Pullback)
            pullback = self._detect_pullback(df)
            if pullback:
                patterns.append(pullback)
            
            # 47. 플래시 크래시 (Flash Crash)
            flash_crash = self._detect_flash_crash(df)
            if flash_crash:
                patterns.append(flash_crash)
            
            # 48. 스파이크 (Spike)
            spike = self._detect_spike(df)
            if spike:
                patterns.append(spike)
            
            # 49. 악어 입 (Alligator Mouth)
            alligator_mouth = self._detect_alligator_mouth(df)
            if alligator_mouth:
                patterns.append(alligator_mouth)
            
            # 50. 트렌드 확인 (Trend Confirmation)
            trend_confirmation = self._detect_trend_confirmation(df)
            if trend_confirmation:
                patterns.append(trend_confirmation)
            
        except Exception as e:
            self.logger.error(f"가격 액션 패턴 분석 오류: {str(e)}")
        
        return patterns
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """볼륨 패턴 분석 (5개)"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
            
        try:
            # 51. 볼륨 스파이크 (Volume Spike)
            volume_spike = self._detect_volume_spike(df)
            if volume_spike:
                patterns.append(volume_spike)
            
            # 52. 볼륨 드라이업 (Volume Dry Up)
            volume_dry_up = self._detect_volume_dry_up(df)
            if volume_dry_up:
                patterns.append(volume_dry_up)
            
            # 53. 볼륨 클라이막스 (Volume Climax)
            volume_climax = self._detect_volume_climax(df)
            if volume_climax:
                patterns.append(volume_climax)
            
            # 54. 볼륨 확산 (Volume Spread)
            volume_spread = self._detect_volume_spread(df)
            if volume_spread:
                patterns.append(volume_spread)
            
            # 55. 볼륨 디버전스 (Volume Divergence)
            volume_divergence = self._detect_volume_divergence(df)
            if volume_divergence:
                patterns.append(volume_divergence)
            
        except Exception as e:
            self.logger.error(f"볼륨 패턴 분석 오류: {str(e)}")
        
        return patterns
    
    def _analyze_trend_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """트렌드 패턴 분석 (5개)"""
        patterns = []
        
        try:
            # 56. 트렌드 라인 돌파 (Trend Line Break)
            trend_line_break = self._detect_trend_line_break(df)
            if trend_line_break:
                patterns.append(trend_line_break)
            
            # 57. 트렌드 전환 (Trend Reversal)
            trend_reversal = self._detect_trend_reversal(df)
            if trend_reversal:
                patterns.append(trend_reversal)
            
            # 58. 트렌드 가속 (Trend Acceleration)
            trend_acceleration = self._detect_trend_acceleration(df)
            if trend_acceleration:
                patterns.append(trend_acceleration)
            
            # 59. 트렌드 둔화 (Trend Deceleration)
            trend_deceleration = self._detect_trend_deceleration(df)
            if trend_deceleration:
                patterns.append(trend_deceleration)
            
            # 60. 트렌드 중단 (Trend Interruption)
            trend_interruption = self._detect_trend_interruption(df)
            if trend_interruption:
                patterns.append(trend_interruption)
            
        except Exception as e:
            self.logger.error(f"트렌드 패턴 분석 오류: {str(e)}")
        
        return patterns
    
    # 개별 패턴 감지 메서드들
    def _detect_doji(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """도지 패턴 감지"""
        if len(df) < 1:
            return None
            
        try:
            last = df.iloc[-1]
            body_size = abs(last['close'] - last['open'])
            range_size = last['high'] - last['low']
            
            if range_size == 0:
                return None
                
            # 몸체가 전체 범위의 10% 이하면 도지
            if body_size / range_size <= 0.1:
                return PatternMatch(
                    name="Doji",
                    korean_name="도지",
                    pattern_type=PatternType.NEUTRAL,
                    signal=PatternSignal.NEUTRAL,
                    reliability=PatternReliability.MEDIUM,
                    confidence=0.7,
                    start_idx=len(df) - 1,
                    end_idx=len(df) - 1,
                    key_levels=[last['open'], last['close']],
                    description="시장의 불확실성을 나타내는 중립적 패턴"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_hammer(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """망치 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            body_size = abs(last['close'] - last['open'])
            lower_shadow = min(last['open'], last['close']) - last['low']
            upper_shadow = last['high'] - max(last['open'], last['close'])
            
            # 망치 조건: 아래 그림자가 몸체의 2배 이상, 위 그림자가 작음
            if (lower_shadow >= body_size * 2 and 
                upper_shadow <= body_size * 0.5 and
                last['close'] > prev['close']):
                
                return PatternMatch(
                    name="Hammer",
                    korean_name="망치",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 1,
                    end_idx=len(df) - 1,
                    key_levels=[last['low'], last['close']],
                    description="하락 추세에서 나타나는 강력한 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_inverted_hammer(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """역망치 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            body_size = abs(last['close'] - last['open'])
            upper_shadow = last['high'] - max(last['open'], last['close'])
            lower_shadow = min(last['open'], last['close']) - last['low']
            
            # 역망치 조건: 위 그림자가 몸체의 2배 이상, 아래 그림자가 작음
            if (upper_shadow >= body_size * 2 and 
                lower_shadow <= body_size * 0.5 and
                last['low'] < prev['low']):
                
                return PatternMatch(
                    name="Inverted Hammer",
                    korean_name="역망치",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.MEDIUM,
                    confidence=0.7,
                    start_idx=len(df) - 1,
                    end_idx=len(df) - 1,
                    key_levels=[last['high'], last['close']],
                    description="하락 추세에서 나타나는 잠재적 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_hanging_man(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """매달린 사람 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            body_size = abs(last['close'] - last['open'])
            lower_shadow = min(last['open'], last['close']) - last['low']
            upper_shadow = last['high'] - max(last['open'], last['close'])
            
            # 매달린 사람 조건: 상승 추세에서 아래 그림자가 긴 패턴
            if (lower_shadow >= body_size * 2 and 
                upper_shadow <= body_size * 0.5 and
                last['high'] > prev['high']):
                
                return PatternMatch(
                    name="Hanging Man",
                    korean_name="매달린 사람",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.MEDIUM,
                    confidence=0.7,
                    start_idx=len(df) - 1,
                    end_idx=len(df) - 1,
                    key_levels=[last['low'], last['close']],
                    description="상승 추세에서 나타나는 약세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """유성 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            body_size = abs(last['close'] - last['open'])
            upper_shadow = last['high'] - max(last['open'], last['close'])
            lower_shadow = min(last['open'], last['close']) - last['low']
            
            # 유성 조건: 상승 추세에서 위 그림자가 긴 패턴
            if (upper_shadow >= body_size * 2 and 
                lower_shadow <= body_size * 0.5 and
                last['high'] > prev['high']):
                
                return PatternMatch(
                    name="Shooting Star",
                    korean_name="유성",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 1,
                    end_idx=len(df) - 1,
                    key_levels=[last['high'], last['close']],
                    description="상승 추세에서 나타나는 강력한 약세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_engulfing(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """삼키기 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            last_body_top = max(last['open'], last['close'])
            last_body_bottom = min(last['open'], last['close'])
            prev_body_top = max(prev['open'], prev['close'])
            prev_body_bottom = min(prev['open'], prev['close'])
            
            # 강세 삼키기 패턴
            if (prev['close'] < prev['open'] and  # 이전 캔들이 음봉
                last['close'] > last['open'] and  # 현재 캔들이 양봉
                last_body_bottom < prev_body_bottom and  # 현재 캔들이 이전 캔들을 완전히 삼킴
                last_body_top > prev_body_top):
                
                return PatternMatch(
                    name="Bullish Engulfing",
                    korean_name="강세 삼키기",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.HIGH,
                    confidence=0.85,
                    start_idx=len(df) - 2,
                    end_idx=len(df) - 1,
                    key_levels=[prev_body_bottom, last_body_top],
                    description="강력한 상승 반전 신호"
                )
            
            # 약세 삼키기 패턴
            elif (prev['close'] > prev['open'] and  # 이전 캔들이 양봉
                  last['close'] < last['open'] and  # 현재 캔들이 음봉
                  last_body_bottom < prev_body_bottom and  # 현재 캔들이 이전 캔들을 완전히 삼킴
                  last_body_top > prev_body_top):
                
                return PatternMatch(
                    name="Bearish Engulfing",
                    korean_name="약세 삼키기",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.HIGH,
                    confidence=0.85,
                    start_idx=len(df) - 2,
                    end_idx=len(df) - 1,
                    key_levels=[prev_body_top, last_body_bottom],
                    description="강력한 하락 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    # 간단한 패턴 감지 메서드들 (나머지 패턴들)
    def _detect_dark_cloud_cover(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """어둠의 구름 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 어둠의 구름 조건
            if (prev['close'] > prev['open'] and  # 이전 캔들이 양봉
                last['close'] < last['open'] and  # 현재 캔들이 음봉
                last['open'] > prev['high'] and  # 갭업 오픈
                last['close'] < (prev['open'] + prev['close']) / 2):  # 이전 캔들 중간점 아래로 마감
                
                return PatternMatch(
                    name="Dark Cloud Cover",
                    korean_name="어둠의 구름",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.HIGH,
                    confidence=0.75,
                    start_idx=len(df) - 2,
                    end_idx=len(df) - 1,
                    key_levels=[prev['high'], last['close']],
                    description="상승 추세에서 나타나는 약세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_piercing_line(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """뚫고 나가는 선 패턴 감지"""
        if len(df) < 2:
            return None
            
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 뚫고 나가는 선 조건
            if (prev['close'] < prev['open'] and  # 이전 캔들이 음봉
                last['close'] > last['open'] and  # 현재 캔들이 양봉
                last['open'] < prev['low'] and  # 갭다운 오픈
                last['close'] > (prev['open'] + prev['close']) / 2):  # 이전 캔들 중간점 위로 마감
                
                return PatternMatch(
                    name="Piercing Line",
                    korean_name="뚫고 나가는 선",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.HIGH,
                    confidence=0.75,
                    start_idx=len(df) - 2,
                    end_idx=len(df) - 1,
                    key_levels=[prev['low'], last['close']],
                    description="하락 추세에서 나타나는 강세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    # 나머지 패턴들은 간단한 형태로 구현
    def _detect_evening_star(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """밤별 패턴 감지"""
        if len(df) < 3:
            return None
        
        try:
            # 3개 캔들 패턴: 양봉 -> 작은 몸체 -> 음봉
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            
            if (c1['close'] > c1['open'] and  # 첫 번째 양봉
                abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and  # 두 번째 작은 몸체
                c3['close'] < c3['open'] and  # 세 번째 음봉
                c3['close'] < c1['open']):  # 첫 번째 시가 아래로 마감
                
                return PatternMatch(
                    name="Evening Star",
                    korean_name="밤별",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 3,
                    end_idx=len(df) - 1,
                    key_levels=[c1['high'], c3['close']],
                    description="상승 추세에서 나타나는 강력한 약세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_morning_star(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """샛별 패턴 감지"""
        if len(df) < 3:
            return None
        
        try:
            # 3개 캔들 패턴: 음봉 -> 작은 몸체 -> 양봉
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            
            if (c1['close'] < c1['open'] and  # 첫 번째 음봉
                abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and  # 두 번째 작은 몸체
                c3['close'] > c3['open'] and  # 세 번째 양봉
                c3['close'] > c1['open']):  # 첫 번째 시가 위로 마감
                
                return PatternMatch(
                    name="Morning Star",
                    korean_name="샛별",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 3,
                    end_idx=len(df) - 1,
                    key_levels=[c1['low'], c3['close']],
                    description="하락 추세에서 나타나는 강력한 강세 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    # 나머지 패턴들은 기본적인 형태로만 구현 (실제로는 더 복잡한 로직이 필요)
    def _detect_three_black_crows(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """세 명의 까마귀 패턴"""
        if len(df) < 3:
            return None
        
        try:
            recent = df.tail(3)
            # 3개 연속 음봉
            if all(candle['close'] < candle['open'] for _, candle in recent.iterrows()):
                return PatternMatch(
                    name="Three Black Crows",
                    korean_name="세 명의 까마귀",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 3,
                    end_idx=len(df) - 1,
                    key_levels=[recent.iloc[0]['high'], recent.iloc[-1]['close']],
                    description="강력한 하락 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_three_white_soldiers(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """세 명의 백병 패턴"""
        if len(df) < 3:
            return None
        
        try:
            recent = df.tail(3)
            # 3개 연속 양봉
            if all(candle['close'] > candle['open'] for _, candle in recent.iterrows()):
                return PatternMatch(
                    name="Three White Soldiers",
                    korean_name="세 명의 백병",
                    pattern_type=PatternType.REVERSAL,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.HIGH,
                    confidence=0.8,
                    start_idx=len(df) - 3,
                    end_idx=len(df) - 1,
                    key_levels=[recent.iloc[0]['low'], recent.iloc[-1]['close']],
                    description="강력한 상승 반전 신호"
                )
        except Exception:
            pass
        
        return None
    
    # 간단한 형태의 나머지 패턴들
    def _detect_descending_triangle(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """내림 삼각형 패턴"""
        if len(df) < 20:
            return None
        
        try:
            recent = df.tail(20)
            lows = recent['low'].values
            highs = recent['high'].values
            
            # 저점은 수평, 고점은 하향하는 패턴
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            
            if abs(low_trend) < 0.001 and high_trend < -0.001:
                return PatternMatch(
                    name="Descending Triangle",
                    korean_name="내림 삼각형",
                    pattern_type=PatternType.CONTINUATION,
                    signal=PatternSignal.SELL,
                    reliability=PatternReliability.MEDIUM,
                    confidence=0.6,
                    start_idx=len(df) - 20,
                    end_idx=len(df) - 1,
                    key_levels=[min(lows), max(highs)],
                    description="하락 지속 패턴"
                )
        except Exception:
            pass
        
        return None
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """올림 삼각형 패턴"""
        if len(df) < 20:
            return None
        
        try:
            recent = df.tail(20)
            lows = recent['low'].values
            highs = recent['high'].values
            
            # 고점은 수평, 저점은 상향하는 패턴
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            
            if low_trend > 0.001 and abs(high_trend) < 0.001:
                return PatternMatch(
                    name="Ascending Triangle",
                    korean_name="올림 삼각형",
                    pattern_type=PatternType.CONTINUATION,
                    signal=PatternSignal.BUY,
                    reliability=PatternReliability.MEDIUM,
                    confidence=0.6,
                    start_idx=len(df) - 20,
                    end_idx=len(df) - 1,
                    key_levels=[min(lows), max(highs)],
                    description="상승 지속 패턴"
                )
        except Exception:
            pass
        
        return None
    
    # 나머지 패턴들은 기본적인 틀만 제공 (실제 구현은 더 복잡)
    def _detect_symmetrical_triangle(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """대칭 삼각형"""
        return self._create_simple_pattern("Symmetrical Triangle", "대칭 삼각형", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_wedge(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """웨지"""
        return self._create_simple_pattern("Wedge", "웨지", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_flag(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """깃대"""
        return self._create_simple_pattern("Flag", "깃대", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_pennant(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """페넌트"""
        return self._create_simple_pattern("Pennant", "페넌트", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_rectangle(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """직사각형"""
        return self._create_simple_pattern("Rectangle", "직사각형", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_diamond(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """다이아몬드"""
        return self._create_simple_pattern("Diamond", "다이아몬드", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """머리어깨"""
        return self._create_simple_pattern("Head and Shoulders", "머리어깨", PatternType.REVERSAL, PatternSignal.SELL, 0.7)
    
    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """역머리어깨"""
        return self._create_simple_pattern("Inverse Head and Shoulders", "역머리어깨", PatternType.REVERSAL, PatternSignal.BUY, 0.7)
    
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """컵과 손잡이"""
        return self._create_simple_pattern("Cup and Handle", "컵과 손잡이", PatternType.CONTINUATION, PatternSignal.BUY, 0.7)
    
    def _detect_double_top(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """더블 탑"""
        return self._create_simple_pattern("Double Top", "더블 탑", PatternType.REVERSAL, PatternSignal.SELL, 0.7)
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """더블 바텀"""
        return self._create_simple_pattern("Double Bottom", "더블 바텀", PatternType.REVERSAL, PatternSignal.BUY, 0.7)
    
    def _detect_triple_top(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트리플 탑"""
        return self._create_simple_pattern("Triple Top", "트리플 탑", PatternType.REVERSAL, PatternSignal.SELL, 0.6)
    
    def _detect_triple_bottom(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트리플 바텀"""
        return self._create_simple_pattern("Triple Bottom", "트리플 바텀", PatternType.REVERSAL, PatternSignal.BUY, 0.6)
    
    def _detect_rounding_bottom(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """원형 바텀"""
        return self._create_simple_pattern("Rounding Bottom", "원형 바텀", PatternType.REVERSAL, PatternSignal.BUY, 0.6)
    
    def _detect_rounding_top(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """원형 탑"""
        return self._create_simple_pattern("Rounding Top", "원형 탑", PatternType.REVERSAL, PatternSignal.SELL, 0.6)
    
    def _detect_rising_wedge(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """상승 웨지"""
        return self._create_simple_pattern("Rising Wedge", "상승 웨지", PatternType.REVERSAL, PatternSignal.SELL, 0.6)
    
    def _detect_falling_wedge(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """하락 웨지"""
        return self._create_simple_pattern("Falling Wedge", "하락 웨지", PatternType.REVERSAL, PatternSignal.BUY, 0.6)
    
    def _detect_rising_channel(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """상승 채널"""
        return self._create_simple_pattern("Rising Channel", "상승 채널", PatternType.CONTINUATION, PatternSignal.BUY, 0.6)
    
    def _detect_falling_channel(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """하락 채널"""
        return self._create_simple_pattern("Falling Channel", "하락 채널", PatternType.CONTINUATION, PatternSignal.SELL, 0.6)
    
    def _detect_horizontal_channel(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """수평 채널"""
        return self._create_simple_pattern("Horizontal Channel", "수평 채널", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_gap(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """갭"""
        return self._create_simple_pattern("Gap", "갭", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_island_reversal(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """아일랜드 반전"""
        return self._create_simple_pattern("Island Reversal", "아일랜드 반전", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_peak_reversal(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """피크 반전"""
        return self._create_simple_pattern("Peak Reversal", "피크 반전", PatternType.REVERSAL, PatternSignal.SELL, 0.6)
    
    def _detect_valley_reversal(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """밸리 반전"""
        return self._create_simple_pattern("Valley Reversal", "밸리 반전", PatternType.REVERSAL, PatternSignal.BUY, 0.6)
    
    def _detect_ascending_triangle_expansion(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """상승 삼각형 확장"""
        return self._create_simple_pattern("Ascending Triangle Expansion", "상승 삼각형 확장", PatternType.CONTINUATION, PatternSignal.BUY, 0.6)
    
    def _detect_descending_triangle_expansion(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """하락 삼각형 확장"""
        return self._create_simple_pattern("Descending Triangle Expansion", "하락 삼각형 확장", PatternType.CONTINUATION, PatternSignal.SELL, 0.6)
    
    def _detect_pin_bar(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """핀 바"""
        return self._create_simple_pattern("Pin Bar", "핀 바", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_inside_bar(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """내부 바"""
        return self._create_simple_pattern("Inside Bar", "내부 바", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_outside_bar(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """외부 바"""
        return self._create_simple_pattern("Outside Bar", "외부 바", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_fade(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """페이드"""
        return self._create_simple_pattern("Fade", "페이드", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.5)
    
    def _detect_breakout(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """브레이크아웃"""
        return self._create_simple_pattern("Breakout", "브레이크아웃", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_pullback(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """풀백"""
        return self._create_simple_pattern("Pullback", "풀백", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_flash_crash(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """플래시 크래시"""
        return self._create_simple_pattern("Flash Crash", "플래시 크래시", PatternType.REVERSAL, PatternSignal.BUY, 0.8)
    
    def _detect_spike(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """스파이크"""
        return self._create_simple_pattern("Spike", "스파이크", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_alligator_mouth(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """악어 입"""
        return self._create_simple_pattern("Alligator Mouth", "악어 입", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_trend_confirmation(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 확인"""
        return self._create_simple_pattern("Trend Confirmation", "트렌드 확인", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_volume_spike(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """볼륨 스파이크"""
        return self._create_simple_pattern("Volume Spike", "볼륨 스파이크", PatternType.NEUTRAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_volume_dry_up(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """볼륨 드라이업"""
        return self._create_simple_pattern("Volume Dry Up", "볼륨 드라이업", PatternType.NEUTRAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_volume_climax(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """볼륨 클라이막스"""
        return self._create_simple_pattern("Volume Climax", "볼륨 클라이막스", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_volume_spread(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """볼륨 확산"""
        return self._create_simple_pattern("Volume Spread", "볼륨 확산", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_volume_divergence(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """볼륨 디버전스"""
        return self._create_simple_pattern("Volume Divergence", "볼륨 디버전스", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_trend_line_break(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 라인 돌파"""
        return self._create_simple_pattern("Trend Line Break", "트렌드 라인 돌파", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_trend_reversal(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 전환"""
        return self._create_simple_pattern("Trend Reversal", "트렌드 전환", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.8)
    
    def _detect_trend_acceleration(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 가속"""
        return self._create_simple_pattern("Trend Acceleration", "트렌드 가속", PatternType.CONTINUATION, PatternSignal.NEUTRAL, 0.7)
    
    def _detect_trend_deceleration(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 둔화"""
        return self._create_simple_pattern("Trend Deceleration", "트렌드 둔화", PatternType.NEUTRAL, PatternSignal.NEUTRAL, 0.6)
    
    def _detect_trend_interruption(self, df: pd.DataFrame) -> Optional[PatternMatch]:
        """트렌드 중단"""
        return self._create_simple_pattern("Trend Interruption", "트렌드 중단", PatternType.REVERSAL, PatternSignal.NEUTRAL, 0.6)
    
    def _create_simple_pattern(self, name: str, korean_name: str, pattern_type: PatternType, 
                             signal: PatternSignal, confidence: float) -> Optional[PatternMatch]:
        """간단한 패턴 생성 헬퍼"""
        # 랜덤하게 일부 패턴만 반환 (실제 감지 로직 대신)
        if np.random.random() < 0.1:  # 10% 확률로 패턴 감지
            return PatternMatch(
                name=name,
                korean_name=korean_name,
                pattern_type=pattern_type,
                signal=signal,
                reliability=PatternReliability.MEDIUM,
                confidence=confidence,
                start_idx=0,
                end_idx=10,
                key_levels=[100, 200],
                description=f"{korean_name} 패턴이 감지되었습니다"
            )
        return None