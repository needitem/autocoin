"""
차트 분석 엔진
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from .performance_optimizer import cached_analysis, monitored_execution, DataFrameOptimizer

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """차트 패턴 타입"""
    # 반전 패턴
    HEAD_AND_SHOULDERS = "헤드앤숄더"
    INVERSE_HEAD_AND_SHOULDERS = "역헤드앤숄더"
    DOUBLE_TOP = "더블탑"
    DOUBLE_BOTTOM = "더블바텀"
    TRIPLE_TOP = "트리플탑"
    TRIPLE_BOTTOM = "트리플바텀"
    
    # 지속 패턴
    ASCENDING_TRIANGLE = "상승삼각형"
    DESCENDING_TRIANGLE = "하락삼각형"
    SYMMETRICAL_TRIANGLE = "대칭삼각형"
    FLAG = "깃발형"
    PENNANT = "페넌트"
    WEDGE_RISING = "상승쐐기"
    WEDGE_FALLING = "하락쐐기"
    
    # 캔들스틱 패턴
    DOJI = "도지"
    HAMMER = "해머"
    SHOOTING_STAR = "슈팅스타"
    ENGULFING_BULLISH = "강세포옴"
    ENGULFING_BEARISH = "약세포옴"
    MORNING_STAR = "샛별"
    EVENING_STAR = "저녁별"

class TrendDirection(Enum):
    """트렌드 방향"""
    UPTREND = "상승"
    DOWNTREND = "하락"
    SIDEWAYS = "횡보"

class SignalStrength(Enum):
    """신호 강도"""
    VERY_STRONG = "매우 강함"
    STRONG = "강함"
    MODERATE = "보통"
    WEAK = "약함"

@dataclass
class ChartPattern:
    pattern_type: PatternType
    confidence: float  # 0.0 ~ 1.0
    start_date: str
    end_date: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str
    signal_strength: SignalStrength

@dataclass
class TrendAnalysis:
    direction: TrendDirection
    strength: float  # 0.0 ~ 1.0
    start_date: str
    duration_days: int
    support_level: Optional[float]
    resistance_level: Optional[float]

@dataclass
class TechnicalAnalysis:
    trend: TrendAnalysis
    patterns: List[ChartPattern]
    support_resistance: Dict[str, List[float]]
    momentum_signals: Dict[str, str]
    volume_analysis: Dict[str, any]
    risk_level: str

class ChartAnalyzer:
    def __init__(self):
        """차트 분석기 초기화"""
        self.min_pattern_length = 10  # 최소 패턴 길이
        self.max_pattern_length = 100  # 최대 패턴 길이
        
        # 고급 설정
        self.pattern_confidence_threshold = 0.6  # 패턴 신뢰도 임계값
        self.volume_weight = 0.3  # 거래량 가중치
        self.price_weight = 0.7  # 가격 가중치
        
        # 알림 설정
        self.alert_patterns = [
            PatternType.HAMMER, PatternType.ENGULFING_BULLISH, PatternType.ENGULFING_BEARISH,
            PatternType.DOUBLE_BOTTOM, PatternType.DOUBLE_TOP, PatternType.HEAD_AND_SHOULDERS
        ]
        
    @monitored_execution
    def analyze_chart(self, ohlcv_data: pd.DataFrame, indicators: Dict = None) -> TechnicalAnalysis:
        """전체 차트 분석"""
        try:
            if ohlcv_data.empty or len(ohlcv_data) < 20:
                logger.warning("분석에 충분한 데이터가 없습니다.")
                return self._get_default_analysis()
            
            # DataFrame 메모리 최적화
            ohlcv_data = DataFrameOptimizer.optimize_memory(ohlcv_data.copy())
            
            # 트렌드 분석
            trend_analysis = self._analyze_trend(ohlcv_data)
            
            # 패턴 분석
            patterns = self._detect_patterns(ohlcv_data)
            
            # 지지/저항 분석
            support_resistance = self._find_support_resistance(ohlcv_data)
            
            # 모멘텀 분석
            momentum_signals = self._analyze_momentum(ohlcv_data, indicators)
            
            # 볼륨 분석
            volume_analysis = self._analyze_volume(ohlcv_data)
            
            # 위험도 평가
            risk_level = self._assess_risk(ohlcv_data, patterns, trend_analysis)
            
            return TechnicalAnalysis(
                trend=trend_analysis,
                patterns=patterns,
                support_resistance=support_resistance,
                momentum_signals=momentum_signals,
                volume_analysis=volume_analysis,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"차트 분석 오류: {str(e)}")
            return self._get_default_analysis()
    
    def _analyze_trend(self, data: pd.DataFrame) -> TrendAnalysis:
        """트렌드 분석"""
        try:
            # 단순 이동평균을 이용한 트렌드 분석
            ma_short = data['close'].rolling(window=20).mean()
            ma_long = data['close'].rolling(window=50).mean()
            
            current_price = data['close'].iloc[-1]
            ma_short_current = ma_short.iloc[-1]
            ma_long_current = ma_long.iloc[-1]
            
            # 트렌드 방향 결정
            if current_price > ma_short_current > ma_long_current:
                direction = TrendDirection.UPTREND
                strength = min(1.0, (current_price - ma_long_current) / ma_long_current * 10)
            elif current_price < ma_short_current < ma_long_current:
                direction = TrendDirection.DOWNTREND
                strength = min(1.0, (ma_long_current - current_price) / ma_long_current * 10)
            else:
                direction = TrendDirection.SIDEWAYS
                strength = 0.3
            
            # 지지/저항 레벨
            recent_data = data.tail(50)
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()
            
            # 트렌드 시작일 (임시)
            start_date = data.index[-30].strftime('%Y-%m-%d') if len(data) > 30 else data.index[0].strftime('%Y-%m-%d')
            
            return TrendAnalysis(
                direction=direction,
                strength=strength,
                start_date=start_date,
                duration_days=30,
                support_level=support_level,
                resistance_level=resistance_level
            )
            
        except Exception as e:
            logger.error(f"트렌드 분석 오류: {str(e)}")
            return TrendAnalysis(
                direction=TrendDirection.SIDEWAYS,
                strength=0.5,
                start_date=datetime.now().strftime('%Y-%m-%d'),
                duration_days=0,
                support_level=None,
                resistance_level=None
            )
    
    @cached_analysis(ttl_seconds=120)
    def _detect_patterns(self, data: pd.DataFrame) -> List[ChartPattern]:
        """차트 패턴 감지"""
        patterns = []
        
        try:
            # 캔들스틱 패턴 감지
            candlestick_patterns = self._detect_candlestick_patterns(data)
            patterns.extend(candlestick_patterns)
            
            # 클래식 패턴 감지
            classic_patterns = self._detect_classic_patterns(data)
            patterns.extend(classic_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"패턴 감지 오류: {str(e)}")
            return []
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[ChartPattern]:
        """고급 캔들스틱 패턴 감지"""
        patterns = []
        
        try:
            recent_data = data.tail(20)  # 최근 20일 데이터로 확대
            
            for i in range(len(recent_data) - 2):
                candle = recent_data.iloc[i]
                next_candle = recent_data.iloc[i + 1] if i + 1 < len(recent_data) else None
                prev_candle = recent_data.iloc[i - 1] if i > 0 else None
                
                # 기본 캔들 정보
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range == 0:
                    continue
                
                # 거래량 가중치 계산
                volume_weight = 1.0
                if 'volume' in candle:
                    avg_volume = recent_data['volume'].tail(10).mean()
                    volume_weight = min(2.0, candle['volume'] / avg_volume) if avg_volume > 0 else 1.0
                
                # 도지 패턴 (개선된 감지)
                if body_size / total_range < 0.05:  # 더 엄격한 기준
                    confidence = 0.8 * volume_weight
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.DOJI,
                        confidence=min(0.95, confidence),
                        start_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        end_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        target_price=None,
                        stop_loss=None,
                        description=f"완전 도지 패턴 - 시장 전환점 신호 (거래량: {volume_weight:.1f}배)",
                        signal_strength=SignalStrength.STRONG if volume_weight > 1.2 else SignalStrength.MODERATE
                    ))
                
                # 해머 패턴 (개선된 감지)
                lower_shadow = min(candle['open'], candle['close']) - candle['low']
                upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                body_ratio = body_size / total_range
                lower_ratio = lower_shadow / total_range
                upper_ratio = upper_shadow / total_range
                
                # 해머: 긴 아래꼬리, 짧은 위꼬리, 작은 몸통
                if (lower_ratio > 0.6 and upper_ratio < 0.1 and body_ratio < 0.3):
                    confidence = 0.85 * volume_weight
                    target_multiplier = 1.03 + (volume_weight - 1) * 0.02  # 거래량에 따른 목표가 조정
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.HAMMER,
                        confidence=min(0.95, confidence),
                        start_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        end_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        target_price=candle['close'] * target_multiplier,
                        stop_loss=candle['low'] * 0.995,
                        description=f"강력한 해머 패턴 - 상승 반전 신호 (거래량: {volume_weight:.1f}배)",
                        signal_strength=SignalStrength.VERY_STRONG if volume_weight > 1.5 else SignalStrength.STRONG
                    ))
                
                # 슈팅스타 패턴
                if (upper_ratio > 0.6 and lower_ratio < 0.1 and body_ratio < 0.3):
                    confidence = 0.85 * volume_weight
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.SHOOTING_STAR,
                        confidence=min(0.95, confidence),
                        start_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        end_date=recent_data.index[i].strftime('%Y-%m-%d'),
                        target_price=candle['close'] * 0.97,
                        stop_loss=candle['high'] * 1.005,
                        description=f"슈팅스타 패턴 - 하락 반전 신호 (거래량: {volume_weight:.1f}배)",
                        signal_strength=SignalStrength.VERY_STRONG if volume_weight > 1.5 else SignalStrength.STRONG
                    ))
            
            # 포용 패턴 (Engulfing)
            if len(recent_data) >= 2:
                prev_candle = recent_data.iloc[-2]
                curr_candle = recent_data.iloc[-1]
                
                # 강세 포용
                if (prev_candle['close'] < prev_candle['open'] and  # 이전: 음봉
                    curr_candle['close'] > curr_candle['open'] and  # 현재: 양봉
                    curr_candle['open'] < prev_candle['close'] and  # 현재 시가 < 이전 종가
                    curr_candle['close'] > prev_candle['open']):    # 현재 종가 > 이전 시가
                    
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.ENGULFING_BULLISH,
                        confidence=0.85,
                        start_date=recent_data.index[-2].strftime('%Y-%m-%d'),
                        end_date=recent_data.index[-1].strftime('%Y-%m-%d'),
                        target_price=curr_candle['close'] * 1.1,
                        stop_loss=prev_candle['low'],
                        description="강세 포용 패턴 - 강한 상승 신호",
                        signal_strength=SignalStrength.VERY_STRONG
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"캔들스틱 패턴 감지 오류: {str(e)}")
            return []
    
    def _detect_classic_patterns(self, data: pd.DataFrame) -> List[ChartPattern]:
        """클래식 차트 패턴 감지"""
        patterns = []
        
        try:
            if len(data) < 50:
                return patterns
            
            # 더블톱/더블바텀 패턴 감지
            peaks, valleys = self._find_peaks_valleys(data)
            
            # 더블톱 패턴
            if len(peaks) >= 2:
                last_peaks = peaks[-2:]
                peak1_price = data['high'].iloc[last_peaks[0]]
                peak2_price = data['high'].iloc[last_peaks[1]]
                
                # 두 고점이 비슷한 높이인지 확인
                if abs(peak1_price - peak2_price) / peak1_price < 0.02:
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.75,
                        start_date=data.index[last_peaks[0]].strftime('%Y-%m-%d'),
                        end_date=data.index[last_peaks[1]].strftime('%Y-%m-%d'),
                        target_price=peak1_price * 0.9,  # 10% 하락 목표
                        stop_loss=peak2_price * 1.02,
                        description="더블톱 패턴 - 하락 반전 신호",
                        signal_strength=SignalStrength.STRONG
                    ))
            
            # 더블바텀 패턴
            if len(valleys) >= 2:
                last_valleys = valleys[-2:]
                valley1_price = data['low'].iloc[last_valleys[0]]
                valley2_price = data['low'].iloc[last_valleys[1]]
                
                # 두 저점이 비슷한 높이인지 확인
                if abs(valley1_price - valley2_price) / valley1_price < 0.02:
                    patterns.append(ChartPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=0.75,
                        start_date=data.index[last_valleys[0]].strftime('%Y-%m-%d'),
                        end_date=data.index[last_valleys[1]].strftime('%Y-%m-%d'),
                        target_price=valley1_price * 1.1,  # 10% 상승 목표
                        stop_loss=valley2_price * 0.98,
                        description="더블바텀 패턴 - 상승 반전 신호",
                        signal_strength=SignalStrength.STRONG
                    ))
            
            # 삼각형 패턴 감지
            triangle_pattern = self._detect_triangle_patterns(data)
            if triangle_pattern:
                patterns.append(triangle_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"클래식 패턴 감지 오류: {str(e)}")
            return []
    
    def _find_peaks_valleys(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """고점과 저점 찾기"""
        try:
            peaks = []
            valleys = []
            
            highs = data['high'].values
            lows = data['low'].values
            
            for i in range(window, len(data) - window):
                # 고점 찾기
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    peaks.append(i)
                
                # 저점 찾기
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    valleys.append(i)
            
            return peaks, valleys
            
        except Exception as e:
            logger.error(f"고점/저점 찾기 오류: {str(e)}")
            return [], []
    
    def _detect_triangle_patterns(self, data: pd.DataFrame) -> Optional[ChartPattern]:
        """삼각형 패턴 감지"""
        try:
            if len(data) < 30:
                return None
            
            recent_data = data.tail(30)
            peaks, valleys = self._find_peaks_valleys(recent_data)
            
            if len(peaks) < 2 or len(valleys) < 2:
                return None
            
            # 상승삼각형: 저점은 올라가고 고점은 수평
            if len(valleys) >= 2:
                valley_trend = np.polyfit(valleys, [recent_data['low'].iloc[v] for v in valleys], 1)[0]
                
                if valley_trend > 0:  # 저점이 상승
                    return ChartPattern(
                        pattern_type=PatternType.ASCENDING_TRIANGLE,
                        confidence=0.7,
                        start_date=recent_data.index[0].strftime('%Y-%m-%d'),
                        end_date=recent_data.index[-1].strftime('%Y-%m-%d'),
                        target_price=recent_data['close'].iloc[-1] * 1.1,
                        stop_loss=recent_data['low'].iloc[valleys[-1]] * 0.98,
                        description="상승삼각형 패턴 - 상승 지속 신호",
                        signal_strength=SignalStrength.MODERATE
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"삼각형 패턴 감지 오류: {str(e)}")
            return None
    
    @cached_analysis(ttl_seconds=180)
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """지지선과 저항선 찾기"""
        try:
            recent_data = data.tail(100)  # 최근 100일 데이터
            
            # 클러스터링을 통한 지지/저항 레벨 찾기
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # 간단한 클러스터링 (가격대별 빈도)
            all_prices = np.concatenate([highs, lows])
            price_range = all_prices.max() - all_prices.min()
            
            # 가격대를 20개 구간으로 나눔
            bins = np.linspace(all_prices.min(), all_prices.max(), 21)
            hist, bin_edges = np.histogram(all_prices, bins=bins)
            
            # 빈도가 높은 가격대를 지지/저항으로 간주
            threshold = np.percentile(hist, 80)
            significant_levels = []
            
            for i, count in enumerate(hist):
                if count >= threshold:
                    level = (bin_edges[i] + bin_edges[i+1]) / 2
                    significant_levels.append(level)
            
            current_price = recent_data['close'].iloc[-1]
            
            # 현재가 기준으로 지지선/저항선 분류
            support_levels = [level for level in significant_levels if level < current_price]
            resistance_levels = [level for level in significant_levels if level > current_price]
            
            return {
                'support': sorted(support_levels, reverse=True)[:3],  # 가까운 순으로 3개
                'resistance': sorted(resistance_levels)[:3]  # 가까운 순으로 3개
            }
            
        except Exception as e:
            logger.error(f"지지/저항 분석 오류: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _analyze_momentum(self, data, indicators: Dict = None) -> Dict[str, str]:
        """모멘텀 분석"""
        try:
            signals = {}
            
            # 데이터 타입 확인 및 변환
            if isinstance(data, dict):
                # dict 형태인 경우 pandas DataFrame으로 변환 시도
                try:
                    if 'close' in data:
                        data = pd.DataFrame(data)
                    else:
                        logger.warning("모멘텀 분석: 'close' 컬럼이 없는 dict 데이터")
                        return {}
                except Exception as conv_error:
                    logger.warning(f"모멘텀 분석: 데이터 변환 실패 - {str(conv_error)}")
                    return {}
            
            # pandas DataFrame이 아닌 경우 처리
            if not isinstance(data, pd.DataFrame):
                logger.warning(f"모멘텀 분석: 지원되지 않는 데이터 타입 - {type(data)}")
                return {}
            
            # 필수 컬럼 확인
            if 'close' not in data.columns:
                logger.warning("모멘텀 분석: 'close' 컬럼이 없습니다")
                return {}
            
            # RSI 분석
            if indicators and 'rsi' in indicators:
                try:
                    rsi_data = indicators['rsi']
                    if hasattr(rsi_data, 'iloc') and len(rsi_data) > 0:
                        rsi_current = rsi_data.iloc[-1]
                        if rsi_current > 70:
                            signals['RSI'] = "과매수 - 매도 신호"
                        elif rsi_current < 30:
                            signals['RSI'] = "과매도 - 매수 신호"
                        else:
                            signals['RSI'] = "중립"
                    elif isinstance(rsi_data, (int, float)):
                        rsi_current = rsi_data
                        if rsi_current > 70:
                            signals['RSI'] = "과매수 - 매도 신호"
                        elif rsi_current < 30:
                            signals['RSI'] = "과매도 - 매수 신호"
                        else:
                            signals['RSI'] = "중립"
                except Exception as rsi_error:
                    logger.warning(f"RSI 분석 오류: {str(rsi_error)}")
            
            # MACD 분석
            if indicators and 'macd' in indicators and 'signal' in indicators:
                try:
                    macd_data = indicators['macd']
                    signal_data = indicators['signal']
                    
                    if (hasattr(macd_data, 'iloc') and hasattr(signal_data, 'iloc') and 
                        len(macd_data) > 0 and len(signal_data) > 0):
                        macd_current = macd_data.iloc[-1]
                        signal_current = signal_data.iloc[-1]
                        
                        if macd_current > signal_current:
                            signals['MACD'] = "상승 모멘텀"
                        else:
                            signals['MACD'] = "하락 모멘텀"
                    elif isinstance(macd_data, (int, float)) and isinstance(signal_data, (int, float)):
                        if macd_data > signal_data:
                            signals['MACD'] = "상승 모멘텀"
                        else:
                            signals['MACD'] = "하락 모멘텀"
                except Exception as macd_error:
                    logger.warning(f"MACD 분석 오류: {str(macd_error)}")
            
            # 이동평균 분석
            if indicators:
                try:
                    ma5 = indicators.get('MA5')
                    ma20 = indicators.get('MA20')
                    
                    if ma5 is not None and ma20 is not None:
                        ma5_current = ma5.iloc[-1] if hasattr(ma5, 'iloc') else ma5
                        ma20_current = ma20.iloc[-1] if hasattr(ma20, 'iloc') else ma20
                        
                        if isinstance(ma5_current, (int, float)) and isinstance(ma20_current, (int, float)):
                            if ma5_current > ma20_current:
                                signals['이동평균'] = "단기 상승세"
                            else:
                                signals['이동평균'] = "단기 하락세"
                except Exception as ma_error:
                    logger.warning(f"이동평균 분석 오류: {str(ma_error)}")
            
            # 가격 모멘텀
            if len(data) >= 5:
                try:
                    close_data = data['close']
                    if len(close_data) >= 5:
                        price_change_5d = (close_data.iloc[-1] - close_data.iloc[-5]) / close_data.iloc[-5] * 100
                        
                        if price_change_5d > 5:
                            signals['가격모멘텀'] = "강한 상승"
                        elif price_change_5d > 2:
                            signals['가격모멘텀'] = "약한 상승"
                        elif price_change_5d < -5:
                            signals['가격모멘텀'] = "강한 하락"
                        elif price_change_5d < -2:
                            signals['가격모멘텀'] = "약한 하락"
                        else:
                            signals['가격모멘텀'] = "보합"
                except Exception as price_error:
                    logger.warning(f"가격 모멘텀 분석 오류: {str(price_error)}")
            
            return signals
            
        except Exception as e:
            logger.error(f"모멘텀 분석 오류: {str(e)}")
            return {}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, any]:
        """거래량 분석"""
        try:
            recent_data = data.tail(20)
            
            # 평균 거래량 대비 현재 거래량
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 거래량 트렌드
            volume_trend = "증가" if volume_ratio > 1.2 else "감소" if volume_ratio < 0.8 else "보통"
            
            # 가격-거래량 관계
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5]
            
            if price_change > 0 and volume_ratio > 1.2:
                volume_signal = "상승 확인"
            elif price_change < 0 and volume_ratio > 1.2:
                volume_signal = "하락 확인"
            elif price_change > 0 and volume_ratio < 0.8:
                volume_signal = "상승 의심"
            elif price_change < 0 and volume_ratio < 0.8:
                volume_signal = "하락 의심"
            else:
                volume_signal = "중립"
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'trend': volume_trend,
                'signal': volume_signal,
                'description': f"평균 대비 {volume_ratio:.1f}배 - {volume_signal}"
            }
            
        except Exception as e:
            logger.error(f"거래량 분석 오류: {str(e)}")
            return {
                'trend': '분석불가',
                'signal': '중립',
                'description': '거래량 분석 실패'
            }
    
    def _assess_risk(self, data: pd.DataFrame, patterns: List[ChartPattern], trend: TrendAnalysis) -> str:
        """위험도 평가"""
        try:
            risk_score = 0
            
            # 변동성 기반 위험도
            recent_data = data.tail(20)
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연환산 변동성
            
            if volatility > 0.5:
                risk_score += 3
            elif volatility > 0.3:
                risk_score += 2
            else:
                risk_score += 1
            
            # 트렌드 강도 기반
            if trend.strength < 0.3:
                risk_score += 2  # 약한 트렌드는 위험
            
            # 패턴 기반
            reversal_patterns = [p for p in patterns if p.pattern_type in [
                PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM,
                PatternType.HEAD_AND_SHOULDERS, PatternType.INVERSE_HEAD_AND_SHOULDERS
            ]]
            
            if reversal_patterns:
                risk_score += 1
            
            # 위험도 분류
            if risk_score >= 5:
                return "높음"
            elif risk_score >= 3:
                return "중간"
            else:
                return "낮음"
                
        except Exception as e:
            logger.error(f"위험도 평가 오류: {str(e)}")
            return "중간"
    
    def _get_default_analysis(self) -> TechnicalAnalysis:
        """기본 분석 결과"""
        return TechnicalAnalysis(
            trend=TrendAnalysis(
                direction=TrendDirection.SIDEWAYS,
                strength=0.5,
                start_date=datetime.now().strftime('%Y-%m-%d'),
                duration_days=0,
                support_level=None,
                resistance_level=None
            ),
            patterns=[],
            support_resistance={'support': [], 'resistance': []},
            momentum_signals={},
            volume_analysis={'trend': '분석불가', 'signal': '중립', 'description': '데이터 부족'},
            risk_level="중간"
        )