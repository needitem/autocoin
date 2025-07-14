"""
체계적인 가격 변동 분석 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    STRONG_BULLISH = "강한 상승"
    BULLISH = "상승"
    SIDEWAYS = "횡보"
    BEARISH = "하락"
    STRONG_BEARISH = "강한 하락"

class VolatilityLevel(Enum):
    VERY_HIGH = "매우 높음"
    HIGH = "높음"
    MEDIUM = "보통"
    LOW = "낮음"
    VERY_LOW = "매우 낮음"

@dataclass
class PriceAnalysis:
    """가격 분석 결과"""
    market: str
    current_price: float
    price_change_24h: float
    price_change_1h: float
    trend_direction: TrendDirection
    volatility_level: VolatilityLevel
    volume_analysis: Dict
    technical_indicators: Dict
    support_resistance: Dict
    market_sentiment: Dict
    risk_assessment: Dict
    key_insights: List[str]
    recommendations: List[str]
    pattern_analysis: Dict
    multi_timeframe_analysis: Optional[Dict] = None

class SystematicPriceAnalyzer:
    """체계적인 가격 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_comprehensive(self, market: str, current_data: Dict, historical_data: Optional[pd.DataFrame] = None, 
                             exchange_api=None, enable_multi_timeframe: bool = True) -> PriceAnalysis:
        """종합적인 가격 분석 수행"""
        try:
            # 기본 가격 정보 추출
            current_price = float(current_data.get('trade_price', 0))
            price_change_24h = float(current_data.get('signed_change_rate', 0)) * 100
            
            # 1시간 변동률 계산 (히스토리컬 데이터가 있는 경우)
            price_change_1h = self._calculate_1h_change(historical_data) if historical_data is not None else 0
            
            # 트렌드 분석
            trend_direction = self._analyze_trend(price_change_24h, price_change_1h)
            
            # 변동성 분석
            volatility_level = self._analyze_volatility(historical_data, price_change_24h)
            
            # 거래량 분석
            volume_analysis = self._analyze_volume(current_data, historical_data)
            
            # 기술적 지표 분석
            technical_indicators = self._analyze_technical_indicators(historical_data)
            
            # 지지/저항 분석
            support_resistance = self._analyze_support_resistance(historical_data)
            
            # 패턴 분석
            pattern_analysis = self._analyze_chart_patterns(historical_data)
            
            # 다중 시간대 분석 (옵션)
            multi_timeframe_analysis = None
            if enable_multi_timeframe and exchange_api:
                multi_timeframe_analysis = self._analyze_multi_timeframe(market, exchange_api)
            
            # 시장 심리 분석
            market_sentiment = self._analyze_market_sentiment(price_change_24h, volume_analysis)
            
            # 리스크 평가
            risk_assessment = self._assess_risk(price_change_24h, volatility_level, volume_analysis)
            
            # 핵심 인사이트 생성
            key_insights = self._generate_key_insights(trend_direction, volatility_level, volume_analysis, technical_indicators, pattern_analysis)
            
            # 투자 권장사항 생성
            recommendations = self._generate_recommendations(trend_direction, risk_assessment, market_sentiment, pattern_analysis)
            
            return PriceAnalysis(
                market=market,
                current_price=current_price,
                price_change_24h=price_change_24h,
                price_change_1h=price_change_1h,
                trend_direction=trend_direction,
                volatility_level=volatility_level,
                volume_analysis=volume_analysis,
                technical_indicators=technical_indicators,
                support_resistance=support_resistance,
                market_sentiment=market_sentiment,
                risk_assessment=risk_assessment,
                key_insights=key_insights,
                recommendations=recommendations,
                pattern_analysis=pattern_analysis,
                multi_timeframe_analysis=multi_timeframe_analysis
            )
            
        except Exception as e:
            self.logger.error(f"가격 분석 중 오류 발생: {str(e)}")
            return self._create_fallback_analysis(market, current_data)
    
    def _calculate_1h_change(self, historical_data: pd.DataFrame) -> float:
        """1시간 가격 변동률 계산"""
        if historical_data is None or len(historical_data) < 2:
            return 0
        
        try:
            # 최근 1시간 데이터 필터링
            recent_data = historical_data.tail(60)  # 1분 데이터 기준
            if len(recent_data) < 2:
                return 0
            
            start_price = recent_data.iloc[0]['close']
            end_price = recent_data.iloc[-1]['close']
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception as e:
            self.logger.error(f"1시간 변동률 계산 오류: {str(e)}")
            return 0
    
    def _analyze_trend(self, price_change_24h: float, price_change_1h: float) -> TrendDirection:
        """트렌드 방향 분석"""
        # 24시간 변동률을 기준으로 트렌드 판단
        if price_change_24h > 10:
            return TrendDirection.STRONG_BULLISH
        elif price_change_24h > 3:
            return TrendDirection.BULLISH
        elif price_change_24h < -10:
            return TrendDirection.STRONG_BEARISH
        elif price_change_24h < -3:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def _analyze_volatility(self, historical_data: Optional[pd.DataFrame], price_change_24h: float) -> VolatilityLevel:
        """변동성 수준 분석"""
        try:
            if historical_data is not None and len(historical_data) > 20:
                # 히스토리컬 데이터 기반 변동성 계산
                returns = historical_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(1440)  # 일일 변동성 (1분 데이터 기준)
                
                if volatility > 0.08:
                    return VolatilityLevel.VERY_HIGH
                elif volatility > 0.05:
                    return VolatilityLevel.HIGH
                elif volatility > 0.03:
                    return VolatilityLevel.MEDIUM
                elif volatility > 0.01:
                    return VolatilityLevel.LOW
                else:
                    return VolatilityLevel.VERY_LOW
            else:
                # 24시간 변동률 기반 변동성 추정
                abs_change = abs(price_change_24h)
                if abs_change > 15:
                    return VolatilityLevel.VERY_HIGH
                elif abs_change > 8:
                    return VolatilityLevel.HIGH
                elif abs_change > 4:
                    return VolatilityLevel.MEDIUM
                elif abs_change > 2:
                    return VolatilityLevel.LOW
                else:
                    return VolatilityLevel.VERY_LOW
                    
        except Exception as e:
            self.logger.error(f"변동성 분석 오류: {str(e)}")
            return VolatilityLevel.MEDIUM
    
    def _analyze_volume(self, current_data: Dict, historical_data: Optional[pd.DataFrame]) -> Dict:
        """거래량 분석"""
        try:
            current_volume = float(current_data.get('acc_trade_volume_24h', 0))
            
            volume_analysis = {
                'current_volume': current_volume,
                'volume_trend': '데이터 부족',
                'volume_level': '보통',
                'volume_price_correlation': '분석 불가'
            }
            
            if historical_data is not None and len(historical_data) > 7:
                # 평균 거래량 계산
                avg_volume = historical_data['volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # 거래량 트렌드 분석
                if volume_ratio > 2:
                    volume_analysis['volume_trend'] = '급증'
                    volume_analysis['volume_level'] = '매우 높음'
                elif volume_ratio > 1.5:
                    volume_analysis['volume_trend'] = '증가'
                    volume_analysis['volume_level'] = '높음'
                elif volume_ratio > 0.8:
                    volume_analysis['volume_trend'] = '유지'
                    volume_analysis['volume_level'] = '보통'
                else:
                    volume_analysis['volume_trend'] = '감소'
                    volume_analysis['volume_level'] = '낮음'
                
                # 가격-거래량 상관관계 분석
                price_changes = historical_data['close'].pct_change().dropna()
                volume_changes = historical_data['volume'].pct_change().dropna()
                
                if len(price_changes) > 10 and len(volume_changes) > 10:
                    correlation = np.corrcoef(price_changes[-10:], volume_changes[-10:])[0, 1]
                    
                    if correlation > 0.3:
                        volume_analysis['volume_price_correlation'] = '양의 상관관계'
                    elif correlation < -0.3:
                        volume_analysis['volume_price_correlation'] = '음의 상관관계'
                    else:
                        volume_analysis['volume_price_correlation'] = '상관관계 없음'
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"거래량 분석 오류: {str(e)}")
            return {
                'current_volume': 0,
                'volume_trend': '분석 실패',
                'volume_level': '알 수 없음',
                'volume_price_correlation': '분석 실패'
            }
    
    def _analyze_technical_indicators(self, historical_data: Optional[pd.DataFrame]) -> Dict:
        """기술적 지표 분석"""
        indicators = {
            'rsi': '데이터 부족',
            'macd': '데이터 부족',
            'bb_position': '데이터 부족',
            'sma_20': 0,
            'sma_50': 0,
            'signal': '중립'
        }
        
        try:
            if historical_data is None or len(historical_data) < 50:
                return indicators
            
            closes = historical_data['close'].values
            
            # RSI 계산
            rsi = self._calculate_rsi(closes)
            if rsi:
                indicators['rsi'] = f"{rsi:.1f}"
                if rsi > 70:
                    indicators['rsi'] += " (과매수)"
                elif rsi < 30:
                    indicators['rsi'] += " (과매도)"
            
            # 이동평균 계산
            if len(closes) >= 20:
                indicators['sma_20'] = np.mean(closes[-20:])
            if len(closes) >= 50:
                indicators['sma_50'] = np.mean(closes[-50:])
            
            # MACD 계산
            macd_line, signal_line = self._calculate_macd(closes)
            if macd_line and signal_line:
                if macd_line > signal_line:
                    indicators['macd'] = "상승 신호"
                else:
                    indicators['macd'] = "하락 신호"
            
            # 볼린저 밴드 위치
            bb_position = self._calculate_bb_position(closes)
            if bb_position:
                indicators['bb_position'] = bb_position
            
            # 종합 신호 생성
            indicators['signal'] = self._generate_technical_signal(indicators, closes[-1])
            
        except Exception as e:
            self.logger.error(f"기술적 지표 분석 오류: {str(e)}")
        
        return indicators
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> Optional[float]:
        """RSI 계산"""
        try:
            if len(closes) < period + 1:
                return None
            
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return None
    
    def _calculate_macd(self, closes: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """MACD 계산"""
        try:
            if len(closes) < 26:
                return None, None
            
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            
            if ema_12 and ema_26:
                macd_line = ema_12 - ema_26
                # 신호선은 MACD의 9일 EMA (단순화)
                signal_line = macd_line * 0.9  # 근사치
                return macd_line, signal_line
            
            return None, None
            
        except Exception:
            return None, None
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> Optional[float]:
        """EMA 계산"""
        try:
            if len(data) < period:
                return None
            
            multiplier = 2 / (period + 1)
            ema = np.mean(data[:period])
            
            for price in data[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception:
            return None
    
    def _calculate_bb_position(self, closes: np.ndarray, period: int = 20) -> Optional[str]:
        """볼린저 밴드 위치 계산"""
        try:
            if len(closes) < period:
                return None
            
            recent_closes = closes[-period:]
            sma = np.mean(recent_closes)
            std = np.std(recent_closes)
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current_price = closes[-1]
            
            if current_price > upper_band:
                return "상단 밴드 근처 (과매수 가능)"
            elif current_price < lower_band:
                return "하단 밴드 근처 (과매도 가능)"
            else:
                bb_position = (current_price - lower_band) / (upper_band - lower_band)
                if bb_position > 0.7:
                    return "상단 근처"
                elif bb_position < 0.3:
                    return "하단 근처"
                else:
                    return "중간 영역"
                    
        except Exception:
            return None
    
    def _generate_technical_signal(self, indicators: Dict, current_price: float) -> str:
        """기술적 지표 기반 종합 신호"""
        signals = []
        
        # RSI 신호
        rsi_str = str(indicators.get('rsi', ''))
        if '과매수' in rsi_str:
            signals.append('매도')
        elif '과매도' in rsi_str:
            signals.append('매수')
        
        # MACD 신호
        macd = indicators.get('macd', '')
        if '상승' in macd:
            signals.append('매수')
        elif '하락' in macd:
            signals.append('매도')
        
        # 이동평균 신호
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        
        if sma_20 > 0 and sma_50 > 0:
            if current_price > sma_20 > sma_50:
                signals.append('매수')
            elif current_price < sma_20 < sma_50:
                signals.append('매도')
        
        # 종합 판단
        buy_signals = signals.count('매수')
        sell_signals = signals.count('매도')
        
        if buy_signals > sell_signals:
            return '매수 신호'
        elif sell_signals > buy_signals:
            return '매도 신호'
        else:
            return '중립'
    
    def _analyze_support_resistance(self, historical_data: Optional[pd.DataFrame]) -> Dict:
        """지지/저항선 분석"""
        result = {
            'support_levels': [],
            'resistance_levels': [],
            'current_level': '분석 불가'
        }
        
        try:
            if historical_data is None or len(historical_data) < 50:
                return result
            
            closes = historical_data['close'].values
            highs = historical_data['high'].values
            lows = historical_data['low'].values
            
            # 최근 50개 데이터로 지지/저항 찾기
            recent_closes = closes[-50:]
            recent_highs = highs[-50:]
            recent_lows = lows[-50:]
            
            # 저항선 (최근 고점들)
            resistance_levels = []
            for i in range(2, len(recent_highs)-2):
                if (recent_highs[i] > recent_highs[i-1] and 
                    recent_highs[i] > recent_highs[i+1] and
                    recent_highs[i] > recent_highs[i-2] and 
                    recent_highs[i] > recent_highs[i+2]):
                    resistance_levels.append(recent_highs[i])
            
            # 지지선 (최근 저점들)
            support_levels = []
            for i in range(2, len(recent_lows)-2):
                if (recent_lows[i] < recent_lows[i-1] and 
                    recent_lows[i] < recent_lows[i+1] and
                    recent_lows[i] < recent_lows[i-2] and 
                    recent_lows[i] < recent_lows[i+2]):
                    support_levels.append(recent_lows[i])
            
            # 상위 3개씩 선택
            result['resistance_levels'] = sorted(resistance_levels, reverse=True)[:3]
            result['support_levels'] = sorted(support_levels, reverse=True)[:3]
            
            # 현재 위치 분석
            current_price = closes[-1]
            if result['resistance_levels'] and result['support_levels']:
                nearest_resistance = min(result['resistance_levels'], key=lambda x: abs(x - current_price))
                nearest_support = min(result['support_levels'], key=lambda x: abs(x - current_price))
                
                if abs(current_price - nearest_resistance) < abs(current_price - nearest_support):
                    result['current_level'] = f"저항선 근처 ({nearest_resistance:,.0f})"
                else:
                    result['current_level'] = f"지지선 근처 ({nearest_support:,.0f})"
            
        except Exception as e:
            self.logger.error(f"지지/저항 분석 오류: {str(e)}")
        
        return result
    
    def _analyze_chart_patterns(self, historical_data: Optional[pd.DataFrame]) -> Dict:
        """차트 패턴 분석"""
        pattern_result = {
            'detected_patterns': [],
            'primary_pattern': None,
            'pattern_signals': [],
            'pattern_reliability': 'LOW'
        }
        
        try:
            if historical_data is None or len(historical_data) < 10:
                return pattern_result
            
            # 패턴 인식기 초기화
            from src.core.pattern_recognition import ComprehensivePatternRecognizer
            recognizer = ComprehensivePatternRecognizer()
            
            # 패턴 분석 실행
            patterns = recognizer.analyze_patterns(historical_data)
            
            if patterns:
                pattern_result['detected_patterns'] = patterns
                
                # 주요 패턴 선택 (가장 신뢰도 높은 패턴)
                primary_pattern = max(patterns, key=lambda p: p.confidence)
                pattern_result['primary_pattern'] = {
                    'name': primary_pattern.korean_name,
                    'type': primary_pattern.pattern_type.value,
                    'signal': primary_pattern.signal.value,
                    'confidence': primary_pattern.confidence,
                    'reliability': primary_pattern.reliability.value,
                    'description': primary_pattern.description
                }
                
                # 패턴 신호 요약
                signals = [p.signal.value for p in patterns if p.signal.value != '중립']
                pattern_result['pattern_signals'] = signals
                
                # 전체 패턴 신뢰도 계산
                avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
                if avg_confidence > 0.8:
                    pattern_result['pattern_reliability'] = 'VERY_HIGH'
                elif avg_confidence > 0.7:
                    pattern_result['pattern_reliability'] = 'HIGH'
                elif avg_confidence > 0.5:
                    pattern_result['pattern_reliability'] = 'MEDIUM'
                else:
                    pattern_result['pattern_reliability'] = 'LOW'
                
                self.logger.info(f"패턴 분석 완료: {len(patterns)}개 패턴 감지")
            
        except Exception as e:
            self.logger.error(f"패턴 분석 오류: {str(e)}")
        
        return pattern_result
    
    def _analyze_multi_timeframe(self, market: str, exchange_api) -> Dict:
        """다중 시간대 분석"""
        multi_timeframe_result = {
            'timeframe_analyses': {},
            'summary': {},
            'conflicts': [],
            'convergence_score': 0.0
        }
        
        try:
            from src.core.multi_timeframe_analyzer import MultiTimeFrameAnalyzer
            analyzer = MultiTimeFrameAnalyzer()
            
            # 비동기 분석을 동기적으로 실행
            import asyncio
            
            # 이벤트 루프가 이미 실행 중인지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, analyzer.analyze_all_timeframes(market, exchange_api))
                    timeframe_analyses = future.result(timeout=30)
            except RuntimeError:
                # 실행 중인 루프가 없으면 직접 실행
                timeframe_analyses = asyncio.run(analyzer.analyze_all_timeframes(market, exchange_api))
            
            if timeframe_analyses:
                # 분석 결과를 직렬화 가능한 형태로 변환
                serializable_analyses = {}
                for tf, analysis in timeframe_analyses.items():
                    serializable_analyses[tf.value] = {
                        'trend_direction': analysis.trend_direction,
                        'trend_strength': analysis.trend_strength,
                        'volume_analysis': analysis.volume_analysis,
                        'pattern_count': analysis.pattern_count,
                        'primary_patterns': analysis.primary_patterns,
                        'support_levels': analysis.support_levels,
                        'resistance_levels': analysis.resistance_levels,
                        'key_insights': analysis.key_insights,
                        'signal_strength': analysis.signal_strength,
                        'signal_direction': analysis.signal_direction,
                        'reliability_score': analysis.reliability_score
                    }
                
                multi_timeframe_result['timeframe_analyses'] = serializable_analyses
                
                # 종합 요약 생성
                summary = analyzer.generate_multi_timeframe_summary(timeframe_analyses)
                multi_timeframe_result['summary'] = summary
                multi_timeframe_result['conflicts'] = summary.get('timeframe_conflicts', [])
                multi_timeframe_result['convergence_score'] = 1.0 - summary.get('signal_convergence', 0.5)
                
                self.logger.info(f"다중 시간대 분석 완료: {len(timeframe_analyses)}개 시간대")
            
        except Exception as e:
            self.logger.error(f"다중 시간대 분석 오류: {str(e)}")
            multi_timeframe_result['error'] = str(e)
        
        return multi_timeframe_result
    
    def _analyze_market_sentiment(self, price_change_24h: float, volume_analysis: Dict) -> Dict:
        """시장 심리 분석"""
        sentiment = {
            'overall': '중립',
            'score': 0.5,
            'confidence': '보통',
            'factors': []
        }
        
        try:
            score = 0.5
            factors = []
            
            # 가격 변동 기반 심리
            if price_change_24h > 10:
                score += 0.3
                factors.append('강한 상승세로 인한 긍정적 심리')
            elif price_change_24h > 5:
                score += 0.2
                factors.append('상승세로 인한 긍정적 심리')
            elif price_change_24h < -10:
                score -= 0.3
                factors.append('강한 하락세로 인한 부정적 심리')
            elif price_change_24h < -5:
                score -= 0.2
                factors.append('하락세로 인한 부정적 심리')
            
            # 거래량 기반 심리
            volume_level = volume_analysis.get('volume_level', '보통')
            if volume_level == '매우 높음':
                factors.append('높은 거래량으로 인한 활발한 시장 참여')
                score += 0.1
            elif volume_level == '낮음':
                factors.append('낮은 거래량으로 인한 관망세')
                score -= 0.1
            
            # 점수 정규화
            score = max(0, min(1, score))
            
            # 전체 심리 결정
            if score > 0.7:
                sentiment['overall'] = '매우 긍정'
                sentiment['confidence'] = '높음'
            elif score > 0.6:
                sentiment['overall'] = '긍정'
                sentiment['confidence'] = '높음'
            elif score > 0.4:
                sentiment['overall'] = '중립'
                sentiment['confidence'] = '보통'
            elif score > 0.3:
                sentiment['overall'] = '부정'
                sentiment['confidence'] = '높음'
            else:
                sentiment['overall'] = '매우 부정'
                sentiment['confidence'] = '높음'
            
            sentiment['score'] = score
            sentiment['factors'] = factors
            
        except Exception as e:
            self.logger.error(f"시장 심리 분석 오류: {str(e)}")
        
        return sentiment
    
    def _assess_risk(self, price_change_24h: float, volatility_level: VolatilityLevel, volume_analysis: Dict) -> Dict:
        """리스크 평가"""
        risk = {
            'level': '보통',
            'score': 0.5,
            'factors': [],
            'recommendations': []
        }
        
        try:
            risk_score = 0.5
            factors = []
            recommendations = []
            
            # 변동성 기반 리스크
            if volatility_level == VolatilityLevel.VERY_HIGH:
                risk_score += 0.3
                factors.append('매우 높은 변동성')
                recommendations.append('포지션 크기 축소 고려')
            elif volatility_level == VolatilityLevel.HIGH:
                risk_score += 0.2
                factors.append('높은 변동성')
                recommendations.append('리스크 관리 강화')
            
            # 가격 변동 기반 리스크
            if abs(price_change_24h) > 15:
                risk_score += 0.2
                factors.append('극심한 가격 변동')
                recommendations.append('단기 거래 주의')
            
            # 거래량 기반 리스크
            volume_level = volume_analysis.get('volume_level', '보통')
            if volume_level == '낮음':
                risk_score += 0.1
                factors.append('낮은 유동성')
                recommendations.append('대량 거래 시 주의')
            
            # 리스크 점수 정규화
            risk_score = max(0, min(1, risk_score))
            
            # 리스크 레벨 결정
            if risk_score > 0.8:
                risk['level'] = '매우 높음'
            elif risk_score > 0.6:
                risk['level'] = '높음'
            elif risk_score > 0.4:
                risk['level'] = '보통'
            elif risk_score > 0.2:
                risk['level'] = '낮음'
            else:
                risk['level'] = '매우 낮음'
            
            risk['score'] = risk_score
            risk['factors'] = factors
            risk['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.error(f"리스크 평가 오류: {str(e)}")
        
        return risk
    
    def _generate_key_insights(self, trend_direction: TrendDirection, volatility_level: VolatilityLevel, 
                             volume_analysis: Dict, technical_indicators: Dict, pattern_analysis: Dict) -> List[str]:
        """핵심 인사이트 생성"""
        insights = []
        
        try:
            # 트렌드 인사이트
            if trend_direction == TrendDirection.STRONG_BULLISH:
                insights.append("🚀 강한 상승 트렌드가 지속되고 있습니다")
            elif trend_direction == TrendDirection.STRONG_BEARISH:
                insights.append("📉 강한 하락 트렌드가 지속되고 있습니다")
            elif trend_direction == TrendDirection.SIDEWAYS:
                insights.append("➡️ 횡보 구간에서 방향성을 찾고 있습니다")
            
            # 변동성 인사이트
            if volatility_level == VolatilityLevel.VERY_HIGH:
                insights.append("⚠️ 매우 높은 변동성으로 급격한 가격 변동 가능")
            elif volatility_level == VolatilityLevel.VERY_LOW:
                insights.append("😴 낮은 변동성으로 안정적인 움직임")
            
            # 거래량 인사이트
            volume_trend = volume_analysis.get('volume_trend', '')
            if volume_trend == '급증':
                insights.append("📊 거래량 급증으로 강한 시장 관심")
            elif volume_trend == '감소':
                insights.append("📉 거래량 감소로 관망세 증가")
            
            # 기술적 지표 인사이트
            signal = technical_indicators.get('signal', '')
            if signal == '매수 신호':
                insights.append("📈 기술적 지표들이 매수 신호를 보이고 있습니다")
            elif signal == '매도 신호':
                insights.append("📉 기술적 지표들이 매도 신호를 보이고 있습니다")
            
            # 패턴 분석 인사이트
            primary_pattern = pattern_analysis.get('primary_pattern')
            if primary_pattern:
                pattern_name = primary_pattern.get('name', '')
                pattern_signal = primary_pattern.get('signal', '')
                confidence = primary_pattern.get('confidence', 0)
                
                if confidence > 0.7:
                    insights.append(f"🔍 {pattern_name} 패턴이 감지되었습니다 (신뢰도: {confidence:.1%})")
                    
                    if pattern_signal in ['강한 매수', '매수']:
                        insights.append(f"📊 {pattern_name} 패턴이 강세 신호를 보이고 있습니다")
                    elif pattern_signal in ['강한 매도', '매도']:
                        insights.append(f"📊 {pattern_name} 패턴이 약세 신호를 보이고 있습니다")
            
            # 패턴 신호 종합
            pattern_signals = pattern_analysis.get('pattern_signals', [])
            if pattern_signals:
                buy_signals = sum(1 for s in pattern_signals if '매수' in s)
                sell_signals = sum(1 for s in pattern_signals if '매도' in s)
                
                if buy_signals > sell_signals:
                    insights.append(f"🎯 감지된 패턴들이 전반적으로 매수 신호를 보이고 있습니다")
                elif sell_signals > buy_signals:
                    insights.append(f"🎯 감지된 패턴들이 전반적으로 매도 신호를 보이고 있습니다")
            
            # 최대 7개 인사이트 반환
            return insights[:7]
            
        except Exception as e:
            self.logger.error(f"인사이트 생성 오류: {str(e)}")
            return ["📊 시장 데이터를 분석 중입니다"]
    
    def _generate_recommendations(self, trend_direction: TrendDirection, risk_assessment: Dict, 
                                market_sentiment: Dict, pattern_analysis: Dict) -> List[str]:
        """투자 권장사항 생성"""
        recommendations = []
        
        try:
            risk_level = risk_assessment.get('level', '보통')
            sentiment = market_sentiment.get('overall', '중립')
            
            # 기본 추천
            if trend_direction == TrendDirection.STRONG_BULLISH and sentiment == '긍정':
                recommendations.append("💰 상승 모멘텀 활용 고려")
            elif trend_direction == TrendDirection.STRONG_BEARISH and sentiment == '부정':
                recommendations.append("🛡️ 손절매 및 포지션 축소 고려")
            elif trend_direction == TrendDirection.SIDEWAYS:
                recommendations.append("⏳ 명확한 방향성 확인 후 진입")
            
            # 리스크 기반 추천
            if risk_level in ['높음', '매우 높음']:
                recommendations.append("⚠️ 리스크 관리 강화 필요")
                recommendations.append("📊 포지션 크기 축소 고려")
            elif risk_level in ['낮음', '매우 낮음']:
                recommendations.append("✅ 안정적인 시장 상황")
            
            # 심리 기반 추천
            if sentiment == '매우 긍정':
                recommendations.append("🔍 과열 구간 주의")
            elif sentiment == '매우 부정':
                recommendations.append("🎯 매수 기회 탐색")
            
            # 패턴 기반 추천
            primary_pattern = pattern_analysis.get('primary_pattern')
            if primary_pattern:
                pattern_type = primary_pattern.get('type', '')
                pattern_signal = primary_pattern.get('signal', '')
                confidence = primary_pattern.get('confidence', 0)
                
                if confidence > 0.7:
                    if pattern_type == '반전':
                        recommendations.append("🔄 트렌드 반전 가능성 주시")
                    elif pattern_type == '지속':
                        recommendations.append("⏭️ 기존 트렌드 지속 가능성 고려")
                    
                    if pattern_signal in ['강한 매수', '매수']:
                        recommendations.append(f"📊 {primary_pattern.get('name', '')} 패턴 매수 신호 고려")
                    elif pattern_signal in ['강한 매도', '매도']:
                        recommendations.append(f"📊 {primary_pattern.get('name', '')} 패턴 매도 신호 고려")
            
            # 패턴 신뢰도 기반 추천
            pattern_reliability = pattern_analysis.get('pattern_reliability', 'LOW')
            if pattern_reliability in ['VERY_HIGH', 'HIGH']:
                recommendations.append("🎯 패턴 신뢰도가 높아 참고 가치 있음")
            elif pattern_reliability == 'LOW':
                recommendations.append("⚠️ 패턴 신뢰도가 낮아 추가 확인 필요")
            
            # 일반적인 추천
            recommendations.extend([
                "📈 기술적 분석과 패턴 분석 종합 판단",
                "💼 분산 투자 및 리스크 관리",
                "📰 뉴스 및 시장 동향 모니터링"
            ])
            
            return recommendations[:8]
            
        except Exception as e:
            self.logger.error(f"추천사항 생성 오류: {str(e)}")
            return ["📊 신중한 분석 후 투자 결정 권장"]
    
    def _create_fallback_analysis(self, market: str, current_data: Dict) -> PriceAnalysis:
        """오류 시 기본 분석 반환"""
        return PriceAnalysis(
            market=market,
            current_price=float(current_data.get('trade_price', 0)),
            price_change_24h=float(current_data.get('signed_change_rate', 0)) * 100,
            price_change_1h=0,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_level=VolatilityLevel.MEDIUM,
            volume_analysis={'current_volume': 0, 'volume_trend': '분석 실패'},
            technical_indicators={'signal': '분석 실패'},
            support_resistance={'current_level': '분석 실패'},
            market_sentiment={'overall': '분석 실패'},
            risk_assessment={'level': '분석 실패'},
            key_insights=['데이터 분석 중 오류 발생'],
            recommendations=['다시 시도해 주세요'],
            pattern_analysis={'detected_patterns': [], 'primary_pattern': None},
            multi_timeframe_analysis=None
        )