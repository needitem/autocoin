"""
AI 기반 가격 예측 모델
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class PredictionConfidence(Enum):
    """예측 신뢰도"""
    VERY_HIGH = "매우 높음"
    HIGH = "높음"
    MEDIUM = "보통"
    LOW = "낮음"
    VERY_LOW = "매우 낮음"

class PredictionDirection(Enum):
    """예측 방향"""
    STRONG_UP = "강한 상승"
    UP = "상승"
    NEUTRAL = "보합"
    DOWN = "하락"
    STRONG_DOWN = "강한 하락"

@dataclass
class PricePrediction:
    """가격 예측 결과"""
    market: str
    current_price: float
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_24h: float
    direction: PredictionDirection
    confidence: PredictionConfidence
    probability_up: float  # 0.0 ~ 1.0
    probability_down: float  # 0.0 ~ 1.0
    key_factors: List[str]
    risk_assessment: str
    timestamp: datetime

class AIPricePredictor:
    """AI 기반 가격 예측기"""
    
    def __init__(self):
        self.model_weights = {
            'technical_indicators': 0.35,
            'price_action': 0.25,
            'volume_analysis': 0.20,
            'market_sentiment': 0.15,
            'time_patterns': 0.05
        }
        
        # 기술적 지표 가중치
        self.indicator_weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bollinger': 0.15,
            'moving_averages': 0.25,
            'stochastic': 0.15
        }
        
        # 학습된 패턴 데이터베이스 (가중치 기반)
        self.pattern_probabilities = {
            'hammer': {'up': 0.75, 'down': 0.25, 'weight': 0.8},
            'doji': {'up': 0.5, 'down': 0.5, 'weight': 0.6},
            'engulfing_bullish': {'up': 0.85, 'down': 0.15, 'weight': 0.9},
            'engulfing_bearish': {'up': 0.15, 'down': 0.85, 'weight': 0.9},
            'shooting_star': {'up': 0.25, 'down': 0.75, 'weight': 0.8},
            'double_bottom': {'up': 0.8, 'down': 0.2, 'weight': 0.85},
            'double_top': {'up': 0.2, 'down': 0.8, 'weight': 0.85}
        }
    
    def predict_price(self, ohlcv_data: pd.DataFrame, indicators: Dict = None, 
                     patterns: List = None, market_data: Dict = None) -> PricePrediction:
        """AI 기반 가격 예측"""
        try:
            if ohlcv_data.empty or len(ohlcv_data) < 20:
                return self._get_default_prediction(ohlcv_data)
            
            current_price = ohlcv_data['close'].iloc[-1]
            market = market_data.get('market', 'UNKNOWN') if market_data else 'UNKNOWN'
            
            # 1. 기술적 지표 분석
            technical_score = self._analyze_technical_indicators(indicators) if indicators else 0.5
            
            # 2. 가격 액션 분석
            price_action_score = self._analyze_price_action(ohlcv_data)
            
            # 3. 볼륨 분석
            volume_score = self._analyze_volume_patterns(ohlcv_data)
            
            # 4. 패턴 분석
            pattern_score = self._analyze_chart_patterns(patterns) if patterns else 0.5
            
            # 5. 시간 패턴 분석
            time_score = self._analyze_time_patterns(ohlcv_data)
            
            # 종합 점수 계산
            combined_score = (
                technical_score * self.model_weights['technical_indicators'] +
                price_action_score * self.model_weights['price_action'] +
                volume_score * self.model_weights['volume_analysis'] +
                pattern_score * self.model_weights['market_sentiment'] +
                time_score * self.model_weights['time_patterns']
            )
            
            # 방향성 및 신뢰도 결정
            direction, confidence = self._determine_direction_confidence(combined_score)
            
            # 확률 계산
            prob_up, prob_down = self._calculate_probabilities(combined_score)
            
            # 가격 예측
            pred_1h, pred_4h, pred_24h = self._predict_future_prices(
                current_price, combined_score, ohlcv_data
            )
            
            # 주요 요인 분석
            key_factors = self._identify_key_factors(
                technical_score, price_action_score, volume_score, pattern_score, time_score
            )
            
            # 위험도 평가
            risk_assessment = self._assess_risk(ohlcv_data, combined_score)
            
            return PricePrediction(
                market=market,
                current_price=current_price,
                predicted_price_1h=pred_1h,
                predicted_price_4h=pred_4h,
                predicted_price_24h=pred_24h,
                direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                probability_down=prob_down,
                key_factors=key_factors,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"AI 예측 오류: {str(e)}")
            return self._get_default_prediction(ohlcv_data)
    
    def _analyze_technical_indicators(self, indicators: Dict) -> float:
        """기술적 지표 분석"""
        try:
            total_score = 0
            total_weight = 0
            
            # RSI 분석
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    rsi_score = 0.8  # 과매도 -> 상승 기대
                elif rsi > 70:
                    rsi_score = 0.2  # 과매수 -> 하락 기대
                else:
                    rsi_score = 0.5 + (50 - abs(rsi - 50)) / 100  # 중립 지역
                
                total_score += rsi_score * self.indicator_weights['rsi']
                total_weight += self.indicator_weights['rsi']
            
            # MACD 분석
            if 'macd' in indicators and 'signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                signal = indicators['signal'].iloc[-1]
                
                macd_score = 0.7 if macd > signal else 0.3
                if len(indicators['macd']) > 1:
                    # 교차 확인
                    prev_diff = indicators['macd'].iloc[-2] - indicators['signal'].iloc[-2]
                    curr_diff = macd - signal
                    if prev_diff < 0 and curr_diff > 0:  # 골든 크로스
                        macd_score = 0.8
                    elif prev_diff > 0 and curr_diff < 0:  # 데드 크로스
                        macd_score = 0.2
                
                total_score += macd_score * self.indicator_weights['macd']
                total_weight += self.indicator_weights['macd']
            
            # 이동평균 분석
            if 'MA5' in indicators and 'MA20' in indicators:
                ma5 = indicators['MA5'].iloc[-1]
                ma20 = indicators['MA20'].iloc[-1]
                
                ma_score = 0.7 if ma5 > ma20 else 0.3
                
                # 기울기 분석
                if len(indicators['MA5']) > 5:
                    ma5_slope = (indicators['MA5'].iloc[-1] - indicators['MA5'].iloc[-5]) / indicators['MA5'].iloc[-5]
                    ma20_slope = (indicators['MA20'].iloc[-1] - indicators['MA20'].iloc[-5]) / indicators['MA20'].iloc[-5]
                    
                    if ma5_slope > 0 and ma20_slope > 0:
                        ma_score += 0.1
                    elif ma5_slope < 0 and ma20_slope < 0:
                        ma_score -= 0.1
                
                total_score += ma_score * self.indicator_weights['moving_averages']
                total_weight += self.indicator_weights['moving_averages']
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"기술적 지표 분석 오류: {str(e)}")
            return 0.5
    
    def _analyze_price_action(self, ohlcv_data: pd.DataFrame) -> float:
        """가격 액션 분석"""
        try:
            if len(ohlcv_data) < 10:
                return 0.5
            
            recent_data = ohlcv_data.tail(10)
            score = 0.5
            
            # 캔들스틱 분석
            last_candle = recent_data.iloc[-1]
            prev_candle = recent_data.iloc[-2]
            
            # 캔들 타입 분석
            body_size = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # 강한 양봉/음봉
                if body_ratio > 0.7:
                    if last_candle['close'] > last_candle['open']:
                        score += 0.2  # 강한 양봉
                    else:
                        score -= 0.2  # 강한 음봉
            
            # 연속성 분석
            up_candles = sum(1 for i in range(len(recent_data)) 
                           if recent_data.iloc[i]['close'] > recent_data.iloc[i]['open'])
            
            if up_candles > 6:
                score += 0.1
            elif up_candles < 4:
                score -= 0.1
            
            # 고점/저점 분석
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # 상승 추세 확인
            if highs[-1] > highs[-3] and lows[-1] > lows[-3]:
                score += 0.15
            elif highs[-1] < highs[-3] and lows[-1] < lows[-3]:
                score -= 0.15
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"가격 액션 분석 오류: {str(e)}")
            return 0.5
    
    def _analyze_volume_patterns(self, ohlcv_data: pd.DataFrame) -> float:
        """볼륨 패턴 분석"""
        try:
            if len(ohlcv_data) < 20:
                return 0.5
            
            recent_data = ohlcv_data.tail(20)
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            
            score = 0.5
            
            # 볼륨 증가와 가격 관계
            if current_volume > avg_volume * 1.5:
                # 높은 거래량
                price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-2]) / recent_data['close'].iloc[-2]
                if price_change > 0:
                    score += 0.2  # 상승 + 높은 거래량
                else:
                    score -= 0.1  # 하락 + 높은 거래량 (약한 신호)
            
            # 거래량 트렌드 분석
            volume_ma5 = recent_data['volume'].tail(5).mean()
            volume_ma15 = recent_data['volume'].tail(15).mean()
            
            if volume_ma5 > volume_ma15 * 1.2:
                score += 0.1  # 거래량 증가 추세
            elif volume_ma5 < volume_ma15 * 0.8:
                score -= 0.05  # 거래량 감소 추세
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"볼륨 패턴 분석 오류: {str(e)}")
            return 0.5
    
    def _analyze_chart_patterns(self, patterns: List) -> float:
        """차트 패턴 분석"""
        try:
            if not patterns:
                return 0.5
            
            total_score = 0
            total_weight = 0
            
            for pattern in patterns:
                pattern_name = pattern.pattern_type.value.lower()
                
                # 패턴 이름 매핑
                pattern_key = None
                if 'hammer' in pattern_name or '해머' in pattern_name:
                    pattern_key = 'hammer'
                elif 'doji' in pattern_name or '도지' in pattern_name:
                    pattern_key = 'doji'
                elif 'engulfing' in pattern_name and ('bullish' in pattern_name or '강세' in pattern_name):
                    pattern_key = 'engulfing_bullish'
                elif 'engulfing' in pattern_name and ('bearish' in pattern_name or '약세' in pattern_name):
                    pattern_key = 'engulfing_bearish'
                elif 'shooting' in pattern_name or '슈팅' in pattern_name:
                    pattern_key = 'shooting_star'
                elif 'double_bottom' in pattern_name or '더블바텀' in pattern_name:
                    pattern_key = 'double_bottom'
                elif 'double_top' in pattern_name or '더블탑' in pattern_name:
                    pattern_key = 'double_top'
                
                if pattern_key and pattern_key in self.pattern_probabilities:
                    pattern_info = self.pattern_probabilities[pattern_key]
                    pattern_score = pattern_info['up']
                    pattern_weight = pattern_info['weight'] * pattern.confidence
                    
                    total_score += pattern_score * pattern_weight
                    total_weight += pattern_weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"차트 패턴 분석 오류: {str(e)}")
            return 0.5
    
    def _analyze_time_patterns(self, ohlcv_data: pd.DataFrame) -> float:
        """시간 패턴 분석"""
        try:
            # 간단한 시간 패턴 분석 (요일, 시간대별)
            current_time = datetime.now()
            score = 0.5
            
            # 요일 패턴 (경험적)
            weekday = current_time.weekday()
            if weekday in [0, 1]:  # 월, 화 - 상승 경향
                score += 0.05
            elif weekday in [4, 6]:  # 금, 일 - 하락 경향
                score -= 0.05
            
            # 시간대 패턴 (한국 시간 기준)
            hour = current_time.hour
            if 9 <= hour <= 11:  # 오전 활발한 시간
                score += 0.03
            elif 21 <= hour <= 23:  # 저녁 활발한 시간
                score += 0.03
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"시간 패턴 분석 오류: {str(e)}")
            return 0.5
    
    def _determine_direction_confidence(self, score: float) -> Tuple[PredictionDirection, PredictionConfidence]:
        """방향성 및 신뢰도 결정"""
        if score >= 0.8:
            return PredictionDirection.STRONG_UP, PredictionConfidence.VERY_HIGH
        elif score >= 0.65:
            return PredictionDirection.UP, PredictionConfidence.HIGH
        elif score >= 0.55:
            return PredictionDirection.UP, PredictionConfidence.MEDIUM
        elif score >= 0.45:
            return PredictionDirection.NEUTRAL, PredictionConfidence.MEDIUM
        elif score >= 0.35:
            return PredictionDirection.DOWN, PredictionConfidence.MEDIUM
        elif score >= 0.2:
            return PredictionDirection.DOWN, PredictionConfidence.HIGH
        else:
            return PredictionDirection.STRONG_DOWN, PredictionConfidence.VERY_HIGH
    
    def _calculate_probabilities(self, score: float) -> Tuple[float, float]:
        """상승/하락 확률 계산"""
        # 시그모이드 함수를 사용한 확률 계산
        prob_up = 1 / (1 + np.exp(-10 * (score - 0.5)))
        prob_down = 1 - prob_up
        
        return prob_up, prob_down
    
    def _predict_future_prices(self, current_price: float, score: float, 
                             ohlcv_data: pd.DataFrame) -> Tuple[float, float, float]:
        """미래 가격 예측"""
        try:
            # 변동성 계산
            returns = ohlcv_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # 방향성 강도
            direction_strength = abs(score - 0.5) * 2
            
            # 예상 변화율
            expected_change_1h = (score - 0.5) * volatility * 0.5 * direction_strength
            expected_change_4h = (score - 0.5) * volatility * 1.0 * direction_strength
            expected_change_24h = (score - 0.5) * volatility * 2.0 * direction_strength
            
            # 미래 가격 계산
            pred_1h = current_price * (1 + expected_change_1h)
            pred_4h = current_price * (1 + expected_change_4h)
            pred_24h = current_price * (1 + expected_change_24h)
            
            return pred_1h, pred_4h, pred_24h
            
        except Exception as e:
            logger.error(f"가격 예측 계산 오류: {str(e)}")
            return current_price, current_price, current_price
    
    def _identify_key_factors(self, tech_score: float, price_score: float, 
                            volume_score: float, pattern_score: float, time_score: float) -> List[str]:
        """주요 요인 식별"""
        factors = []
        
        if tech_score > 0.7:
            factors.append("긍정적 기술적 지표")
        elif tech_score < 0.3:
            factors.append("부정적 기술적 지표")
        
        if price_score > 0.7:
            factors.append("강한 상승 가격 액션")
        elif price_score < 0.3:
            factors.append("약한 가격 액션")
        
        if volume_score > 0.7:
            factors.append("거래량 증가 지원")
        elif volume_score < 0.3:
            factors.append("거래량 부족")
        
        if pattern_score > 0.7:
            factors.append("상승 패턴 확인")
        elif pattern_score < 0.3:
            factors.append("하락 패턴 경고")
        
        if not factors:
            factors.append("혼재된 신호")
        
        return factors[:3]  # 최대 3개 요인
    
    def _assess_risk(self, ohlcv_data: pd.DataFrame, score: float) -> str:
        """위험도 평가"""
        try:
            # 변동성 기반 위험도
            returns = ohlcv_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # 신호 명확성
            signal_clarity = abs(score - 0.5) * 2
            
            if volatility > 0.05 and signal_clarity < 0.3:
                return "높음 - 높은 변동성과 불분명한 신호"
            elif volatility > 0.03:
                return "중간 - 보통 변동성"
            else:
                return "낮음 - 낮은 변동성과 명확한 신호"
                
        except Exception as e:
            logger.error(f"위험도 평가 오류: {str(e)}")
            return "중간 - 평가 불가"
    
    def _get_default_prediction(self, ohlcv_data: pd.DataFrame) -> PricePrediction:
        """기본 예측 결과"""
        current_price = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0
        
        return PricePrediction(
            market="UNKNOWN",
            current_price=current_price,
            predicted_price_1h=current_price,
            predicted_price_4h=current_price,
            predicted_price_24h=current_price,
            direction=PredictionDirection.NEUTRAL,
            confidence=PredictionConfidence.LOW,
            probability_up=0.5,
            probability_down=0.5,
            key_factors=["데이터 부족"],
            risk_assessment="평가 불가",
            timestamp=datetime.now()
        )