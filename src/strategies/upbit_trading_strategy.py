"""
Upbit Trading Strategy Module

This module implements trading strategies for the Upbit exchange.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyupbit
import time
from requests.exceptions import RequestException
from src.core.strategy_manager import StrategyManager

# Configure logging
logger = logging.getLogger(__name__)

class UpbitTradingStrategy:
    """Class for implementing Upbit trading strategies."""
    
    def __init__(self) -> None:
        """Initialize the UpbitTradingStrategy."""
        self.logger = logging.getLogger(__name__)
        self.strategy_manager = StrategyManager()
        self.last_request_time = 0
        self.request_interval = 0.1  # 100ms
        self.cache = {}
        self.cache_duration = 300  # 5분
    
    def _get_cached_data(self, market: str) -> Optional[Dict[str, Any]]:
        """캐시된 데이터를 가져옵니다."""
        if market in self.cache:
            cached_data = self.cache[market]
            if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                return cached_data['data']
        return None
    
    def _cache_data(self, market: str, data: Dict[str, Any]):
        """데이터를 캐시에 저장합니다."""
        self.cache[market] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _wait_for_request(self):
        """API 요청 간격을 제어합니다."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_ohlcv_with_retry(self, market: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """재시도 로직이 포함된 OHLCV 데이터 조회 함수입니다."""
        for attempt in range(max_retries):
            try:
                self._wait_for_request()
                # 일봉 데이터 사용 (분봉 대신)
                df = pyupbit.get_ohlcv(market, interval="day", count=30)
                if df is not None and not df.empty:
                    return df
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 지수 백오프
                    self.logger.warning(f"{market} 데이터 조회 재시도 중... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    
            except RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"{market} 데이터 조회 중 오류 발생, 재시도 중... ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"{market} 데이터 조회 실패: {str(e)}")
            except Exception as e:
                self.logger.error(f"{market} 데이터 조회 중 예상치 못한 오류 발생: {str(e)}")
                break
                
        return None
    
    def analyze_market(self, market: str) -> Dict[str, Any]:
        """Analyze market and generate trading signals."""
        try:
            # Get market analysis
            analysis = self.strategy_manager.get_analysis(market)
            
            if not analysis or 'signals' not in analysis:
                self.logger.warning(f"{market} 분석 실패")
                return {}
            
            # Extract signals
            signals = analysis['signals']
            ma_signal = signals.get('ma', {}).get('signal', 'NEUTRAL')
            rsi_signal = signals.get('rsi', {}).get('signal', 'NEUTRAL')
            macd_signal = signals.get('macd', {}).get('signal', 'NEUTRAL')
            
            # Calculate confidence based on signal agreement
            bullish_count = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] 
                              if signal == 'BULLISH')
            bearish_count = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] 
                              if signal == 'BEARISH')
            
            # Determine action and confidence
            if bullish_count >= 2:
                action = 'BUY'
                confidence = bullish_count / 3 * 100
            elif bearish_count >= 2:
                action = 'SELL'
                confidence = bearish_count / 3 * 100
            else:
                action = 'HOLD'
                confidence = 0
            
            self.logger.info(f"{market} 분석 완료: {action} (신뢰도: {confidence:.1f}%)")
            
            return {
                'action': action,
                'confidence': confidence / 100,  # Convert to decimal
                'signals': {
                    'ma': signals['ma'],
                    'rsi': signals['rsi'],
                    'macd': signals['macd']
                },
                'risk': analysis.get('risk', {'risk_score': 0.0, 'volatility': 0.0})
            }
            
        except Exception as e:
            self.logger.error(f"{market} 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _get_default_signals(self) -> Dict[str, Any]:
        """기본 신호를 반환합니다."""
        return {
            'signals': {
                'action': 'HOLD',
                'confidence': 0.0,
                'risk_level': 'UNKNOWN'
            }
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기술적 지표를 계산합니다."""
        try:
            if len(df) < 20:  # 최소 데이터 포인트 확인
                return {}
            
            # 데이터 타입 변환 확인
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # NaN 값 제거
            df = df.dropna()
            
            if len(df) < 20:  # NaN 제거 후 다시 확인
                return {}
                
            # 이동평균선
            ma5 = df['close'].rolling(window=5).mean().iloc[-1]
            ma20 = df['close'].rolling(window=20).mean().iloc[-1]
            ma60 = df['close'].rolling(window=20).mean().iloc[-1]  # 일봉이므로 60일은 너무 김
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # 볼린저 밴드
            middle = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # 거래량 분석
            volume = float(df['volume'].iloc[-1])
            volume_ma20 = float(df['volume'].rolling(window=20).mean().iloc[-1])
            volume_ratio = volume / volume_ma20 if volume_ma20 > 0 else 1.0
            
            indicators = {
                'MA': {
                    'MA5': float(ma5),
                    'MA20': float(ma20),
                    'MA60': float(ma60)
                },
                'RSI': float(rsi),
                'BB': {
                    'upper': float(upper.iloc[-1]),
                    'middle': float(middle.iloc[-1]),
                    'lower': float(lower.iloc[-1])
                },
                'MACD': {
                    'MACD': float(macd.iloc[-1]),
                    'signal': float(signal.iloc[-1]),
                    'histogram': float(histogram.iloc[-1])
                },
                'volume': {
                    'current': volume,
                    'MA20': volume_ma20,
                    'ratio': volume_ratio
                }
            }
            
            # NaN 값 체크
            for category, values in indicators.items():
                if isinstance(values, dict):
                    if any(pd.isna(v) for v in values.values()):
                        self.logger.warning(f"{category} 지표에 NaN 값이 있습니다")
                        return {}
                elif pd.isna(values):
                    self.logger.warning(f"{category} 지표가 NaN 입니다")
                    return {}
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            return {}
    
    def _generate_signals(self,
                         df: pd.DataFrame,
                         indicators: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 분석을 기반으로 매매 신호를 생성합니다."""
        try:
            signals = []
            confidence_scores = []
            
            current_price = df['close'].iloc[-1]
            
            # 이동평균선 신호
            ma_data = indicators.get('MA', {})
            if ma_data:
                ma5 = ma_data['MA5']
                ma20 = ma_data['MA20']
                ma60 = ma_data['MA60']
                
                if ma5 > ma20 and ma20 > ma60:
                    signals.append('BUY')
                    confidence_scores.append(0.6)
                elif ma5 < ma20 and ma20 < ma60:
                    signals.append('SELL')
                    confidence_scores.append(0.6)
            
            # RSI 신호
            rsi = indicators.get('RSI')
            if rsi is not None:
                if rsi < 30:
                    signals.append('BUY')
                    confidence_scores.append(0.7)
                elif rsi > 70:
                    signals.append('SELL')
                    confidence_scores.append(0.7)
            
            # 볼린저 밴드 신호
            bb_data = indicators.get('BB', {})
            if bb_data:
                upper = bb_data['upper']
                lower = bb_data['lower']
                
                if current_price < lower:
                    signals.append('BUY')
                    confidence_scores.append(0.6)
                elif current_price > upper:
                    signals.append('SELL')
                    confidence_scores.append(0.6)
            
            # MACD 신호
            macd_data = indicators.get('MACD', {})
            if macd_data:
                macd = macd_data['MACD']
                signal = macd_data['signal']
                
                if macd > signal:
                    signals.append('BUY')
                    confidence_scores.append(0.5)
                elif macd < signal:
                    signals.append('SELL')
                    confidence_scores.append(0.5)
            
            # 거래량 신호
            volume_data = indicators.get('volume', {})
            if volume_data:
                volume_ratio = volume_data['ratio']
                if volume_ratio > 2.0:  # 거래량 급증
                    if current_price > df['close'].iloc[-2]:
                        signals.append('BUY')
                        confidence_scores.append(0.6)
                    else:
                        signals.append('SELL')
                        confidence_scores.append(0.6)
            
            # 최종 신호 계산
            if not signals:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'risk_level': self._calculate_risk_level(df, indicators)
                }
            
            # 신호 카운트
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            # 평균 신뢰도 계산
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # 행동 결정
            if buy_signals > sell_signals:
                action = 'BUY'
            elif sell_signals > buy_signals:
                action = 'SELL'
            else:
                action = 'HOLD'
                avg_confidence = 0.0
            
            return {
                'action': action,
                'confidence': float(avg_confidence),
                'risk_level': self._calculate_risk_level(df, indicators),
                'analysis': {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'total_signals': len(signals)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'risk_level': 'UNKNOWN'
            }
    
    def _calculate_risk_level(self,
                            df: pd.DataFrame,
                            indicators: Dict[str, Any]) -> str:
        """시장 상황을 기반으로 위험도를 계산합니다."""
        try:
            # 변동성 계산
            returns = df['close'].pct_change()
            volatility = returns.std() * 100
            
            # RSI 가져오기
            rsi = indicators.get('RSI', 50)
            
            # 거래량 비율 가져오기
            volume_ratio = indicators.get('volume', {}).get('ratio', 1.0)
            
            # 위험도 점수 계산
            risk_score = 0
            
            # 변동성 기여도
            if volatility > 5:
                risk_score += 3
            elif volatility > 3:
                risk_score += 2
            elif volatility > 1:
                risk_score += 1
            
            # RSI 기여도
            if rsi < 20 or rsi > 80:
                risk_score += 2
            elif rsi < 30 or rsi > 70:
                risk_score += 1
            
            # 거래량 기여도
            if volume_ratio > 3:
                risk_score += 2
            elif volume_ratio > 2:
                risk_score += 1
            
            # 위험도 레벨 결정
            if risk_score >= 5:
                return 'HIGH'
            elif risk_score >= 3:
                return 'MEDIUM'
            else:
                return 'LOW'
            
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return 'UNKNOWN'

    def execute_strategy(self, market: str) -> Dict[str, Any]:
        """
        Execute trading strategy for a specific market.
        
        Args:
            market (str): Market code (e.g., 'KRW-BTC')
            
        Returns:
            Dict[str, Any]: Strategy execution results
        """
        try:
            # Analyze market
            analysis = self.analyze_market(market)
            if not analysis:
                return {}
            
            # Get account information
            account = self.upbit.get_account()
            
            # Generate order parameters based on analysis
            if analysis['signals']['action'] != 'HOLD':
                order_params = self._generate_order_params(
                    market,
                    analysis['signals'],
                    account
                )
                
                if order_params:
                    # Place order
                    order = self.upbit.place_order(**order_params)
                    
                    return {
                        'market': market,
                        'action': analysis['signals']['action'],
                        'confidence': analysis['signals']['confidence'],
                        'order': order,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'market': market,
                'action': 'HOLD',
                'confidence': analysis['signals']['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing strategy for {market}: {str(e)}")
            return {}

    def _generate_order_params(self,
                             market: str,
                             signals: Dict[str, Any],
                             account: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate order parameters based on signals and account balance."""
        try:
            # Get current price
            ticker = self.upbit.get_ticker(market)[0]
            current_price = ticker['trade_price']
            
            # Find relevant balance
            krw_balance = next((item for item in account if item['currency'] == 'KRW'), None)
            asset_balance = next((item for item in account if item['currency'] == market.split('-')[1]), None)
            
            if signals['action'] == 'BUY':
                if not krw_balance:
                    return None
                
                # Calculate order amount (1% to 5% of balance based on confidence)
                available_krw = float(krw_balance['balance'])
                order_ratio = min(0.01 + (signals['confidence'] * 0.04), 0.05)
                order_amount = available_krw * order_ratio
                
                return {
                    'market': market,
                    'side': 'bid',
                    'price': str(current_price),
                    'volume': str(order_amount / current_price),
                    'ord_type': 'limit'
                }
                
            elif signals['action'] == 'SELL':
                if not asset_balance:
                    return None
                
                # Calculate order amount (5% to 20% of balance based on confidence)
                available_volume = float(asset_balance['balance'])
                order_ratio = min(0.05 + (signals['confidence'] * 0.15), 0.2)
                order_volume = available_volume * order_ratio
                
                return {
                    'market': market,
                    'side': 'ask',
                    'price': str(current_price),
                    'volume': str(order_volume),
                    'ord_type': 'limit'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating order parameters: {str(e)}")
            return None 