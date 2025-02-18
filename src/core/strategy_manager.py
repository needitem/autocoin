"""
Strategy Manager for Trading Analysis

This module handles trading strategy analysis and performance metrics.
"""

import pyupbit
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data.db_manager import DBManager

class StrategyManager:
    def __init__(self):
        """Initialize strategy manager."""
        self.db = DBManager()
    
    def get_analysis(self, market: str) -> Dict[str, Any]:
        """Get strategy analysis for a market."""
        try:
            # Get historical data (최근 200일 데이터)
            df = pyupbit.get_ohlcv(market, interval="day", count=200)
            
            if df is None or len(df) < 2:
                return self._get_default_analysis()
            
            # 종가 기준으로 정규화
            df['close_norm'] = df['close'] / df['close'].iloc[0]
            
            # Calculate indicators
            ma_signals = self._analyze_moving_averages(df)
            rsi_signals = self._analyze_rsi(df)
            macd_signals = self._analyze_macd(df)
            risk_metrics = self._analyze_risk(df)
            
            # Get performance metrics from database
            perf_metrics = self.db.get_latest_performance()
            if not perf_metrics:
                perf_metrics = {
                    'win_rate': 0.0,
                    'return_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'trading_days': 0
                }
            
            return {
                'signals': {
                    'ma': ma_signals,
                    'rsi': rsi_signals,
                    'macd': macd_signals
                },
                'risk': risk_metrics,
                'performance': perf_metrics
            }
        
        except Exception as e:
            print(f"Error in strategy analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving averages."""
        try:
            # Calculate moving averages using normalized prices
            df['MA5'] = df['close_norm'].rolling(window=5).mean()
            df['MA20'] = df['close_norm'].rolling(window=20).mean()
            df['MA60'] = df['close_norm'].rolling(window=60).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate price position relative to MAs
            price_above_ma5 = latest['close_norm'] > latest['MA5']
            price_above_ma20 = latest['close_norm'] > latest['MA20']
            ma5_above_ma20 = latest['MA5'] > latest['MA20']
            
            # Determine trend and strength
            if price_above_ma5 and price_above_ma20 and ma5_above_ma20:
                signal = "BULLISH"
                strength = "강세"
            elif not price_above_ma5 and not price_above_ma20 and not ma5_above_ma20:
                signal = "BEARISH"
                strength = "약세"
            else:
                signal = "NEUTRAL"
                strength = "중립"
            
            # Calculate MA trend (percentage change)
            ma_trend = ((latest['MA5'] - prev['MA5']) / prev['MA5']) * 100
            
            return {
                'signal': signal,
                'strength': strength,
                'value': round(ma_trend, 2),
                'ma5': latest['MA5'] * df['close'].iloc[0],
                'ma20': latest['MA20'] * df['close'].iloc[0],
                'ma60': latest['MA60'] * df['close'].iloc[0]
            }
        
        except Exception as e:
            print(f"Error in MA analysis: {str(e)}")
            return {'signal': "NEUTRAL", 'strength': "분석 실패", 'value': 0.0}
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RSI."""
        try:
            # Calculate price changes
            delta = df['close_norm'].diff()
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Get latest RSI
            latest_rsi = rsi.iloc[-1]
            
            # Determine signal and strength
            if latest_rsi >= 70:
                signal = "BEARISH"
                strength = "과매수"
            elif latest_rsi <= 30:
                signal = "BULLISH"
                strength = "과매도"
            else:
                signal = "NEUTRAL"
                strength = "중립"
            
            return {
                'signal': signal,
                'strength': strength,
                'value': round(latest_rsi, 2)
            }
        
        except Exception as e:
            print(f"Error in RSI analysis: {str(e)}")
            return {'signal': "NEUTRAL", 'strength': "분석 실패", 'value': 50.0}
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD."""
        try:
            # Calculate MACD using normalized prices
            exp1 = df['close_norm'].ewm(span=12, adjust=False).mean()
            exp2 = df['close_norm'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            # Get latest values
            latest_macd = macd.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            prev_macd = macd.iloc[-2]
            
            # Calculate relative MACD value (as percentage of price)
            latest_price_norm = df['close_norm'].iloc[-1]
            macd_pct = (latest_macd / latest_price_norm) * 100
            signal_pct = (latest_signal / latest_price_norm) * 100
            
            # Calculate MACD momentum
            macd_momentum = ((latest_macd - prev_macd) / abs(prev_macd)) * 100 if abs(prev_macd) > 0.0001 else 0
            
            # Determine signal and strength based on MACD and signal line crossover
            if latest_macd > latest_signal:
                if macd_momentum > 0:
                    signal = "BULLISH"
                    strength = "상승추세"
                else:
                    signal = "NEUTRAL"
                    strength = "약세반등"
            elif latest_macd < latest_signal:
                if macd_momentum < 0:
                    signal = "BEARISH"
                    strength = "하락추세"
                else:
                    signal = "NEUTRAL"
                    strength = "약세조정"
            else:
                signal = "NEUTRAL"
                strength = "중립"
            
            return {
                'signal': signal,
                'strength': strength,
                'value': round(macd_pct, 2),  # MACD as percentage of price
                'signal_line': round(signal_pct, 2),  # Signal line as percentage of price
                'momentum': round(macd_momentum, 2)
            }
        
        except Exception as e:
            print(f"Error in MACD analysis: {str(e)}")
            return {'signal': "NEUTRAL", 'strength': "분석 실패", 'value': 0.0, 'signal_line': 0.0, 'momentum': 0.0}
    
    def _analyze_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk metrics."""
        try:
            # Calculate daily returns using normalized prices
            returns = df['close_norm'].pct_change().dropna()
            
            # Calculate volatility (20-day rolling)
            volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
            current_volatility = volatility.iloc[-1]
            
            # Calculate risk score (0-10)
            # 50% 연간 변동성을 기준으로 리스크 점수 계산
            risk_score = min(current_volatility / 5, 10)
            
            return {
                'volatility': round(current_volatility, 2),
                'risk_score': round(risk_score, 1)
            }
        
        except Exception as e:
            print(f"Error in risk analysis: {str(e)}")
            return {'volatility': 0.0, 'risk_score': 0.0}
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when data is unavailable."""
        return {
            'signals': {
                'ma': {'signal': "NEUTRAL", 'strength': "데이터 없음", 'value': 0.0},
                'rsi': {'signal': "NEUTRAL", 'strength': "데이터 없음", 'value': 50.0},
                'macd': {'signal': "NEUTRAL", 'strength': "데이터 없음", 'value': 0.0}
            },
            'risk': {
                'volatility': 0.0,
                'risk_score': 0.0
            },
            'performance': {
                'win_rate': 0.0,
                'return_rate': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'trading_days': 0
            }
        }