"""
Upbit Trading Strategy Module

This module implements trading strategies for the Upbit exchange.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from src.exchange.exchange_api import UpbitAPI

class UpbitTradingStrategy:
    """Class for implementing Upbit trading strategies."""
    
    def __init__(self) -> None:
        """Initialize the UpbitTradingStrategy."""
        self.logger = logging.getLogger(__name__)
        self.upbit = UpbitAPI()
        
    def analyze_market(self, market: str) -> Dict[str, Any]:
        """
        Analyze market and generate trading signals.
        
        Args:
            market (str): Market symbol (e.g., 'KRW-BTC')
            
        Returns:
            Dict[str, Any]: Analysis results and trading signals
        """
        try:
            # Get market data
            candles = self.upbit.get_candles_minutes(market, unit=15, count=200)  # 15분봉 200개
            orderbook = self.upbit.get_orderbook(market)
            market_index = self.upbit.get_market_index(market)
            
            if not candles:
                return {
                    'signals': {
                        'action': 'HOLD',
                        'confidence': 0.0,
                        'risk_level': 'UNKNOWN'
                    }
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signals
            signals = self._generate_signals(df, indicators, orderbook, market_index)
            
            return {
                'signals': signals,
                'indicators': indicators,
                'metadata': {
                    'market': market,
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(df)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market {market}: {str(e)}")
            return {
                'signals': {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'risk_level': 'UNKNOWN'
                }
            }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            # Moving Averages
            df['MA5'] = df['trade_price'].rolling(window=5).mean()
            df['MA20'] = df['trade_price'].rolling(window=20).mean()
            df['MA60'] = df['trade_price'].rolling(window=60).mean()
            
            # RSI
            delta = df['trade_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['trade_price'].rolling(window=20).mean()
            df['BB_std'] = df['trade_price'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # MACD
            exp1 = df['trade_price'].ewm(span=12, adjust=False).mean()
            exp2 = df['trade_price'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['Signal']
            
            # Volume analysis
            df['Volume_MA20'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['candle_acc_trade_volume'] / df['Volume_MA20']
            
            # Get current values
            current = df.iloc[-1]
            
            return {
                'MA': {
                    'MA5': float(current['MA5']),
                    'MA20': float(current['MA20']),
                    'MA60': float(current['MA60'])
                },
                'RSI': float(current['RSI']),
                'BB': {
                    'upper': float(current['BB_upper']),
                    'middle': float(current['BB_middle']),
                    'lower': float(current['BB_lower'])
                },
                'MACD': {
                    'MACD': float(current['MACD']),
                    'signal': float(current['Signal']),
                    'histogram': float(current['MACD_hist'])
                },
                'volume': {
                    'current': float(current['candle_acc_trade_volume']),
                    'MA20': float(current['Volume_MA20']),
                    'ratio': float(current['Volume_ratio'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _generate_signals(self,
                         df: pd.DataFrame,
                         indicators: Dict[str, Any],
                         orderbook: List[Dict[str, Any]],
                         market_index: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on technical analysis."""
        try:
            signals = []
            confidence_scores = []
            
            current_price = df['trade_price'].iloc[-1]
            
            # Moving Average signals
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
            
            # RSI signals
            rsi = indicators.get('RSI')
            if rsi is not None:
                if rsi < 30:
                    signals.append('BUY')
                    confidence_scores.append(0.7)
                elif rsi > 70:
                    signals.append('SELL')
                    confidence_scores.append(0.7)
            
            # Bollinger Bands signals
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
            
            # MACD signals
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
            
            # Volume signals
            volume_data = indicators.get('volume', {})
            if volume_data:
                volume_ratio = volume_data['ratio']
                if volume_ratio > 2.0:  # Volume spike
                    if current_price > df['trade_price'].iloc[-2]:
                        signals.append('BUY')
                        confidence_scores.append(0.6)
                    else:
                        signals.append('SELL')
                        confidence_scores.append(0.6)
            
            # Market pressure signals
            if market_index:
                pressure = market_index.get('buy_sell_pressure', 0)
                if pressure > 20:
                    signals.append('BUY')
                    confidence_scores.append(0.5)
                elif pressure < -20:
                    signals.append('SELL')
                    confidence_scores.append(0.5)
            
            # Calculate final signal
            if not signals:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'risk_level': self._calculate_risk_level(df, indicators)
                }
            
            # Count signals
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Determine action
            if buy_signals > sell_signals:
                action = 'BUY'
            elif sell_signals > buy_signals:
                action = 'SELL'
            else:
                action = 'HOLD'
            
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
        """Calculate risk level based on market conditions."""
        try:
            # Calculate volatility
            returns = df['trade_price'].pct_change()
            volatility = returns.std() * 100
            
            # Get RSI
            rsi = indicators.get('RSI', 50)
            
            # Get volume ratio
            volume_ratio = indicators.get('volume', {}).get('ratio', 1.0)
            
            # Calculate risk score
            risk_score = 0
            
            # Volatility contribution
            if volatility > 5:
                risk_score += 3
            elif volatility > 3:
                risk_score += 2
            elif volatility > 1:
                risk_score += 1
            
            # RSI contribution
            if rsi < 20 or rsi > 80:
                risk_score += 2
            elif rsi < 30 or rsi > 70:
                risk_score += 1
            
            # Volume contribution
            if volume_ratio > 3:
                risk_score += 2
            elif volume_ratio > 2:
                risk_score += 1
            
            # Determine risk level
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