"""
Technical Analysis Module

This module handles technical analysis of market data, including various indicators
and pattern recognition.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

class TechnicalAnalyzer:
    """Class for performing technical analysis on market data."""
    
    def __init__(self) -> None:
        """Initialize the TechnicalAnalyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and analyze technical indicators for market data.
        
        Args:
            market_data (Dict[str, Any]): Market data including OHLCV
            
        Returns:
            Dict[str, Any]: Technical analysis results
        """
        try:
            if not market_data.get('ohlcv'):
                return {
                    'indicators': {},
                    'signals': {},
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'error': 'No OHLCV data available'
                    }
                }
            
            # Convert OHLCV data to DataFrame
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Convert columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate indicators
            indicators = {}
            
            # Moving Averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            
            current_price = float(df['close'].iloc[-1])
            ma20 = float(df['MA20'].iloc[-1])
            ma50 = float(df['MA50'].iloc[-1])
            
            indicators['MA20'] = {
                'value': ma20,
                'position': 'ABOVE' if current_price > ma20 else 'BELOW'
            }
            
            indicators['MA50'] = {
                'value': ma50,
                'position': 'ABOVE' if current_price > ma50 else 'BELOW'
            }
            
            # RSI (14-period)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = float(rsi.iloc[-1])
            indicators['RSI'] = {
                'value': current_rsi,
                'condition': 'OVERBOUGHT' if current_rsi > 70 else ('OVERSOLD' if current_rsi < 30 else 'NEUTRAL')
            }
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            current_macd = float(macd.iloc[-1])
            current_signal = float(signal.iloc[-1])
            
            indicators['MACD'] = {
                'value': current_macd,
                'signal': current_signal,
                'histogram': float(current_macd - current_signal),
                'trend': 'BULLISH' if current_macd > current_signal else 'BEARISH'
            }
            
            # Bollinger Bands
            ma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])
            current_middle = float(ma20.iloc[-1])
            
            indicators['BB'] = {
                'upper': current_upper,
                'middle': current_middle,
                'lower': current_lower,
                'width': float((current_upper - current_lower) / current_middle),
                'position': 'ABOVE' if current_price > current_upper else ('BELOW' if current_price < current_lower else 'INSIDE')
            }
            
            # Generate signals based on indicators
            signals = self._generate_indicator_signals(indicators)
            
            return {
                'indicators': indicators,
                'signals': signals,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'timeframe': market_data.get('timeframe', 'UNKNOWN')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical indicators: {str(e)}")
            return {
                'indicators': {},
                'signals': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _generate_indicator_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators."""
        signals = {
            'MA_TREND': 'NEUTRAL',
            'RSI_SIGNAL': 'NEUTRAL',
            'MACD_SIGNAL': 'NEUTRAL',
            'BB_SIGNAL': 'NEUTRAL',
            'OVERALL': 'NEUTRAL'
        }
        
        # Moving Average signals
        if indicators.get('MA20', {}).get('position') == 'ABOVE' and \
           indicators.get('MA50', {}).get('position') == 'ABOVE':
            signals['MA_TREND'] = 'BULLISH'
        elif indicators.get('MA20', {}).get('position') == 'BELOW' and \
             indicators.get('MA50', {}).get('position') == 'BELOW':
            signals['MA_TREND'] = 'BEARISH'
        
        # RSI signals
        rsi_condition = indicators.get('RSI', {}).get('condition')
        if rsi_condition == 'OVERSOLD':
            signals['RSI_SIGNAL'] = 'BUY'
        elif rsi_condition == 'OVERBOUGHT':
            signals['RSI_SIGNAL'] = 'SELL'
        
        # MACD signals
        macd_trend = indicators.get('MACD', {}).get('trend')
        if macd_trend == 'BULLISH':
            signals['MACD_SIGNAL'] = 'BUY'
        elif macd_trend == 'BEARISH':
            signals['MACD_SIGNAL'] = 'SELL'
        
        # Bollinger Bands signals
        bb_position = indicators.get('BB', {}).get('position')
        if bb_position == 'BELOW':
            signals['BB_SIGNAL'] = 'BUY'
        elif bb_position == 'ABOVE':
            signals['BB_SIGNAL'] = 'SELL'
        
        # Calculate overall signal
        buy_signals = sum(1 for signal in signals.values() if signal in ['BUY', 'BULLISH'])
        sell_signals = sum(1 for signal in signals.values() if signal in ['SELL', 'BEARISH'])
        
        if buy_signals > sell_signals:
            signals['OVERALL'] = 'BUY'
        elif sell_signals > buy_signals:
            signals['OVERALL'] = 'SELL'
        
        return signals 