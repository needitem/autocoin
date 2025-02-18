"""
Technical Analysis Module

This module provides technical analysis tools for cryptocurrency trading,
including various technical indicators and trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

class TechnicalAnalyzer:
    """Class for performing technical analysis on cryptocurrency price data."""
    
    def __init__(self) -> None:
        """Initialize the TechnicalAnalyzer with default settings."""
        self.logger = logging.getLogger(__name__)

    def analyze_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate various technical indicators.
        
        Args:
            market_data (Dict[str, Any]): Market data containing OHLCV information
            
        Returns:
            Dict[str, Any]: Technical indicators and analysis results
        """
        try:
            # Convert OHLCV data to DataFrame
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Calculate various indicators
            moving_averages = self._calculate_moving_averages(df)
            oscillators = self._calculate_oscillators(df)
            momentum = self._calculate_momentum_indicators(df)
            volatility = self._calculate_volatility_indicators(df)
            
            return {
                'moving_averages': moving_averages,
                'oscillators': oscillators,
                'momentum': momentum,
                'volatility': volatility,
                'metadata': {
                    'symbol': market_data['symbol'],
                    'timeframe': market_data['timeframe'],
                    'calculation_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise Exception(f"Failed to calculate technical indicators: {str(e)}")

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various moving averages."""
        try:
            # Simple Moving Averages
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()
            sma_200 = df['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            return {
                'sma': {
                    'sma_20': float(sma_20.iloc[-1]),
                    'sma_50': float(sma_50.iloc[-1]),
                    'sma_200': float(sma_200.iloc[-1])
                },
                'ema': {
                    'ema_12': float(ema_12.iloc[-1]),
                    'ema_26': float(ema_26.iloc[-1])
                },
                'macd': {
                    'macd_line': float(macd_line.iloc[-1]),
                    'signal_line': float(signal_line.iloc[-1]),
                    'histogram': float(macd_histogram.iloc[-1])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {str(e)}")
            return {}

    def _calculate_oscillators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate oscillator indicators."""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            k = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            d = k.rolling(window=3).mean()
            
            return {
                'rsi': {
                    'value': float(rsi.iloc[-1]),
                    'is_overbought': bool(rsi.iloc[-1] > 70),
                    'is_oversold': bool(rsi.iloc[-1] < 30)
                },
                'stochastic': {
                    'k': float(k.iloc[-1]),
                    'd': float(d.iloc[-1]),
                    'is_overbought': bool(k.iloc[-1] > 80),
                    'is_oversold': bool(k.iloc[-1] < 20)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillators: {str(e)}")
            return {}

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        try:
            # Rate of Change
            roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Momentum
            momentum = df['close'] - df['close'].shift(10)
            
            return {
                'roc': {
                    'value': float(roc.iloc[-1]),
                    'is_positive': bool(roc.iloc[-1] > 0)
                },
                'momentum': {
                    'value': float(momentum.iloc[-1]),
                    'is_positive': bool(momentum.iloc[-1] > 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            return {}

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility indicators."""
        try:
            # Bollinger Bands
            middle_band = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            upper_band = middle_band + (std_dev * 2)
            lower_band = middle_band - (std_dev * 2)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            current_price = df['close'].iloc[-1]
            
            return {
                'bollinger_bands': {
                    'upper': float(upper_band.iloc[-1]),
                    'middle': float(middle_band.iloc[-1]),
                    'lower': float(lower_band.iloc[-1]),
                    'width': float((upper_band.iloc[-1] - lower_band.iloc[-1]) / middle_band.iloc[-1]),
                    'is_above_upper': bool(current_price > upper_band.iloc[-1]),
                    'is_below_lower': bool(current_price < lower_band.iloc[-1])
                },
                'atr': {
                    'value': float(atr.iloc[-1]),
                    'percent_of_price': float(atr.iloc[-1] / current_price * 100)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {str(e)}")
            return {} 