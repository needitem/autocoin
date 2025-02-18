"""
Pattern Analysis Module

This module analyzes cryptocurrency price charts for various technical patterns
such as support/resistance levels, chart patterns, and candlestick patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime

class PatternAnalyzer:
    """Class for analyzing chart patterns in cryptocurrency price data."""
    
    def __init__(self) -> None:
        """Initialize the PatternAnalyzer with default settings."""
        self.logger = logging.getLogger(__name__)

    def analyze_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for various patterns.
        
        Args:
            market_data (Dict[str, Any]): Market data containing OHLCV information
            
        Returns:
            Dict[str, Any]: Detected patterns and analysis results
        """
        try:
            # Convert OHLCV data to DataFrame
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Analyze different pattern types
            support_resistance = self._find_support_resistance(df)
            chart_patterns = self._identify_chart_patterns(df)
            candlestick_patterns = self._analyze_candlestick_patterns(df)
            
            return {
                'support_resistance': support_resistance,
                'chart_patterns': chart_patterns,
                'candlestick_patterns': candlestick_patterns,
                'metadata': {
                    'symbol': market_data['symbol'],
                    'timeframe': market_data['timeframe'],
                    'analysis_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
            raise Exception(f"Failed to analyze patterns: {str(e)}")

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Find support and resistance levels using price action.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, List[float]]: Support and resistance levels
        """
        try:
            # Calculate pivot points
            highs = df['high'].values
            lows = df['low'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            window = 5  # Number of candles to look before and after
            
            for i in range(window, len(df) - window):
                # Check for resistance
                if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] > highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
                
                # Check for support
                if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] < lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)))
            support_levels = sorted(list(set(support_levels)))
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}

    def _identify_chart_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify common chart patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Dict[str, Any]]: List of identified patterns
        """
        patterns = []
        try:
            # Head and Shoulders pattern
            h_and_s = self._find_head_and_shoulders(df)
            if h_and_s:
                patterns.extend(h_and_s)
            
            # Double Top/Bottom pattern
            double_patterns = self._find_double_patterns(df)
            if double_patterns:
                patterns.extend(double_patterns)
            
            # Triangle patterns
            triangle_patterns = self._find_triangle_patterns(df)
            if triangle_patterns:
                patterns.extend(triangle_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying chart patterns: {str(e)}")
            return []

    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze candlestick patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Dict[str, Any]]: List of identified candlestick patterns
        """
        patterns = []
        try:
            # Analyze last 3 candles for patterns
            last_candles = df.tail(3)
            
            # Doji pattern
            if self._is_doji(last_candles.iloc[-1]):
                patterns.append({
                    'pattern': 'DOJI',
                    'timestamp': last_candles.iloc[-1]['timestamp'],
                    'significance': 'MEDIUM'
                })
            
            # Hammer pattern
            if self._is_hammer(last_candles.iloc[-1]):
                patterns.append({
                    'pattern': 'HAMMER',
                    'timestamp': last_candles.iloc[-1]['timestamp'],
                    'significance': 'HIGH'
                })
            
            # Engulfing pattern
            if len(last_candles) >= 2:
                if self._is_engulfing(last_candles.iloc[-2:]):
                    patterns.append({
                        'pattern': 'ENGULFING',
                        'timestamp': last_candles.iloc[-1]['timestamp'],
                        'significance': 'HIGH'
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {str(e)}")
            return []

    def _is_doji(self, candle: pd.Series) -> bool:
        """
        Check if a candle is a doji pattern.
        
        Args:
            candle (pd.Series): Single candle data
            
        Returns:
            bool: True if the candle is a doji
        """
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        return body <= total_range * 0.1

    def _is_hammer(self, candle: pd.Series) -> bool:
        """
        Check if a candle is a hammer pattern.
        
        Args:
            candle (pd.Series): Single candle data
            
        Returns:
            bool: True if the candle is a hammer
        """
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return (lower_wick > body * 2) and (upper_wick < body * 0.5)

    def _is_engulfing(self, candles: pd.DataFrame) -> bool:
        """
        Check if two candles form an engulfing pattern.
        
        Args:
            candles (pd.DataFrame): Two consecutive candles
            
        Returns:
            bool: True if the candles form an engulfing pattern
        """
        prev_candle = candles.iloc[0]
        curr_candle = candles.iloc[1]
        
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        
        is_bullish_engulfing = (
            curr_candle['close'] > curr_candle['open'] and
            prev_candle['close'] < prev_candle['open'] and
            curr_body > prev_body * 1.5
        )
        
        is_bearish_engulfing = (
            curr_candle['close'] < curr_candle['open'] and
            prev_candle['close'] > prev_candle['open'] and
            curr_body > prev_body * 1.5
        )
        
        return is_bullish_engulfing or is_bearish_engulfing

    def _find_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find Head and Shoulders patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Dict[str, Any]]: Identified Head and Shoulders patterns
        """
        # Implementation for Head and Shoulders pattern detection
        # This is a placeholder for the actual implementation
        return []

    def _find_double_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find Double Top and Double Bottom patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Dict[str, Any]]: Identified Double patterns
        """
        # Implementation for Double Top/Bottom pattern detection
        # This is a placeholder for the actual implementation
        return []

    def _find_triangle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find Triangle patterns (Ascending, Descending, Symmetric).
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Dict[str, Any]]: Identified Triangle patterns
        """
        # Implementation for Triangle pattern detection
        # This is a placeholder for the actual implementation
        return [] 