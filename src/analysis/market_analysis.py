"""
Market Analysis Module

This module provides market analysis functionality for cryptocurrency trading,
including price analysis, volume analysis, and trend detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

class MarketAnalyzer:
    """Class for analyzing cryptocurrency market data."""
    
    def __init__(self) -> None:
        """Initialize the MarketAnalyzer with default settings."""
        self.logger = logging.getLogger(__name__)

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data to generate insights.
        
        Args:
            market_data (Dict[str, Any]): Market data containing OHLCV information
            
        Returns:
            Dict[str, Any]: Market analysis results
        """
        try:
            # Convert OHLCV data to DataFrame
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Perform various analyses
            price_analysis = self._analyze_price(df)
            volume_analysis = self._analyze_volume(df)
            trend_analysis = self._analyze_trend(df)
            support_resistance = self._find_support_resistance(df)
            volatility = self._calculate_volatility(df)
            
            return {
                'price_analysis': price_analysis,
                'volume_analysis': volume_analysis,
                'trend_analysis': trend_analysis,
                'support_resistance': support_resistance,
                'volatility': volatility,
                'metadata': {
                    'symbol': market_data['symbol'],
                    'timeframe': market_data['timeframe'],
                    'analysis_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            raise Exception(f"Failed to analyze market: {str(e)}")

    def _analyze_price(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price action and patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Price analysis results
        """
        try:
            # Calculate basic price metrics
            current_price = df['close'].iloc[-1]
            price_change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            high_price = df['high'].max()
            low_price = df['low'].min()
            
            # Calculate price statistics
            price_mean = df['close'].mean()
            price_std = df['close'].std()
            price_skew = df['close'].skew()
            
            # Detect price patterns
            higher_highs = self._check_higher_highs(df)
            lower_lows = self._check_lower_lows(df)
            
            return {
                'current_price': float(current_price),
                'price_change_percent': float(price_change),
                'high_price': float(high_price),
                'low_price': float(low_price),
                'statistics': {
                    'mean': float(price_mean),
                    'std': float(price_std),
                    'skew': float(price_skew)
                },
                'patterns': {
                    'higher_highs': higher_highs,
                    'lower_lows': lower_lows
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price: {str(e)}")
            return {}

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading volume patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Volume analysis results
        """
        try:
            # Calculate volume metrics
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_change = (current_volume - df['volume'].iloc[-2]) / df['volume'].iloc[-2] * 100
            
            # Analyze volume trend
            volume_trend = "INCREASING" if current_volume > avg_volume * 1.1 else \
                         "DECREASING" if current_volume < avg_volume * 0.9 else \
                         "STABLE"
            
            # Calculate volume concentration
            volume_std = df['volume'].std()
            volume_concentration = float(volume_std / avg_volume)
            
            return {
                'current_volume': float(current_volume),
                'average_volume': float(avg_volume),
                'volume_change_percent': float(volume_change),
                'volume_trend': volume_trend,
                'volume_concentration': volume_concentration,
                'is_high_volume': bool(current_volume > avg_volume * 1.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            return {}

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market trend using various indicators.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Trend analysis results
        """
        try:
            # Calculate moving averages
            ma_20 = df['close'].rolling(window=20).mean()
            ma_50 = df['close'].rolling(window=50).mean()
            ma_200 = df['close'].rolling(window=200).mean()
            
            # Determine trend based on moving averages
            current_price = df['close'].iloc[-1]
            trend = "STRONGLY_BULLISH" if current_price > ma_20.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1] else \
                   "BULLISH" if current_price > ma_20.iloc[-1] > ma_50.iloc[-1] else \
                   "STRONGLY_BEARISH" if current_price < ma_20.iloc[-1] < ma_50.iloc[-1] < ma_200.iloc[-1] else \
                   "BEARISH" if current_price < ma_20.iloc[-1] < ma_50.iloc[-1] else \
                   "NEUTRAL"
            
            # Calculate trend strength
            trend_strength = abs((current_price - ma_50.iloc[-1]) / ma_50.iloc[-1] * 100)
            
            return {
                'trend': trend,
                'trend_strength': float(trend_strength),
                'moving_averages': {
                    'ma_20': float(ma_20.iloc[-1]),
                    'ma_50': float(ma_50.iloc[-1]),
                    'ma_200': float(ma_200.iloc[-1])
                },
                'is_golden_cross': bool(ma_50.iloc[-1] > ma_200.iloc[-1] and 
                                      ma_50.iloc[-2] <= ma_200.iloc[-2]),
                'is_death_cross': bool(ma_50.iloc[-1] < ma_200.iloc[-1] and 
                                     ma_50.iloc[-2] >= ma_200.iloc[-2])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return {}

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Find support and resistance levels.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, List[float]]: Support and resistance levels
        """
        try:
            # Find local maxima and minima
            window = 20
            highs = df['high'].values
            lows = df['low'].values
            
            resistance_levels = []
            support_levels = []
            
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
                'support': [float(level) for level in support_levels],
                'resistance': [float(level) for level in resistance_levels]
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}

    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various volatility metrics.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, float]: Volatility metrics
        """
        try:
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate volatility metrics
            daily_volatility = returns.std() * np.sqrt(365)
            rolling_volatility = returns.rolling(window=30).std() * np.sqrt(365)
            
            # Calculate true range
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            average_true_range = df['true_range'].rolling(window=14).mean().iloc[-1]
            
            return {
                'daily_volatility': float(daily_volatility),
                'current_volatility': float(rolling_volatility.iloc[-1]),
                'average_true_range': float(average_true_range),
                'is_high_volatility': bool(rolling_volatility.iloc[-1] > rolling_volatility.mean())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return {}

    def _check_higher_highs(self, df: pd.DataFrame) -> bool:
        """
        Check if the market is making higher highs.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if market is making higher highs
        """
        try:
            # Get last 3 significant highs
            highs = df['high'].rolling(window=5).max()
            significant_highs = []
            last_high = float('-inf')
            
            for high in reversed(highs.values[-20:]):
                if high > last_high:
                    significant_highs.append(high)
                    last_high = high
                if len(significant_highs) >= 3:
                    break
            
            return len(significant_highs) >= 3 and \
                   all(significant_highs[i] > significant_highs[i+1] 
                       for i in range(len(significant_highs)-1))
                   
        except Exception as e:
            self.logger.error(f"Error checking higher highs: {str(e)}")
            return False

    def _check_lower_lows(self, df: pd.DataFrame) -> bool:
        """
        Check if the market is making lower lows.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if market is making lower lows
        """
        try:
            # Get last 3 significant lows
            lows = df['low'].rolling(window=5).min()
            significant_lows = []
            last_low = float('inf')
            
            for low in reversed(lows.values[-20:]):
                if low < last_low:
                    significant_lows.append(low)
                    last_low = low
                if len(significant_lows) >= 3:
                    break
            
            return len(significant_lows) >= 3 and \
                   all(significant_lows[i] < significant_lows[i+1] 
                       for i in range(len(significant_lows)-1))
                   
        except Exception as e:
            self.logger.error(f"Error checking lower lows: {str(e)}")
            return False 