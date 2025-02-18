"""
Market Analysis Module

This module handles market analysis, including trend analysis, volatility,
and market conditions.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

class MarketAnalyzer:
    """Class for analyzing market conditions and trends."""
    
    def __init__(self) -> None:
        """Initialize the MarketAnalyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions and trends.
        
        Args:
            market_data (Dict[str, Any]): Market data including OHLCV
            
        Returns:
            Dict[str, Any]: Market analysis results
        """
        try:
            if not market_data.get('ohlcv'):
                return {
                    'market_conditions': {},
                    'volatility': {},
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
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(df)
            
            # Analyze volatility
            volatility = self._analyze_volatility(df)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(df)
            
            return {
                'market_conditions': market_conditions,
                'volatility': volatility,
                'volume_analysis': volume_analysis,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'timeframe': market_data.get('timeframe', 'UNKNOWN')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {
                'market_conditions': {},
                'volatility': {},
                'volume_analysis': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions including trend and momentum."""
        try:
            # Calculate short-term and long-term trends
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            current_price = float(df['close'].iloc[-1])
            sma20 = float(df['SMA20'].iloc[-1])
            sma50 = float(df['SMA50'].iloc[-1])
            
            # Determine trend
            if current_price > sma20 and sma20 > sma50:
                trend = 'STRONG_UPTREND'
            elif current_price > sma20:
                trend = 'UPTREND'
            elif current_price < sma20 and sma20 < sma50:
                trend = 'STRONG_DOWNTREND'
            elif current_price < sma20:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            # Calculate momentum
            momentum = (current_price - float(df['close'].iloc[-20])) / float(df['close'].iloc[-20]) * 100
            
            # Calculate price change
            price_change_24h = (current_price - float(df['close'].iloc[-24])) / float(df['close'].iloc[-24]) * 100
            
            return {
                'trend': trend,
                'momentum': float(momentum),
                'price_change_24h': float(price_change_24h),
                'above_sma20': bool(current_price > sma20),
                'above_sma50': bool(current_price > sma50),
                'sma20_slope': float((sma20 - float(df['SMA20'].iloc[-2])) / float(df['SMA20'].iloc[-2]) * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility."""
        try:
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility measures
            daily_volatility = float(df['returns'].std() * 100)
            
            # Calculate Average True Range (ATR)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = float(df['tr'].rolling(window=14).mean().iloc[-1])
            
            # Calculate Bollinger Band width
            middle_band = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            bb_width = ((middle_band + 2 * std) - (middle_band - 2 * std)) / middle_band * 100
            
            # Calculate rolling volatility
            rolling_std = df['returns'].rolling(window=30).std()
            rolling_volatility = float(rolling_std.mean() * 100)
            
            return {
                'daily_volatility': daily_volatility,
                'atr': atr,
                'atr_percent': float(atr / df['close'].iloc[-1] * 100),
                'bb_width': float(bb_width.iloc[-1]),
                'is_high_volatility': bool(daily_volatility > rolling_volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            return {}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading volume patterns."""
        try:
            # Calculate volume metrics
            avg_volume = float(df['volume'].mean())
            current_volume = float(df['volume'].iloc[-1])
            volume_sma = float(df['volume'].rolling(window=20).mean().iloc[-1])
            
            # Calculate volume trend
            volume_change = (current_volume - volume_sma) / volume_sma * 100
            
            # Determine if volume is rising
            recent_volumes = df['volume'].iloc[-3:].astype(float).tolist()
            is_volume_rising = all(recent_volumes[i] <= recent_volumes[i+1] for i in range(len(recent_volumes)-1))
            
            # Calculate price-volume relationship
            up_mask = df['close'] > df['close'].shift()
            down_mask = df['close'] < df['close'].shift()
            
            price_up_volume = float(df.loc[up_mask, 'volume'].mean())
            price_down_volume = float(df.loc[down_mask, 'volume'].mean())
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_sma20': volume_sma,
                'volume_change_percent': float(volume_change),
                'is_volume_rising': is_volume_rising,
                'price_up_volume_ratio': float(price_up_volume / price_down_volume) if price_down_volume > 0 else 1.0,
                'is_volume_significant': bool(current_volume > volume_sma * 1.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            return {} 