"""
Trader AI Strategy Module

This module implements an AI trading strategy that mimics human trader decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime

class TraderAIStrategy:
    """Class implementing AI trading strategy."""
    
    def __init__(self) -> None:
        """Initialize the TraderAIStrategy."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market and generate trading decisions.
        
        Args:
            market_data (Dict[str, Any]): Market data including OHLCV
            
        Returns:
            Dict[str, Any]: Trading decision and analysis
        """
        try:
            # Extract market features
            features = self._extract_market_features(market_data)
            
            # Generate signals based on features
            signals = self._generate_signals(features)
            
            # Add market analysis
            signals.update(self._analyze_market_conditions(market_data))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return self._get_empty_signals()
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features from market data."""
        try:
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Rename columns to match Upbit API response
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            
            features = {}
            
            # Price momentum
            features['price_momentum'] = df['close'].pct_change().iloc[-1]
            
            # Volume momentum
            features['volume_momentum'] = df['volume'].pct_change().iloc[-1]
            
            # Price volatility
            features['volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Moving averages
            ma20 = df['close'].rolling(20).mean()
            ma50 = df['close'].rolling(50).mean()
            features['ma20_dist'] = (df['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
            features['ma50_dist'] = (df['close'].iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]
            
            # Bollinger Bands
            std20 = df['close'].rolling(20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            features['bb_position'] = (df['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_hist'] = macd.iloc[-1] - signal.iloc[-1]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _generate_signals(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signals based on features."""
        try:
            # Initialize signal strength
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            
            # RSI signals
            if features.get('rsi', 50) < 30:
                buy_signals += 1
            elif features.get('rsi', 50) > 70:
                sell_signals += 1
            total_signals += 1
            
            # Moving average signals
            if features.get('ma20_dist', 0) > 0 and features.get('ma50_dist', 0) > 0:
                buy_signals += 1
            elif features.get('ma20_dist', 0) < 0 and features.get('ma50_dist', 0) < 0:
                sell_signals += 1
            total_signals += 1
            
            # Bollinger Bands signals
            if features.get('bb_position', 0.5) < 0.2:
                buy_signals += 1
            elif features.get('bb_position', 0.5) > 0.8:
                sell_signals += 1
            total_signals += 1
            
            # MACD signals
            if features.get('macd_hist', 0) > 0:
                buy_signals += 1
            elif features.get('macd_hist', 0) < 0:
                sell_signals += 1
            total_signals += 1
            
            # Calculate confidence
            max_signals = max(buy_signals, sell_signals)
            confidence = max_signals / total_signals if total_signals > 0 else 0
            
            # Determine action
            if buy_signals > sell_signals:
                action = 'BUY'
            elif sell_signals > buy_signals:
                action = 'SELL'
            else:
                action = 'HOLD'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return self._get_empty_signals()
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions."""
        try:
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Rename columns to match Upbit API response
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            
            # Calculate market trend
            ma20 = df['close'].rolling(20).mean()
            ma50 = df['close'].rolling(50).mean()
            current_price = df['close'].iloc[-1]
            
            if current_price > ma20.iloc[-1] > ma50.iloc[-1]:
                trend = 'STRONG_UPTREND'
            elif current_price > ma20.iloc[-1]:
                trend = 'UPTREND'
            elif current_price < ma20.iloc[-1] < ma50.iloc[-1]:
                trend = 'STRONG_DOWNTREND'
            elif current_price < ma20.iloc[-1]:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            # Calculate volatility
            volatility = df['close'].pct_change().std() * 100
            
            # Calculate volume trend
            volume_change = (df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] - 1) * 100
            
            return {
                'market_conditions': {
                    'trend': trend,
                    'volatility': float(volatility),
                    'volume_change': float(volume_change)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {'market_conditions': {}}
    
    def _get_empty_signals(self) -> Dict[str, Any]:
        """Return empty signals structure."""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'market_conditions': {}
        } 