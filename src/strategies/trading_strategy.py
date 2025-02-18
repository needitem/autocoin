"""
Trading Strategy Module

This module implements various trading strategies for cryptocurrency trading,
combining technical analysis, market sentiment, and risk management.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import numpy as np

class TradingStrategy:
    """Class for implementing trading strategies."""
    
    def __init__(self) -> None:
        """Initialize the TradingStrategy with default settings."""
        self.logger = logging.getLogger(__name__)
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def generate_trading_signals(self, 
                               market_data: Dict[str, Any],
                               technical_analysis: Dict[str, Any],
                               sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market analysis.
        
        Args:
            market_data (Dict[str, Any]): Current market data
            technical_analysis (Dict[str, Any]): Technical analysis results
            sentiment_analysis (Dict[str, Any]): Market sentiment analysis
            
        Returns:
            Dict[str, Any]: Trading signals and recommendations
        """
        try:
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(
                market_data,
                technical_analysis,
                sentiment_analysis
            )
            
            # Generate trading signals
            signals = self._generate_signals(market_conditions)
            
            # Calculate risk parameters
            risk_params = self._calculate_risk_parameters(
                market_data,
                signals
            )
            
            return {
                'signals': signals,
                'risk_parameters': risk_params,
                'market_conditions': market_conditions,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': market_data['symbol'],
                    'timeframe': market_data['timeframe']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            raise Exception(f"Failed to generate trading signals: {str(e)}")

    def _analyze_market_conditions(self,
                                 market_data: Dict[str, Any],
                                 technical_analysis: Dict[str, Any],
                                 sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions.
        
        Args:
            market_data (Dict[str, Any]): Current market data
            technical_analysis (Dict[str, Any]): Technical analysis results
            sentiment_analysis (Dict[str, Any]): Market sentiment analysis
            
        Returns:
            Dict[str, Any]: Market conditions analysis
        """
        try:
            # Analyze trend
            trend = technical_analysis.get('trend_analysis', {})
            trend_direction = trend.get('trend', 'NEUTRAL')
            trend_strength = trend.get('trend_strength', 0)
            
            # Analyze momentum
            momentum = technical_analysis.get('momentum', {})
            roc = momentum.get('roc', {}).get('value', 0)
            adx = momentum.get('adx', {}).get('value', 0)
            
            # Analyze volatility
            volatility = technical_analysis.get('volatility', {})
            bb = volatility.get('bollinger_bands', {})
            atr = volatility.get('atr', {}).get('value', 0)
            
            # Analyze sentiment
            sentiment = sentiment_analysis.get('sentiment', 'NEUTRAL')
            sentiment_score = sentiment_analysis.get('sentiment_score', 50)
            
            return {
                'trend': {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'is_strong': trend_strength > 25
                },
                'momentum': {
                    'roc': roc,
                    'adx': adx,
                    'is_strong': adx > 25
                },
                'volatility': {
                    'atr': atr,
                    'bb_width': bb.get('width', 0),
                    'is_high': bb.get('width', 0) > 0.05
                },
                'sentiment': {
                    'overall': sentiment,
                    'score': sentiment_score,
                    'is_positive': sentiment_score > 50
                },
                'trading_conditions': self._evaluate_trading_conditions(
                    trend_direction,
                    trend_strength,
                    adx,
                    sentiment_score
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}

    def _generate_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market conditions.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals
        """
        try:
            # Check if conditions are suitable for trading
            if market_conditions['trading_conditions'] == 'POOR':
                return {
                    'action': 'WAIT',
                    'reason': 'Poor trading conditions',
                    'confidence': 0.0
                }
            
            # Generate signals based on trend and momentum
            if market_conditions['trend']['is_strong'] and \
               market_conditions['momentum']['is_strong']:
                if market_conditions['trend']['direction'] == 'BULLISH':
                    action = 'BUY'
                    confidence = 0.8
                elif market_conditions['trend']['direction'] == 'BEARISH':
                    action = 'SELL'
                    confidence = 0.8
                else:
                    action = 'WAIT'
                    confidence = 0.0
            else:
                action = 'WAIT'
                confidence = 0.0
            
            # Adjust confidence based on sentiment
            if market_conditions['sentiment']['is_positive'] and action == 'BUY':
                confidence += 0.1
            elif not market_conditions['sentiment']['is_positive'] and action == 'SELL':
                confidence += 0.1
            
            # Adjust signals based on volatility
            if market_conditions['volatility']['is_high']:
                confidence *= 0.8  # Reduce confidence in high volatility
            
            return {
                'action': action,
                'confidence': min(confidence, 1.0),
                'reason': self._get_signal_reason(
                    action,
                    market_conditions
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                'action': 'WAIT',
                'confidence': 0.0,
                'reason': f"Error: {str(e)}"
            }

    def _calculate_risk_parameters(self,
                                 market_data: Dict[str, Any],
                                 signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk management parameters.
        
        Args:
            market_data (Dict[str, Any]): Current market data
            signals (Dict[str, Any]): Generated trading signals
            
        Returns:
            Dict[str, Any]: Risk management parameters
        """
        try:
            if signals['action'] == 'WAIT':
                return {
                    'position_size': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0
                }
            
            # Get current price
            current_price = market_data.get('current_price', 0)
            if not current_price:
                return {
                    'position_size': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0
                }
            
            # Calculate position size based on confidence
            base_position_size = 0.02  # 2% of portfolio
            position_size = base_position_size * signals['confidence']
            
            # Calculate stop loss and take profit
            atr = market_data.get('atr', current_price * 0.02)  # Default to 2% if ATR not available
            
            if signals['action'] == 'BUY':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:  # SELL
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            return {
                'position_size': float(position_size),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'risk_reward_ratio': 1.5,  # (3 * ATR) / (2 * ATR)
                'max_loss_percent': float(position_size * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }

    def _evaluate_trading_conditions(self,
                                   trend: str,
                                   trend_strength: float,
                                   adx: float,
                                   sentiment_score: float) -> str:
        """
        Evaluate overall trading conditions.
        
        Args:
            trend (str): Current market trend
            trend_strength (float): Strength of the trend
            adx (float): Average Directional Index value
            sentiment_score (float): Market sentiment score
            
        Returns:
            str: Trading conditions evaluation
        """
        try:
            # Count favorable conditions
            favorable_conditions = 0
            
            if trend in ['BULLISH', 'STRONGLY_BULLISH', 'BEARISH', 'STRONGLY_BEARISH']:
                favorable_conditions += 1
            
            if trend_strength > 25:
                favorable_conditions += 1
            
            if adx > 25:
                favorable_conditions += 1
            
            if 40 <= sentiment_score <= 60:  # Neutral sentiment
                favorable_conditions += 1
            elif (sentiment_score > 60 and trend == 'BULLISH') or \
                 (sentiment_score < 40 and trend == 'BEARISH'):
                favorable_conditions += 2  # Extra point for aligned sentiment
            
            # Evaluate conditions
            if favorable_conditions >= 4:
                return "EXCELLENT"
            elif favorable_conditions == 3:
                return "GOOD"
            elif favorable_conditions == 2:
                return "FAIR"
            else:
                return "POOR"
            
        except Exception as e:
            self.logger.error(f"Error evaluating trading conditions: {str(e)}")
            return "UNKNOWN"

    def _get_signal_reason(self,
                          action: str,
                          market_conditions: Dict[str, Any]) -> str:
        """
        Get the reason for the generated trading signal.
        
        Args:
            action (str): Trading action
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            str: Reason for the trading signal
        """
        try:
            if action == 'WAIT':
                if market_conditions['trading_conditions'] == 'POOR':
                    return "Poor trading conditions"
                return "No clear trading opportunity"
            
            reasons = []
            
            if market_conditions['trend']['is_strong']:
                reasons.append(f"Strong {market_conditions['trend']['direction'].lower()} trend")
            
            if market_conditions['momentum']['is_strong']:
                reasons.append("Strong momentum")
            
            if market_conditions['sentiment']['is_positive'] and action == 'BUY':
                reasons.append("Positive market sentiment")
            elif not market_conditions['sentiment']['is_positive'] and action == 'SELL':
                reasons.append("Negative market sentiment")
            
            if market_conditions['volatility']['is_high']:
                reasons.append("High volatility - exercise caution")
            
            return "; ".join(reasons) if reasons else "Multiple technical factors aligned"
            
        except Exception as e:
            self.logger.error(f"Error getting signal reason: {str(e)}")
            return "Technical analysis based signal" 