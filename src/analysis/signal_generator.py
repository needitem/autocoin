"""
Signal Generator Module

This module handles the generation of trading signals based on technical analysis
and market conditions.
"""

from typing import Dict, List, Any
import logging
from datetime import datetime

class SignalGenerator:
    """Class for generating trading signals."""
    
    def __init__(self) -> None:
        """Initialize the SignalGenerator."""
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self,
                        market_data: Dict[str, Any],
                        technical_analysis: Dict[str, Any],
                        sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on technical analysis and market conditions.
        
        Args:
            market_data (Dict[str, Any]): Market data including OHLCV
            technical_analysis (Dict[str, Any]): Technical analysis results
            sentiment_data (Dict[str, Any]): Market sentiment data
            
        Returns:
            Dict[str, Any]: Trading signals and analysis
        """
        try:
            if not market_data.get('ohlcv') or not technical_analysis:
                return {
                    'signals': {},
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'error': 'Insufficient data for signal generation'
                    }
                }
            
            # Get technical signals
            tech_signals = technical_analysis.get('signals', {})
            
            # Get market conditions
            market_conditions = technical_analysis.get('market_conditions', {})
            
            # Get sentiment
            sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
            sentiment_score = sentiment_data.get('sentiment_score', 50)
            
            # Generate signals
            signals = self._evaluate_signals(tech_signals, market_conditions, sentiment, sentiment_score)
            
            # Calculate risk parameters
            risk_parameters = self._calculate_risk_parameters(market_data, technical_analysis)
            
            return {
                'signals': signals,
                'risk_parameters': risk_parameters,
                'strategy_type': self._determine_strategy_type(signals, risk_parameters),
                'market_conditions': market_conditions,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'timeframe': market_data.get('timeframe', 'UNKNOWN')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {
                'signals': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _evaluate_signals(self,
                         tech_signals: Dict[str, str],
                         market_conditions: Dict[str, Any],
                         sentiment: str,
                         sentiment_score: float) -> Dict[str, Any]:
        """Evaluate and combine different signals to generate trading decisions."""
        try:
            # Initialize signal counters
            buy_signals = 0
            sell_signals = 0
            signal_strength = 0
            
            # Evaluate technical signals
            if tech_signals.get('OVERALL') == 'BUY':
                buy_signals += 1
                signal_strength += 0.4
            elif tech_signals.get('OVERALL') == 'SELL':
                sell_signals += 1
                signal_strength += 0.4
            
            # Evaluate market conditions
            trend = market_conditions.get('trend', 'UNKNOWN')
            if trend in ['STRONG_UPTREND', 'UPTREND']:
                buy_signals += 1
                signal_strength += 0.3
            elif trend in ['STRONG_DOWNTREND', 'DOWNTREND']:
                sell_signals += 1
                signal_strength += 0.3
            
            # Evaluate sentiment
            if sentiment == 'BULLISH' or sentiment_score > 70:
                buy_signals += 1
                signal_strength += 0.3
            elif sentiment == 'BEARISH' or sentiment_score < 30:
                sell_signals += 1
                signal_strength += 0.3
            
            # Determine action
            if buy_signals > sell_signals and signal_strength >= 0.6:
                action = 'BUY'
                confidence = signal_strength
            elif sell_signals > buy_signals and signal_strength >= 0.6:
                action = 'SELL'
                confidence = signal_strength
            else:
                action = 'HOLD'
                confidence = max(0.5, signal_strength)
            
            return {
                'action': action,
                'confidence': confidence,
                'factors': {
                    'technical': tech_signals.get('OVERALL', 'NEUTRAL'),
                    'trend': trend,
                    'sentiment': sentiment
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating signals: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'factors': {}
            }
    
    def _calculate_risk_parameters(self,
                                 market_data: Dict[str, Any],
                                 technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk parameters for trading decisions."""
        try:
            # Get volatility data
            volatility = technical_analysis.get('volatility', {})
            
            # Determine risk level
            daily_volatility = volatility.get('daily_volatility', 0)
            if daily_volatility > 5:
                risk_level = 'HIGH'
            elif daily_volatility > 2:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Calculate position size based on risk
            if risk_level == 'HIGH':
                max_position_size = 0.02  # 2% of portfolio
            elif risk_level == 'MEDIUM':
                max_position_size = 0.05  # 5% of portfolio
            else:
                max_position_size = 0.1   # 10% of portfolio
            
            return {
                'risk_level': risk_level,
                'max_position_size': max_position_size,
                'stop_loss_percent': daily_volatility * 2,  # 2x daily volatility
                'take_profit_percent': daily_volatility * 3  # 3x daily volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'risk_level': 'UNKNOWN',
                'max_position_size': 0.02,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 6.0
            }
    
    def _determine_strategy_type(self,
                               signals: Dict[str, Any],
                               risk_parameters: Dict[str, Any]) -> str:
        """Determine the type of trading strategy to use."""
        try:
            action = signals.get('action', 'HOLD')
            confidence = signals.get('confidence', 0.0)
            risk_level = risk_parameters.get('risk_level', 'UNKNOWN')
            
            if action == 'HOLD':
                return 'NEUTRAL'
            
            if confidence > 0.8:
                if risk_level == 'LOW':
                    return 'AGGRESSIVE'
                else:
                    return 'MODERATE'
            elif confidence > 0.6:
                if risk_level == 'HIGH':
                    return 'CONSERVATIVE'
                else:
                    return 'MODERATE'
            else:
                return 'CONSERVATIVE'
                
        except Exception as e:
            self.logger.error(f"Error determining strategy type: {str(e)}")
            return 'CONSERVATIVE' 