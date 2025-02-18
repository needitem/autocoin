"""
Investment Strategy Module

This module implements various investment strategies for cryptocurrency trading,
including different risk levels and trading styles.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

class TradingStrategyType(Enum):
    """Enumeration of available trading strategy types."""
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class InvestmentStrategy:
    """Class for implementing investment strategies."""
    
    def __init__(self, strategy_type: TradingStrategyType = TradingStrategyType.MODERATE) -> None:
        """
        Initialize the InvestmentStrategy with default settings.
        
        Args:
            strategy_type (TradingStrategyType): Type of trading strategy to use
        """
        self.logger = logging.getLogger(__name__)
        self.strategy_type = strategy_type
        
        # Strategy configurations
        self.strategy_configs = {
            TradingStrategyType.SCALPING: {
                'position_size': 0.02,  # 2% of portfolio per trade
                'take_profit': 0.005,   # 0.5% profit target
                'stop_loss': 0.003,     # 0.3% stop loss
                'max_trades_per_day': 20,
                'min_volume': 1000000,   # Minimum 24h volume
                'timeframes': ['1m', '5m'],
                'indicators': ['rsi', 'macd', 'bollinger'],
                'risk_level': 'high'
            },
            TradingStrategyType.SWING: {
                'position_size': 0.05,   # 5% of portfolio per trade
                'take_profit': 0.02,     # 2% profit target
                'stop_loss': 0.01,       # 1% stop loss
                'max_trades_per_day': 5,
                'min_volume': 500000,    # Minimum 24h volume
                'timeframes': ['1h', '4h'],
                'indicators': ['ma_cross', 'rsi', 'support_resistance'],
                'risk_level': 'medium'
            },
            TradingStrategyType.POSITION: {
                'position_size': 0.1,    # 10% of portfolio per trade
                'take_profit': 0.05,     # 5% profit target
                'stop_loss': 0.03,       # 3% stop loss
                'max_trades_per_day': 2,
                'min_volume': 100000,    # Minimum 24h volume
                'timeframes': ['4h', '1d'],
                'indicators': ['trend', 'moving_averages', 'fundamentals'],
                'risk_level': 'low'
            },
            TradingStrategyType.CONSERVATIVE: {
                'position_size': 0.03,   # 3% of portfolio per trade
                'take_profit': 0.015,    # 1.5% profit target
                'stop_loss': 0.01,       # 1% stop loss
                'max_trades_per_day': 3,
                'min_volume': 1000000,   # Minimum 24h volume
                'timeframes': ['15m', '1h'],
                'indicators': ['ma_cross', 'rsi', 'volume'],
                'risk_level': 'low'
            },
            TradingStrategyType.MODERATE: {
                'position_size': 0.05,   # 5% of portfolio per trade
                'take_profit': 0.025,    # 2.5% profit target
                'stop_loss': 0.015,      # 1.5% stop loss
                'max_trades_per_day': 5,
                'min_volume': 500000,    # Minimum 24h volume
                'timeframes': ['5m', '15m'],
                'indicators': ['macd', 'bollinger', 'support_resistance'],
                'risk_level': 'medium'
            },
            TradingStrategyType.AGGRESSIVE: {
                'position_size': 0.08,   # 8% of portfolio per trade
                'take_profit': 0.04,     # 4% profit target
                'stop_loss': 0.02,       # 2% stop loss
                'max_trades_per_day': 10,
                'min_volume': 250000,    # Minimum 24h volume
                'timeframes': ['1m', '5m'],
                'indicators': ['momentum', 'volatility', 'trend'],
                'risk_level': 'high'
            }
        }

    def generate_signals(self,
                        market_data: Dict[str, Any],
                        technical_analysis: Dict[str, Any],
                        sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on the selected strategy.
        
        Args:
            market_data (Dict[str, Any]): Current market data
            technical_analysis (Dict[str, Any]): Technical analysis results
            sentiment_analysis (Dict[str, Any]): Market sentiment analysis
            
        Returns:
            Dict[str, Any]: Trading signals and recommendations
        """
        try:
            config = self.strategy_configs[self.strategy_type]
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(
                market_data,
                technical_analysis,
                sentiment_analysis
            )
            
            # Generate strategy-specific signals
            if self.strategy_type == TradingStrategyType.SCALPING:
                signals = self._generate_scalping_signals(market_conditions)
            elif self.strategy_type == TradingStrategyType.SWING:
                signals = self._generate_swing_signals(market_conditions)
            elif self.strategy_type == TradingStrategyType.POSITION:
                signals = self._generate_position_signals(market_conditions)
            elif self.strategy_type == TradingStrategyType.CONSERVATIVE:
                signals = self._generate_conservative_signals(market_conditions)
            elif self.strategy_type == TradingStrategyType.MODERATE:
                signals = self._generate_moderate_signals(market_conditions)
            elif self.strategy_type == TradingStrategyType.AGGRESSIVE:
                signals = self._generate_aggressive_signals(market_conditions)
            else:
                signals = self._generate_general_signals(market_conditions)
            
            return {
                'strategy_type': self.strategy_type.value,
                'signals': signals,
                'risk_parameters': {
                    'position_size': config['position_size'],
                    'take_profit': config['take_profit'],
                    'stop_loss': config['stop_loss']
                },
                'market_conditions': market_conditions,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'timeframes': config['timeframes'],
                    'indicators_used': config['indicators']
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
            config = self.strategy_configs[self.strategy_type]
            
            # Check volume requirements
            volume_24h = market_data.get('volume_24h', 0)
            volume_sufficient = volume_24h >= config['min_volume']
            
            # Analyze trend
            trend = technical_analysis.get('trend_analysis', {}).get('trend', 'NEUTRAL')
            trend_strength = technical_analysis.get('trend_analysis', {}).get('trend_strength', 0)
            
            # Check volatility
            volatility = technical_analysis.get('volatility', {}).get('daily_volatility', 0)
            high_volatility = volatility > 0.02  # 2% daily volatility threshold
            
            # Analyze sentiment
            sentiment = sentiment_analysis.get('sentiment', 'NEUTRAL')
            sentiment_score = sentiment_analysis.get('sentiment_score', 50)
            
            return {
                'volume_sufficient': volume_sufficient,
                'trend': {
                    'direction': trend,
                    'strength': trend_strength,
                    'is_strong_trend': trend_strength > 25
                },
                'volatility': {
                    'value': volatility,
                    'is_high': high_volatility
                },
                'sentiment': {
                    'overall': sentiment,
                    'score': sentiment_score,
                    'is_positive': sentiment_score > 50
                },
                'trading_conditions': self._evaluate_trading_conditions(
                    volume_sufficient,
                    trend,
                    volatility,
                    sentiment_score
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}

    def _generate_scalping_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for scalping strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for scalping
        """
        try:
            # Check if market conditions are suitable for scalping
            if not market_conditions['volume_sufficient']:
                return {'action': 'WAIT', 'reason': 'Insufficient volume for scalping'}
            
            if market_conditions['volatility']['is_high']:
                return {'action': 'WAIT', 'reason': 'Volatility too high for scalping'}
            
            # Generate scalping signals based on market conditions
            if market_conditions['trend']['is_strong_trend']:
                action = 'BUY' if market_conditions['trend']['direction'] == 'BULLISH' else \
                        'SELL' if market_conditions['trend']['direction'] == 'BEARISH' else \
                        'WAIT'
            else:
                action = 'WAIT'
            
            return {
                'action': action,
                'timeframe': '1m',
                'strategy': 'SCALPING',
                'conditions_met': all([
                    market_conditions['volume_sufficient'],
                    not market_conditions['volatility']['is_high'],
                    market_conditions['trend']['is_strong_trend']
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scalping signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_swing_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for swing trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for swing trading
        """
        try:
            # Check if market conditions are suitable for swing trading
            if not market_conditions['volume_sufficient']:
                return {'action': 'WAIT', 'reason': 'Insufficient volume for swing trading'}
            
            # Generate swing trading signals based on market conditions
            if market_conditions['trend']['is_strong_trend'] and \
               market_conditions['sentiment']['is_positive']:
                action = 'BUY' if market_conditions['trend']['direction'] == 'BULLISH' else \
                        'SELL' if market_conditions['trend']['direction'] == 'BEARISH' else \
                        'WAIT'
            else:
                action = 'WAIT'
            
            return {
                'action': action,
                'timeframe': '4h',
                'strategy': 'SWING',
                'conditions_met': all([
                    market_conditions['volume_sufficient'],
                    market_conditions['trend']['is_strong_trend'],
                    market_conditions['sentiment']['is_positive']
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating swing trading signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_position_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for position trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for position trading
        """
        try:
            # Check if market conditions are suitable for position trading
            if not market_conditions['volume_sufficient']:
                return {'action': 'WAIT', 'reason': 'Insufficient volume for position trading'}
            
            # Generate position trading signals based on market conditions
            if market_conditions['trend']['is_strong_trend'] and \
               market_conditions['sentiment']['is_positive'] and \
               not market_conditions['volatility']['is_high']:
                action = 'BUY' if market_conditions['trend']['direction'] == 'BULLISH' else \
                        'SELL' if market_conditions['trend']['direction'] == 'BEARISH' else \
                        'WAIT'
            else:
                action = 'WAIT'
            
            return {
                'action': action,
                'timeframe': '1d',
                'strategy': 'POSITION',
                'conditions_met': all([
                    market_conditions['volume_sufficient'],
                    market_conditions['trend']['is_strong_trend'],
                    market_conditions['sentiment']['is_positive'],
                    not market_conditions['volatility']['is_high']
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating position trading signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_conservative_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for conservative trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for conservative trading
        """
        try:
            # Conservative strategy requires strong confirmation
            if market_conditions['trading_conditions'] not in ['EXCELLENT', 'GOOD']:
                return {'action': 'WAIT', 'reason': 'Conditions not optimal for conservative trading'}
            
            if market_conditions['volatility']['is_high']:
                return {'action': 'WAIT', 'reason': 'Volatility too high for conservative trading'}
            
            action = 'BUY' if all([
                market_conditions['trend']['direction'] == 'BULLISH',
                market_conditions['sentiment']['is_positive'],
                not market_conditions['volatility']['is_high']
            ]) else 'WAIT'
            
            return {
                'action': action,
                'timeframe': '1h',
                'strategy': 'CONSERVATIVE',
                'conditions_met': action == 'BUY'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating conservative signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_moderate_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for moderate trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for moderate trading
        """
        try:
            # Moderate strategy requires good conditions
            if market_conditions['trading_conditions'] == 'POOR':
                return {'action': 'WAIT', 'reason': 'Poor trading conditions'}
            
            action = 'BUY' if all([
                market_conditions['trend']['direction'] in ['BULLISH', 'STRONGLY_BULLISH'],
                market_conditions['sentiment']['score'] > 40
            ]) else 'SELL' if all([
                market_conditions['trend']['direction'] in ['BEARISH', 'STRONGLY_BEARISH'],
                market_conditions['sentiment']['score'] < 60
            ]) else 'WAIT'
            
            return {
                'action': action,
                'timeframe': '15m',
                'strategy': 'MODERATE',
                'conditions_met': action in ['BUY', 'SELL']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating moderate signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_aggressive_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for aggressive trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for aggressive trading
        """
        try:
            # Aggressive strategy can trade in most conditions
            action = 'BUY' if any([
                market_conditions['trend']['direction'] == 'BULLISH',
                market_conditions['sentiment']['is_positive']
            ]) else 'SELL' if any([
                market_conditions['trend']['direction'] == 'BEARISH',
                not market_conditions['sentiment']['is_positive']
            ]) else 'WAIT'
            
            return {
                'action': action,
                'timeframe': '5m',
                'strategy': 'AGGRESSIVE',
                'conditions_met': action in ['BUY', 'SELL']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating aggressive signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _generate_general_signals(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals for general trading strategy.
        
        Args:
            market_conditions (Dict[str, Any]): Current market conditions
            
        Returns:
            Dict[str, Any]: Trading signals for general trading
        """
        try:
            config = self.strategy_configs[self.strategy_type]
            
            # Check basic conditions
            if not market_conditions['volume_sufficient']:
                return {'action': 'WAIT', 'reason': 'Insufficient volume'}
            
            # Generate signals based on risk level
            if config['risk_level'] == 'low':
                signals = self._generate_conservative_signals(market_conditions)
            elif config['risk_level'] == 'medium':
                signals = self._generate_moderate_signals(market_conditions)
            else:  # high risk
                signals = self._generate_aggressive_signals(market_conditions)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating general trading signals: {str(e)}")
            return {'action': 'WAIT', 'reason': f"Error: {str(e)}"}

    def _evaluate_trading_conditions(self,
                                   volume_sufficient: bool,
                                   trend: str,
                                   volatility: float,
                                   sentiment_score: float) -> str:
        """
        Evaluate overall trading conditions.
        
        Args:
            volume_sufficient (bool): Whether volume meets minimum requirements
            trend (str): Current market trend
            volatility (float): Current market volatility
            sentiment_score (float): Current sentiment score
            
        Returns:
            str: Trading conditions evaluation
        """
        try:
            if not volume_sufficient:
                return "POOR"
            
            # Count favorable conditions
            favorable_conditions = 0
            
            if trend in ['BULLISH', 'STRONGLY_BULLISH']:
                favorable_conditions += 1
            
            if volatility < 0.02:  # Low volatility is generally favorable
                favorable_conditions += 1
            
            if sentiment_score > 50:
                favorable_conditions += 1
            
            # Evaluate conditions
            if favorable_conditions >= 3:
                return "EXCELLENT"
            elif favorable_conditions == 2:
                return "GOOD"
            elif favorable_conditions == 1:
                return "FAIR"
            else:
                return "POOR"
            
        except Exception as e:
            self.logger.error(f"Error evaluating trading conditions: {str(e)}")
            return "UNKNOWN" 