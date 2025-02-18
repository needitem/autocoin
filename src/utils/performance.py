"""
Performance Monitoring Module

This module handles performance monitoring and metrics calculation
for the cryptocurrency trading application.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class PerformanceMonitor:
    """Class for monitoring trading performance and calculating metrics."""
    
    def __init__(self) -> None:
        """Initialize the PerformanceMonitor with default settings."""
        self.logger = logging.getLogger(__name__)
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()

    def calculate_trading_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate various trading performance metrics.
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            
        Returns:
            Dict[str, Any]: Calculated performance metrics
        """
        try:
            if not trades:
                return self._get_empty_metrics()
            
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            
            # Calculate profit metrics
            total_profit = sum(t.get('pnl', 0) for t in trades)
            max_profit = max((t.get('pnl', 0) for t in trades), default=0)
            max_loss = min((t.get('pnl', 0) for t in trades), default=0)
            
            # Calculate win rate and risk metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = (
                sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) /
                abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
                if sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) != 0
                else float('inf')
            )
            
            # Calculate drawdown
            drawdown = self._calculate_drawdown(trades)
            
            # Calculate trade duration metrics
            durations = self._calculate_trade_durations(trades)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
                'max_drawdown': drawdown['max_drawdown'],
                'current_drawdown': drawdown['current_drawdown'],
                'average_trade_duration': durations['average'],
                'max_trade_duration': durations['max'],
                'min_trade_duration': durations['min'],
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(trades),
                'expectancy': self._calculate_expectancy(trades),
                'metadata': {
                    'calculation_time': datetime.now().isoformat(),
                    'period_start': min(t['entry_time'] for t in trades),
                    'period_end': max(t.get('exit_time', datetime.now().isoformat()) for t in trades),
                    'number_of_trades': total_trades
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating trading metrics: {str(e)}")
            return self._get_empty_metrics()

    def _calculate_drawdown(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate maximum drawdown from trade history.
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            
        Returns:
            Dict[str, float]: Maximum and current drawdown
        """
        try:
            # Calculate cumulative profits
            cumulative = np.cumsum([t.get('pnl', 0) for t in trades])
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdowns
            drawdowns = running_max - cumulative
            
            return {
                'max_drawdown': float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0,
                'current_drawdown': float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}

    def _calculate_trade_durations(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate trade duration statistics.
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            
        Returns:
            Dict[str, float]: Trade duration statistics
        """
        try:
            durations = []
            
            for trade in trades:
                if trade.get('exit_time') and trade.get('entry_time'):
                    entry = datetime.fromisoformat(trade['entry_time'])
                    exit = datetime.fromisoformat(trade['exit_time'])
                    duration = (exit - entry).total_seconds() / 3600  # Convert to hours
                    durations.append(duration)
            
            if durations:
                return {
                    'average': float(np.mean(durations)),
                    'max': float(np.max(durations)),
                    'min': float(np.min(durations))
                }
            
            return {'average': 0.0, 'max': 0.0, 'min': 0.0}
        except Exception as e:
            self.logger.error(f"Error calculating trade durations: {str(e)}")
            return {'average': 0.0, 'max': 0.0, 'min': 0.0}

    def _calculate_sharpe_ratio(self, trades: List[Dict[str, Any]], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio for the trading strategy.
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sharpe Ratio
        """
        try:
            if not trades:
                return 0.0
            
            # Calculate daily returns
            returns = [t.get('pnl', 0) for t in trades]
            
            if not returns:
                return 0.0
            
            # Calculate annualized metrics
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Assume 252 trading days per year
            sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
            
            return float(sharpe)
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe Ratio: {str(e)}")
            return 0.0

    def _calculate_risk_reward_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate Risk-Reward Ratio for the trading strategy.
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            
        Returns:
            float: Risk-Reward Ratio
        """
        try:
            if not trades:
                return 0.0
            
            winning_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0]
            
            if not winning_trades or not losing_trades:
                return 0.0
            
            avg_win = np.mean(winning_trades)
            avg_loss = np.mean(losing_trades)
            
            if avg_loss == 0:
                return float('inf')
            
            return float(avg_win / avg_loss)
        except Exception as e:
            self.logger.error(f"Error calculating Risk-Reward Ratio: {str(e)}")
            return 0.0

    def _calculate_expectancy(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate trading expectancy (average profit/loss per trade).
        
        Args:
            trades (List[Dict[str, Any]]): List of trade records
            
        Returns:
            float: Trading expectancy
        """
        try:
            if not trades:
                return 0.0
            
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            return float(total_pnl / len(trades))
        except Exception as e:
            self.logger.error(f"Error calculating expectancy: {str(e)}")
            return 0.0

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """
        Get empty metrics dictionary with zero values.
        
        Returns:
            Dict[str, Any]: Empty metrics
        """
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'average_trade_duration': 0.0,
            'max_trade_duration': 0.0,
            'min_trade_duration': 0.0,
            'sharpe_ratio': 0.0,
            'risk_reward_ratio': 0.0,
            'expectancy': 0.0,
            'metadata': {
                'calculation_time': datetime.now().isoformat(),
                'period_start': None,
                'period_end': None,
                'number_of_trades': 0
            }
        }

    def track_metric(self, metric_name: str, value: float, timestamp: Optional[str] = None) -> None:
        """
        Track a performance metric over time.
        
        Args:
            metric_name (str): Name of the metric
            value (float): Metric value
            timestamp (Optional[str]): Timestamp for the metric
        """
        try:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': timestamp or datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error tracking metric {metric_name}: {str(e)}")

    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric_name (str): Name of the metric
            limit (int): Number of historical values to retrieve
            
        Returns:
            List[Dict[str, Any]]: Historical metric values
        """
        try:
            return list(reversed(self.metrics[metric_name]))[:limit]
        except Exception as e:
            self.logger.error(f"Error retrieving metric history for {metric_name}: {str(e)}")
            return []

    def calculate_system_metrics(self) -> Dict[str, Any]:
        """
        Calculate system performance metrics.
        
        Returns:
            Dict[str, Any]: System performance metrics
        """
        try:
            uptime = datetime.now() - self.start_time
            
            return {
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime),
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error calculating system metrics: {str(e)}")
            return {
                'uptime_seconds': 0,
                'uptime_formatted': '0:00:00',
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat()
            } 