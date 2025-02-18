"""
Performance Analysis Module

This module handles the calculation of trading performance metrics such as
Sharpe ratio, drawdown, win rate, and other key performance indicators.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

class PerformanceAnalyzer:
    """Class for analyzing trading performance metrics."""
    
    def __init__(self) -> None:
        """Initialize the PerformanceAnalyzer."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, ohlcv_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate various performance metrics from OHLCV data.
        
        Args:
            ohlcv_data (List[Dict[str, Any]]): List of OHLCV data points
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            if not ohlcv_data:
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'error': 'No data available'
                    }
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            
            # Convert columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate metrics
            sharpe_ratio = self._calculate_sharpe_ratio(df['returns'])
            max_drawdown = self._calculate_max_drawdown(df['close'])
            win_rate = self._calculate_win_rate(df['returns'])
            profit_factor = self._calculate_profit_factor(df['returns'])
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                    'num_trades': len(df)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio."""
        try:
            # Assuming daily data and risk-free rate of 2%
            risk_free_rate = 0.02 / 365
            
            excess_returns = returns - risk_free_rate
            if len(excess_returns) > 0:
                sharpe_ratio = np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
                return float(sharpe_ratio)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate Maximum Drawdown percentage."""
        try:
            # Calculate the running maximum
            running_max = prices.expanding().max()
            
            # Calculate the percentage drawdown
            drawdown = ((running_max - prices) / running_max) * 100
            
            # Get the maximum drawdown
            max_drawdown = drawdown.max()
            
            return float(max_drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate Win Rate percentage."""
        try:
            if len(returns) > 0:
                # Calculate the number of winning trades
                winning_trades = (returns > 0).sum()
                
                # Calculate win rate
                win_rate = (winning_trades / len(returns)) * 100
                
                return float(win_rate)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate Profit Factor."""
        try:
            # Separate winning and losing trades
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            if len(losing_trades) > 0 and abs(losing_trades.sum()) > 0:
                # Calculate profit factor
                profit_factor = abs(winning_trades.sum() / losing_trades.sum())
                return float(profit_factor)
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {str(e)}")
            return 1.0 