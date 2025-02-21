"""
Trading strategy implementation
"""

import pandas as pd
import numpy as np
from typing import Literal

class Strategy:
    def __init__(self):
        """Initialize strategy parameters."""
        self.short_window = 20
        self.long_window = 50
        self.rsi_period = 14
        self.rsi_upper = 70
        self.rsi_lower = 30

    def analyze(self, df: pd.DataFrame) -> Literal['BUY', 'SELL', 'HOLD']:
        """
        Analyze market data and generate trading signals.
        Uses a combination of Moving Average Crossover and RSI.
        """
        try:
            # Calculate indicators
            df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
            df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
            df['RSI'] = self._calculate_rsi(df['close'])
            
            # Get latest values
            current_short_ma = df['SMA_short'].iloc[-1]
            prev_short_ma = df['SMA_short'].iloc[-2]
            current_long_ma = df['SMA_long'].iloc[-1]
            prev_long_ma = df['SMA_long'].iloc[-2]
            current_rsi = df['RSI'].iloc[-1]
            
            # Generate signals
            if (current_short_ma > current_long_ma and 
                prev_short_ma <= prev_long_ma and 
                current_rsi < self.rsi_upper):
                return 'BUY'
            elif (current_short_ma < current_long_ma and 
                  prev_short_ma >= prev_long_ma and 
                  current_rsi > self.rsi_lower):
                return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            print(f"Error in strategy analysis: {e}")
            return 'HOLD'

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50)  # Neutral RSI value 