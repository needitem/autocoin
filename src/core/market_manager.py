"""
Market Manager

This module manages market data and operations.
"""

from typing import List, Dict, Any, Optional
import pyupbit
import logging
from datetime import datetime

class MarketManager:
    """Class for managing market data and operations."""
    
    def __init__(self):
        """Initialize the market manager."""
        self.logger = logging.getLogger(__name__)
    
    def get_markets(self) -> List[str]:
        """Get list of available markets."""
        try:
            markets = pyupbit.get_tickers(fiat="KRW")
            return markets if markets else []
        except Exception as e:
            self.logger.error(f"시장 목록 조회 실패: {str(e)}")
            return []
    
    def get_market_data(self, market: str) -> Optional[Dict[str, Any]]:
        """Get market data for specified market."""
        try:
            # Get current ticker
            ticker = pyupbit.get_current_price(market)
            if not ticker:
                return None
            
            # Get daily OHLCV
            df = pyupbit.get_ohlcv(market, interval="day", count=1)
            if df is None or df.empty:
                return None
            
            # Get ticker info
            ticker_info = pyupbit.get_ohlcv(market, interval="day", count=1)
            
            return {
                'symbol': market,
                'current_price': float(ticker),
                'open': float(df['open'].iloc[-1]),
                'high': float(df['high'].iloc[-1]),
                'low': float(df['low'].iloc[-1]),
                'volume_24h': float(df['volume'].iloc[-1]),
                'value_24h': float(df['value'].iloc[-1]),
                'price_change_24h': float((ticker - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100),
                'volume_change_24h': 0.0,  # Need historical data for comparison
                'market_cap': float(df['value'].iloc[-1]),  # Using trading value as market cap
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            return None 