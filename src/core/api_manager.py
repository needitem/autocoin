"""
API Request Manager Module

This module manages API requests to ensure rate limiting.
"""

import time
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
import pyupbit
import pandas as pd
from threading import Lock

class UpbitAPIManager:
    """Singleton class for managing Upbit API requests with rate limiting."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)
            self.last_request_time = {}
            self.request_interval = 1.0  # 1초
            self.initialized = True
    
    def _wait_for_rate_limit(self, request_type: str):
        """Wait if necessary to comply with rate limiting."""
        current_time = time.time()
        last_time = self.last_request_time.get(request_type, 0)
        
        # 마지막 요청 이후 경과 시간 계산
        elapsed = current_time - last_time
        
        # 필요한 대기 시간 계산
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            time.sleep(wait_time)
        
        # 요청 시간 업데이트
        self.last_request_time[request_type] = time.time()
    
    def get_current_price(self, market: str) -> Optional[float]:
        """Get current price with rate limiting."""
        try:
            self._wait_for_rate_limit('current_price')
            return pyupbit.get_current_price(market)
        except Exception as e:
            self.logger.error(f"Error getting current price for {market}: {str(e)}")
            return None
    
    def get_ohlcv(self, market: str, interval: str = "day", count: int = 1) -> Optional[pd.DataFrame]:
        """Get OHLCV data with rate limiting."""
        try:
            self._wait_for_rate_limit('ohlcv')
            return pyupbit.get_ohlcv(market, interval=interval, count=count)
        except Exception as e:
            self.logger.error(f"Error getting OHLCV data for {market}: {str(e)}")
            return None
    
    def get_orderbook(self, market: str) -> Optional[Dict[str, Any]]:
        """Get orderbook data with rate limiting."""
        try:
            self._wait_for_rate_limit('orderbook')
            return pyupbit.get_orderbook(market)
        except Exception as e:
            self.logger.error(f"Error getting orderbook for {market}: {str(e)}")
            return None
    
    def get_tickers(self, fiat: str = "KRW") -> List[str]:
        """Get list of tickers with rate limiting."""
        try:
            self._wait_for_rate_limit('tickers')
            tickers = pyupbit.get_tickers(fiat=fiat)
            return sorted(tickers) if tickers else []
        except Exception as e:
            self.logger.error(f"Error getting tickers: {str(e)}")
            return [] 