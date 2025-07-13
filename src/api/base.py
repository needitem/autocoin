"""
거래소 API 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd

class ExchangeAPI(ABC):
    """거래소 API 추상 클래스"""
    
    def __init__(self, access_key: str = None, secret_key: str = None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.exchange_name = "Unknown"
        
    @abstractmethod
    def get_markets(self) -> List[Dict]:
        """사용 가능한 마켓 목록 조회"""
        pass
    
    @abstractmethod
    def get_ticker(self, market: str) -> Optional[Dict]:
        """현재가 정보 조회"""
        pass
    
    @abstractmethod
    def get_orderbook(self, market: str) -> Optional[Dict]:
        """호가 정보 조회"""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, market: str, interval: str = 'minute1', count: int = 200) -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        pass
    
    @abstractmethod
    def get_balance(self, currency: str = 'KRW') -> float:
        """잔고 조회"""
        pass
    
    @abstractmethod
    def place_order(self, market: str, side: str, ord_type: str, volume: float = None, price: float = None) -> dict:
        """주문 실행"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Dict]:
        """주문 정보 조회"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        pass
    
    def get_exchange_name(self) -> str:
        """거래소 이름 반환"""
        return self.exchange_name
    
    def format_market_code(self, base: str, quote: str = 'KRW') -> str:
        """마켓 코드 포맷
        
        Args:
            base: 기준 통화 (예: BTC)
            quote: 견적 통화 (예: KRW)
            
        Returns:
            str: 포맷된 마켓 코드
        """
        # 기본적으로 대부분의 거래소는 QUOTE-BASE 형식 사용
        return f"{quote}-{base}"