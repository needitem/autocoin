"""
빗썸(Bithumb) API 클라이언트
"""

import os
import time
import base64
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.api.base import ExchangeAPI
from src.utils.logger import get_logger
from src.db.database import DatabaseManager

load_dotenv()

class BithumbAPI(ExchangeAPI):
    """빗썸 API 클라이언트"""
    
    def __init__(self, access_key: str = None, secret_key: str = None, 
                 db_path: str = 'market_data.db', verbose: bool = False):
        """빗썸 API 초기화
        
        Args:
            access_key (str): API 접근 키
            secret_key (str): API 비밀 키
            db_path (str): 데이터베이스 파일 경로
            verbose (bool): 상세 로깅 여부
        """
        super().__init__(access_key, secret_key)
        
        self.exchange_name = "Bithumb"
        self.verbose = verbose
        self.logger = get_logger('bithumb_api')
        
        # API 키 설정
        if not access_key or not secret_key:
            self._load_api_keys()
            
        # 데이터베이스 매니저 초기화
        self.db_manager = DatabaseManager(db_path=db_path, verbose=verbose)
        
        # API 엔드포인트
        self.public_api_url = "https://api.bithumb.com/public"
        self.private_api_url = "https://api.bithumb.com"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms
        
        self.logger.info("빗썸 API 클라이언트 초기화 완료")
    
    def _load_api_keys(self):
        """환경 변수에서 API 키 로드"""
        self.access_key = os.getenv('BITHUMB_API_KEY')
        self.secret_key = os.getenv('BITHUMB_SECRET_KEY')
        
        if not self.access_key or not self.secret_key:
            self.logger.warning("빗썸 API 키가 설정되지 않았습니다. 일부 기능이 제한됩니다.")
    
    def _wait_for_rate_limit(self):
        """API 요청 속도 제한"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def _send_public_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Public API 요청"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.public_api_url}{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == '0000':
                return data.get('data')
            else:
                self.logger.error(f"API 오류: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Public API 요청 실패: {str(e)}")
            return None
    
    def _send_private_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Private API 요청"""
        try:
            if not self.access_key or not self.secret_key:
                self.logger.error("API 키가 설정되지 않았습니다")
                return None
            
            self._wait_for_rate_limit()
            
            # 빗썸 Private API 인증 로직 구현
            # 참고: https://apidocs.bithumb.com/docs/authentication
            
            # 현재는 기본 구현만 제공
            self.logger.warning("Private API 인증 구현 필요")
            return None
            
        except Exception as e:
            self.logger.error(f"Private API 요청 실패: {str(e)}")
            return None
    
    def get_markets(self) -> List[Dict]:
        """사용 가능한 마켓 목록 조회"""
        try:
            # 빗썸은 개별 API가 없으므로 ticker 정보에서 추출
            data = self._send_public_request("/ticker/ALL_KRW")
            
            if not data:
                return []
            
            markets = []
            for symbol, info in data.items():
                if symbol == 'date':
                    continue
                    
                markets.append({
                    'market': f"KRW-{symbol}",
                    'korean_name': symbol,  # 빗썸은 한글명 제공 안함
                    'english_name': symbol
                })
            
            return markets
            
        except Exception as e:
            self.logger.error(f"마켓 목록 조회 실패: {str(e)}")
            return []
    
    def get_ticker(self, market: str) -> Optional[Dict]:
        """현재가 정보 조회
        
        Args:
            market (str): 마켓 코드 (예: KRW-BTC)
            
        Returns:
            Dict: 현재가 정보
        """
        try:
            # 마켓 코드에서 심볼 추출 (KRW-BTC -> BTC)
            symbol = market.split('-')[1] if '-' in market else market
            
            data = self._send_public_request(f"/ticker/{symbol}_KRW")
            
            if not data:
                return None
            
            # 빗썸 응답을 업비트 형식으로 변환
            return {
                'market': market,
                'trade_price': float(data.get('closing_price', 0)),
                'signed_change_rate': float(data.get('fluctate_rate_24H', 0)) / 100,  # 퍼센트를 비율로
                'acc_trade_volume_24h': float(data.get('units_traded_24H', 0)),
                'high_price': float(data.get('max_price', 0)),
                'low_price': float(data.get('min_price', 0)),
                'opening_price': float(data.get('opening_price', 0)),
                'prev_closing_price': float(data.get('prev_closing_price', 0))
            }
            
        except Exception as e:
            self.logger.error(f"현재가 조회 실패 ({market}): {str(e)}")
            return None
    
    def get_orderbook(self, market: str) -> Optional[Dict]:
        """호가 정보 조회"""
        try:
            symbol = market.split('-')[1] if '-' in market else market
            
            data = self._send_public_request(f"/orderbook/{symbol}_KRW")
            
            if not data:
                return None
            
            # 빗썸 호가 데이터 변환
            orderbook = {
                'market': market,
                'timestamp': int(data.get('timestamp', 0)),
                'total_ask_size': 0,
                'total_bid_size': 0,
                'orderbook_units': []
            }
            
            # 매도호가
            asks = data.get('asks', [])
            bids = data.get('bids', [])
            
            for i in range(min(len(asks), len(bids))):
                unit = {
                    'ask_price': float(asks[i]['price']),
                    'bid_price': float(bids[i]['price']),
                    'ask_size': float(asks[i]['quantity']),
                    'bid_size': float(bids[i]['quantity'])
                }
                orderbook['orderbook_units'].append(unit)
                orderbook['total_ask_size'] += unit['ask_size']
                orderbook['total_bid_size'] += unit['bid_size']
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"호가 정보 조회 실패 ({market}): {str(e)}")
            return None
    
    def fetch_ohlcv(self, market: str, interval: str = 'minute1', count: int = 200) -> pd.DataFrame:
        """OHLCV 데이터 조회
        
        Args:
            market (str): 마켓 코드 (예: KRW-BTC)
            interval (str): 시간 간격 (minute1, minute3, minute5, minute10, minute30, hour, day)
            count (int): 데이터 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            symbol = market.split('-')[1] if '-' in market else market
            
            # 빗썸 캔들스틱 차트 인터벌 매핑
            interval_map = {
                'minute1': '1m',
                'minute3': '3m',
                'minute5': '5m',
                'minute10': '10m',
                'minute30': '30m',
                'hour': '1h',
                'day': '24h'
            }
            
            bithumb_interval = interval_map.get(interval, '1m')
            
            # 빗썸 캔들스틱 API 엔드포인트
            endpoint = f"/candlestick/{symbol}_KRW/{bithumb_interval}"
            data = self._send_public_request(endpoint)
            
            if not data:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # 데이터 변환
            df_data = []
            for candle in data[:count]:
                # candle = [timestamp, open, close, high, low, volume]
                df_data.append({
                    'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[3]),
                    'low': float(candle[4]),
                    'close': float(candle[2]),
                    'volume': float(candle[5])
                })
            
            if not df_data:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # DataFrame 생성
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"OHLCV 데이터 조회 실패 ({market}): {str(e)}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def get_balance(self, currency: str = 'KRW') -> float:
        """잔고 조회"""
        try:
            # Private API 필요
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 없어 잔고를 조회할 수 없습니다")
                return 0.0
            
            # TODO: Private API 구현
            return 0.0
            
        except Exception as e:
            self.logger.error(f"잔고 조회 실패 ({currency}): {str(e)}")
            return 0.0
    
    def place_order(self, market: str, side: str, ord_type: str, volume: float = None, price: float = None) -> dict:
        """주문 실행"""
        try:
            # Private API 필요
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 없어 주문을 실행할 수 없습니다")
                return {}
            
            # TODO: Private API 구현
            return {}
            
        except Exception as e:
            self.logger.error(f"주문 실행 실패: {str(e)}")
            return {}
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """주문 정보 조회"""
        try:
            # Private API 필요
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 없어 주문 정보를 조회할 수 없습니다")
                return None
            
            # TODO: Private API 구현
            return None
            
        except Exception as e:
            self.logger.error(f"주문 정보 조회 실패 ({order_id}): {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            # Private API 필요
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 없어 주문을 취소할 수 없습니다")
                return False
            
            # TODO: Private API 구현
            return False
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패 ({order_id}): {str(e)}")
            return False
    
    def format_market_code(self, base: str, quote: str = 'KRW') -> str:
        """빗썸 마켓 코드 포맷
        
        빗썸은 기본적으로 SYMBOL_KRW 형식을 사용하지만,
        통일성을 위해 KRW-SYMBOL 형식으로 변환
        """
        return f"{quote}-{base}"