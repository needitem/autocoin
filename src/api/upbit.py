"""
UPbit 종합 트레이딩 시스템 (버전 2.0)

기능 요약:
1. 실시간 시장 데이터 수집 및 처리
2. 150+ 기술적 지표 통합 분석
3. SQLite 기반 데이터 저장
4. 자동화된 리스크 관리
5. 멀티타임프레임 분석 지원
6. 커스텀 전략 빌더 인터페이스
"""

import os
from dotenv import load_dotenv
import jwt
import uuid
import hashlib
import requests
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import urllib.parse
import traceback
from src.db.database import DatabaseManager
from src.utils.logger import get_logger, DEBUG, INFO
from src.api.base import ExchangeAPI
import json

# .env 파일 로드
load_dotenv()

class UpbitTradingSystem(ExchangeAPI):
    def __init__(self, access_key: str = None, secret_key: str = None, 
                 db_path: str = 'market_data.db', verbose: bool = False):
        """업비트 트레이딩 시스템 초기화
        
        Args:
            access_key (str): API 접근 키
            secret_key (str): API 비밀 키
            db_path (str): 데이터베이스 파일 경로
            verbose (bool): 상세 로깅 여부
        """
        super().__init__(access_key, secret_key)
        
        self.exchange_name = "Upbit"
        self.verbose = verbose
        self.logger = get_logger('upbit_trading')
        
        # 로깅 레벨 설정
        if verbose:
            self.logger.setLevel(DEBUG)
        else:
            self.logger.setLevel(INFO)
        
        # API 키 설정
        if not access_key or not secret_key:
            self._load_api_keys()
            
        # 데이터베이스 매니저 초기화
        self.db_manager = DatabaseManager(
            db_path=db_path if db_path else 'market_data.db',
            verbose=verbose
        )
        
        self.logger.info("API keys loaded successfully")
        
        self.server_url = "https://api.upbit.com/v1"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms
        self.max_retries = 3  # 최대 재시도 횟수 추가
        
        # 마지막 요청 정보 초기화
        self._last_request = {
            'method': 'GET',
            'endpoint': '',
            'params': None,
            'data': None,
            'headers': None
        }

    def _log_verbose(self, message: str):
        """상세 로그 메시지 기록"""
        if self.verbose:
            self.logger.debug(message)
            
    def _log_error(self, message: str, error: Exception = None):
        """에러 로그 메시지 기록"""
        error_msg = f"{message}: {str(error)}" if error else message
        self.logger.error(error_msg)
        if self.verbose and error:
            self.logger.error(traceback.format_exc())

    def _log_info(self, message: str):
        """정보 로그 메시지 기록"""
        self.logger.info(message)

    def _log_warning(self, message: str):
        """경고 로그 메시지 기록"""
        self.logger.warning(message)

    def _wait_for_rate_limit(self):
        """API 요청 속도 제한"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def _create_jwt_token(self, query_hash: Optional[str] = None) -> str:
        """JWT 토큰 생성"""
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }
        
        if query_hash:
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'

        jwt_token = jwt.encode(payload, self.secret_key)
        return jwt_token

    def _create_query_hash(self, params: dict) -> str:
        """쿼리 해시 생성"""
        query_string = self._dict_to_query_string(params)
        query_hash = hashlib.sha512(query_string.encode()).hexdigest()
        return query_hash

    def _dict_to_query_string(self, params: dict) -> str:
        """딕셔너리를 쿼리 스트링으로 변환"""
        return urllib.parse.urlencode(params)

    def _send_request(self, method: str, endpoint: str, params: Optional[Dict] = None, timeout: int = 30) -> Optional[Dict]:
        """API 요청 전송"""
        try:
            # 현재 요청 정보 저장
            self._last_request = {
                'method': method,
                'endpoint': endpoint,
                'params': params,
                'timeout': timeout
            }
            
            url = f"{self.server_url}{endpoint}"
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return self.handle_api_error(e)

    def get_accounts(self) -> List[Dict]:
        """전체 계좌 조회"""
        try:
            self._log_verbose("Getting all accounts")
            
            if not (self.access_key and self.secret_key):
                self._log_verbose("Using simulated accounts for dummy mode")
                return [
                    {'currency': 'KRW', 'balance': '1000000', 'locked': '0', 
                     'avg_buy_price': '0', 'avg_buy_price_modified': False, 'unit_currency': 'KRW'},
                    {'currency': 'BTC', 'balance': '0.1', 'locked': '0', 
                     'avg_buy_price': '50000000', 'avg_buy_price_modified': False, 'unit_currency': 'KRW'}
                ]
            
            response = self._send_request('GET', '/accounts')
            return response
        except Exception as e:
            self._log_verbose(f"Error getting accounts: {str(e)}")
            self._log_error(f"Failed to get accounts: {str(e)}")
            return []

    def get_order_chance(self, market: str) -> Dict:
        """주문 가능 정보 조회"""
        try:
            self._log_verbose(f"Getting order chance for {market}")
            
            params = {'market': market}
            response = self._send_request('GET', '/orders/chance', params=params)
            return response
        except Exception as e:
            self._log_verbose(f"Error getting order chance: {str(e)}")
            self._log_error(f"Failed to get order chance: {str(e)}")
            return {}

    def get_order(self, uuid: Optional[str] = None, identifier: Optional[str] = None) -> Dict:
        """개별 주문 조회"""
        try:
            self._log_verbose(f"Getting order details for {uuid or identifier}")
            
            params = {}
            if uuid:
                params['uuid'] = uuid
            if identifier:
                params['identifier'] = identifier
            
            response = self._send_request('GET', '/order', params=params)
            return response
        except Exception as e:
            self._log_verbose(f"Error getting order details: {str(e)}")
            self._log_error(f"Failed to get order details: {str(e)}")
            return {}

    def get_orders(self, market: Optional[str] = None, state: Optional[str] = None,
                  states: Optional[List[str]] = None, uuids: Optional[List[str]] = None,
                  identifiers: Optional[List[str]] = None, page: int = 1,
                  limit: int = 100, order_by: str = 'desc') -> List[Dict]:
        """주문 리스트 조회"""
        try:
            self._log_verbose(f"Getting orders for {market or 'all markets'}")
            
            params = {
                'page': page,
                'limit': limit,
                'order_by': order_by
            }
            
            if market:
                params['market'] = market
            if state:
                params['state'] = state
            if states:
                params['states[]'] = states
            if uuids:
                params['uuids[]'] = uuids
            if identifiers:
                params['identifiers[]'] = identifiers
            
            response = self._send_request('GET', '/orders', params=params)
            return response
        except Exception as e:
            self._log_verbose(f"Error getting orders: {str(e)}")
            self._log_error(f"Failed to get orders: {str(e)}")
            return []

    def get_pending_orders(self, market: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회"""
        try:
            self._log_verbose(f"Getting pending orders for {market or 'all markets'}")
            
            params = {'state': 'wait'}
            if market:
                params['market'] = market
            
            response = self._send_request('GET', '/orders', params=params)
            return response
        except Exception as e:
            self._log_verbose(f"Error getting pending orders: {str(e)}")
            self._log_error(f"Failed to get pending orders: {str(e)}")
            return []

    def place_order(self, market: str, side: str, ord_type: str, volume: float = None, price: float = None) -> dict:
        """주문 실행"""
        try:
            if not self.access_key or not self.secret_key:
                self._log_verbose("API 키가 설정되지 않았습니다")
                return {}
            
            data = {
                'market': market,
                'side': side,
                'ord_type': ord_type
            }
            
            if volume is not None:
                data['volume'] = str(volume)
            if price is not None:
                data['price'] = str(price)
            
            response = self._send_request('POST', '/orders', data=data)
            return response
            
        except Exception as e:
            self._log_verbose(f"주문 중 오류 발생: {str(e)}")
            return {}

    def get_orderbook(self, market: str) -> dict:
        """호가 정보 조회"""
        try:
            url = f"{self.server_url}/orderbook"
            params = {'markets': market}
            
            self._wait_for_rate_limit()
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                orderbooks = response.json()
                return orderbooks[0] if orderbooks else {}
            else:
                self._log_verbose(f"호가 정보 조회 실패: {response.status_code}")
                return {}
            
        except Exception as e:
            self._log_verbose(f"호가 정보 조회 중 오류 발생: {str(e)}")
            return {}

    def get_trades_ticks(self, market: str, to: Optional[str] = None,
                        count: int = 100, cursor: Optional[str] = None,
                        days_ago: Optional[int] = None) -> List[Dict]:
        """최근 체결 내역 조회"""
        try:
            self._log_verbose(f"Getting trade ticks for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            
            if to:
                params['to'] = to
            if cursor:
                params['cursor'] = cursor
            if days_ago:
                params['daysAgo'] = days_ago
            
            return self._send_request('GET', '/trades/ticks', params=params)
        except Exception as e:
            self._log_verbose(f"Error getting trade ticks: {str(e)}")
            self._log_error(f"Failed to get trade ticks: {str(e)}")
            return []

    def get_daily_ohlcv(self, market: str, count: int = 200) -> pd.DataFrame:
        """일봉 데이터 조회
        
        Args:
            market (str): 마켓 코드
            count (int, optional): 캔들 개수. Defaults to 200.
            
        Returns:
            pd.DataFrame: 일봉 데이터
        """
        try:
            self._log_verbose(f"Getting daily OHLCV for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            
            response = self._send_request('GET', '/candles/days', params=params)
            
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.set_index('timestamp')
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self._log_verbose(f"Error getting daily OHLCV: {str(e)}")
            self._log_error(f"Failed to get daily OHLCV: {str(e)}")
            return pd.DataFrame()

    def get_weekly_ohlcv(self, market: str, to: Optional[str] = None,
                        count: int = 200, cursor: Optional[str] = None) -> pd.DataFrame:
        """주봉 데이터 조회"""
        try:
            self._log_verbose(f"Getting weekly OHLCV for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            
            if to:
                params['to'] = to
            if cursor:
                params['cursor'] = cursor
            
            response = self._send_request('GET', '/candles/weeks', params=params)
            
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.set_index('timestamp')
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self._log_verbose(f"Error getting weekly OHLCV: {str(e)}")
            self._log_error(f"Failed to get weekly OHLCV: {str(e)}")
            return pd.DataFrame()

    def get_monthly_ohlcv(self, market: str, to: Optional[str] = None,
                         count: int = 200, cursor: Optional[str] = None) -> pd.DataFrame:
        """월봉 데이터 조회"""
        try:
            self._log_verbose(f"Getting monthly OHLCV for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            
            if to:
                params['to'] = to
            if cursor:
                params['cursor'] = cursor
            
            response = self._send_request('GET', '/candles/months', params=params)
            
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.set_index('timestamp')
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self._log_verbose(f"Error getting monthly OHLCV: {str(e)}")
            self._log_error(f"Failed to get monthly OHLCV: {str(e)}")
            return pd.DataFrame()

    def get_markets(self, is_details: bool = True) -> List[Dict]:
        """마켓 코드 조회
        
        Args:
            is_details (bool, optional): 상세 정보 포함 여부. Defaults to True.
            
        Returns:
            List[Dict]: 마켓 정보 목록
        """
        try:
            self._log_verbose("마켓 코드 조회 중...")
            
            # 캐시 키 생성
            cache_key = f"markets:{'details' if is_details else 'simple'}"
            
            # 캐시된 데이터 확인
            cached_data = self.db_manager.get_cache(cache_key)
            if cached_data:
                return cached_data
            
            # API 요청
            response = self._send_request(
                method='GET',
                endpoint='/market/all',
                params={'isDetails': str(is_details).lower()}
            )
            
            if response:
                # KRW 마켓만 필터링
                krw_markets = [market for market in response if market['market'].startswith('KRW-')]
                
                # 캐시 저장 (1시간)
                self.db_manager.set_cache(cache_key, krw_markets, 3600)
                    
                self._log_info(f"{len(krw_markets)}개의 KRW 마켓을 가져왔습니다")
                return krw_markets
            
            self._log_warning("마켓 정보를 가져오지 못했습니다")
            return []
                
        except Exception as e:
            self._log_error("마켓 코드 조회 중 오류 발생", e)
            return []

    def get_minute_candles(self, market: str, unit: int = 1, count: int = 200) -> list:
        """Get minute candles for a market."""
        try:
            self._log_verbose(f"Getting {unit}-minute candles for {market}")
            
            self._wait_for_rate_limit()
            
            url = f"{self.server_url}/candles/minutes/{unit}"
            params = {
                'market': market,
                'count': count
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            self.last_request_time = time.time()
            return response.json()
            
        except Exception as e:
            self._log_verbose(f"Error getting minute candles: {str(e)}")
            return []

    def get_second_candles(self, market: str, to: Optional[str] = None,
                          count: int = 200) -> pd.DataFrame:
        """초 캔들 조회"""
        try:
            self._log_verbose(f"Getting second candles for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            if to:
                params['to'] = to
            
            response = self._send_request('GET', '/candles/seconds/1', params=params)
            
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.set_index('timestamp')
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self._log_verbose(f"Error getting second candles: {str(e)}")
            self._log_error(f"Failed to get second candles: {str(e)}")
            return pd.DataFrame()

    def get_year_candles(self, market: str, to: Optional[str] = None,
                        count: int = 200) -> pd.DataFrame:
        """연봉 데이터 조회"""
        try:
            self._log_verbose(f"Getting yearly OHLCV for {market}")
            
            params = {
                'market': market,
                'count': count
            }
            if to:
                params['to'] = to
            
            response = self._send_request('GET', '/candles/years', params=params)
            
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['candle_date_time_utc'])
                df = df.set_index('timestamp')
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self._log_verbose(f"Error getting yearly OHLCV: {str(e)}")
            self._log_error(f"Failed to get yearly OHLCV: {str(e)}")
            return pd.DataFrame()

    def get_tickers_by_quote(self, quotes: Optional[List[str]] = None) -> List[Dict]:
        """마켓 단위 현재가 정보 조회"""
        try:
            self._log_verbose(f"Getting tickers by quote: {quotes}")
            
            params = {}
            if quotes:
                params['quotes'] = ','.join(quotes)
            
            return self._send_request('GET', '/ticker/quotes', params=params)
        except Exception as e:
            self._log_verbose(f"Error getting tickers by quote: {str(e)}")
            self._log_error(f"Failed to get tickers by quote: {str(e)}")
            return []

    def get_orderbook_units(self) -> List[Dict]:
        """호가 단위 정보 조회"""
        try:
            self._log_verbose("Getting orderbook units")
            
            return self._send_request('GET', '/orderbook/units')
        except Exception as e:
            self._log_verbose(f"Error getting orderbook units: {str(e)}")
            self._log_error(f"Failed to get orderbook units: {str(e)}")
            return []

    def get_balance(self, currency: str = 'KRW') -> float:
        """특정 화폐의 잔고 조회"""
        try:
            accounts = self.get_accounts()
            for account in accounts:
                if account['currency'] == currency:
                    return float(account['balance'])
            return 0.0
        except Exception as e:
            self._log_verbose(f"Error getting balance: {str(e)}")
            self._log_error(f"Failed to get balance: {str(e)}")
            return 0.0

    def fetch_ohlcv(self, market: str, interval: str = 'minute1', count: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data for a market.
        
        Args:
            market (str): Market code (e.g. "KRW-BTC")
            interval (str, optional): Time interval. Defaults to 'minute1'
            count (int, optional): Number of candles. Defaults to 200
            
        Returns:
            pd.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
        """
        try:
            self._log_verbose(f"Fetching OHLCV data for {market}")
            
            # API 엔드포인트 결정
            if interval.startswith('minute'):
                unit = interval.replace('minute', '')
                endpoint = f'/candles/minutes/{unit}'
            elif interval == 'day':
                endpoint = '/candles/days'
            elif interval == 'week':
                endpoint = '/candles/weeks'
            elif interval == 'month':
                endpoint = '/candles/months'
            else:
                raise ValueError(f"Invalid interval: {interval}")
            
            # API 호출
            params = {
                'market': market,
                'count': count
            }
            
            response = self._send_request('GET', endpoint, params=params)
            
            if not response:
                self._log_warning(f"No data available for {market}")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            self._log_verbose(f"Received response: {response[:2]}")  # 처음 2개 항목만 로깅
            
            # 응답이 리스트인지 확인
            if not isinstance(response, list):
                self._log_error("API response is not a list")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # 데이터 변환
            data = []
            for candle in response:
                try:
                    data.append({
                        'timestamp': pd.to_datetime(candle['candle_date_time_utc']),
                        'open': float(candle['opening_price']),
                        'high': float(candle['high_price']),
                        'low': float(candle['low_price']),
                        'close': float(candle['trade_price']),
                        'volume': float(candle['candle_acc_trade_volume'])
                    })
                except (KeyError, ValueError) as e:
                    self._log_error(f"Error processing candle data: {e}")
                    continue
            
            if not data:
                self._log_warning("No valid candle data found")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # DataFrame 생성 및 정렬
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            self._log_verbose(f"Created DataFrame with {len(df)} rows")
            return df
            
        except Exception as e:
            self._log_error(f"Failed to fetch OHLCV data: {str(e)}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def invalidate_cache(self, key: str):
        """캐시 무효화
        
        Args:
            key (str): 캐시 키
        """
        try:
            self.db_manager.delete_cache(key)
            if self.verbose:
                print(f"캐시 무효화 완료: {key}")
        except Exception as e:
            if self.verbose:
                print(f"캐시 무효화 중 오류 발생: {str(e)}")

    def check_data_consistency(self, data: pd.DataFrame) -> bool:
        """데이터 일관성 검사"""
        try:
            # 필수 컬럼 존재 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self._log_error("Missing required columns")
                return False

            # 인덱스가 timestamp인지 확인
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    self._log_error(f"Failed to convert index to DatetimeIndex: {str(e)}")
                    return False

            # 데이터 타입 검사
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    self._log_error(f"Column {col} is not numeric")
                    return False

            # 가격 관계 검사 (high >= low, high >= open, high >= close)
            invalid_rows = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close'])
            )
            if invalid_rows.any():
                self._log_error("Invalid price relationships found")
                return False

            # 음수 값 검사
            if (data[numeric_columns] < 0).any().any():
                self._log_error("Negative values found")
                return False

            # 거래량 검사
            if (data['volume'] == 0).all():
                self._log_warning("All volume values are zero")
                return True  # 거래량이 0이어도 데이터는 유효할 수 있음

            return True
        except Exception as e:
            self._log_error("Error checking data consistency", e)
            return False

    def save_to_database(self, data: pd.DataFrame, market: str, interval: str) -> bool:
        """데이터베이스에 저장"""
        try:
            # DataFrame을 복사하여 수정
            df_to_save = data.copy()
            
            # timestamp 인덱스를 문자열로 변환
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                df_to_save = df_to_save.reset_index()
                df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 데이터를 JSON 형식으로 변환
            json_data = df_to_save.to_json(orient='records')
            
            # 데이터베이스에 저장
            return self.db_manager.save_data(market, interval, json_data)
            
        except Exception as e:
            self._log_error(f"Error saving to database: {str(e)}")
            return False

    def load_from_database(self, market: str, interval: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """데이터베이스에서 데이터 로드"""
        try:
            key = f"{market}:{interval}"
            json_data = self.db_manager.get_data(market, interval)
            
            if not json_data:
                self._log_info(f"No data found for {market}:{interval}")
                return None
            
            # DataFrame으로 변환
            df = pd.read_json(json_data, orient='records')
            
            # timestamp를 datetime으로 변환하고 인덱스로 설정
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 시간 범위 필터링
            if start_time:
                df = df[df.index >= start_time]
            if end_time:
                df = df[df.index <= end_time]
            
            self._log_info(f"Data loaded from database for {market}:{interval}")
            return df
            
        except Exception as e:
            self._log_error(f"Error loading from database: {str(e)}")
            return None

    def get_market_data(self, market: str) -> Optional[pd.DataFrame]:
        """시장 데이터 조회

        Args:
            market (str): 마켓 코드 (ex. KRW-BTC)

        Returns:
            Optional[pd.DataFrame]: 시장 데이터를 담은 DataFrame. 실패 시 None 반환
        """
        try:
            # 총 청크 수 조회
            total_chunks = self.db_manager.get_total_chunks(market)
            if total_chunks is None:
                self.logger.info(f"No data found for {market}")
                return None
            
            total_chunks = int(total_chunks)
            chunks_data = []
            
            # 각 청크 로드
            for i in range(total_chunks):
                chunk_key = f"{market}_chunk_{i}"
                chunk_data = self.db_manager.get_chunk(market, i)
                
                if chunk_data is None:
                    self.logger.error(f"Missing chunk {i} for {market}")
                    continue
                
                try:
                    chunk_records = json.loads(chunk_data)
                    chunks_data.extend(chunk_records)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding chunk {i}: {str(e)}")
                    continue
                
            if not chunks_data:
                self.logger.error("No valid chunks found")
                return None
            
            # DataFrame 생성
            df = pd.DataFrame(chunks_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 데이터 정렬
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            return None

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """기술적 지표 계산"""
        try:
            if not self.check_data_consistency(data):
                self._log_error("Data consistency check failed")
                return {}

            # 이동평균선
            ma_data = {
                'MA5': data['close'].rolling(window=5).mean(),
                'MA10': data['close'].rolling(window=10).mean(),
                'MA20': data['close'].rolling(window=20).mean(),
                'MA60': data['close'].rolling(window=60).mean(),
                'MA120': data['close'].rolling(window=120).mean()
            }

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # 볼린저 밴드
            bb_period = 20
            bb_std = 2
            middle_band = data['close'].rolling(window=bb_period).mean()
            std = data['close'].rolling(window=bb_period).std()
            upper_band = middle_band + (std * bb_std)
            lower_band = middle_band - (std * bb_std)

            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            return {
                'moving_averages': ma_data,
                'rsi': rsi,
                'bollinger_bands': {
                    'upper': upper_band,
                    'middle': middle_band,
                    'lower': lower_band
                },
                'macd': {
                    'macd': macd_line,
                    'signal': signal_line,
                    'hist': histogram
                }
            }
        except Exception as e:
            self._log_error("Error calculating technical indicators", e)
            return {}

    def handle_api_error(self, error: Exception, retry_count: int = 0, max_retries: int = 3, retry_delay: float = 0.1) -> Optional[Any]:
        """API 에러 처리 및 재시도

        Args:
            error (Exception): 발생한 예외
            retry_count (int): 현재 재시도 횟수
            max_retries (int): 최대 재시도 횟수
            retry_delay (float): 재시도 대기 시간 (초)

        Returns:
            Optional[Any]: 성공 시 응답 데이터, 실패 시 None
        """
        self.logger.error(f"API error occurred: {str(error)}")
        
        if retry_count >= max_retries:
            self.logger.error(f"Max retries ({max_retries}) exceeded")
            return None
        
        # 지수 백오프로 대기 시간 계산
        wait_time = retry_delay * (2 ** retry_count)
        time.sleep(wait_time)
        
        try:
            # 마지막 요청 정보가 있는 경우에만 재시도
            if hasattr(self, '_last_request'):
                response = requests.get(
                    f"{self.server_url}{self._last_request['endpoint']}",
                    params=self._last_request['params'],
                    timeout=self._last_request['timeout']
                )
                response.raise_for_status()
                return response.json()
            
            return None
            
        except Exception as e:
            return self.handle_api_error(e, retry_count + 1, max_retries, retry_delay)

    def get_current_price(self, market: str, timeout: int = 30, max_retries: int = 3) -> Optional[Dict]:
        """현재가 조회

        Args:
            market (str): 마켓 코드 (ex. KRW-BTC)
            timeout (int, optional): 요청 타임아웃 (초). Defaults to 30.
            max_retries (int, optional): 최대 재시도 횟수. Defaults to 3.

        Returns:
            Optional[Dict]: 현재가 정보를 담은 딕셔너리. 실패 시 None 반환
        """
        try:
            url = f"{self.server_url}/ticker"
            params = {"markets": market}
            
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                self.logger.error("Empty response received")
                return None
            
            ticker = data[0]
            required_fields = {'market', 'trade_price', 'signed_change_rate'}
            
            if not all(field in ticker for field in required_fields):
                self.logger.error(f"Missing required fields in response: {ticker}")
                return None
            
            return {
                'market': ticker['market'],
                'trade_price': float(ticker['trade_price']),
                'signed_change_rate': float(ticker['signed_change_rate'])
            }
            
        except requests.exceptions.RequestException as e:
            return self.handle_api_error(e, retry_count=0, max_retries=max_retries)
        except (ValueError, KeyError, TypeError) as e:
            self.logger.error(f"Error processing response: {str(e)}")
            return None

    def save_market_data(self, market: str, data: pd.DataFrame) -> bool:
        """시장 데이터 저장"""
        try:
            return self.db_manager.save_market_data(market, data)
        except Exception as e:
            self._log_error(f"Error saving market data: {str(e)}")
            return False

    def load_market_data(self, market: str, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """시장 데이터 로드"""
        try:
            return self.db_manager.load_market_data(market, start_time, end_time)
        except Exception as e:
            self._log_error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()

    def _load_api_keys(self):
        """환경 변수에서 API 키 로드"""
        try:
            self.access_key = os.getenv('UPBIT_ACCESS_KEY')
            self.secret_key = os.getenv('UPBIT_SECRET_KEY')
            
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 설정되지 않았습니다. 일부 기능이 제한됩니다.")
                self.access_key = None
                self.secret_key = None
            else:
                self.logger.info("API 키 로드 성공")
                
        except Exception as e:
            self.logger.error(f"API 키 로드 실패: {e}")
            self.access_key = None
            self.secret_key = None

    def get_ticker(self, market: str) -> Optional[Dict]:
        """현재가 정보 조회"""
        try:
            self._log_verbose(f"현재가 조회 중: {market}")
            
            # 캐시 키 생성
            cache_key = f"ticker:{market}"
            
            # 캐시된 데이터 확인
            cached_data = self.db_manager.get_cache(cache_key)
            if cached_data:
                return cached_data
            
            # API 요청
            response = self._send_request(
                method='GET',
                endpoint='/ticker',
                params={'markets': market}
            )
            
            if response and isinstance(response, list) and len(response) > 0:
                ticker = response[0]
                result = {
                    'market': ticker['market'],
                    'trade_price': float(ticker['trade_price']),
                    'signed_change_rate': float(ticker['signed_change_rate']),
                    'acc_trade_volume_24h': float(ticker['acc_trade_volume_24h']),
                    'high_price': float(ticker['high_price']),
                    'low_price': float(ticker['low_price'])
                }
                
                # 캐시 저장 (10초)
                self.db_manager.set_cache(cache_key, result, 10)
                return result
            
            self._log_warning(f"현재가 데이터를 가져오지 못했습니다: {market}")
            return None
                
        except Exception as e:
            self._log_error(f"현재가 조회 중 오류 발생: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소 (ExchangeAPI 인터페이스 구현)"""
        try:
            if not self.access_key or not self.secret_key:
                self.logger.warning("API 키가 없어 주문을 취소할 수 없습니다")
                return False
                
            params = {'uuid': order_id}
            query_string = self._dict_to_query_string(params)
            query_hash = self._create_query_hash(params)
            
            headers = {
                'Authorization': f'Bearer {self._create_jwt_token(query_hash)}'
            }
            
            response = requests.delete(
                f"{self.server_url}/order",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"주문 취소 실패: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"주문 취소 중 오류 발생: {str(e)}")
            return False 