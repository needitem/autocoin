"""
Core trading functionality
"""

from typing import Dict, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.api.base import ExchangeAPI
from src.api.upbit import UpbitTradingSystem
from src.api.bithumb import BithumbAPI
from src.core.strategy import Strategy
from src.utils.logger import get_logger, DEBUG, INFO

class TradingManager:
    def __init__(self, api: ExchangeAPI = None, exchange: str = 'upbit', verbose: bool = False):
        """거래 관리자 초기화
        
        Args:
            api (ExchangeAPI, optional): 거래소 API. Defaults to None.
            exchange (str, optional): 거래소 이름 ('upbit' 또는 'bithumb'). Defaults to 'upbit'.
            verbose (bool, optional): 상세 로깅 여부. Defaults to False.
        """
        self.exchange_name = exchange.lower()
        
        # API 초기화
        if api:
            self.api = api
        else:
            if self.exchange_name == 'bithumb':
                self.api = BithumbAPI(verbose=verbose)
            else:
                self.api = UpbitTradingSystem(verbose=verbose)
                
        self.logger = get_logger('trading')
        self.strategy = Strategy()
        
        # 로깅 레벨 설정
        if verbose:
            self.logger.setLevel(DEBUG)
        else:
            self.logger.setLevel(INFO)
        
        # 기본 마켓 목록
        self.default_markets = [
            {'market': 'KRW-BTC', 'korean_name': '비트코인', 'english_name': 'Bitcoin'},
            {'market': 'KRW-ETH', 'korean_name': '이더리움', 'english_name': 'Ethereum'},
            {'market': 'KRW-XRP', 'korean_name': '리플', 'english_name': 'Ripple'},
            {'market': 'KRW-DOGE', 'korean_name': '도지코인', 'english_name': 'Dogecoin'},
            {'market': 'KRW-ADA', 'korean_name': '에이다', 'english_name': 'Cardano'}
        ]
        
        self.logger.info(f"{self.api.get_exchange_name()} 거래 관리자가 초기화되었습니다.")
        
    def get_markets(self) -> List[Dict]:
        """사용 가능한 마켓 목록 조회
        
        Returns:
            list: 마켓 목록
        """
        try:
            markets = self.api.get_markets()
            if not markets:
                if self.logger:
                    self.logger.warning("마켓 목록을 가져올 수 없어 기본값을 사용합니다")
                return self.default_markets
            
            # KRW 마켓만 필터링
            krw_markets = [
                market for market in markets 
                if isinstance(market, dict) and market.get('market', '').startswith('KRW-')
            ]
            
            if not krw_markets:
                if self.logger:
                    self.logger.warning("KRW 마켓이 없어 기본값을 사용합니다")
                return self.default_markets
            
            if self.logger:
                self.logger.info(f"총 {len(krw_markets)}개의 KRW 마켓을 불러왔습니다")
            
            return krw_markets
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"마켓 목록 조회 중 오류 발생: {str(e)}")
            return self.default_markets

    def get_market_data(self, market: str) -> Optional[Dict]:
        """시장 데이터 조회"""
        try:
            # 현재가 정보 조회
            ticker = self.api.get_ticker(market)
            if not ticker:
                if self.logger:
                    self.logger.error(f"현재가 정보를 가져올 수 없습니다: {market}")
                return None

            # OHLCV 데이터 조회
            ohlcv = self.api.fetch_ohlcv(market, interval='minute1', count=200)
            if ohlcv is None or ohlcv.empty:
                if self.logger:
                    self.logger.warning(f"OHLCV 데이터를 가져올 수 없습니다: {market}")
                return ticker

            # 데이터 통합
            result = {
                'market': market,
                'trade_price': ticker['trade_price'],
                'signed_change_rate': ticker['signed_change_rate'],
                'acc_trade_volume_24h': ticker['acc_trade_volume_24h'],
                'high_price': ticker['high_price'],
                'low_price': ticker['low_price'],
                'ohlcv': ohlcv
            }

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting market data: {str(e)}")
            return None

    def get_ohlcv(self, market: str, interval: str = 'minute1', count: int = 200) -> Optional[pd.DataFrame]:
        """OHLCV 데이터 조회"""
        try:
            return self.api.fetch_ohlcv(market, interval=interval, count=count)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting OHLCV data: {str(e)}")
            return None

    def calculate_indicators(self, ohlcv: pd.DataFrame) -> Dict:
        """기술적 지표 계산"""
        try:
            if ohlcv is None or ohlcv.empty:
                return {}

            result = {}

            # 이동평균선
            ma_periods = [5, 10, 20, 60, 120]
            ma_data = pd.DataFrame()
            for period in ma_periods:
                ma_data[f'MA{period}'] = ohlcv['close'].rolling(window=period).mean()
            result['ma'] = ma_data

            # RSI
            delta = ohlcv['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))

            # 볼린저 밴드
            ma20 = ohlcv['close'].rolling(window=20).mean()
            std20 = ohlcv['close'].rolling(window=20).std()
            result['bollinger'] = {
                'upper': ma20 + (std20 * 2),
                'middle': ma20,
                'lower': ma20 - (std20 * 2)
            }

            # MACD
            exp1 = ohlcv['close'].ewm(span=12, adjust=False).mean()
            exp2 = ohlcv['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            result['macd'] = {
                'macd': macd,
                'signal': signal,
                'hist': macd - signal
            }

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def place_order(self, market: str, side: str, ord_type: str, volume: float, price: Optional[float] = None) -> bool:
        """주문 실행"""
        try:
            return self.api.place_order(market, side, ord_type, volume, price)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error placing order: {str(e)}")
            return False

    def execute_strategy(self, market: str, current_price: float) -> Dict:
        """Execute trading strategy."""
        try:
            ohlcv = self.api.fetch_ohlcv(market)
            if ohlcv is not None:
                signal = self.strategy.analyze(ohlcv)
                
                if signal == 'BUY':
                    balance = self.api.upbit.get_balance()
                    if balance:
                        amount = min(balance, 10000)  # Limit order size
                        result = self.api.execute_order('market_buy', market, amount)
                        if not result.get('error'):
                            return {'action': 'BUY', 'amount': amount, 'confidence': 0.8}
                
                elif signal == 'SELL':
                    balance = self.api.upbit.get_balance(market.split('-')[1])
                    if balance:
                        result = self.api.execute_order('market_sell', market, balance)
                        if not result.get('error'):
                            return {'action': 'SELL', 'amount': balance * current_price, 'confidence': 0.8}
            
            return {'action': 'HOLD', 'amount': 0, 'confidence': 0.5}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing strategy: {e}")
            return {'action': 'HOLD', 'amount': 0, 'confidence': 0.5}

    def execute_manual_trade(self, market: str, amount: float, is_buy: bool = True) -> bool:
        """Execute manual trade."""
        try:
            if is_buy:
                result = self.api.execute_order('market_buy', market, amount)
            else:
                result = self.api.execute_order('market_sell', market, amount)
            return not bool(result.get('error'))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing manual trade: {e}")
            return False

    def process_market_data(self, raw_data):
        """Process raw market data into standardized format.
        
        Args:
            raw_data (dict): Raw market data from API
            
        Returns:
            dict: Processed market data
        """
        try:
            processed_data = {
                'market': raw_data['market'],
                'timestamp': raw_data['candle_date_time_utc'],
                'open': raw_data['opening_price'],
                'high': raw_data['high_price'],
                'low': raw_data['low_price'],
                'close': raw_data['trade_price'],
                'volume': raw_data['candle_acc_trade_volume']
            }
            if self.logger:
                self.logger.info(f"Processed market data: {processed_data}")
            return processed_data
        except KeyError as e:
            if self.logger:
                self.logger.error(f"Error processing market data: {e}")
            return None
            
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators from market data.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            dict: Technical indicators
        """
        try:
            # 이동평균선 계산
            ma_periods = [5, 10, 20, 60, 120]
            ma_data = {}
            for period in ma_periods:
                ma_data[f'MA{period}'] = data['close'].rolling(window=period).mean()
            
            # RSI 계산
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD 계산
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'ma': ma_data,
                'rsi': rsi,
                'macd': {
                    'macd': macd,
                    'signal': signal,
                    'histogram': macd - signal
                }
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating technical indicators: {e}")
            return {
                'ma': {},
                'rsi': pd.Series(),
                'macd': {
                    'macd': pd.Series(),
                    'signal': pd.Series(),
                    'histogram': pd.Series()
                }
            }
    
    def switch_exchange(self, exchange: str, verbose: bool = None) -> bool:
        """거래소 전환
        
        Args:
            exchange (str): 거래소 이름 ('upbit' 또는 'bithumb')
            verbose (bool, optional): 상세 로깅 여부
            
        Returns:
            bool: 전환 성공 여부
        """
        try:
            self.exchange_name = exchange.lower()
            verbose = verbose if verbose is not None else (self.logger.level == DEBUG)
            
            if self.exchange_name == 'bithumb':
                self.api = BithumbAPI(verbose=verbose)
            else:
                self.api = UpbitTradingSystem(verbose=verbose)
            
            self.logger.info(f"{self.api.get_exchange_name()}로 전환되었습니다.")
            return True
            
        except Exception as e:
            self.logger.error(f"거래소 전환 중 오류 발생: {str(e)}")
            return False
    
    def get_current_exchange(self) -> str:
        """현재 선택된 거래소 이름 반환"""
        return self.api.get_exchange_name() 