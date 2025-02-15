import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.base_url = "https://api.upbit.com/v1"
        self.session = requests.Session()
    
    def get_candles(self, symbol: str, interval: str = "minutes/60", count: int = 200) -> Optional[pd.DataFrame]:
        """캔들 데이터 조회"""
        try:
            if not symbol:
                logger.error("심볼이 지정되지 않았습니다.")
                return None
                
            url = f"{self.base_url}/candles/{interval}"
            params = {
                "market": f"KRW-{symbol}",
                "count": count
            }
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"API 요청 실패: {response.status_code}")
                return None
                
            data = response.json()
            if not data:
                logger.error("데이터가 비어있습니다.")
                return None
                
            # 데이터프레임 생성 및 전처리
            df = pd.DataFrame(data)
            
            # 필요한 컬럼 확인
            required_columns = [
                'candle_date_time_kst',
                'opening_price',
                'high_price',
                'low_price',
                'trade_price',
                'candle_acc_trade_volume'
            ]
            
            if not all(col in df.columns for col in required_columns):
                logger.error("필수 컬럼이 누락되었습니다.")
                return None
            
            # 컬럼 표준화
            df = self._standardize_columns(df)
            
            # 시간순 정렬
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 중복 제거
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            if len(df) == 0:
                logger.error("처리된 데이터가 비어있습니다.")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"캔들 데이터를 가져오는데 실패했습니다: {str(e)}")
            return None
    
    def get_orderbook(self, symbol: str) -> dict:
        """호가 데이터 가져오기"""
        try:
            url = f"{self.base_url}/orderbook"
            params = {
                "markets": f"KRW-{symbol}"
            }
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                return None
                
            return response.json()[0]
            
        except Exception as e:
            logger.error(f"호가 데이터 가져오기 실패: {str(e)}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 표준화"""
        column_mapping = {
            'candle_date_time_kst': 'timestamp',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        }
        
        # 필요한 컬럼만 선택
        df = df[list(column_mapping.keys())]
        
        # 컬럼명 변경
        df = df.rename(columns=column_mapping)
        
        return df 