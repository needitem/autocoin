"""
공통 유틸리티 함수들
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """데이터 검증 유틸리티"""
    
    @staticmethod
    def validate_ohlcv(data: pd.DataFrame) -> bool:
        """OHLCV 데이터 유효성 검사"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not isinstance(data, pd.DataFrame):
            return False
            
        if not all(col in data.columns for col in required_columns):
            return False
            
        if data.empty:
            return False
            
        # 가격 데이터 검증
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
            if (data[col] < 0).any():
                return False
                
        # 논리적 검증 (high >= low, high >= open, high >= close 등)
        if (data['high'] < data['low']).any():
            return False
        if (data['high'] < data['open']).any():
            return False
        if (data['high'] < data['close']).any():
            return False
            
        return True
    
    @staticmethod
    def validate_price_data(prices: Union[pd.Series, List, np.ndarray]) -> bool:
        """가격 데이터 유효성 검사"""
        if isinstance(prices, (list, np.ndarray)):
            prices = pd.Series(prices)
            
        if prices.empty:
            return False
            
        if not pd.api.types.is_numeric_dtype(prices):
            return False
            
        if (prices <= 0).any():
            return False
            
        if prices.isnull().any():
            return False
            
        return True

class DataConverter:
    """데이터 변환 유틸리티"""
    
    @staticmethod
    def to_dataframe(data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """다양한 형태의 데이터를 DataFrame으로 변환"""
        try:
            if isinstance(data, dict):
                return pd.DataFrame([data])
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({'value': data})
            else:
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"데이터 변환 실패: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        """OHLCV 데이터 정규화"""
        normalized = data.copy()
        
        # 컬럼명 소문자로 통일
        column_mapping = {
            'Open': 'open', 'HIGH': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'High': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'VOLUME': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in normalized.columns:
                normalized = normalized.rename(columns={old_col: new_col})
        
        # 인덱스가 시간이 아닌 경우 처리
        if 'timestamp' in normalized.columns:
            normalized['timestamp'] = pd.to_datetime(normalized['timestamp'])
            normalized.set_index('timestamp', inplace=True)
        
        return normalized

class ErrorHandler:
    """에러 처리 유틸리티"""
    
    @staticmethod
    def safe_api_call(func, *args, **kwargs):
        """안전한 API 호출 래퍼"""
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"연결 오류: {e}")
            return None
        except TimeoutError as e:
            logger.error(f"타임아웃 오류: {e}")
            return None
        except ValueError as e:
            logger.error(f"값 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return None
    
    @staticmethod
    def log_and_return_fallback(error: Exception, fallback_value=None, context=""):
        """에러 로깅 후 기본값 반환"""
        logger.error(f"{context} 오류: {error}")
        return fallback_value

class MathUtils:
    """수학 계산 유틸리티"""
    
    @staticmethod
    def safe_divide(numerator, denominator, default=0):
        """안전한 나눗셈"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def calculate_percentage_change(old_value, new_value):
        """퍼센트 변화율 계산"""
        if old_value == 0:
            return 0
        return ((new_value - old_value) / old_value) * 100