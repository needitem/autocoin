import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, periods: list = [5, 20, 60]) -> dict:
        """이동평균선 계산"""
        ma_data = {}
        for period in periods:
            ma_data[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return ma_data
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> tuple:
        """MACD 계산"""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20) -> tuple:
        """볼린저 밴드 계산"""
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        return upper, ma, lower
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame) -> float:
        """추세 강도 계산 (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # +DM, -DM
        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        
        # ADX
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        return adx.iloc[-1] 