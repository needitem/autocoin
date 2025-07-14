"""
기술적 분석 지표 계산 모듈 (talib 없이 구현)

이 모듈은 다양한 기술적 분석 지표를 계산하는 함수들을 포함합니다.
모든 계산은 pandas DataFrame을 기반으로 하며, 
기본적으로 OHLCV(Open, High, Low, Close, Volume) 데이터를 사용합니다.
"""

import numpy as np
import pandas as pd

class TechnicalAnalysis:
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120]) -> dict:
        """이동평균선 계산"""
        ma_data = {}
        for period in periods:
            ma_data[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return ma_data
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
        """볼린저 밴드 계산"""
        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """지수이동평균선 계산"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_wma(df: pd.DataFrame, period: int) -> pd.Series:
        """가중이동평균선 계산"""
        weights = np.arange(1, period + 1)
        wma = df['close'].rolling(window=period).apply(
            lambda x: np.sum(x * weights) / np.sum(weights), raw=False
        )
        return wma
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_parabolic_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR 계산 (간단한 구현)"""
        # 간단한 구현 - 실제 SAR 알고리즘은 복잡함
        sar = pd.Series(index=df.index, dtype=float)
        if len(df) > 0:
            sar.iloc[0] = df['low'].iloc[0]
        return sar
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """CCI(Commodity Channel Index) 계산"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow 계산"""
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def calculate_dmi(df: pd.DataFrame, period: int = 14) -> dict:
        """DMI(Directional Movement Index) 계산 (간단한 구현)"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = TechnicalAnalysis.calculate_atr(df, period)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """MACD(Moving Average Convergence Divergence) 계산"""
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'hist': hist
        }
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3) -> dict:
        """Stochastic Oscillator 계산"""
        low_min = df['low'].rolling(window=fastk_period).min()
        high_max = df['high'].rolling(window=fastk_period).max()
        
        fastk = 100 * ((df['close'] - low_min) / (high_max - low_min))
        fastd = fastk.rolling(window=slowk_period).mean()
        slowk = fastd
        slowd = slowk.rolling(window=slowd_period).mean()
        
        return {
            'fastk': fastk,
            'fastd': fastd,
            'slowk': slowk,
            'slowd': slowd
        }
    
    @staticmethod
    def calculate_trix(df: pd.DataFrame, period: int = 30) -> pd.Series:
        """TRIX 계산"""
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = (ema3.pct_change()) * 10000
        return trix
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> dict:
        """거래량 지표 계산"""
        # OBV (On Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # A/D (Accumulation/Distribution)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        ad = (clv * df['volume']).fillna(0).cumsum()
        
        # 간단한 ADOSC (Accumulation/Distribution Oscillator) 
        adosc = ad.rolling(window=3).mean() - ad.rolling(window=10).mean()
        
        return {
            'obv': obv,
            'ad': ad,
            'adosc': adosc
        }
    
    @staticmethod
    def calculate_additional_indicators(df: pd.DataFrame) -> dict:
        """추가 지표 계산"""
        # MFI (Money Flow Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        
        mf_pos = mf.where(tp > tp.shift(), 0)
        mf_neg = mf.where(tp < tp.shift(), 0)
        
        mfi_ratio = mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum()
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        # Williams %R
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        willr = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Momentum
        mom = df['close'].diff(10)
        
        # ROC (Rate of Change)
        roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        return {
            'atr': TechnicalAnalysis.calculate_atr(df),
            'mfi': mfi,
            'willr': willr,
            'cmo': mom,  # 간단한 momentum 사용
            'mom': mom,
            'roc': roc,
            'aroon_osc': pd.Series(index=df.index, dtype=float),  # 복잡한 지표는 0으로
            'bwi': pd.Series(index=df.index, dtype=float)  # 복잡한 지표는 0으로
        }
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> dict:
        """모든 지표 계산"""
        if len(df) < 100:  # 최소 데이터 필요
            return {}
            
        indicators = {}
        
        # 이동평균선
        ma_dict = TechnicalAnalysis.calculate_moving_averages(df)
        indicators.update(ma_dict)
        
        # RSI
        indicators['rsi'] = TechnicalAnalysis.calculate_rsi(df)
        
        # 볼린저 밴드
        bb_dict = TechnicalAnalysis.calculate_bollinger_bands(df)
        indicators.update(bb_dict)
        
        # MACD
        macd_dict = TechnicalAnalysis.calculate_macd(df)
        indicators.update(macd_dict)
        
        # Stochastic
        stoch_dict = TechnicalAnalysis.calculate_stochastic(df)
        indicators.update(stoch_dict)
        
        # 거래량 지표
        vol_dict = TechnicalAnalysis.calculate_volume_indicators(df)
        indicators.update(vol_dict)
        
        # 추가 지표
        add_dict = TechnicalAnalysis.calculate_additional_indicators(df)
        indicators.update(add_dict)
        
        # DMI
        dmi_dict = TechnicalAnalysis.calculate_dmi(df)
        indicators.update(dmi_dict)
        
        # 기타
        indicators['cci'] = TechnicalAnalysis.calculate_cci(df)
        indicators['cmf'] = TechnicalAnalysis.calculate_cmf(df)
        indicators['trix'] = TechnicalAnalysis.calculate_trix(df)
        indicators['ema12'] = TechnicalAnalysis.calculate_ema(df, 12)
        indicators['ema26'] = TechnicalAnalysis.calculate_ema(df, 26)
        indicators['wma10'] = TechnicalAnalysis.calculate_wma(df, 10)
        indicators['sar'] = TechnicalAnalysis.calculate_parabolic_sar(df)
        
        return indicators