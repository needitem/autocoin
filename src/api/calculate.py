"""
기술적 분석 지표 계산 모듈

이 모듈은 다양한 기술적 분석 지표를 계산하는 함수들을 포함합니다.
모든 계산은 pandas DataFrame을 기반으로 하며, 
기본적으로 OHLCV(Open, High, Low, Close, Volume) 데이터를 사용합니다.
"""

import numpy as np
import pandas as pd
import talib

class TechnicalAnalysis:
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120]) -> dict:
        """이동평균선 계산"""
        ma_data = {}
        for period in periods:
            ma_data[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return ma_data

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict:
        """볼린저 밴드 계산"""
        df = df.copy()
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = (upper - lower) / middle * 100
        
        return {
            'upper': upper.fillna(method='bfill'),
            'middle': middle.fillna(method='bfill'),
            'lower': lower.fillna(method='bfill'),
            'bandwidth': bandwidth.fillna(method='bfill')
        }

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> dict:
        """일목균형표 계산"""
        high_values = df['high']
        low_values = df['low']
        
        # 전환선 (Conversion Line, Tenkan-sen)
        period9_high = high_values.rolling(window=9).max()
        period9_low = low_values.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # 기준선 (Base Line, Kijun-sen)
        period26_high = high_values.rolling(window=26).max()
        period26_low = low_values.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # 선행스팬1 (Leading Span A, Senkou Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # 선행스팬2 (Leading Span B, Senkou Span B)
        period52_high = high_values.rolling(window=52).max()
        period52_low = low_values.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # 후행스팬 (Lagging Span, Chikou Span)
        chikou_span = df['close'].shift(-26)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> dict:
        """피봇 포인트 계산"""
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = 2 * pivot - df['low']
        r2 = pivot + (df['high'] - df['low'])
        s1 = 2 * pivot - df['high']
        s2 = pivot - (df['high'] - df['low'])
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }

    @staticmethod
    def calculate_parabolic_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR 계산"""
        return talib.SAR(df['high'], df['low'], acceleration, maximum)

    @staticmethod
    def calculate_price_channels(df: pd.DataFrame, period: int = 20) -> dict:
        """가격 채널 계산"""
        df = df.copy()
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {
            'upper': upper.fillna(method='bfill'),
            'middle': middle.fillna(method='bfill'),
            'lower': lower.fillna(method='bfill')
        }

    @staticmethod
    def calculate_disparity(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """이격도 계산"""
        ma = df['close'].rolling(window=period).mean()
        return (df['close'] / ma * 100) - 100

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """CCI(Commodity Channel Index) 계산"""
        return talib.CCI(df['high'], df['low'], df['close'], period)

    @staticmethod
    def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow 계산"""
        return talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    @staticmethod
    def calculate_dmi(df: pd.DataFrame, period: int = 14) -> dict:
        """DMI(Directional Movement Index) 계산"""
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], period)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], period)
        adx = talib.ADX(df['high'], df['low'], df['close'], period)
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx
        }

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """MACD(Moving Average Convergence Divergence) 계산"""
        macd, signal, hist = talib.MACD(df['close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        
        return {
            'macd': macd,
            'signal': signal,
            'hist': hist
        }

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI 계산"""
        df = df.copy()
        close_delta = df['close'].diff()
        
        gain = (close_delta.where(close_delta > 0, 0)).fillna(0)
        loss = (-close_delta.where(close_delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.clip(0, 100).fillna(method='bfill')

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3) -> dict:
        """Stochastic Oscillator 계산"""
        fastk, fastd = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period, slowk_period)
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period, slowk_period, slowd_period)
        
        return {
            'fast': {'k': fastk, 'd': fastd},
            'slow': {'k': slowk, 'd': slowd}
        }

    @staticmethod
    def calculate_trix(df: pd.DataFrame, period: int = 30) -> pd.Series:
        """TRIX 계산"""
        return talib.TRIX(df['close'], period)

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> dict:
        """거래량 관련 지표 계산"""
        try:
            # 데이터 검증
            if not isinstance(df, pd.DataFrame):
                raise ValueError("입력은 DataFrame 형식이어야 합니다")
            
            required_columns = ['close', 'volume', 'high', 'low']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"필수 컬럼이 누락되었습니다: {required_columns}")
            
            if df.empty:
                return {
                    'obv': pd.Series(dtype=float),
                    'ad': pd.Series(dtype=float),
                    'adosc': pd.Series(dtype=float)
                }
            
            # NaN 값 처리
            df = df.copy()
            for col in required_columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 거래량 지표 계산
            obv = talib.OBV(df['close'], df['volume'])
            ad = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            adosc = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
            
            # NaN 값 처리
            obv = obv.fillna(method='ffill').fillna(0)
            ad = ad.fillna(method='ffill').fillna(0)
            adosc = adosc.fillna(method='ffill').fillna(0)
            
            return {
                'obv': obv,
                'ad': ad,
                'adosc': adosc
            }
            
        except Exception as e:
            print(f"거래량 지표 계산 중 오류 발생: {str(e)}")
            return {
                'obv': pd.Series(dtype=float),
                'ad': pd.Series(dtype=float),
                'adosc': pd.Series(dtype=float)
            }

    @staticmethod
    def calculate_psychological_indicators(df: pd.DataFrame, period: int = 20) -> dict:
        """심리도 지표 계산"""
        try:
            # 불필요한 복사 제거
            close_diff = df['close'].diff()
            up_days = (close_diff > 0).astype(float)
            
            # 메모리 효율적인 계산
            new_psychological = up_days.rolling(window=period, min_periods=1).mean() * 100
            investment_psychological = new_psychological.ewm(span=period, min_periods=1, adjust=False).mean()
            
            # NaN 값 처리 개선
            new_psychological = new_psychological.fillna(50)
            investment_psychological = investment_psychological.fillna(50)
            
            return {
                'new_psychological': new_psychological.clip(0, 100),
                'investment_psychological': investment_psychological.clip(0, 100)
            }
        except Exception as e:
            print(f"Error calculating psychological indicators: {e}")
            return {
                'new_psychological': pd.Series(50, index=df.index),
                'investment_psychological': pd.Series(50, index=df.index)
            }

    @staticmethod
    def calculate_additional_indicators(df: pd.DataFrame) -> dict:
        """추가 기술적 지표 계산"""
        df = df.copy()
        return {
            'atr': talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).fillna(method='bfill'),
            'mfi': talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14).fillna(method='bfill'),
            'willr': talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14).fillna(method='bfill'),
            'cmo': talib.CMO(df['close'], timeperiod=14).fillna(method='bfill'),
            'mom': talib.MOM(df['close'], timeperiod=10).fillna(method='bfill'),
            'roc': talib.ROC(df['close'], timeperiod=10).fillna(method='bfill'),
            'aroon_osc': talib.AROONOSC(df['high'], df['low'], timeperiod=14).fillna(method='bfill'),
            'bwi': talib.BOP(df['open'], df['high'], df['low'], df['close']).fillna(method='bfill')
        }

    @staticmethod
    def calculate_elder_ray_index(df: pd.DataFrame, period: int = 13) -> dict:
        """Elder Ray Index 계산"""
        ema = df['close'].ewm(span=period, adjust=False).mean()
        bull_power = df['high'] - ema
        bear_power = df['low'] - ema
        
        return {
            'ema': ema,
            'bull_power': bull_power,
            'bear_power': bear_power
        }

    @staticmethod
    def calculate_mass_index(df: pd.DataFrame, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
        """Mass Index 계산"""
        high_low = df['high'] - df['low']
        ema1 = high_low.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        ema_ratio = ema1 / ema2
        mass_index = ema_ratio.rolling(window=sum_period).sum()
        return mass_index

    @staticmethod
    def calculate_nco(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Net Change Oscillator (NCO) 계산"""
        price_change = df['close'].diff(period)
        abs_price_change = abs(df['close'].diff(period))
        nco = (price_change / abs_price_change) * 100
        return nco

    @staticmethod
    def calculate_nvi(df: pd.DataFrame) -> pd.Series:
        """Negative Volume Index (NVI) 계산"""
        nvi = pd.Series(100.0, index=df.index)
        for i in range(1, len(df)):
            if df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        return nvi

    @staticmethod
    def calculate_pvi(df: pd.DataFrame) -> pd.Series:
        """Positive Volume Index (PVI) 계산"""
        pvi = pd.Series(100.0, index=df.index)
        for i in range(1, len(df)):
            if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        return pvi

    @staticmethod
    def calculate_rmi(df: pd.DataFrame, period: int = 20, momentum_period: int = 5) -> pd.Series:
        """Relative Momentum Index (RMI) 계산"""
        momentum = df['close'].diff(momentum_period)
        up_momentum = momentum.where(momentum > 0, 0).rolling(window=period).mean()
        down_momentum = -momentum.where(momentum < 0, 0).rolling(window=period).mean()
        rmi = 100 - (100 / (1 + up_momentum / down_momentum))
        return rmi

    @staticmethod
    def calculate_rvi(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Relative Volatility Index 계산"""
        df = df.copy()
        close_diff = df['close'].diff()
        std = df['close'].rolling(window=period).std()
        
        up_vol = std.where(close_diff > 0, 0)
        down_vol = std.where(close_diff < 0, 0)
        
        up_avg = up_vol.rolling(window=period).mean()
        down_avg = down_vol.rolling(window=period).mean()
        
        rvi = 100 * up_avg / (up_avg + down_avg)
        
        return rvi.clip(0, 100).fillna(method='bfill')

    @staticmethod
    def calculate_sonar(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict:
        """SONAR 지표 계산"""
        df = df.copy()
        sonar = df['close'].ewm(span=period).mean()
        signal = sonar.ewm(span=period//2).mean()
        
        std = df['close'].rolling(window=period).std()
        upper = sonar + (std * std_dev)
        lower = sonar - (std * std_dev)
        
        return {
            'sonar': sonar.fillna(method='bfill'),
            'signal': signal.fillna(method='bfill'),
            'upper': upper.fillna(method='bfill'),
            'lower': lower.fillna(method='bfill')
        }

    @staticmethod
    def calculate_stochastic_momentum(df: pd.DataFrame, k_period: int = 10, d_period: int = 3, smooth_period: int = 3) -> dict:
        """Stochastic Momentum Index 계산"""
        highest_high = df['high'].rolling(window=k_period).max()
        lowest_low = df['low'].rolling(window=k_period).min()
        
        distance = df['close'] - (highest_high + lowest_low) / 2
        range_sum = (highest_high - lowest_low) / 2
        
        smi = 100 * (distance / range_sum).rolling(window=smooth_period).mean()
        signal = smi.rolling(window=d_period).mean()
        
        return {
            'smi': smi,
            'signal': signal
        }

    @staticmethod
    def calculate_stochastic_rsi(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> dict:
        """Stochastic RSI 계산"""
        rsi = TechnicalAnalysis.calculate_rsi(df, period)
        stoch_rsi = pd.Series(index=df.index)
        
        for i in range(period, len(rsi)):
            rsi_window = rsi[i-period+1:i+1]
            if not rsi_window.isna().all():
                highest_rsi = rsi_window.max()
                lowest_rsi = rsi_window.min()
                if highest_rsi != lowest_rsi:
                    stoch_rsi[i] = (rsi[i] - lowest_rsi) / (highest_rsi - lowest_rsi)
                else:
                    stoch_rsi[i] = 0
        
        k = stoch_rsi.rolling(window=smooth_k).mean() * 100
        d = k.rolling(window=smooth_d).mean()
        
        return {
            'k': k,
            'd': d
        }

    @staticmethod
    def calculate_vr(df: pd.DataFrame, period: int = 26) -> pd.Series:
        """Volume Ratio (VR) 계산"""
        close_change = df['close'].diff()
        up_volume = df['volume'].where(close_change > 0, 0)
        down_volume = df['volume'].where(close_change < 0, 0)
        
        up_sum = up_volume.rolling(window=period).sum()
        down_sum = down_volume.rolling(window=period).sum()
        
        vr = 100 * up_sum / down_sum
        return vr

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> dict:
        """모든 기술적 지표 계산"""
        indicators = {
            'moving_averages': TechnicalAnalysis.calculate_moving_averages(df),
            'bollinger_bands': TechnicalAnalysis.calculate_bollinger_bands(df),
            'ichimoku': TechnicalAnalysis.calculate_ichimoku(df),
            'pivot_points': TechnicalAnalysis.calculate_pivot_points(df),
            'parabolic_sar': TechnicalAnalysis.calculate_parabolic_sar(df),
            'price_channels': TechnicalAnalysis.calculate_price_channels(df),
            'disparity': TechnicalAnalysis.calculate_disparity(df),
            'cci': TechnicalAnalysis.calculate_cci(df),
            'cmf': TechnicalAnalysis.calculate_cmf(df),
            'dmi': TechnicalAnalysis.calculate_dmi(df),
            'macd': TechnicalAnalysis.calculate_macd(df),
            'rsi': TechnicalAnalysis.calculate_rsi(df),
            'stochastic': TechnicalAnalysis.calculate_stochastic(df),
            'trix': TechnicalAnalysis.calculate_trix(df),
            'volume': TechnicalAnalysis.calculate_volume_indicators(df),
            'psychological': TechnicalAnalysis.calculate_psychological_indicators(df),
            'additional': TechnicalAnalysis.calculate_additional_indicators(df),
            'elder_ray': TechnicalAnalysis.calculate_elder_ray_index(df),
            'mass_index': TechnicalAnalysis.calculate_mass_index(df),
            'nco': TechnicalAnalysis.calculate_nco(df),
            'nvi': TechnicalAnalysis.calculate_nvi(df),
            'pvi': TechnicalAnalysis.calculate_pvi(df),
            'rmi': TechnicalAnalysis.calculate_rmi(df),
            'rvi': TechnicalAnalysis.calculate_rvi(df),
            'sonar': TechnicalAnalysis.calculate_sonar(df),
            'stochastic_momentum': TechnicalAnalysis.calculate_stochastic_momentum(df),
            'stochastic_rsi': TechnicalAnalysis.calculate_stochastic_rsi(df),
            'vr': TechnicalAnalysis.calculate_vr(df)
        }
        return indicators 