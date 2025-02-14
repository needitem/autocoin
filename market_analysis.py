import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from database import Database
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        """시장 분석기 초기화"""
        self.db = Database()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
    def get_candles(self, symbol: str, interval: str = 'days', count: int = 200) -> Optional[pd.DataFrame]:
        """캔들 데이터 가져오기"""
        try:
            url = f"https://api.upbit.com/v1/candles/{interval}"
            params = {
                "market": f"KRW-{symbol}",
                "count": count
            }
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error("캔들 데이터를 가져오는데 실패했습니다.")
                return None
                
            data = response.json()
            df = pd.DataFrame(data)
            df['trade_date'] = pd.to_datetime(df['candle_date_time_kst'])
            df = df.sort_values('trade_date')
            
            # 기술적 지표 계산
            self._calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 가져오기 실패: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame):
        """기술적 지표 계산"""
        try:
            # 기본 OHLCV 데이터 준비
            df['open'] = df['opening_price']
            df['high'] = df['high_price']
            df['low'] = df['low_price']
            df['close'] = df['trade_price']
            df['volume'] = df['candle_acc_trade_volume']
            
            # 이동평균선
            for period in [5, 10, 20, 60, 120]:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
                
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            period = 20
            df['BB_middle'] = df['close'].rolling(window=period).mean()
            df['BB_std'] = df['close'].rolling(window=period).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # 스토캐스틱
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['STOCH_K'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
            df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
            
            # ATR (Average True Range)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
            
            # OBV (On Balance Volume)
            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # ADX (Average Directional Index)
            # +DM
            up_move = df['high'].diff()
            down_move = df['low'].diff()
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            tr = pd.DataFrame({
                'tr1': tr1,
                'tr2': tr2,
                'tr3': tr3
            }).max(axis=1)
            
            period = 14
            smoothed_tr = tr.rolling(window=period).mean()
            smoothed_pos_dm = pd.Series(pos_dm).rolling(window=period).mean()
            smoothed_neg_dm = pd.Series(neg_dm).rolling(window=period).mean()
            
            pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
            neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
            
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            df['ADX'] = dx.rolling(window=period).mean()
            
            # 이격도
            for period in [5, 20, 60]:
                df[f'Disparity{period}'] = (df['close'] / df[f'MA{period}'] - 1) * 100
                
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")

    def detect_market_manipulation(self, df: pd.DataFrame) -> Dict:
        """시장 조작 감지"""
        try:
            # 이상치 탐지를 위한 특성 선택
            features = ['volume', 'ATR', 'OBV']
            X = df[features].fillna(method='ffill')
            
            # 데이터 정규화
            X_scaled = self.scaler.fit_transform(X)
            
            # 이상치 탐지
            anomalies = self.anomaly_detector.fit_predict(X_scaled)
            anomaly_indices = np.where(anomalies == -1)[0]
            
            # 이상치 분석
            anomaly_details = []
            if len(anomaly_indices) > 0:
                for idx in anomaly_indices:
                    anomaly_details.append({
                        'timestamp': df['trade_date'].iloc[idx],
                        'price': df['close'].iloc[idx],
                        'volume': df['volume'].iloc[idx],
                        'atr': df['ATR'].iloc[idx],
                        'obv': df['OBV'].iloc[idx]
                    })
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_ratio': len(anomaly_indices) / len(df),
                'anomaly_details': anomaly_details
            }
            
        except Exception as e:
            logger.error(f"시장 조작 감지 실패: {e}")
            return {'error': str(e)}

    def analyze_market_trend(self, df: pd.DataFrame) -> Dict:
        """시장 추세 분석"""
        try:
            current_price = df['close'].iloc[-1]
            
            # 이동평균선 기반 추세 분석
            ma_trend = self._analyze_ma_trend(df)
            
            # MACD 분석
            macd_analysis = self._analyze_macd(df)
            
            # RSI 분석
            rsi = df['RSI'].iloc[-1]
            rsi_signal = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
            
            # 볼린저 밴드 분석
            bb_analysis = self._analyze_bollinger_bands(df)
            
            # 거래량 분석
            volume_analysis = self._analyze_volume(df)
            
            # 추세 강도 분석
            adx = df['ADX'].iloc[-1]
            trend_strength = "강함" if adx > 25 else "약함"
            
            return {
                'current_price': current_price,
                'ma_trend': ma_trend,
                'macd_analysis': macd_analysis,
                'rsi': {
                    'value': rsi,
                    'signal': rsi_signal
                },
                'bollinger_bands': bb_analysis,
                'volume_analysis': volume_analysis,
                'trend_strength': {
                    'adx': adx,
                    'interpretation': trend_strength
                }
            }
            
        except Exception as e:
            logger.error(f"시장 추세 분석 실패: {e}")
            return {'error': str(e)}

    def _analyze_ma_trend(self, df: pd.DataFrame) -> Dict:
        """이동평균선 추세 분석"""
        try:
            current_price = df['close'].iloc[-1]
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma60 = df['MA60'].iloc[-1]
            
            # 이동평균선 배열 확인
            if ma5 > ma20 > ma60:
                trend = "강세"
                strength = "상승추세"
            elif ma5 < ma20 < ma60:
                trend = "약세"
                strength = "하락추세"
            else:
                trend = "중립"
                strength = "횡보추세"
                
            # 이동평균선 정배열/역배열 강도
            ma_strength = abs((ma5 - ma60) / ma60 * 100)
            
            return {
                'trend': trend,
                'strength': strength,
                'ma_strength': round(ma_strength, 2),
                'price_vs_ma5': round((current_price / ma5 - 1) * 100, 2),
                'price_vs_ma20': round((current_price / ma20 - 1) * 100, 2),
                'price_vs_ma60': round((current_price / ma60 - 1) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"이동평균선 추세 분석 실패: {e}")
            return {'error': str(e)}

    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        """MACD 분석"""
        try:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_signal'].iloc[-1]
            hist = df['MACD_hist'].iloc[-1]
            
            # MACD 신호 해석
            if hist > 0 and hist > df['MACD_hist'].iloc[-2]:
                signal_type = "강한 매수"
            elif hist > 0:
                signal_type = "매수"
            elif hist < 0 and hist < df['MACD_hist'].iloc[-2]:
                signal_type = "강한 매도"
            else:
                signal_type = "매도"
                
            return {
                'macd': round(macd, 2),
                'signal': round(signal, 2),
                'histogram': round(hist, 2),
                'interpretation': signal_type
            }
            
        except Exception as e:
            logger.error(f"MACD 분석 실패: {e}")
            return {'error': str(e)}

    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """볼린저 밴드 분석"""
        try:
            current_price = df['close'].iloc[-1]
            upper = df['BB_upper'].iloc[-1]
            middle = df['BB_middle'].iloc[-1]
            lower = df['BB_lower'].iloc[-1]
            
            # 밴드 내 위치
            band_position = (current_price - lower) / (upper - lower) * 100
            
            # 신호 해석
            if current_price > upper:
                signal = "과매수"
            elif current_price < lower:
                signal = "과매도"
            else:
                signal = "중립"
                
            return {
                'upper': round(upper, 2),
                'middle': round(middle, 2),
                'lower': round(lower, 2),
                'band_position': round(band_position, 2),
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"볼린저 밴드 분석 실패: {e}")
            return {'error': str(e)}

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """거래량 분석"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            # 거래량 증감 추세
            volume_trend = "중립"
            if current_volume > avg_volume * 2:
                volume_trend = "매우 높음"
            elif current_volume > avg_volume * 1.5:
                volume_trend = "높음"
            elif current_volume < avg_volume * 0.5:
                volume_trend = "매우 낮음"
            elif current_volume < avg_volume * 0.8:
                volume_trend = "낮음"
                
            # OBV 분석
            obv = df['OBV'].iloc[-1]
            obv_sma = df['OBV'].rolling(window=20).mean().iloc[-1]
            obv_trend = "상승" if obv > obv_sma else "하락"
            
            return {
                'current_volume': round(current_volume, 2),
                'average_volume': round(avg_volume, 2),
                'volume_trend': volume_trend,
                'volume_ratio': round(current_volume / avg_volume * 100, 2),
                'obv': {
                    'current': round(obv, 2),
                    'trend': obv_trend
                }
            }
            
        except Exception as e:
            logger.error(f"거래량 분석 실패: {e}")
            return {'error': str(e)}

    def analyze_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """지지/저항 레벨 분석"""
        try:
            # 최근 데이터에서 피봇 포인트 찾기
            recent_df = df.tail(window)
            pivot = {
                'high': recent_df['high'].max(),
                'low': recent_df['low'].min(),
                'close': recent_df['close'].iloc[-1]
            }
            
            # 피봇 포인트 기반 지지/저항 계산
            pp = (pivot['high'] + pivot['low'] + pivot['close']) / 3
            r1 = 2 * pp - pivot['low']  # 1차 저항
            r2 = pp + (pivot['high'] - pivot['low'])  # 2차 저항
            s1 = 2 * pp - pivot['high']  # 1차 지지
            s2 = pp - (pivot['high'] - pivot['low'])  # 2차 지지
            
            current_price = df['close'].iloc[-1]
            
            # 현재가 기준 레벨 해석
            levels = [s2, s1, pp, r1, r2]
            current_level = "Unknown"
            next_resistance = None
            next_support = None
            
            for i, level in enumerate(levels):
                if current_price < level:
                    next_resistance = level
                    if i > 0:
                        next_support = levels[i-1]
                    break
                elif i == len(levels) - 1:
                    next_support = level
            
            return {
                'pivot_point': round(pp, 2),
                'resistance': {
                    'r1': round(r1, 2),
                    'r2': round(r2, 2)
                },
                'support': {
                    's1': round(s1, 2),
                    's2': round(s2, 2)
                },
                'current_price': round(current_price, 2),
                'next_resistance': round(next_resistance, 2) if next_resistance else None,
                'next_support': round(next_support, 2) if next_support else None
            }
            
        except Exception as e:
            logger.error(f"지지/저항 레벨 분석 실패: {e}")
            return {'error': str(e)}

def get_candles(symbol, interval='days', count=200):
    """
    캔들 데이터를 가져옵니다.
    interval: minutes1, minutes3, minutes5, minutes15, minutes30, minutes60, minutes240, days, weeks, months
    """
    try:
        url = f"https://api.upbit.com/v1/candles/{interval}"
        params = {
            "market": f"KRW-{symbol}",
            "count": count
        }
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print("캔들 데이터를 가져오는데 실패했습니다.")
            return None
            
        data = response.json()
        df = pd.DataFrame(data)
        df['trade_date'] = pd.to_datetime(df['candle_date_time_kst'])
        df = df.sort_values('trade_date')
        return df
        
    except Exception as e:
        print(f"데이터 가져오기 실패: {e}")
        return None

def get_orderbook(symbol):
    """
    호가 데이터를 가져옵니다.
    """
    try:
        url = "https://api.upbit.com/v1/orderbook"
        params = {
            "markets": f"KRW-{symbol}"
        }
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return None
            
        return response.json()[0]
        
    except Exception as e:
        print(f"호가 데이터 가져오기 실패: {e}")
        return None

def analyze_trend(df):
    """
    추세를 분석합니다.
    """
    if df is None or len(df) < 20:
        return None
        
    # 이동평균선 계산
    df['MA5'] = df['trade_price'].rolling(window=5).mean()
    df['MA20'] = df['trade_price'].rolling(window=20).mean()
    df['MA60'] = df['trade_price'].rolling(window=60).mean()
    
    current_price = df['trade_price'].iloc[-1]
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    
    # 이동평균선 배열 확인
    trend = "중립"
    if ma5 > ma20 > ma60:
        trend = "상승"
    elif ma5 < ma20 < ma60:
        trend = "하락"
        
    # 이동평균선 정배열/역배열 강도
    ma_strength = abs((ma5 - ma60) / ma60 * 100)
    
    return {
        'trend': trend,
        'strength': round(ma_strength, 2),
        'price_vs_ma5': round((current_price / ma5 - 1) * 100, 2),
        'price_vs_ma20': round((current_price / ma20 - 1) * 100, 2),
        'price_vs_ma60': round((current_price / ma60 - 1) * 100, 2)
    }

def analyze_volume_profile(df):
    """
    거래량 프로파일을 분석합니다.
    """
    if df is None or len(df) < 20:
        return None
        
    # 최근 20일 평균 거래량
    avg_volume = df['candle_acc_trade_volume'].tail(20).mean()
    current_volume = df['candle_acc_trade_volume'].iloc[-1]
    
    # 거래량 증감 추세
    volume_trend = "중립"
    if current_volume > avg_volume * 1.5:
        volume_trend = "매우 높음"
    elif current_volume > avg_volume * 1.2:
        volume_trend = "높음"
    elif current_volume < avg_volume * 0.8:
        volume_trend = "낮음"
    elif current_volume < avg_volume * 0.5:
        volume_trend = "매우 낮음"
        
    return {
        'trend': volume_trend,
        'current': current_volume,
        'average': avg_volume,
        'ratio': round(current_volume / avg_volume * 100, 2)
    }

def analyze_orderbook(orderbook):
    """
    호가창을 분석합니다.
    """
    if orderbook is None:
        return None
        
    # 매수/매도 총량 계산
    total_bid_size = sum(item['bid_size'] for item in orderbook['orderbook_units'])
    total_ask_size = sum(item['ask_size'] for item in orderbook['orderbook_units'])
    
    # 호가 비율
    ratio = total_bid_size / total_ask_size if total_ask_size > 0 else 0
    
    # 매수/매도 세력 판단
    pressure = "중립"
    if ratio > 1.5:
        pressure = "강한 매수세"
    elif ratio > 1.2:
        pressure = "매수세"
    elif ratio < 0.67:
        pressure = "강한 매도세"
    elif ratio < 0.83:
        pressure = "매도세"
    
    # 호가 구간별 분석
    current_price = orderbook['orderbook_units'][0]['ask_price']  # 현재가 기준
    
    # 매도 호가 구간별 분석 (1%, 2%, 3% 단위)
    ask_walls = []
    cumulative_ask = 0
    for unit in orderbook['orderbook_units']:
        price_diff = (unit['ask_price'] - current_price) / current_price * 100
        cumulative_ask += unit['ask_size']
        if unit['ask_size'] > total_ask_size * 0.1:  # 매도벽 기준: 전체의 10% 이상
            ask_walls.append({
                'price': unit['ask_price'],
                'size': unit['ask_size'],
                'diff': price_diff
            })
    
    # 매수 호가 구간별 분석
    bid_walls = []
    cumulative_bid = 0
    for unit in orderbook['orderbook_units']:
        price_diff = (current_price - unit['bid_price']) / current_price * 100
        cumulative_bid += unit['bid_size']
        if unit['bid_size'] > total_bid_size * 0.1:  # 매수벽 기준: 전체의 10% 이상
            bid_walls.append({
                'price': unit['bid_price'],
                'size': unit['bid_size'],
                'diff': price_diff
            })
    
    # 구간별 누적 비율 계산
    zones = {
        'ask_1p': sum(u['ask_size'] for u in orderbook['orderbook_units'] if (u['ask_price'] - current_price) / current_price <= 0.01),
        'ask_2p': sum(u['ask_size'] for u in orderbook['orderbook_units'] if (u['ask_price'] - current_price) / current_price <= 0.02),
        'ask_3p': sum(u['ask_size'] for u in orderbook['orderbook_units'] if (u['ask_price'] - current_price) / current_price <= 0.03),
        'bid_1p': sum(u['bid_size'] for u in orderbook['orderbook_units'] if (current_price - u['bid_price']) / current_price <= 0.01),
        'bid_2p': sum(u['bid_size'] for u in orderbook['orderbook_units'] if (current_price - u['bid_price']) / current_price <= 0.02),
        'bid_3p': sum(u['bid_size'] for u in orderbook['orderbook_units'] if (current_price - u['bid_price']) / current_price <= 0.03),
    }
        
    return {
        'pressure': pressure,
        'bid_total': round(total_bid_size, 2),
        'ask_total': round(total_ask_size, 2),
        'ratio': round(ratio, 2),
        'current_price': current_price,
        'ask_walls': ask_walls,
        'bid_walls': bid_walls,
        'zones': zones
    }

def calculate_rsi(df, period=14):
    """
    RSI(Relative Strength Index) 계산
    """
    delta = df['trade_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    """
    MACD(Moving Average Convergence Divergence) 계산
    """
    exp1 = df['trade_price'].ewm(span=12, adjust=False).mean()
    exp2 = df['trade_price'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(df, period=20):
    """
    볼린저 밴드 계산
    """
    ma = df['trade_price'].rolling(window=period).mean()
    std = df['trade_price'].rolling(window=period).std()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return upper, ma, lower

def analyze_candlestick_patterns(df):
    """
    캔들 패턴 분석
    """
    patterns = []
    
    # 최근 3개 캔들 데이터
    recent = df.tail(3)
    
    # 장대 양봉/음봉 판단
    last_candle = recent.iloc[-1]
    price_change = last_candle['trade_price'] - last_candle['opening_price']
    body_ratio = abs(price_change) / (last_candle['high_price'] - last_candle['low_price'])
    
    if body_ratio > 0.7:  # 실체부가 70% 이상
        if price_change > 0:
            patterns.append({"pattern": "장대 양봉", "strength": "강세"})
        else:
            patterns.append({"pattern": "장대 음봉", "strength": "약세"})
    
    # 망치형/역망치형 패턴
    if last_candle['trade_price'] > last_candle['opening_price']:  # 양봉
        shadow_ratio = (last_candle['high_price'] - last_candle['trade_price']) / (last_candle['trade_price'] - last_candle['low_price'])
        if shadow_ratio < 0.3:
            patterns.append({"pattern": "망치형", "strength": "강세"})
        elif shadow_ratio > 3:
            patterns.append({"pattern": "역망치형", "strength": "약세"})
    
    # 도지 패턴
    body = abs(last_candle['trade_price'] - last_candle['opening_price'])
    total_range = last_candle['high_price'] - last_candle['low_price']
    if body / total_range < 0.1:
        patterns.append({"pattern": "도지", "strength": "중립"})
    
    return patterns

def analyze_chart(df):
    """
    차트를 종합적으로 분석합니다.
    """
    if df is None or len(df) < 60:
        return None
        
    current_price = df['trade_price'].iloc[-1]
    
    # RSI 계산
    rsi = calculate_rsi(df)
    current_rsi = rsi.iloc[-1]
    
    # MACD 계산
    macd, signal = calculate_macd(df)
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    macd_cross = (macd.iloc[-2] - signal.iloc[-2]) * (current_macd - current_signal) < 0
    
    # 볼린저 밴드 계산
    upper, ma, lower = calculate_bollinger_bands(df)
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    
    # 이동평균선 계산 (기존 코드 활용)
    df['MA5'] = df['trade_price'].rolling(window=5).mean()
    df['MA20'] = df['trade_price'].rolling(window=20).mean()
    df['MA60'] = df['trade_price'].rolling(window=60).mean()
    
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    
    # 캔들 패턴 분석
    patterns = analyze_candlestick_patterns(df)
    
    return {
        'current_price': current_price,
        'rsi': {
            'value': round(current_rsi, 2),
            'state': "과매수" if current_rsi > 70 else "과매도" if current_rsi < 30 else "중립"
        },
        'macd': {
            'value': round(current_macd, 2),
            'signal': round(current_signal, 2),
            'cross': macd_cross,
            'direction': "상향" if current_macd > current_signal else "하향"
        },
        'bollinger': {
            'upper': round(current_upper, 2),
            'lower': round(current_lower, 2),
            'position': "상단" if current_price >= current_upper else "하단" if current_price <= current_lower else "중간"
        },
        'moving_averages': {
            'ma5': round(ma5, 2),
            'ma20': round(ma20, 2),
            'ma60': round(ma60, 2),
            'trend': "상승" if ma5 > ma20 > ma60 else "하락" if ma5 < ma20 < ma60 else "중립"
        },
        'patterns': patterns
    }

def analyze_market_manipulation(orderbook, df):
    """
    시장 조작 가능성을 분석합니다.
    """
    if orderbook is None or df is None:
        return None
        
    suspicious_patterns = []
    
    # 호가창 깊이 분석
    current_price = orderbook['orderbook_units'][0]['ask_price']
    
    # 매도 호가 분석
    ask_distribution = []
    for unit in orderbook['orderbook_units']:
        price_diff = (unit['ask_price'] - current_price) / current_price * 100
        ask_distribution.append({
            'price': unit['ask_price'],
            'size': unit['ask_size'],
            'diff': price_diff
        })
    
    # 매수 호가 분석
    bid_distribution = []
    for unit in orderbook['orderbook_units']:
        price_diff = (current_price - unit['bid_price']) / current_price * 100
        bid_distribution.append({
            'price': unit['bid_price'],
            'size': unit['bid_size'],
            'diff': price_diff
        })
    
    # 가짜 매물 의심 패턴 분석
    total_ask = sum(item['ask_size'] for item in orderbook['orderbook_units'])
    total_bid = sum(item['bid_size'] for item in orderbook['orderbook_units'])
    
    # 1. 비정상적으로 큰 단일 호가
    for ask in ask_distribution:
        if ask['size'] > total_ask * 0.3:  # 전체의 30% 이상
            suspicious_patterns.append({
                'type': '대량 매도물량',
                'price': ask['price'],
                'size': ask['size'],
                'confidence': 'medium'
            })
            
    for bid in bid_distribution:
        if bid['size'] > total_bid * 0.3:  # 전체의 30% 이상
            suspicious_patterns.append({
                'type': '대량 매수물량',
                'price': bid['price'],
                'size': bid['size'],
                'confidence': 'medium'
            })
    
    # 2. 호가창 불균형 분석
    ask_std = np.std([ask['size'] for ask in ask_distribution])
    bid_std = np.std([bid['size'] for bid in bid_distribution])
    ask_mean = np.mean([ask['size'] for ask in ask_distribution])
    bid_mean = np.mean([bid['size'] for bid in bid_distribution])
    
    if ask_std > ask_mean * 2:  # 매도 호가의 편차가 매우 큰 경우
        suspicious_patterns.append({
            'type': '매도호가 불균형',
            'details': '특정 구간에 매물 쏠림',
            'confidence': 'high'
        })
        
    if bid_std > bid_mean * 2:  # 매수 호가의 편차가 매우 큰 경우
        suspicious_patterns.append({
            'type': '매수호가 불균형',
            'details': '특정 구간에 매물 쏠림',
            'confidence': 'high'
        })
    
    # 3. 거래량 패턴 분석
    recent_volumes = df['candle_acc_trade_volume'].tail(10)
    avg_volume = recent_volumes.mean()
    max_volume = recent_volumes.max()
    
    if max_volume > avg_volume * 3:  # 최근 거래량이 평균의 3배 이상
        suspicious_patterns.append({
            'type': '비정상 거래량',
            'details': '단기간 거래량 급증',
            'confidence': 'high'
        })
    
    # 4. 가격 변동 패턴 분석
    recent_prices = df['trade_price'].tail(10)
    price_changes = recent_prices.pct_change()
    
    if abs(price_changes.iloc[-1]) > 0.05:  # 5% 이상의 급격한 가격 변동
        suspicious_patterns.append({
            'type': '급격한 가격 변동',
            'details': f"{abs(price_changes.iloc[-1]*100):.1f}% 변동",
            'confidence': 'medium'
        })
    
    return {
        'patterns': suspicious_patterns,
        'ask_distribution': ask_distribution,
        'bid_distribution': bid_distribution
    }

def print_market_analysis(symbol):
    """
    시장 분석 결과를 출력합니다.
    """
    analyzer = MarketAnalyzer()
    
    print(f"\n=== {symbol} 시장 분석 ===")
    print(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 일봉 데이터 분석
    daily_df = get_candles(symbol, 'days', 60)
    if daily_df is not None:
        # 차트 분석
        chart_analysis = analyze_chart(daily_df)
        if chart_analysis:
            print("\n[차트 분석]")
            print(f"현재가: {chart_analysis['current_price']:,.2f}원")
            
            # RSI
            print(f"\nRSI(14): {chart_analysis['rsi']['value']:.2f} ({chart_analysis['rsi']['state']})")
            
            # MACD
            print(f"\nMACD:")
            print(f"  MACD: {chart_analysis['macd']['value']:.2f}")
            print(f"  Signal: {chart_analysis['macd']['signal']:.2f}")
            print(f"  방향: {chart_analysis['macd']['direction']}")
            if chart_analysis['macd']['cross']:
                print("  ⚠️ MACD 골든/데드 크로스 임박")
            
            # 볼린저 밴드
            print(f"\n볼린저 밴드:")
            print(f"  상단: {chart_analysis['bollinger']['upper']:,.2f}")
            print(f"  하단: {chart_analysis['bollinger']['lower']:,.2f}")
            print(f"  현재 위치: {chart_analysis['bollinger']['position']}")
            
            # 이동평균선
            print(f"\n이동평균선:")
            print(f"  5일선: {chart_analysis['moving_averages']['ma5']:,.2f}")
            print(f"  20일선: {chart_analysis['moving_averages']['ma20']:,.2f}")
            print(f"  60일선: {chart_analysis['moving_averages']['ma60']:,.2f}")
            print(f"  추세: {chart_analysis['moving_averages']['trend']}")
            
            # 캔들 패턴
            if chart_analysis['patterns']:
                print("\n캔들 패턴:")
                for pattern in chart_analysis['patterns']:
                    print(f"  {pattern['pattern']} ({pattern['strength']})")
        
        # 거래량 분석
        volume_analysis = analyze_volume_profile(daily_df)
        if volume_analysis:
            print("\n[거래량 분석]")
            print(f"거래량 추세: {volume_analysis['trend']}")
            print(f"현재 거래량: {volume_analysis['current']:,.0f}")
            print(f"20일 평균 거래량: {volume_analysis['average']:,.0f}")
            print(f"평균 대비: {volume_analysis['ratio']}%")
    
    # 호가 분석
    orderbook = get_orderbook(symbol)
    if orderbook:
        order_analysis = analyze_orderbook(orderbook)
        manipulation_analysis = analyze_market_manipulation(orderbook, daily_df)
        
        if order_analysis:
            print("\n[호가 분석]")
            print(f"현재가: {order_analysis['current_price']:,.2f}원")
            print(f"매수/매도 압력: {order_analysis['pressure']}")
            print(f"총 매수잔량: {order_analysis['bid_total']:,.2f}")
            print(f"총 매도잔량: {order_analysis['ask_total']:,.2f}")
            print(f"매수/매도 비율: {order_analysis['ratio']:.2f}")
            
            print("\n[구간별 호가 분석]")
            print("매도 호가:")
            print(f"  1% 이내: {order_analysis['zones']['ask_1p']:,.2f}")
            print(f"  2% 이내: {order_analysis['zones']['ask_2p']:,.2f}")
            print(f"  3% 이내: {order_analysis['zones']['ask_3p']:,.2f}")
            print("매수 호가:")
            print(f"  1% 이내: {order_analysis['zones']['bid_1p']:,.2f}")
            print(f"  2% 이내: {order_analysis['zones']['bid_2p']:,.2f}")
            print(f"  3% 이내: {order_analysis['zones']['bid_3p']:,.2f}")
            
            if order_analysis['ask_walls']:
                print("\n[주요 매도벽]")
                for wall in order_analysis['ask_walls']:
                    print(f"  {wall['price']:,.2f}원 (+{wall['diff']:.2f}%): {wall['size']:,.2f}")
                    
            if order_analysis['bid_walls']:
                print("\n[주요 매수벽]")
                for wall in order_analysis['bid_walls']:
                    print(f"  {wall['price']:,.2f}원 (-{wall['diff']:.2f}%): {wall['size']:,.2f}")
        
        if manipulation_analysis and manipulation_analysis['patterns']:
            print("\n[세력 및 이상 징후 분석]")
            for pattern in manipulation_analysis['patterns']:
                confidence_emoji = "🔴" if pattern['confidence'] == 'high' else "🟡"
                if 'price' in pattern:
                    print(f"{confidence_emoji} {pattern['type']}: {pattern['price']:,.2f}원에 {pattern['size']:,.2f} {symbol}")
                else:
                    print(f"{confidence_emoji} {pattern['type']}: {pattern['details']}")
            
            # 호가 분포 분석
            ask_sizes = [ask['size'] for ask in manipulation_analysis['ask_distribution']]
            bid_sizes = [bid['size'] for bid in manipulation_analysis['bid_distribution']]
            
            print("\n[호가 분포 분석]")
            print("매도 호가 편차:", f"{np.std(ask_sizes):,.2f}")
            print("매수 호가 편차:", f"{np.std(bid_sizes):,.2f}")
            
            # 가짜 매물 가능성이 있는 구간 표시
            print("\n[주의 구간]")
            for ask in manipulation_analysis['ask_distribution']:
                if ask['size'] > np.mean(ask_sizes) * 2:
                    print(f"⚠️ {ask['price']:,.2f}원 매도물량 집중: {ask['size']:,.2f}")
            for bid in manipulation_analysis['bid_distribution']:
                if bid['size'] > np.mean(bid_sizes) * 2:
                    print(f"⚠️ {bid['price']:,.2f}원 매수물량 집중: {bid['size']:,.2f}")
    
    print("\n[투자 전략 제안]")
    if daily_df is not None and chart_analysis and volume_analysis and order_analysis:
        # RSI 기반 전략
        if chart_analysis['rsi']['state'] == "과매수":
            print("💡 RSI 과매수 구간으로 조정 가능성 있음")
        elif chart_analysis['rsi']['state'] == "과매도":
            print("💡 RSI 과매도 구간으로 반등 가능성 있음")
        
        # MACD 기반 전략
        if chart_analysis['macd']['cross']:
            if chart_analysis['macd']['direction'] == "상향":
                print("💡 MACD 골든크로스 임박, 상승 전환 가능성")
            else:
                print("💡 MACD 데드크로스 임박, 하락 전환 가능성")
        
        # 볼린저 밴드 기반 전략
        if chart_analysis['bollinger']['position'] == "상단":
            print("💡 볼린저 밴드 상단에 위치, 과매수 주의")
        elif chart_analysis['bollinger']['position'] == "하단":
            print("💡 볼린저 밴드 하단에 위치, 반등 가능성")
        
        # 캔들 패턴 기반 전략
        for pattern in chart_analysis['patterns']:
            if pattern['strength'] == "강세":
                print(f"💡 {pattern['pattern']} 발생, 상승 추세 예상")
            elif pattern['strength'] == "약세":
                print(f"💡 {pattern['pattern']} 발생, 하락 추세 예상")
        
        # 거래량 기준 추가 전략
        if volume_analysis['trend'] in ["매우 높음", "높음"]:
            print("💡 거래량이 높아 단기적으로 변동성이 커질 수 있습니다.")
        elif volume_analysis['trend'] in ["매우 낮음", "낮음"]:
            print("💡 거래량이 낮아 큰 방향성을 잡기 어려울 수 있습니다.")
        
        # 세력 동향 기반 전략 추가
        if manipulation_analysis and manipulation_analysis['patterns']:
            print("\n[세력 동향 참고사항]")
            for pattern in manipulation_analysis['patterns']:
                if pattern['confidence'] == 'high':
                    if '매도' in pattern['type']:
                        print(f"💡 강한 매도세력 감지, 상방 돌파가 어려울 수 있음")
                    elif '매수' in pattern['type']:
                        print(f"💡 강한 매수세력 감지, 하방 지지 예상")
                    elif '거래량' in pattern['type']:
                        print(f"💡 단기 변동성 확대 가능성 높음")
    
    print("\n⚠️ 주의: 이 분석은 참고용이며, 실제 투자는 더 다양한 지표와 함께 종합적으로 판단하시기 바랍니다.")

if __name__ == "__main__":
    symbol = input("코인 심볼을 입력하세요 (예: BTC, XRP, ETH): ").strip().upper()
    print_market_analysis(symbol) 