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

class MarketAnalysis:
    def __init__(self):
        """시장 분석기 초기화"""
        self.db = Database()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.base_url = "https://api.upbit.com/v1"
        
    def get_candles(self, symbol: str, interval: str = 'minutes/60', count: int = 200) -> Optional[pd.DataFrame]:
        """캔들 데이터 가져오기"""
        try:
            url = f"{self.base_url}/candles/{interval}"
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
            df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
            df = df.sort_values('timestamp')
            
            # 컬럼 표준화
            df = self._standardize_columns(df)
            
            # 기술적 지표 계산
            self._calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 가져오기 실패: {str(e)}")
            return None

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
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

    def analyze_chart(self, df: pd.DataFrame) -> Dict:
        """차트를 종합적으로 분석"""
        try:
            if df is None or len(df) < 20:
                return None

            # 기술적 지표 계산
            self._calculate_technical_indicators(df)
            
            # 현재가
            current_price = df['close'].iloc[-1]
            
            # 변동성 계산 (20일 기준)
            volatility = df['close'].pct_change().rolling(window=20).std() * 100
            current_volatility = volatility.iloc[-1]
            
            # 추세 강도 계산
            trend_strength = self._calculate_trend_strength(df)
            
            # 시장 심리 분석
            market_sentiment = {
                'volatility': current_volatility,
                'trend_strength': trend_strength,
                'trend_direction': 'bullish' if trend_strength > 60 else 'bearish' if trend_strength < 40 else 'neutral',
                'rsi_status': 'oversold' if df['RSI'].iloc[-1] < 30 else 'overbought' if df['RSI'].iloc[-1] > 70 else 'neutral',
                'macd_signal': 'buy' if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'sell'
            }
            
            # 기술적 지표 추출
            technical_indicators = {
                'moving_averages': {
                    f'MA{period}': df[f'MA{period}'].iloc[-1] 
                    for period in [5, 10, 20, 60, 120]
                },
                'rsi': df['RSI'].iloc[-1],
                'macd': {
                    'macd': df['MACD'].iloc[-1],
                    'signal': df['MACD_signal'].iloc[-1],
                    'histogram': df['MACD_hist'].iloc[-1]
                },
                'bollinger_bands': {
                    'upper': df['BB_upper'].iloc[-1],
                    'middle': df['BB_middle'].iloc[-1],
                    'lower': df['BB_lower'].iloc[-1]
                },
                'trend_strength': trend_strength
            }
            
            # 패턴 분석
            patterns = self._analyze_chart_patterns(df)
            
            # 지지/저항 레벨
            support_resistance = self._analyze_support_resistance(df)
            
            return {
                'current_price': current_price,
                'technical_indicators': technical_indicators,
                'patterns': patterns,
                'support_resistance': support_resistance,
                'market_sentiment': market_sentiment
            }
            
        except Exception as e:
            logger.error(f"차트 분석 실패: {str(e)}")
            return None

    def analyze_orderbook(self, orderbook: dict) -> dict:
        """호가 데이터 분석"""
        try:
            if not orderbook:
                return None
            
            # 매수/매도 총량 계산
            total_bid_size = sum(float(bid['bid_size']) for bid in orderbook['orderbook_units'])
            total_ask_size = sum(float(ask['ask_size']) for ask in orderbook['orderbook_units'])
            
            # 매수/매도 비율
            bid_ask_ratio = total_bid_size / total_ask_size if total_ask_size > 0 else 1.0
            
            # 호가 집중도 분석
            bid_prices = [float(unit['bid_price']) for unit in orderbook['orderbook_units']]
            ask_prices = [float(unit['ask_price']) for unit in orderbook['orderbook_units']]
            bid_sizes = [float(unit['bid_size']) for unit in orderbook['orderbook_units']]
            ask_sizes = [float(unit['ask_size']) for unit in orderbook['orderbook_units']]
            
            # 매수 호가 집중도 (상위 3개 호가의 비중)
            bid_concentration = sum(bid_sizes[:3]) / total_bid_size if total_bid_size > 0 else 0
            
            # 매도 호가 집중도 (상위 3개 호가의 비중)
            ask_concentration = sum(ask_sizes[:3]) / total_ask_size if total_ask_size > 0 else 0
            
            # 호가 스프레드
            spread = (ask_prices[0] - bid_prices[0]) / bid_prices[0]
            
            # 매수 벽 분석
            bid_walls = []
            for i, (price, size) in enumerate(zip(bid_prices, bid_sizes)):
                if i > 0 and size > bid_sizes[i-1] * 2:  # 이전 호가보다 2배 이상 큰 경우
                    bid_walls.append({'price': price, 'size': size})
            
            # 매도 벽 분석
            ask_walls = []
            for i, (price, size) in enumerate(zip(ask_prices, ask_sizes)):
                if i > 0 and size > ask_sizes[i-1] * 2:  # 이전 호가보다 2배 이상 큰 경우
                    ask_walls.append({'price': price, 'size': size})
            
            return {
                'bid_ask_ratio': bid_ask_ratio,
                'bid_concentration': bid_concentration,
                'ask_concentration': ask_concentration,
                'spread': spread,
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'analysis': {
                    'pressure': 'buy' if bid_ask_ratio > 1.2 else 'sell' if bid_ask_ratio < 0.8 else 'neutral',
                    'liquidity': 'high' if spread < 0.001 else 'low' if spread > 0.005 else 'medium',
                    'volatility_risk': 'high' if len(bid_walls) + len(ask_walls) > 3 else 'low'
                }
            }
            
        except Exception as e:
            logger.error(f"호가 분석 중 오류 발생: {str(e)}")
            return None

    def _calculate_fear_greed_index(self, df: pd.DataFrame) -> float:
        """공포/탐욕 지수 계산"""
        try:
            # RSI 기반 공포/탐욕 (0-100)
            rsi = df['RSI'].iloc[-1]
            rsi_score = (rsi - 50) * 2  # -100 ~ 100 변환
            
            # 변동성 기반 공포/탐욕
            volatility = df['close'].pct_change().std() * 100
            vol_score = 100 - min(volatility * 10, 100)  # 변동성이 높을수록 공포
            
            # 이동평균선 배열 기반 공포/탐욕
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma60 = df['MA60'].iloc[-1]
            
            if ma5 > ma20 > ma60:
                ma_score = 80  # 강한 상승 추세
            elif ma5 < ma20 < ma60:
                ma_score = 20  # 강한 하락 추세
            else:
                ma_score = 50  # 중립
            
            # MACD 기반 공포/탐욕
            macd_score = 70 if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 30
            
            # 종합 점수 계산 (가중치 적용)
            final_score = (
                rsi_score * 0.3 +
                vol_score * 0.2 +
                ma_score * 0.3 +
                macd_score * 0.2
            )
            
            return max(0, min(100, final_score))  # 0-100 범위로 제한
            
        except Exception as e:
            logger.error(f"공포/탐욕 지수 계산 중 오류: {str(e)}")
            return 50.0  # 기본값

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

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> None:
        """기술적 지표 계산"""
        try:
            # 이동평균선
            for period in [5, 10, 20, 60, 120]:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
            
            # ADX 계산
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # +DM, -DM
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # +DI, -DI
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['ADX'] = dx.rolling(window=14).mean().fillna(50)  # 기본값 50으로 설정
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # 볼린저 밴드
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # 결측값 처리
            df.ffill(inplace=True)  # 앞의 값으로 채우기
            df.bfill(inplace=True)  # 뒤의 값으로 채우기
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {str(e)}")

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
                        'timestamp': df['timestamp'].iloc[idx],
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

    def _analyze_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
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
            
            return {
                'pivot': pp,
                'resistance': [r1, r2],
                'support': [s1, s2]
            }
            
        except Exception as e:
            logger.error(f"지지/저항 레벨 분석 실패: {str(e)}")
            return {
                'pivot': df['close'].iloc[-1],
                'resistance': [],
                'support': []
            }

    def _analyze_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """차트 패턴 분석"""
        try:
            patterns = []
            current_price = df['close'].iloc[-1]
            
            # 이동평균선 배열 확인
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma60 = df['MA60'].iloc[-1]
            
            # 골든 크로스/데드 크로스 확인
            if df['MA5'].iloc[-2] < df['MA20'].iloc[-2] and ma5 > ma20:
                patterns.append({
                    'name': '골든 크로스',
                    'pattern_type': 'bullish',
                    'reliability': 'high',
                    'target': current_price * 1.05
                })
            elif df['MA5'].iloc[-2] > df['MA20'].iloc[-2] and ma5 < ma20:
                patterns.append({
                    'name': '데드 크로스',
                    'pattern_type': 'bearish',
                    'reliability': 'high',
                    'target': current_price * 0.95
                })
            
            # RSI 다이버전스 확인
            rsi = df['RSI'].iloc[-5:]
            price = df['close'].iloc[-5:]
            
            if price.iloc[-1] > price.iloc[0] and rsi.iloc[-1] < rsi.iloc[0]:
                patterns.append({
                    'name': 'RSI 하락 다이버전스',
                    'pattern_type': 'bearish',
                    'reliability': 'medium',
                    'target': current_price * 0.97
                })
            elif price.iloc[-1] < price.iloc[0] and rsi.iloc[-1] > rsi.iloc[0]:
                patterns.append({
                    'name': 'RSI 상승 다이버전스',
                    'pattern_type': 'bullish',
                    'reliability': 'medium',
                    'target': current_price * 1.03
                })
            
            # 볼린저 밴드 스퀴즈 확인
            bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            if bb_width.iloc[-1] < bb_width.iloc[-20:].mean() * 0.8:
                if current_price > df['BB_middle'].iloc[-1]:
                    patterns.append({
                        'name': '볼린저 밴드 스퀴즈',
                        'pattern_type': 'bullish',
                        'reliability': 'medium',
                        'target': current_price * 1.04
                    })
                else:
                    patterns.append({
                        'name': '볼린저 밴드 스퀴즈',
                        'pattern_type': 'bearish',
                        'reliability': 'medium',
                        'target': current_price * 0.96
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"차트 패턴 분석 실패: {str(e)}")
            return []

    def _generate_price_predictions(self, df: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """가격 예측"""
        try:
            predictions = []
            current_price = df['close'].iloc[-1]
            
            for pattern in patterns:
                prediction = {
                    'pattern': pattern['name'],
                    'type': pattern['pattern_type'],
                    'target_price': pattern['target'],
                    'predicted_range': {
                        'low': min(current_price, pattern['target']) * 0.98,
                        'high': max(current_price, pattern['target']) * 1.02,
                        'most_likely': pattern['target']
                    },
                    'confidence': 0.7 if pattern['reliability'] == 'high' else 0.5,
                    'timeframe': '1-3일',
                    'key_levels': {
                        'resistance': [
                            df['high'].tail(20).max(),
                            df['BB_upper'].iloc[-1]
                        ],
                        'support': [
                            df['low'].tail(20).min(),
                            df['BB_lower'].iloc[-1]
                        ]
                    }
                }
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"가격 예측 생성 실패: {str(e)}")
            return []

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """추세 강도 분석"""
        try:
            # ADX 기반 추세 강도 계산
            adx = df['ADX'].iloc[-1]
            
            # 이동평균선 배열 확인
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma60 = df['MA60'].iloc[-1]
            
            # 이동평균선 기울기 계산
            ma5_slope = (df['MA5'].iloc[-1] - df['MA5'].iloc[-5]) / df['MA5'].iloc[-5] * 100
            ma20_slope = (df['MA20'].iloc[-1] - df['MA20'].iloc[-5]) / df['MA20'].iloc[-5] * 100
            
            # 추세 점수 계산 (0-100)
            trend_score = 0
            
            # ADX 기반 점수 (0-40)
            trend_score += min(adx, 40)
            
            # 이동평균선 배열 기반 점수 (0-30)
            if ma5 > ma20 > ma60:  # 상승 추세
                trend_score += 30
            elif ma5 < ma20 < ma60:  # 하락 추세
                trend_score += 20
            else:  # 횡보
                trend_score += 10
                
            # 이동평균선 기울기 기반 점수 (0-30)
            slope_score = (abs(ma5_slope) + abs(ma20_slope)) / 2
            trend_score += min(slope_score, 30)
            
            return round(trend_score, 1)
            
        except Exception as e:
            logger.error(f"추세 강도 계산 중 오류 발생: {str(e)}")
            return 50.0  # 기본값 반환

if __name__ == "__main__":
    symbol = input("코인 심볼을 입력하세요 (예: BTC, XRP, ETH): ").strip().upper()
    print_market_analysis(symbol) 