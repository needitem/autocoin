import logging
import pandas as pd

class TradingManager:
    def __init__(self, api=None, verbose: bool = False):
        """거래 매니저 초기화"""
        self.api = api
        self.verbose = verbose
        self.logger = logging.getLogger('trading')
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def get_market_data(self, market: str) -> dict:
        """시장 데이터 조회"""
        try:
            if not self.api:
                self.logger.error("API가 초기화되지 않았습니다")
                return None

            # 현재가 조회
            ticker = self.api.get_ticker(market)
            if not ticker:
                self.logger.error(f"티커 데이터를 가져올 수 없습니다: {market}")
                return None

            # OHLCV 데이터 조회
            ohlcv = self.api.get_daily_ohlcv(market)
            if not ohlcv:
                self.logger.error(f"OHLCV 데이터를 가져올 수 없습니다: {market}")
                return None

            # 지표 계산
            indicators = self.calculate_indicators(ohlcv)

            return {
                'current_price': ticker['trade_price'],
                'open': ticker['opening_price'],
                'high': ticker['high_price'],
                'low': ticker['low_price'],
                'volume': ticker['acc_trade_volume_24h'],
                'change_rate': ticker['signed_change_rate'],
                'indicators': indicators,
                'ohlcv_df': ohlcv
            }

        except Exception as e:
            self.logger.error(f"시장 데이터 조회 중 오류 발생: {str(e)}")
            return None

    def calculate_indicators(self, ohlcv_df: pd.DataFrame) -> dict:
        """기술적 지표 계산"""
        try:
            if ohlcv_df.empty:
                return {}

            # 이동평균선
            ma5 = ohlcv_df['close'].rolling(window=5).mean()
            ma10 = ohlcv_df['close'].rolling(window=10).mean()
            ma20 = ohlcv_df['close'].rolling(window=20).mean()

            # 볼린저 밴드
            ma20 = ohlcv_df['close'].rolling(window=20).mean()
            std20 = ohlcv_df['close'].rolling(window=20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20

            # RSI
            delta = ohlcv_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            exp1 = ohlcv_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = ohlcv_df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            # 변동성
            volatility = std20 / ma20 * 100

            return {
                'moving_averages': {
                    'MA5': ma5,
                    'MA10': ma10,
                    'MA20': ma20
                },
                'bollinger_bands': {
                    'upper': upper,
                    'middle': ma20,
                    'lower': lower,
                    'volatility': volatility.iloc[-1]
                },
                'rsi': rsi,
                'macd': {
                    'macd': macd,
                    'signal': signal,
                    'histogram': macd - signal
                }
            }

        except Exception as e:
            self.logger.error(f"지표 계산 중 오류 발생: {str(e)}")
            return {} 