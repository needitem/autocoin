import logging
from typing import Dict
from market_analysis import MarketAnalysis
from investment_strategy import InvestmentStrategy
from trading_strategy import TradingStrategy
import math
import pandas as pd
import random

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.market_analysis = MarketAnalysis()
    
    def analyze_market(self, symbol: str, strategy: TradingStrategy) -> Dict:
        """시장 종합 분석"""
        try:
            if not symbol:
                logger.error("심볼이 지정되지 않았습니다.")
                return None
            
            # strategy가 TradingStrategy 열거형인 경우 InvestmentStrategy 객체로 변환
            if isinstance(strategy, TradingStrategy):
                strategy = InvestmentStrategy(strategy)
            
            # 데이터 가져오기
            df = self.market_analysis.get_candles(symbol)
            if df is None or len(df) == 0:
                logger.error("캔들 데이터를 가져오는데 실패했습니다.")
                return None
            
            # 기본 분석 수행
            analysis_result = self.market_analysis.analyze_chart(df)
            if analysis_result is None:
                logger.error("차트 분석에 실패했습니다.")
                return None
            
            # 호가 데이터 분석
            orderbook = self.market_analysis.get_orderbook(symbol)
            orderbook_analysis = self.market_analysis.analyze_orderbook(orderbook) if orderbook else None
            
            # 투자 전략 추천
            strategy_recommendation = self._generate_strategy_recommendation(
                df, 
                analysis_result['current_price'], 
                analysis_result['technical_indicators']['moving_averages'],
                analysis_result['technical_indicators']['rsi'],
                analysis_result.get('patterns', []),  # patterns가 없을 수 있음
                orderbook_analysis,
                strategy
            )
            
            # 예측 데이터를 Plotly 차트용으로 변환
            plotly_data = {
                'x': [],  # 시간
                'actual': [],  # 실제 가격
                'bullish': [],  # 낙관적 시나리오
                'bearish': [],  # 비관적 시나리오
                'most_likely': []  # 가능성 높은 시나리오
            }

            # 과거 데이터 추가 (최근 20일)
            past_data = df.tail(20)
            for index, row in past_data.iterrows():
                plotly_data['x'].append(pd.to_datetime(index))
                plotly_data['actual'].append(float(row['close']))
                plotly_data['bullish'].append(None)
                plotly_data['bearish'].append(None)
                plotly_data['most_likely'].append(None)

            # 현재 시점 데이터 추가
            current_time = pd.to_datetime(df.index[-1])
            current_price = float(analysis_result['current_price'])
            plotly_data['x'].append(current_time)
            plotly_data['actual'].append(current_price)
            plotly_data['bullish'].append(current_price)
            plotly_data['bearish'].append(current_price)
            plotly_data['most_likely'].append(current_price)

            # 향후 14일 예측 데이터 생성
            for day in range(1, 15):
                try:
                    prediction_time = current_time + pd.Timedelta(days=day)
                    plotly_data['x'].append(prediction_time)
                    
                    # 시나리오별 가격 계산
                    bullish_price = current_price * (1 + (0.02 * day))  # 낙관적: 일 2% 상승
                    bearish_price = current_price * (1 - (0.015 * day))  # 비관적: 일 1.5% 하락
                    likely_price = current_price * (1 + (0.005 * day))  # 가능성 높은: 일 0.5% 상승
                    
                    plotly_data['actual'].append(None)
                    plotly_data['bullish'].append(float(bullish_price))
                    plotly_data['bearish'].append(float(bearish_price))
                    plotly_data['most_likely'].append(float(likely_price))

                    # 중간값 추가 (6시간 간격)
                    for hour in range(6, 24, 6):
                        inter_time = prediction_time - pd.Timedelta(hours=24-hour)
                        progress_ratio = hour / 24
                        
                        plotly_data['x'].append(inter_time)
                        plotly_data['actual'].append(None)
                        plotly_data['bullish'].append(float(
                            current_price + (bullish_price - current_price) * progress_ratio
                        ))
                        plotly_data['bearish'].append(float(
                            current_price + (bearish_price - current_price) * progress_ratio
                        ))
                        plotly_data['most_likely'].append(float(
                            current_price + (likely_price - current_price) * progress_ratio
                        ))

                except Exception as e:
                    logger.error(f"{day}일 예측 계산 중 오류: {str(e)}")
                    continue

            # 시간순 정렬
            sorted_indices = sorted(range(len(plotly_data['x'])), key=lambda k: plotly_data['x'][k])
            for key in plotly_data.keys():
                plotly_data[key] = [plotly_data[key][i] for i in sorted_indices]

            return {
                'df': df,
                'current_price': current_price,
                'technical_indicators': analysis_result['technical_indicators'],
                'pattern_analysis': {
                    'patterns': analysis_result.get('patterns', []),
                    'predictions': []
                },
                'orderbook_analysis': orderbook_analysis,
                'strategy_recommendation': strategy_recommendation,
                'prediction_data': plotly_data,  # Plotly 차트용 데이터
                'prediction_metadata': {
                    'start_date': current_time - pd.Timedelta(days=20),
                    'end_date': current_time + pd.Timedelta(days=14),
                    'scenarios': {
                        'actual': {'name': '실제 가격', 'color': '#000000', 'style': 'solid', 'width': 2},
                        'bullish': {'name': '낙관적 시나리오', 'color': '#4CAF50', 'style': 'dash', 'width': 1},
                        'bearish': {'name': '비관적 시나리오', 'color': '#F44336', 'style': 'dash', 'width': 1},
                        'most_likely': {'name': '가능성 높은 시나리오', 'color': '#2196F3', 'style': 'solid', 'width': 2}
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"시장 분석 중 오류 발생: {str(e)}")
            return None
    
    def _generate_strategy_recommendation(self, df, current_price, ma_data, rsi, patterns, orderbook_analysis, strategy):
        """투자 전략 추천 생성"""
        try:
            # 시장 상황 분석
            market_condition = self._analyze_market_condition(df, rsi, orderbook_analysis)
            
            # 전략별 진입/청산 가격 조정
            strategy_type = strategy.strategy_type.value
            if strategy_type == 'SCALPING':
                entry_ratio = 0.3  # 현재가 대비 하락폭
                exit_ratio = 0.5   # 현재가 대비 상승폭
            elif strategy_type == 'SWING':
                entry_ratio = 0.5
                exit_ratio = 1.0
            elif strategy_type == 'POSITION':
                entry_ratio = 1.0
                exit_ratio = 2.0
            else:
                entry_ratio = 0.5
                exit_ratio = 1.0
            
            # 시장 상황에 따른 조정
            if market_condition['trend'] == 'BULLISH':
                entry_ratio *= 0.8  # 상승장에서는 진입 가격을 높게
                exit_ratio *= 1.2   # 목표가도 높게
            elif market_condition['trend'] == 'BEARISH':
                entry_ratio *= 1.2  # 하락장에서는 진입 가격을 낮게
                exit_ratio *= 0.8   # 목표가도 낮게
            
            # 최적 진입 가격 계산
            entry_levels = self._calculate_optimal_entry_levels(
                df, current_price, rsi, orderbook_analysis, entry_ratio
            )
            
            # 최적 청산 가격 계산
            exit_levels = self._calculate_exit_levels(
                df, current_price, rsi, orderbook_analysis, exit_ratio
            )
            
            return {
                'strategy_type': strategy_type,
                'market_condition': market_condition,
                'entry_levels': entry_levels,
                'exit_levels': exit_levels,
                'stop_loss': round(current_price * 0.95, 0)  # 기본 손절가
            }
            
        except Exception as e:
            logger.error(f"전략 추천 생성 중 오류: {str(e)}")
            return None

    def _calculate_exit_levels(self, df, current_price, rsi, orderbook_analysis, exit_ratio):
        """최적 청산 가격 계산"""
        try:
            # 볼린저 밴드 계산
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
            
            # 피보나치 레벨 계산
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            fib_levels = {
                '0.236': recent_high - (recent_high - recent_low) * 0.236,
                '0.382': recent_high - (recent_high - recent_low) * 0.382,
                '0.618': recent_high - (recent_high - recent_low) * 0.618
            }
            
            # 주요 저항선 계산
            resistance_levels = [
                df['BB_upper'].iloc[-1],  # 볼린저 상단
                fib_levels['0.618'],      # 피보나치 0.618
                fib_levels['0.382'],      # 피보나치 0.382
                recent_high               # 최근 고점
            ]
            
            # 호가 데이터에서 강한 매도벽 위치 확인
            ask_walls = orderbook_analysis.get('strong_resistance_levels', [])
            resistance_levels.extend(ask_walls)
            
            # 현재가보다 높은 저항선들만 필터링
            valid_resistances = [price for price in resistance_levels if price > current_price]
            valid_resistances.sort()  # 낮은 가격순 정렬
            
            # 청산 강도 계산
            def calculate_exit_strength(price):
                strength = 0
                
                # RSI 기반 강도
                if rsi > 70:  # 과매수 구간
                    strength += 0.3
                elif rsi < 30:  # 과매도 구간
                    strength -= 0.2
                
                # 볼린저 밴드 기반 강도
                bb_position = (price - df['BB_middle'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_middle'].iloc[-1])
                if bb_position > 0.8:  # 상단 근처
                    strength += 0.2
                
                # 피보나치 레벨 기반 강도
                if abs(price - fib_levels['0.618']) / current_price < 0.01:
                    strength += 0.15
                elif abs(price - fib_levels['0.382']) / current_price < 0.01:
                    strength += 0.1
                
                # 호가 데이터 기반 강도
                if price in ask_walls:
                    strength += 0.2
                
                return strength
            
            # 최적 청산 가격 3개 선정
            exit_levels = []
            for i, price in enumerate(valid_resistances[:5]):
                strength = calculate_exit_strength(price)
                exit_price = round(price, 0)
                
                if i == 0:  # 첫 번째 저항선
                    ratio = 0.3
                    description = "1차 청산"
                elif i == 1:  # 두 번째 저항선
                    ratio = 0.4
                    description = "2차 청산"
                else:  # 세 번째 저항선
                    ratio = 0.3
                    description = "3차 청산"
                
                change_rate = (exit_price - current_price) / current_price * 100
                
                exit_levels.append({
                    'price': exit_price,
                    'ratio': ratio,
                    'description': f"{description} (현재가 대비 +{change_rate:.1f}%)",
                    'strength': strength,
                    'reasons': [
                        f'RSI {rsi:.1f} - {"과매수 구간" if rsi > 70 else "과매도 구간" if rsi < 30 else "중립 구간"}',
                        f'매도세/매수세 비율: {1/orderbook_analysis["bid_ask_ratio"]:.2f}',
                        f'볼린저 밴드 상단 대비: {((price - df["BB_upper"].iloc[-1]) / df["BB_upper"].iloc[-1] * 100):.1f}%',
                        f'거래량 프로필: {self._analyze_volume_profile(df, price)}',
                        f'피보나치 레벨 근접도: {min(abs(price - fib_levels["0.618"]), abs(price - fib_levels["0.382"])) / current_price * 100:.1f}%'
                    ]
                })
                
                if len(exit_levels) >= 3:
                    break
            
            # 강도에 따라 정렬
            exit_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return exit_levels
            
        except Exception as e:
            logger.error(f"최적 청산 가격 계산 중 오류: {str(e)}")
            return []

    def _calculate_predicted_price(self, df, current_price, minutes, orderbook_analysis, rsi):
        """시간대별 예측 가격 계산"""
        try:
            # 시간 관련 변수들을 먼저 정의
            current_time = pd.to_datetime(df.index[-1])
            
            # 14일 예측을 위해 minutes를 일 단위로 변환 (1일 = 1440분)
            if minutes > 60:  # 60분 이상이면 일 단위로 처리
                days = minutes / 1440  # 분을 일로 변환
                prediction_time = current_time + pd.Timedelta(days=days)
            else:
                prediction_time = current_time + pd.Timedelta(minutes=int(minutes))
            
            # 이동평균선 계산
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # MACD 계산
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 볼린저 밴드 계산
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
            
            # 기술적 지표 기반 가격 변동 예측
            price_factors = {
                'trend': self._calculate_trend_factor(df),
                'momentum': self._calculate_momentum_factor(df, rsi),
                'support_resistance': self._calculate_support_resistance_factor(df, current_price),
                'volume': self._calculate_volume_factor(df),
                'orderbook': self._calculate_orderbook_factor(orderbook_analysis)['pressure']  # pressure 값만 사용
            }
            
            # 시간에 따른 가중치 조정
            time_weight = math.log(minutes + 1) / 10
            
            # 예측 가격 계산 - 더 큰 변동성 허용
            weighted_change = sum(price_factors.values()) * time_weight
            base_predicted_price = current_price * (1 + weighted_change)
            
            # 추세 기반 가격 조정
            trend_direction = 1 if df['close'].iloc[-1] > df['close'].iloc[-5] else -1
            trend_strength = abs(df['close'].pct_change().mean()) * 100
            
            # 변동성 범위 확대 (더 큰 가격 변화 허용)
            volatility = max(0.001, df['close'].pct_change().std() * 2)  # 최소 변동성 보장
            max_change = volatility * math.sqrt(minutes/10) * (1 + trend_strength)
            
            # RSI 기반 추가 조정
            rsi_adjustment = 0
            if rsi > 70:
                rsi_adjustment = -0.001 * (rsi - 70)
            elif rsi < 30:
                rsi_adjustment = 0.001 * (30 - rsi)
            
            # 호가 데이터 기반 조정
            orderbook_adjustment = (orderbook_analysis.get('bid_ask_ratio', 1.0) - 1.0) * 0.002
            
            # 최종 예측 가격 계산
            predicted_price = base_predicted_price * (1 + rsi_adjustment + orderbook_adjustment)
            predicted_price = max(
                current_price * (1 - max_change),
                min(predicted_price, current_price * (1 + max_change))
            )
            
            # 장기 예측을 위한 시나리오 생성
            scenarios = self._generate_price_scenarios(df, current_price, minutes, rsi)
            
            # 가장 가능성 높은 시나리오 선택
            predicted_price = scenarios['most_likely']['price']
            
            # 중간값 생성 시 시나리오 기반 보간
            intermediate_predictions = []
            total_points = int(minutes) if minutes <= 60 else int(minutes/60)  # 60분 이상이면 시간 단위로
            
            for point in range(1, total_points):
                try:
                    if minutes > 60:
                        inter_time = current_time + pd.Timedelta(hours=point)
                    else:
                        inter_time = current_time + pd.Timedelta(minutes=point)
                    
                    # 각 시나리오의 중간값 계산
                    bullish_price = scenarios['bullish']['intermediate_prices'][point-1]
                    bearish_price = scenarios['bearish']['intermediate_prices'][point-1]
                    likely_price = scenarios['most_likely']['intermediate_prices'][point-1]
                    
                    # 시나리오별 가중치 적용
                    weighted_price = (
                        bullish_price * scenarios['bullish']['probability'] +
                        bearish_price * scenarios['bearish']['probability'] +
                        likely_price * scenarios['most_likely']['probability']
                    )
                    
                    intermediate_predictions.append({
                        'timestamp': inter_time,
                        'price': round(float(weighted_price), 2),
                        'type': 'interpolated',
                        'scenarios': {
                            'bullish': round(float(bullish_price), 2),
                            'bearish': round(float(bearish_price), 2),
                            'most_likely': round(float(likely_price), 2)
                        }
                    })
                except Exception as e:
                    logger.error(f"중간값 계산 중 오류: {str(e)}")
                    continue

            prediction_data = {
                'price': round(float(predicted_price), 2),
                'timestamp': prediction_time,
                'type': 'predicted',
                'scenarios': scenarios,
                'intermediate_points': intermediate_predictions
            }
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"가격 예측 계산 중 오류 발생: {str(e)}")
            return self._generate_fallback_prediction(df, current_price, minutes)

    def _generate_price_scenarios(self, df, current_price, minutes, rsi):
        """가격 시나리오 생성"""
        try:
            # 기본 변동성 계산
            volatility = max(0.001, df['close'].pct_change().std() * 2)
            trend = df['close'].pct_change().mean()
            
            # 시간에 따른 변동성 조정
            time_factor = math.sqrt(minutes/1440) if minutes > 60 else math.sqrt(minutes/60)
            adjusted_volatility = volatility * time_factor
            
            # 낙관적 시나리오 (Bullish)
            bullish_change = adjusted_volatility * 2 + abs(trend) * time_factor
            if rsi < 30:  # 과매도 상태면 상승 가능성 증가
                bullish_change *= 1.5
            bullish_price = current_price * (1 + bullish_change)
            
            # 비관적 시나리오 (Bearish)
            bearish_change = adjusted_volatility * 2 - abs(trend) * time_factor
            if rsi > 70:  # 과매수 상태면 하락 가능성 증가
                bearish_change *= 1.5
            bearish_price = current_price * (1 - bearish_change)
            
            # 가장 가능성 높은 시나리오
            likely_change = trend * time_factor
            likely_price = current_price * (1 + likely_change)
            
            # 시나리오별 중간값 생성
            total_points = int(minutes) if minutes <= 60 else int(minutes/60)
            
            def generate_intermediate_prices(start_price, end_price, points):
                prices = []
                for i in range(points):
                    ratio = (i + 1) / points
                    # 비선형 보간으로 자연스러운 가격 변화 생성
                    adj_ratio = math.pow(ratio, 1.1) if end_price > start_price else math.pow(ratio, 0.9)
                    price = start_price + (end_price - start_price) * adj_ratio
                    # 약간의 노이즈 추가
                    noise = random.uniform(-0.0001, 0.0001) * price
                    prices.append(price + noise)
                return prices

            # 시나리오별 확률 계산
            bullish_prob = 0.2 + (30 - min(rsi, 30)) / 100  # RSI가 낮을수록 상승 확률 증가
            bearish_prob = 0.2 + (max(rsi, 70) - 70) / 100  # RSI가 높을수록 하락 확률 증가
            likely_prob = 1 - (bullish_prob + bearish_prob)  # 나머지 확률

            return {
                'bullish': {
                    'price': bullish_price,
                    'probability': bullish_prob,
                    'intermediate_prices': generate_intermediate_prices(current_price, bullish_price, total_points)
                },
                'bearish': {
                    'price': bearish_price,
                    'probability': bearish_prob,
                    'intermediate_prices': generate_intermediate_prices(current_price, bearish_price, total_points)
                },
                'most_likely': {
                    'price': likely_price,
                    'probability': likely_prob,
                    'intermediate_prices': generate_intermediate_prices(current_price, likely_price, total_points)
                }
            }
            
        except Exception as e:
            logger.error(f"시나리오 생성 중 오류: {str(e)}")
            return self._generate_fallback_scenarios(current_price)

    def _generate_fallback_scenarios(self, current_price):
        """기본 시나리오 생성 (오류 발생 시)"""
        return {
            'bullish': {'price': current_price * 1.05, 'probability': 0.3, 'intermediate_prices': []},
            'bearish': {'price': current_price * 0.95, 'probability': 0.3, 'intermediate_prices': []},
            'most_likely': {'price': current_price, 'probability': 0.4, 'intermediate_prices': []}
        }

    def _calculate_trend_factor(self, df):
        """추세 요인 계산"""
        try:
            # 단기 추세 (MA5 기울기)
            short_trend = (df['MA5'].iloc[-1] - df['MA5'].iloc[-5]) / df['MA5'].iloc[-5]
            
            # 중기 추세 (MA20 기울기)
            medium_trend = (df['MA20'].iloc[-1] - df['MA20'].iloc[-20]) / df['MA20'].iloc[-20]
            
            # MACD 방향성
            macd_trend = 1 if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else -1
            
            return (short_trend * 0.5 + medium_trend * 0.3) * macd_trend
            
        except Exception as e:
            logger.error(f"추세 요인 계산 중 오류: {str(e)}")
            return 0

    def _calculate_momentum_factor(self, df, rsi):
        """모멘텀 요인 계산"""
        try:
            # RSI 기반 과매수/과매도
            rsi_factor = 0
            if rsi > 70:
                rsi_factor = -0.002 * (rsi - 70)
            elif rsi < 30:
                rsi_factor = 0.002 * (30 - rsi)
            
            # 볼린저 밴드 위치
            bb_position = (df['close'].iloc[-1] - df['BB_middle'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_middle'].iloc[-1])
            bb_factor = -0.001 * bb_position
            
            return rsi_factor + bb_factor
            
        except Exception as e:
            logger.error(f"모멘텀 요인 계산 중 오류: {str(e)}")
            return 0

    def _calculate_support_resistance_factor(self, df, current_price):
        """지지/저항 요인 계산"""
        try:
            # 최근 고점/저점 식별
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            
            # 현재가 위치에 따른 압력
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            
            return -0.001 * (price_position - 0.5)
            
        except Exception as e:
            logger.error(f"지지/저항 요인 계산 중 오류: {str(e)}")
            return 0

    def _calculate_volume_factor(self, df):
        """거래량 요인 계산"""
        try:
            # 거래량 증감
            volume_change = df['volume'].pct_change().iloc[-1]
            
            # 거래량 가중 평균 대비
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            return 0.001 * (volume_change + (volume_ratio - 1))
            
        except Exception as e:
            logger.error(f"거래량 요인 계산 중 오류: {str(e)}")
            return 0

    def _calculate_orderbook_factor(self, orderbook_analysis):
        """호가 요인 계산"""
        try:
            # 매수/매도 비율 (1보다 크면 매수세 강함, 작으면 매도세 강함)
            bid_ask_ratio = orderbook_analysis.get('bid_ask_ratio', 1.0)
            
            # 매수 집중도 계산
            bid_concentration = {
                'strength': orderbook_analysis.get('support_strength', 1.0),  # 매수벽 강도
                'depth': orderbook_analysis.get('support_depth', 0),         # 매수벽 두께
                'volume': orderbook_analysis.get('cumulative_bid_volume', 0),# 누적 매수량
                'levels': orderbook_analysis.get('strong_support_levels', []),# 강한 매수 구간들
                'density': orderbook_analysis.get('bid_density', 1.0)        # 호가별 주문량 밀집도
            }
            
            # 매도 집중도 계산
            ask_concentration = {
                'strength': orderbook_analysis.get('resistance_strength', 1.0),  # 매도벽 강도
                'depth': orderbook_analysis.get('resistance_depth', 0),         # 매도벽 두께
                'volume': orderbook_analysis.get('cumulative_ask_volume', 0),   # 누적 매도량
                'levels': orderbook_analysis.get('strong_resistance_levels', []),# 강한 매도 구간들
                'density': orderbook_analysis.get('ask_density', 1.0)           # 호가별 주문량 밀집도
            }
            
            # 매수 압력 계산
            bid_pressure = (
                (bid_concentration['strength'] - 1.0) * 0.4 +    # 매수벽 강도 (기준값 1.0)
                (bid_concentration['density'] - 1.0) * 0.3 +     # 매수 물량 밀집도
                (bid_ask_ratio - 1.0) * 0.3                      # 매수/매도 비율의 영향
            ) * 0.001
            
            # 매도 압력 계산
            ask_pressure = (
                (ask_concentration['strength'] - 1.0) * 0.4 +    # 매도벽 강도 (기준값 1.0)
                (ask_concentration['density'] - 1.0) * 0.3 +     # 매도 물량 밀집도
                (1.0 - min(bid_ask_ratio, 1.0)) * 0.3           # 매도 우세 정도
            ) * 0.001
            
            # 전체 호가 압력 계산 (양수: 매수압력 우세, 음수: 매도압력 우세)
            orderbook_pressure = bid_pressure - ask_pressure
            
            return {
                'pressure': orderbook_pressure,
                'details': {
                    'bid_concentration': bid_concentration,
                    'ask_concentration': ask_concentration,
                    'explanation': {
                        'bid_pressure': f"매수세 {bid_pressure*1000:.1f}% " + (
                            "강함" if bid_pressure > 0 else "약함"
                        ),
                        'ask_pressure': f"매도세 {ask_pressure*1000:.1f}% " + (
                            "강함" if ask_pressure > 0 else "약함"
                        ),
                        'overall': (
                            "매수세 우세" if orderbook_pressure > 0 
                            else "매도세 우세" if orderbook_pressure < 0 
                            else "균형상태"
                        ),
                        'strength': abs(orderbook_pressure*1000),  # 압력의 강도
                        'key_levels': {
                            'support': bid_concentration['levels'],     # 주요 매수 구간
                            'resistance': ask_concentration['levels']   # 주요 매도 구간
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"호가 요인 계산 중 오류: {str(e)}")
            return {'pressure': 0, 'details': {}}

    def _calculate_prediction_confidence(self, minutes, orderbook_analysis):
        """예측 신뢰도 계산"""
        try:
            # 기본 신뢰도 (시간이 길어질수록 감소)
            base_confidence = max(0.3, 1 - (minutes / 120))
            
            # 호가 데이터 품질 기반 신뢰도 조정
            orderbook_quality = min(1.0, orderbook_analysis.get('orderbook_depth', 0) / 1000)
            
            # 최종 신뢰도 계산
            confidence = base_confidence * (0.7 + 0.3 * orderbook_quality)
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"신뢰도 계산 중 오류 발생: {str(e)}")
            return 0.5

    def _get_price_signals(self, predicted_price, current_price, orderbook_analysis):
        """가격 예측에 대한 신호 분석"""
        signals = []
        change_rate = (predicted_price - current_price) / current_price * 100
        
        # 가격 변동 기반 신호
        if change_rate > 1:
            signals.append(f"상승 추세 강화 ({change_rate:.1f}%)")
        elif change_rate < -1:
            signals.append(f"하락 추세 강화 ({change_rate:.1f}%)")
            
        # 호가 데이터 기반 신호
        bid_strength = orderbook_analysis.get('support_strength', 1.0)
        ask_strength = orderbook_analysis.get('resistance_strength', 1.0)
        
        if bid_strength > 1.2:
            signals.append(f"매수세 강화 (강도: {bid_strength:.2f})")
        if ask_strength > 1.2:
            signals.append(f"매도세 강화 (강도: {ask_strength:.2f})")
            
        return signals

    def _generate_fallback_prediction(self, df, current_price, minutes):
        """기본 예측 데이터 생성 (오류 발생 시)"""
        try:
            current_time = pd.to_datetime(df.index[-1])
            
            # 14일 예측을 위해 minutes를 일 단위로 변환
            if minutes > 60:
                days = minutes / 1440
                prediction_time = current_time + pd.Timedelta(days=days)
            else:
                prediction_time = current_time + pd.Timedelta(minutes=int(minutes))

            # 기본 시나리오 생성
            scenarios = {
                'bullish': {
                    'price': current_price * 1.05,
                    'probability': 0.3,
                    'intermediate_prices': [current_price * (1 + 0.05 * i/10) for i in range(1, 11)]
                },
                'bearish': {
                    'price': current_price * 0.95,
                    'probability': 0.3,
                    'intermediate_prices': [current_price * (1 - 0.05 * i/10) for i in range(1, 11)]
                },
                'most_likely': {
                    'price': current_price,
                    'probability': 0.4,
                    'intermediate_prices': [current_price for _ in range(10)]
                }
            }

            # 중간값 생성
            intermediate_predictions = []
            total_points = int(minutes/60) if minutes > 60 else int(minutes)
            
            for point in range(1, total_points):
                try:
                    if minutes > 60:
                        inter_time = current_time + pd.Timedelta(hours=point)
                    else:
                        inter_time = current_time + pd.Timedelta(minutes=point)
                        
                    # 기본 중간값 계산
                    progress_ratio = point / total_points
                    weighted_price = (
                        scenarios['bullish']['price'] * scenarios['bullish']['probability'] +
                        scenarios['bearish']['price'] * scenarios['bearish']['probability'] +
                        scenarios['most_likely']['price'] * scenarios['most_likely']['probability']
                    )
                    
                    intermediate_predictions.append({
                        'timestamp': inter_time,
                        'price': round(float(weighted_price), 2),
                        'type': 'interpolated',
                        'scenarios': {
                            'bullish': round(float(scenarios['bullish']['price']), 2),
                            'bearish': round(float(scenarios['bearish']['price']), 2),
                            'most_likely': round(float(scenarios['most_likely']['price']), 2)
                        }
                    })
                except Exception as e:
                    logger.error(f"기본 중간값 계산 중 오류: {str(e)}")
                    continue

            return {
                'price': round(float(scenarios['most_likely']['price']), 2),
                'timestamp': prediction_time,
                'type': 'predicted',
                'scenarios': scenarios,
                'intermediate_points': intermediate_predictions
            }
            
        except Exception as e:
            logger.error(f"기본 예측 생성 중 오류: {str(e)}")
            return {
                'price': current_price,
                'timestamp': current_time + pd.Timedelta(minutes=int(minutes)),
                'type': 'predicted',
                'scenarios': self._generate_fallback_scenarios(current_price),
                'intermediate_points': []
            }

    def _calculate_optimal_entry_levels(self, df, current_price, rsi, orderbook_analysis, entry_ratio):
        """최적 진입 가격 계산"""
        try:
            # 볼린저 밴드 계산
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
            
            # 피보나치 레벨 계산
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            fib_382 = recent_high - (recent_high - recent_low) * 0.382
            fib_500 = recent_high - (recent_high - recent_low) * 0.500
            fib_618 = recent_high - (recent_high - recent_low) * 0.618
            
            # 주요 지지선 계산
            support_levels = [
                df['BB_lower'].iloc[-1],  # 볼린저 하단
                fib_382, fib_500, fib_618,  # 피보나치 레벨
                df['MA20'].iloc[-1],  # 20일 이동평균
                df['MA10'].iloc[-1],  # 10일 이동평균
            ]
            
            # 호가 데이터에서 강한 매수벽 위치 확인
            bid_walls = orderbook_analysis.get('strong_support_levels', [])
            support_levels.extend(bid_walls)
            
            # 현재가보다 낮은 지지선들만 필터링
            valid_supports = [price for price in support_levels if price < current_price]
            valid_supports.sort(reverse=True)  # 높은 가격순 정렬
            
            # 매수 강도 계산
            def calculate_entry_strength(price):
                strength = 0
                
                # RSI 기반 강도
                if rsi > 70:  # 과매수 구간
                    strength -= 0.2
                elif rsi < 30:  # 과매도 구간
                    strength += 0.3
                
                # 볼린저 밴드 기반 강도
                bb_position = (price - df['BB_lower'].iloc[-1]) / (df['BB_middle'].iloc[-1] - df['BB_lower'].iloc[-1])
                if bb_position < 0.2:  # 하단 근처
                    strength += 0.2
                
                # 피보나치 레벨 기반 강도
                if abs(price - fib_618) / current_price < 0.01:  # 0.618 레벨 근처
                    strength += 0.15
                elif abs(price - fib_500) / current_price < 0.01:  # 0.5 레벨 근처
                    strength += 0.1
                
                # 이동평균선 기반 강도
                if abs(price - df['MA20'].iloc[-1]) / current_price < 0.01:
                    strength += 0.15
                
                # 호가 데이터 기반 강도
                if price in bid_walls:
                    strength += 0.2
                
                return strength
            
            # 최적 진입 가격 3개 선정
            entry_levels = []
            for i, price in enumerate(valid_supports[:5]):  # 상위 5개 지지선 검토
                strength = calculate_entry_strength(price)
                entry_price = round(price, 0)
                
                if i == 0:  # 첫 번째 지지선 (가장 높은 가격)
                    ratio = 0.4
                    description = "1차 진입"
                elif i == 1:  # 두 번째 지지선
                    ratio = 0.3
                    description = "2차 진입"
                else:  # 세 번째 지지선
                    ratio = 0.3
                    description = "3차 진입"
                
                change_rate = (entry_price - current_price) / current_price * 100
                
                entry_levels.append({
                    'price': entry_price,
                    'ratio': ratio,
                    'description': f"{description} (현재가 대비 {change_rate:.1f}%)",
                    'strength': strength,
                    'reasons': [
                        f'RSI {rsi:.1f} - {"과매도 구간" if rsi < 30 else "과매수 구간" if rsi > 70 else "중립 구간"}',
                        f'매수세/매도세 비율: {orderbook_analysis["bid_ask_ratio"]:.2f}',
                        f'볼린저 밴드 하단 대비: {((price - df["BB_lower"].iloc[-1]) / df["BB_lower"].iloc[-1] * 100):.1f}%',
                        f'거래량 20일 평균 대비 {df["volume"].iloc[-1]/df["volume"].rolling(20).mean().iloc[-1]*100:.1f}%',
                        f'피보나치 레벨 근접도: {min(abs(price - fib_618), abs(price - fib_500), abs(price - fib_382)) / current_price * 100:.1f}%'
                    ]
                })
                
                if len(entry_levels) >= 3:
                    break
            
            # 강도에 따라 정렬
            entry_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return entry_levels
            
        except Exception as e:
            logger.error(f"최적 진입 가격 계산 중 오류: {str(e)}")
            return []

    def _analyze_market_manipulation(self, df, orderbook_analysis, current_price):
        """세력 개입 가능성 분석"""
        try:
            manipulation_signals = []
            
            # 1. 거래량 급증 패턴 분석
            volume_mean = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].iloc[-1]
            volume_ratio = recent_volume / volume_mean
            
            if volume_ratio > 2.5:
                manipulation_signals.append({
                    'type': '거래량 급증',
                    'strength': min(volume_ratio / 3, 1.0),
                    'description': f'20일 평균 대비 {volume_ratio:.1f}배 거래량 급증',
                    'details': f'평균: {volume_mean:.0f}, 현재: {recent_volume:.0f}'
                })

            # 2. 호가창 불균형 분석
            bid_walls = orderbook_analysis.get('strong_support_levels', [])
            ask_walls = orderbook_analysis.get('strong_resistance_levels', [])
            
            # 매수/매도 벽 크기 비교
            bid_wall_size = sum(wall['size'] for wall in bid_walls) if isinstance(bid_walls, list) else 0
            ask_wall_size = sum(wall['size'] for wall in ask_walls) if isinstance(ask_walls, list) else 0
            
            if bid_wall_size > ask_wall_size * 3 or ask_wall_size > bid_wall_size * 3:
                manipulation_signals.append({
                    'type': '호가 불균형',
                    'strength': 0.8,
                    'description': '한쪽으로 치우친 극단적인 호가 불균형',
                    'details': f'매수벽/매도벽 비율: {bid_wall_size/max(ask_wall_size, 1):.1f}'
                })

            # 3. 가격 급등락 패턴 분석
            price_changes = df['close'].pct_change().iloc[-5:]
            sudden_changes = price_changes[abs(price_changes) > 0.03]  # 3% 이상 변동
            
            if len(sudden_changes) >= 2:
                manipulation_signals.append({
                    'type': '가격 급등락',
                    'strength': 0.7,
                    'description': f'최근 5분봉 중 {len(sudden_changes)}회 급격한 가격 변동',
                    'details': f'최대 변동폭: {max(abs(sudden_changes))*100:.1f}%'
                })

            # 4. 체결 패턴 분석
            if 'trade_history' in orderbook_analysis:
                trades = orderbook_analysis['trade_history']
                large_trades = [t for t in trades if t['size'] > volume_mean * 0.1]  # 평균 거래량의 10% 이상
                
                if large_trades:
                    manipulation_signals.append({
                        'type': '대량 체결',
                        'strength': 0.9,
                        'description': f'최근 {len(large_trades)}건의 대량 체결 발생',
                        'details': f'최대 체결량: {max(t["size"] for t in large_trades):.0f}'
                    })

            # 5. 시간대별 거래 패턴
            time_pattern = df['volume'].groupby(df.index.hour).mean()
            current_hour = df.index[-1].hour
            hour_volume_ratio = df['volume'].iloc[-1] / time_pattern[current_hour]
            
            if hour_volume_ratio > 3:
                manipulation_signals.append({
                    'type': '시간대비 거래량 급증',
                    'strength': min(hour_volume_ratio / 4, 1.0),
                    'description': f'해당 시간대 평균 대비 {hour_volume_ratio:.1f}배 거래량',
                    'details': f'시간대 평균: {time_pattern[current_hour]:.0f}'
                })

            # 6. 호가창 두께 변화
            depth_changes = orderbook_analysis.get('depth_changes', [])
            if depth_changes:
                sudden_depth_changes = [c for c in depth_changes if abs(c) > 0.5]  # 50% 이상 변화
                if sudden_depth_changes:
                    manipulation_signals.append({
                        'type': '호가창 두께 급변',
                        'strength': 0.75,
                        'description': '호가창 두께의 급격한 변화 감지',
                        'details': f'변화율: {max(abs(sudden_depth_changes))*100:.1f}%'
                    })

            # 세력 개입 가능성 종합 점수 계산
            if manipulation_signals:
                total_strength = sum(signal['strength'] for signal in manipulation_signals)
                avg_strength = total_strength / len(manipulation_signals)
                
                return {
                    'manipulation_score': round(avg_strength * 100, 1),  # 0-100 점수
                    'signals': manipulation_signals,
                    'summary': {
                        'level': '강' if avg_strength > 0.7 else '중' if avg_strength > 0.4 else '약',
                        'main_type': manipulation_signals[0]['type'],
                        'signal_count': len(manipulation_signals)
                    },
                    'recommendation': {
                        'action': '주의' if avg_strength > 0.6 else '관망',
                        'message': (
                            '세력 개입 가능성이 높으므로 거래 주의' if avg_strength > 0.6
                            else '일부 세력 신호 감지, 모니터링 필요' if avg_strength > 0.3
                            else '정상적인 거래 흐름'
                        )
                    }
                }
            
            return {
                'manipulation_score': 0,
                'signals': [],
                'summary': {'level': '없음', 'main_type': None, 'signal_count': 0},
                'recommendation': {'action': '정상', 'message': '특이사항 없음'}
            }
            
        except Exception as e:
            logger.error(f"세력 개입 분석 중 오류: {str(e)}")
            return None

    def _analyze_market_condition(self, df, rsi, orderbook_analysis):
        """시장 상황 분석"""
        try:
            # 추세 분석
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            short_ma = df['MA5'].iloc[-1]
            long_ma = df['MA20'].iloc[-1]
            trend = 'BULLISH' if short_ma > long_ma else 'BEARISH'
            
            # 변동성 분석
            volatility = df['close'].pct_change().std() * 100
            
            # 거래량 분석
            volume_trend = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            # 호가 분석
            bid_ask_ratio = orderbook_analysis.get('bid_ask_ratio', 1.0)
            
            # 시장 상황 설명 생성
            description = (
                f"{'상승' if trend == 'BULLISH' else '하락'}추세, "
                f"변동성 {volatility:.1f}%, "
                f"거래량 {'증가' if volume_trend.iloc[-1] > 1 else '감소'} 중, "
                f"{'매수우위' if bid_ask_ratio > 1.1 else '매도우위' if bid_ask_ratio < 0.9 else '균형'} 시장"
            )
            
            return {
                'trend': trend,
                'volatility': volatility,
                'volume_trend': volume_trend.iloc[-1],
                'bid_ask_ratio': bid_ask_ratio,
                'description': description,
                'details': {
                    'ma_trend': {
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'direction': 'UP' if short_ma > long_ma else 'DOWN',
                        'strength': abs(short_ma - long_ma) / long_ma * 100
                    },
                    'volume_analysis': {
                        'current_volume': df['volume'].iloc[-1],
                        'avg_volume': df['volume'].rolling(20).mean().iloc[-1],
                        'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                    },
                    'rsi_condition': {
                        'value': rsi,
                        'status': (
                            '과매수' if rsi > 70 
                            else '과매도' if rsi < 30 
                            else '중립'
                        )
                    },
                    'orderbook_condition': {
                        'bid_ask_ratio': bid_ask_ratio,
                        'status': (
                            '매수우위' if bid_ask_ratio > 1.1 
                            else '매도우위' if bid_ask_ratio < 0.9 
                            else '균형'
                        )
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"시장 상황 분석 중 오류: {str(e)}")
            return {
                'trend': 'NEUTRAL',
                'volatility': 0,
                'volume_trend': 1.0,
                'bid_ask_ratio': 1.0,
                'description': '분석 실패',
                'details': {}
            } 