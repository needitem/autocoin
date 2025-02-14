import streamlit as st
import time
from datetime import datetime
from fear_greed import calculate_returns_and_volatility, FearGreedIndex, get_candles
from market_analysis import analyze_chart, analyze_orderbook, get_orderbook
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from investment_strategy import InvestmentStrategy, format_strategy_message, TradingStrategy

st.set_page_config(
    page_title="암호화폐 시장 분석",
    page_icon="📈",
    layout="wide"
)

def create_candlestick_chart(df):
    """캔들스틱 차트 생성"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('가격', '거래량'),
                       row_width=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df['trade_date'],
                                open=df['opening_price'],
                                high=df['high_price'],
                                low=df['low_price'],
                                close=df['trade_price'],
                                name='캔들'), row=1, col=1)

    fig.add_trace(go.Bar(x=df['trade_date'],
                        y=df['candle_acc_trade_volume'],
                        name='거래량'), row=2, col=1)

    # 이동평균선 추가
    for period in [5, 20, 60]:
        ma = df['trade_price'].rolling(window=period).mean()
        fig.add_trace(go.Scatter(x=df['trade_date'], y=ma,
                               name=f'MA{period}',
                               line=dict(width=1)), row=1, col=1)

    fig.update_layout(
        title='가격 차트',
        yaxis_title='가격',
        yaxis2_title='거래량',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def analyze_market(symbol: str, strategy: InvestmentStrategy):
    """시장 분석 실행"""
    
    # 캔들 데이터 분석
    candles_data = get_candles(f"KRW-{symbol}")
    if not candles_data:
        st.error("데이터를 가져올 수 없습니다.")
        return None
        
    # 캔들 데이터를 DataFrame으로 변환
    df = pd.DataFrame(candles_data)
    df['trade_date'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values('trade_date')
    
    # 호가 데이터 분석
    orderbook = get_orderbook(symbol)
    if not orderbook:
        st.error("호가 데이터를 가져올 수 없습니다.")
        return None
    
    # 차트 분석
    chart_analysis = analyze_chart(df)
    
    # 공포탐욕지수 계산
    prices, volumes = calculate_returns_and_volatility(candles_data)
    fg_index = FearGreedIndex()
    fg_result = fg_index.calculate(prices, volumes)
    
    current_price = float(orderbook['orderbook_units'][0]['ask_price'])
    fg_index_value = fg_result['fear_greed_index'] * 100
    rsi = chart_analysis['rsi']['value']
    
    # 투자 전략 계산
    volatility = chart_analysis['bollinger']['upper'] / chart_analysis['bollinger']['lower'] - 1
    trend_strength = "강함" if chart_analysis['moving_averages']['trend'] == "상승" else "약함"
    
    ma_data = {
        'MA5': chart_analysis['moving_averages']['ma5'],
        'MA20': chart_analysis['moving_averages']['ma20'],
        'MA60': chart_analysis['moving_averages']['ma60']
    }
    
    # 호가 분석 추가
    orderbook_analysis = analyze_orderbook(orderbook)
    
    recommendation = strategy.get_strategy_recommendation(
        current_price=current_price,
        fg_index=fg_index_value,
        rsi=rsi,
        volatility=volatility * 100,
        trend_strength=trend_strength,
        ma_data=ma_data,
        orderbook_analysis=orderbook_analysis,  # 호가 분석 결과 전달
        total_assets=10000000
    )
    
    # 매수/매도 가격을 전략 기반으로 설정
    buy_prices = [level.price for level in recommendation.entry_levels]
    sell_prices = [level.price for level in recommendation.exit_levels]

    return {
        'df': df,
        'current_price': current_price,
        'fg_index': fg_index_value,
        'rsi': rsi,
        'buy_prices': buy_prices,
        'sell_prices': sell_prices,
        'chart_analysis': chart_analysis,
        'orderbook': orderbook,
        'strategy_recommendation': recommendation,
        'trend_strength': trend_strength,
        'ma_data': ma_data
    }

def main():
    st.title("📊 실시간 암호화폐 시장 분석")
    
    # 세션 상태 초기화
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    
    # 사이드바 설정
    st.sidebar.title("설정")
    symbol = st.sidebar.selectbox(
        "코인 선택",
        ["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "MATIC"]
    )
    
    # 코인이 변경되면 분석 초기화
    if st.session_state.current_symbol != symbol:
        st.session_state.current_symbol = symbol
        st.session_state.last_analysis = None
        st.session_state.running = False
    
    # 전략 선택 추가
    strategy_type = st.sidebar.selectbox(
        "트레이딩 전략",
        [
            TradingStrategy.SCALPING.value,
            TradingStrategy.DAYTRADING.value,
            TradingStrategy.SWING.value,
            TradingStrategy.POSITION.value
        ]
    )
    
    # 전략 객체 생성
    strategy = InvestmentStrategy(TradingStrategy(strategy_type))
    
    # 전략 추천 표시
    if st.session_state.last_analysis:
        last_analysis = st.session_state.last_analysis
        # 볼린저 밴드로 변동성 계산
        volatility = (last_analysis['chart_analysis']['bollinger']['upper'] / 
                     last_analysis['chart_analysis']['bollinger']['lower'] - 1) * 100
        
        recommended_strategy = strategy.get_strategy_description(
            fg_index=last_analysis['fg_index'],
            rsi=last_analysis['rsi'],
            volatility=volatility,  # 계산된 변동성 사용
            trend_strength="강함" if last_analysis['chart_analysis']['moving_averages']['trend'] == "상승" else "약함"
        )
        st.sidebar.markdown(recommended_strategy)
    
    # 전략 설명 추가
    strategy_descriptions = {
        TradingStrategy.SCALPING.value: """
        📈 스캘핑 전략 (초단타)
        - 목표 수익: 0.5~1.5%
        - 보유기간: 1시간 이내
        - 투자비중: 총자산의 30%
        - 특징: 빈번한 거래, 작은 수익 실현
        """,
        TradingStrategy.DAYTRADING.value: """
        📊 데이트레이딩 전략 (단타)
        - 목표 수익: 1~3%
        - 보유기간: 1일 이내
        - 투자비중: 총자산의 40%
        - 특징: 일간 변동성 활용
        """,
        TradingStrategy.SWING.value: """
        💹 스윙 트레이딩 전략 (중기)
        - 목표 수익: 3~8%
        - 보유기간: 1-7일
        - 투자비중: 총자산의 50%
        - 특징: 추세 추종
        """,
        TradingStrategy.POSITION.value: """
        🏦 포지션 트레이딩 전략 (장기)
        - 목표 수익: 5~15%
        - 보유기간: 7일 이상
        - 투자비중: 총자산의 60%
        - 특징: 장기 추세 활용
        """
    }
    
    st.sidebar.markdown(strategy_descriptions[strategy_type])
    
    update_interval = st.sidebar.slider(
        "업데이트 주기 (초)",
        min_value=10,
        max_value=300,
        value=60
    )
    
    # 컨테이너 미리 생성
    placeholder = st.empty()
    
    if st.sidebar.button("분석 시작/중지"):
        st.session_state.running = not st.session_state.running
    
    # 분석 실행
    while st.session_state.running:
        with placeholder.container():
            analysis = analyze_market(symbol, strategy)
            
            if analysis:
                st.session_state.last_analysis = analysis
                
                # UI 구성
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 실시간 가격 정보")
                    st.metric(
                        "현재가",
                        f"{analysis['current_price']:,.2f}원",
                        f"{((analysis['current_price']/analysis['df']['trade_price'].iloc[-2])-1)*100:.2f}%"
                    )
                    
                    # 캔들스틱 차트
                    fig = create_candlestick_chart(analysis['df'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 시장 지표")
                    
                    # 게이지 차트로 표시
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("공포탐욕지수", f"{analysis['fg_index']:.1f}")
                        
                    with col2_2:
                        st.metric("RSI", f"{analysis['rsi']:.1f}")
                    
                    # 차트 분석 결과 표시
                    st.subheader("📊 차트 분석")
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # 이동평균선 분석
                        st.write("이동평균선")
                        ma_trend = analysis['chart_analysis']['moving_averages']['trend']
                        ma_color = "🟢" if ma_trend == "상승" else "🔴" if ma_trend == "하락" else "⚪"
                        st.write(f"{ma_color} 추세: {ma_trend}")
                        
                        # 볼린저 밴드 분석
                        st.write("볼린저 밴드")
                        bb = analysis['chart_analysis']['bollinger']
                        bb_position = bb['position']
                        bb_color = "🟢" if bb_position == "하단" else "🔴" if bb_position == "상단" else "⚪"
                        volatility = (bb['upper'] / bb['lower'] - 1) * 100
                        st.write(f"{bb_color} 위치: {bb_position}")
                        st.write(f"변동성: {volatility:.1f}%")
                    
                    with chart_col2:
                        # RSI 분석
                        st.write("RSI")
                        rsi_value = analysis['rsi']
                        rsi_color = "🟢" if rsi_value <= 30 else "🔴" if rsi_value >= 70 else "⚪"
                        rsi_state = "과매도" if rsi_value <= 30 else "과매수" if rsi_value >= 70 else "중립"
                        st.write(f"{rsi_color} {rsi_value:.1f} ({rsi_state})")
                        
                        # 공포탐욕 지수
                        st.write("공포탐욕지수")
                        fg_value = analysis['fg_index']
                        fg_color = "🟢" if fg_value <= 30 else "🔴" if fg_value >= 70 else "⚪"
                        fg_state = "공포" if fg_value <= 30 else "탐욕" if fg_value >= 70 else "중립"
                        st.write(f"{fg_color} {fg_value:.1f} ({fg_state})")
                    
                    # 호가 분석 결과
                    st.write("호가 분석")
                    order_col1, order_col2 = st.columns(2)
                    
                    with order_col1:
                        # 매수 세력
                        buy_pressure = analysis['orderbook'].get('buy_pressure', 1.0)
                        sell_pressure = analysis['orderbook'].get('sell_pressure', 1.0)
                        pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0
                        
                        pressure_color = "🟢" if pressure_ratio > 1.2 else "🔴" if pressure_ratio < 0.8 else "⚪"
                        pressure_state = "매수세 강함" if pressure_ratio > 1.2 else "매도세 강함" if pressure_ratio < 0.8 else "중립"
                        st.write(f"{pressure_color} 매수/매도 비율: {pressure_ratio:.2f}")
                        st.write(f"상태: {pressure_state}")
                    
                    with order_col2:
                        # 지지/저항 레벨
                        support_levels = analysis['orderbook'].get('support_levels', [])
                        resistance_levels = analysis['orderbook'].get('resistance_levels', [])
                        
                        if support_levels:
                            nearest_support = max([p for p in support_levels if p < analysis['current_price']], default=0)
                            if nearest_support > 0:
                                support_diff = (nearest_support / analysis['current_price'] - 1) * 100
                                st.write(f"가까운 지지선: {nearest_support:,.0f}원 ({support_diff:.1f}%)")
                        
                        if resistance_levels:
                            nearest_resistance = min([p for p in resistance_levels if p > analysis['current_price']], default=0)
                            if nearest_resistance > 0:
                                resistance_diff = (nearest_resistance / analysis['current_price'] - 1) * 100
                                st.write(f"가까운 저항선: {nearest_resistance:,.0f}원 ({resistance_diff:.1f}%)")
                
                # 매수/매도 가격
                st.subheader("💰 매수/매도 추천 가격")
                
                # 전략 근거 설명
                st.markdown("""
                ##### 📈 전략 선정 근거
                """)
                strategy_col3, strategy_col4 = st.columns(2)
                
                with strategy_col3:
                    st.write("기술적 지표")
                    st.markdown(f"""
                    - RSI: {analysis['rsi']:.1f} ({'과매도' if analysis['rsi'] <= 30 else '과매수' if analysis['rsi'] >= 70 else '중립'})
                    - 이동평균선: {analysis['chart_analysis']['moving_averages']['trend']} 추세
                    - 볼린저밴드: {analysis['chart_analysis']['bollinger']['position']} 위치
                    - 변동성: {volatility:.1f}%
                    """)
                
                with strategy_col4:
                    st.write("시장 상황")
                    st.markdown(f"""
                    - 공포탐욕지수: {analysis['fg_index']:.1f} ({fg_state})
                    - 매수/매도 세력: {pressure_ratio:.2f} ({pressure_state})
                    - 추세 강도: {analysis['trend_strength']}
                    """)
                
                price_col1, price_col2 = st.columns(2)
                
                with price_col1:
                    st.write("매수 전략")
                    for i, (price, level) in enumerate(zip(analysis['buy_prices'], 
                                                         analysis['strategy_recommendation'].entry_levels), 1):
                        with st.expander(f"{level.description} ({((price/analysis['current_price'])-1)*100:.1f}%)"):
                            st.metric(
                                "매수가",
                                f"{price:,.2f}원",
                                f"{((price/analysis['current_price'])-1)*100:.1f}%"
                            )
                            st.caption(f"투자금액: {level.ratio * analysis['strategy_recommendation'].investment_amount:,.0f}원")
                            
                            # 매수 가격 선정 근거
                            st.markdown("##### 가격 선정 근거")
                            
                            # 지지선 근처 여부 확인
                            nearest_support = max([p for p in support_levels if p < price], default=0)
                            if nearest_support > 0 and nearest_support > price * 0.99:
                                st.markdown(f"- 지지선 ({nearest_support:,.0f}원) 근처")
                            
                            # 이동평균선 기반 설명
                            ma5, ma20, ma60 = (analysis['ma_data']['MA5'], 
                                              analysis['ma_data']['MA20'], 
                                              analysis['ma_data']['MA60'])
                            
                            # 이동평균선 상태 설명
                            ma_conditions = []
                            if price < ma5 < ma20:
                                ma_conditions.append("- 5일선 하향돌파 구간 (약세)")
                            elif ma5 < price < ma20:
                                ma_conditions.append("- 20일선 지지 구간 (중립)")
                            elif price > ma20 > ma5:
                                ma_conditions.append("- 20일선 상향돌파 구간 (강세)")
                            
                            if ma_conditions:
                                st.markdown("\n".join(ma_conditions))
                            
                            # 매수세/매도세 기반 설명
                            if pressure_ratio > 1.2:
                                st.markdown("- 매수세가 강해 상승 가능성")
                            elif pressure_ratio < 0.8:
                                st.markdown("- 매도세가 강해 추가 하락 가능성")
                            else:
                                st.markdown("- 매수/매도 세력 균형 상태")
                            
                            # RSI 기반 설명
                            if analysis['rsi'] <= 30:
                                st.markdown("- RSI 과매도 구간으로 매수 적절")
                            elif analysis['rsi'] >= 70:
                                st.markdown("- RSI 과매수 구간으로 매수 신중")
                            else:
                                st.markdown(f"- RSI {analysis['rsi']:.1f} 중립 구간")
                            
                            # 볼린저 밴드 기반 설명
                            bb_position = analysis['chart_analysis']['bollinger']['position']
                            if bb_position == "하단":
                                st.markdown("- 볼린저 밴드 하단 지지 구간")
                            elif bb_position == "상단":
                                st.markdown("- 볼린저 밴드 상단 저항 구간")
                
                with price_col2:
                    st.write("매도 전략")
                    for i, (price, level) in enumerate(zip(analysis['sell_prices'],
                                                         analysis['strategy_recommendation'].exit_levels), 1):
                        with st.expander(f"{level.description} (+{((price/analysis['current_price'])-1)*100:.1f}%)"):
                            st.metric(
                                "매도가",
                                f"{price:,.2f}원",
                                f"+{((price/analysis['current_price'])-1)*100:.1f}%"
                            )
                            
                            # 매도 가격 선정 근거
                            st.markdown("##### 가격 선정 근거")
                            
                            # 저항선 근처 여부 확인
                            nearest_resistance = min([p for p in resistance_levels if p > price], default=0)
                            if nearest_resistance > 0 and nearest_resistance < price * 1.01:
                                st.markdown(f"- 저항선 ({nearest_resistance:,.0f}원) 근처")
                            
                            # 이동평균선 기반 설명
                            if price > ma20 > ma5:
                                st.markdown("- 20일선 상향돌파로 상승 추세")
                            elif ma60 > price > ma20:
                                st.markdown("- 60일선 저항으로 상승 제한")
                            
                            # 매수세/매도세 기반 설명
                            if pressure_ratio > 1.2:
                                st.markdown("- 매수세 강해 상승 여력 존재")
                            elif pressure_ratio < 0.8:
                                st.markdown("- 매도세 강해 하락 가능성")
                            else:
                                st.markdown("- 매수/매도 세력 균형 상태")
                            
                            # RSI 기반 설명
                            if analysis['rsi'] >= 70:
                                st.markdown("- RSI 과매수 구간으로 매도 적절")
                            elif analysis['rsi'] <= 30:
                                st.markdown("- RSI 과매도 구간으로 매도 신중")
                            
                            # 볼린저 밴드 기반 설명
                            if bb_position == "상단":
                                st.markdown("- 볼린저 밴드 상단으로 매도 고려")
                            elif bb_position == "하단":
                                st.markdown("- 볼린저 밴드 하단으로 매도 신중")
                
                # 손절가 정보
                st.info(f"""
                💡 손절가: {analysis['strategy_recommendation'].stop_loss:,.0f}원 
                ({((analysis['strategy_recommendation'].stop_loss/analysis['current_price'])-1)*100:.1f}%)
                
                손절가 선정 근거:
                - 최저 매수가의 97% 또는 주요 이동평균선의 98% 중 높은 가격
                - 이동평균선 기반 지지선 고려
                - 변동성 구간 반영
                """)
            
            time.sleep(update_interval)
            st.rerun()

if __name__ == "__main__":
    main() 