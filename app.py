import streamlit as st
import time
import logging
from datetime import datetime
import plotly.graph_objects as go
from investment_strategy import InvestmentStrategy, TradingStrategy
from market_analyzer import MarketAnalyzer
from chart_visualizer import ChartVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="암호화폐 시장 분석",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title('암호화폐 자동매매 시스템')
    
    # 세션 상태 초기화
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'running' not in st.session_state:
        st.session_state.running = True
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = "BTC"  # 기본값 설정
    
    # 코인 선택 (사이드바로 이동)
    st.session_state.current_symbol = st.sidebar.selectbox(
        "코인 선택",
        ["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE", "MATIC"],
        index=0
    )
    
    # 전략 선택
    strategy_options = {
        "스캘핑 (단타)": "SCALPING",
        "스윙": "SWING",
        "포지션": "POSITION",
        "보수적": "CONSERVATIVE",
        "중립적": "MODERATE",
        "공격적": "AGGRESSIVE"
    }
    
    strategy_type = st.selectbox(
        "매매 전략 선택",
        options=list(strategy_options.keys()),
        index=0
    )
    
    # 선택된 표시 텍스트를 enum 값으로 변환
    strategy_enum_value = strategy_options[strategy_type]
    
    # 전략 설명
    strategy_descriptions = {
        "SCALPING": """
        스캘핑 전략: 단기 변동성을 이용한 빈번한 거래
        - 짧은 시간 동안의 가격 변동 활용
        - 작은 수익을 자주 실현
        - 빠른 진입과 퇴출이 핵심
        """,
        "SWING": """
        스윙 전략: 중기 추세를 이용한 거래
        - 수일에서 수주 동안의 추세 활용
        - 기술적 분석과 차트 패턴 중시
        - 적절한 진입/퇴출 포인트 포착이 중요
        """,
        "POSITION": """
        포지션 전략: 장기 추세를 이용한 거래
        - 수주에서 수개월 동안의 큰 추세 활용
        - 펀더멘털 분석 중시
        - 높은 수익을 위해 충분한 인내심 필요
        """,
        "CONSERVATIVE": """
        보수적 전략: 보수적인 투자 전략
        - 낮은 위험을 선호
        - 장기적인 관점에서 투자
        """,
        "MODERATE": """
        중립적 전략: 중립적인 투자 전략
        - 중립적인 자산 배분
        - 중간 수준의 위험과 수익 기대
        """,
        "AGGRESSIVE": """
        공격적 전략: 공격적인 투자 전략
        - 높은 위험을 선호
        - 높은 수익을 위해 과도한 리스크 허용
        """,
    }
    st.sidebar.markdown(strategy_descriptions[strategy_enum_value])
    
    # 업데이트 주기 설정
    update_interval = st.sidebar.slider(
        "업데이트 주기 (초)",
        min_value=10,
        max_value=300,
        value=20
    )
    
    # 분석기 및 시각화 도구 초기화
    market_analyzer = MarketAnalyzer()
    chart_visualizer = ChartVisualizer()
    
    # 컨테이너 미리 생성
    placeholder = st.empty()
    
    # 분석 실행
    while st.session_state.running:
        with placeholder.container():
            # 시장 분석 실행
            analysis = market_analyzer.analyze_market(
                st.session_state.current_symbol, 
                InvestmentStrategy(getattr(TradingStrategy, strategy_enum_value))
            )
            
            if analysis:
                st.session_state.last_analysis = analysis
                
                # 기본 정보 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("현재가", f"{analysis['current_price']:,}원")
                with col2:
                    rsi = analysis['technical_indicators']['rsi']
                    st.metric("RSI", f"{rsi:.1f}")
                with col3:
                    trend_strength = analysis['technical_indicators'].get('trend_strength', 50.0)  # 기본값 50.0
                    trend_color = "🟢" if trend_strength > 60 else "🔴" if trend_strength < 40 else "🟡"
                    st.metric(
                        "추세 강도", 
                        f"{trend_color} {trend_strength:.1f}",
                        help="0-100 사이의 값. 높을수록 강한 추세"
                    )
                with col4:
                    macd = analysis['technical_indicators']['macd']['macd']
                    st.metric("MACD", f"{macd:.1f}")
                
                # 차트 섹션
                st.subheader("📈 가격 차트 분석")
                tab1, tab2, tab3 = st.tabs(["캔들스틱", "기술적 지표", "예측"])
                
                with tab1:
                    # 캔들스틱 차트
                    candlestick_fig = chart_visualizer.create_candlestick_chart(analysis['df'])
                    st.plotly_chart(candlestick_fig, use_container_width=True)
                
                with tab2:
                    # 기술적 지표 차트
                    indicators_fig = chart_visualizer.create_technical_indicators_chart(
                        analysis['df'],
                        analysis['technical_indicators']
                    )
                    st.plotly_chart(indicators_fig, use_container_width=True)
                
                with tab3:
                    # 예측 차트
                    if analysis['pattern_analysis'] and analysis['pattern_analysis']['patterns']:
                        prediction_fig = chart_visualizer.create_prediction_chart(
                            analysis['df'],
                            analysis['pattern_analysis'],
                            analysis['current_price']
                        )
                        st.plotly_chart(prediction_fig, use_container_width=True)
                    else:
                        st.info("현재 예측 데이터가 없습니다.")
                
                # 패턴 분석 결과
                st.subheader("📊 차트 패턴 분석")
                if analysis['pattern_analysis'] and analysis['pattern_analysis']['patterns']:
                    for pattern in analysis['pattern_analysis']['patterns']:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            pattern_type = "상승" if pattern['pattern_type'] == 'bullish' else "하락"
                            st.write(f"🎯 {pattern['name']} ({pattern_type} 패턴)")
                            st.caption(f"신뢰도: {pattern['reliability']}")
                        with col2:
                            if 'target' in pattern:
                                target_price = float(pattern['target'])
                                target_percent = (target_price/analysis['current_price'] - 1) * 100
                                st.metric("목표가", f"{target_price:,.0f}원", f"{target_percent:+.1f}%")
                else:
                    st.info("현재 특별한 차트 패턴이 발견되지 않았습니다.")
                
                # 호가 분석
                if analysis['orderbook_analysis']:
                    st.subheader("📚 호가 분석")
                    col1, col2 = st.columns(2)
                    with col1:
                        bid_ratio = analysis['orderbook_analysis']['bid_ask_ratio']
                        st.metric("매수/매도 비율", f"{bid_ratio:.2f}")
                    with col2:
                        bid_conc = analysis['orderbook_analysis']['bid_concentration']
                        st.metric("매수 집중도", f"{bid_conc:.1%}")
                
                # 투자 전략 추천
                if analysis['strategy_recommendation']:
                    st.subheader("💡 투자 전략 추천")
                    rec = analysis['strategy_recommendation']
                    
                    # 전략 개요와 근거
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # 전략 선정 근거
                        st.markdown("#### 📊 전략 선정 근거")
                        
                        # 시장 상황 요약
                        market_status = []
                        
                        # 호가 분석 기반
                        if analysis['orderbook_analysis']:
                            bid_ratio = analysis['orderbook_analysis']['bid_ask_ratio']
                            if bid_ratio > 1.2:
                                market_status.append("• 매수세 우위 (매수/매도 비율: {:.2f})".format(bid_ratio))
                            elif bid_ratio < 0.8:
                                market_status.append("• 매도세 우위 (매수/매도 비율: {:.2f})".format(bid_ratio))
                        
                        # RSI 기반
                        rsi = analysis['technical_indicators']['rsi']
                        if rsi > 70:
                            market_status.append(f"• 과매수 구간 (RSI: {rsi:.1f})")
                        elif rsi < 30:
                            market_status.append(f"• 과매도 구간 (RSI: {rsi:.1f})")
                        
                        # 패턴 분석 기반
                        if analysis['pattern_analysis']['patterns']:
                            for pattern in analysis['pattern_analysis']['patterns']:
                                pattern_type = "상승" if pattern['pattern_type'] == 'bullish' else "하락"
                                market_status.append(f"• {pattern['name']} ({pattern_type} 패턴, 신뢰도: {pattern['reliability']})")
                        
                        # 추세 강도
                        trend_strength = analysis['technical_indicators'].get('trend_strength', 50)
                        trend_direction = "상승" if trend_strength > 60 else "하락" if trend_strength < 40 else "중립"
                        market_status.append(f"• {trend_direction} 추세 (강도: {trend_strength:.1f})")
                        
                        # 변동성
                        volatility = analysis['technical_indicators'].get('volatility', 0)
                        market_status.append(f"• 변동성: {volatility:.1f}%")
                        
                        # MACD 신호
                        macd = analysis['technical_indicators']['macd']
                        if macd['macd'] > macd['signal']:
                            market_status.append("• MACD 매수 신호")
                        else:
                            market_status.append("• MACD 매도 신호")
                        
                        # 시장 상황 표시
                        st.markdown("\n".join(market_status))
                        
                    with col2:
                        # 핵심 지표
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("전략 유형", rec['strategy_type'])
                            st.metric("신뢰도", f"{rec['confidence_score']:.1%}")
                        with col2:
                            st.metric("보유기간", rec['holding_period'])
                            st.metric("리스크", f"{rec['risk_ratio']:.1%}")
                    
                    # 구분선
                    st.markdown("---")
                    
                    # 매매 전략 상세
                    col1, col2 = st.columns(2)
                    
                    # 진입 전략
                    with col1:
                        st.markdown("#### 📥 진입 전략")
                        for level in rec['entry_levels']:
                            with st.expander(level['description']):
                                price_diff = (level['price']/analysis['current_price'] - 1) * 100
                                st.metric(
                                    "진입 가격",
                                    f"{level['price']:,.0f}원",
                                    f"{price_diff:+.1f}% (비중: {level['ratio']:.0%})"
                                )
                                
                                # 진입 가격 설정 근거
                                st.markdown("##### 📊 진입 가격 설정 근거:")
                                for reason in level['reasons']:
                                    st.markdown(f"• {reason}")
                        
                        # 손절가
                        stop_loss_percent = (rec['stop_loss']/analysis['current_price'] - 1) * 100
                        st.metric("🛑 손절가", f"{rec['stop_loss']:,.0f}원", f"{stop_loss_percent:+.1f}%")
                    
                    # 청산 전략
                    with col2:
                        st.markdown("#### 📤 청산 전략")
                        for level in rec['exit_levels']:
                            price_diff = (level['price']/analysis['current_price'] - 1) * 100
                            st.metric(
                                level['description'],
                                f"{level['price']:,.0f}원",
                                f"{price_diff:+.1f}%"
                            )
                        
                        # 투자 규모
                        st.metric("💰 투자 규모", f"{rec['investment_amount']:,.0f}원")
                    
                    # 주의사항
                    st.info("""
                    ⚠️ 주의사항:
                    - 투자는 본인의 판단과 책임하에 진행하세요.
                    - 시장 상황에 따라 전략을 유연하게 조정하세요.
                    - 설정된 손절가를 반드시 준수하세요.
                    """)
            
            time.sleep(update_interval)
            st.rerun()

if __name__ == "__main__":
    main() 