"""
Streamlit 기반 암호화폐 트레이딩 애플리케이션
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.api.upbit import UpbitTradingSystem
from src.core.trading import TradingManager
import numpy as np
import plotly.graph_objects as go

class AutoCoinApp:
    def __init__(self, trading_manager: TradingManager = None):
        """앱 초기화"""
        self.trading_manager = trading_manager or TradingManager()
        self.setup_page()

    def setup_page(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="AutoCoin Trading",
            page_icon="📈",
            layout="wide"
        )
        
        # 헤더 레이아웃
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.title("AutoCoin Trading")
        with col2:
            current_exchange = self.trading_manager.get_current_exchange()
            st.metric("현재 거래소", current_exchange)
        with col3:
            # 거래소 선택 드롭다운
            exchange_options = ["Upbit", "Bithumb"]
            selected_exchange = st.selectbox(
                "거래소 선택",
                exchange_options,
                index=exchange_options.index(current_exchange)
            )
            
            # 거래소 변경
            if selected_exchange != current_exchange:
                if self.trading_manager.switch_exchange(selected_exchange):
                    st.success(f"{selected_exchange}로 전환되었습니다.")
                    st.rerun()
        with col4:
            # 종료 버튼
            if st.button("🛑 종료", type="secondary", use_container_width=True):
                st.error("앱을 종료합니다...")
                st.balloons()
                import time
                import os
                import signal
                import subprocess
                
                # 메시지 표시
                with st.spinner("프로그램을 종료하는 중..."):
                    time.sleep(2)
                
                # 브라우저 닫기 (Windows)
                try:
                    subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                                 capture_output=True, text=True)
                    subprocess.run(["taskkill", "/f", "/im", "msedge.exe"], 
                                 capture_output=True, text=True)
                    subprocess.run(["taskkill", "/f", "/im", "firefox.exe"], 
                                 capture_output=True, text=True)
                except:
                    pass
                
                # Streamlit 프로세스 종료
                try:
                    # 현재 프로세스 종료
                    os.kill(os.getpid(), signal.SIGTERM)
                except:
                    # 강제 종료
                    os._exit(0)

    def render_market_selector(self):
        """마켓 선택 UI"""
        markets = self.trading_manager.get_markets()
        market_codes = [m['market'] for m in markets if m['market'].startswith('KRW-')]
        selected_market = st.selectbox("마켓 선택", market_codes)
        return selected_market

    def render_market_data(self, market: str):
        """시장 데이터 표시"""
        try:
            data = self.trading_manager.get_market_data(market)
            if data is None or not isinstance(data, dict):
                st.warning("시장 데이터를 불러올 수 없습니다")
                return

            # 필수 필드 확인
            required_fields = ['trade_price', 'signed_change_rate', 'acc_trade_volume_24h', 'high_price', 'low_price']
            if not all(field in data for field in required_fields):
                st.warning("일부 시장 데이터가 누락되었습니다")
                return

            # 데이터 유효성 검사
            for field in required_fields:
                if not isinstance(data[field], (int, float)) or data[field] < 0:
                    st.warning("일부 시장 데이터가 유효하지 않습니다")
                    return

            # 기본 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "현재가",
                    f"{data['trade_price']:,} KRW",
                    f"{data['signed_change_rate']*100:.2f}%"
                )
            with col2:
                st.metric(
                    "거래량",
                    f"{data['acc_trade_volume_24h']:,.0f}"
                )
            with col3:
                st.metric(
                    "고가/저가",
                    f"{data['high_price']:,} / {data['low_price']:,}"
                )

            # 뉴스 섹션 추가
            from src.ui.components.news import render_news_section
            render_news_section(market, data['signed_change_rate']*100)

        except Exception as e:
            st.error(f"시장 데이터 표시 중 오류가 발생했습니다: {str(e)}")
            return

    def render_trading_interface(self, market: str):
        """거래 인터페이스"""
        with st.expander("매매 설정", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                order_type = st.radio("주문 유형", ["시장가", "지정가"])
                side = st.radio("매수/매매", ["매수", "매도"])
                
            with col2:
                amount = st.number_input("주문 수량", min_value=0.0001, step=0.0001)
                if order_type == "지정가":
                    price = st.number_input("주문 가격", min_value=1, step=1000)
                
            if st.button("주문 실행"):
                try:
                    result = self.trading_manager.place_order(
                        market=market,
                        side=side,
                        ord_type=order_type,
                        volume=amount,
                        price=price if order_type == "지정가" else None
                    )
                    if result:
                        st.success("주문이 성공적으로 실행되었습니다")
                    else:
                        st.error("주문 실행에 실패했습니다")
                except Exception as e:
                    st.error(f"주문 중 오류가 발생했습니다: {str(e)}")

    def render_technical_analysis(self, market: str):
        """기술적 분석 차트"""
        ohlcv = self.trading_manager.get_ohlcv(market, count=200)
        if ohlcv is None or ohlcv.empty:
            st.warning("차트 데이터를 불러올 수 없습니다")
            return

        # 기술적 지표 계산
        indicators = self.trading_manager.calculate_indicators(ohlcv)
        
        # 캔들스틱 차트 생성
        fig = go.Figure()
        
        # 캔들스틱 추가
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['close'],
            name='가격',
            increasing_line_color='red',
            decreasing_line_color='blue'
        ))
        
        # 이동평균선 추가
        if indicators and 'MA5' in indicators:
            fig.add_trace(go.Scatter(
                x=ohlcv.index,
                y=indicators['MA5'],
                mode='lines',
                name='MA5',
                line=dict(color='orange', width=1)
            ))
        
        if indicators and 'MA20' in indicators:
            fig.add_trace(go.Scatter(
                x=ohlcv.index,
                y=indicators['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='green', width=1)
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{market} 가격 차트',
            yaxis_title='가격 (KRW)',
            xaxis_title='시간',
            height=600,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI 차트
        if indicators and 'rsi' in indicators:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=ohlcv.index,
                y=indicators['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # RSI 기준선
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도")
            
            rsi_fig.update_layout(
                title='RSI',
                yaxis_title='RSI',
                xaxis_title='시간',
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(rsi_fig, use_container_width=True)

    def _render_technical_analysis(self, indicators: dict):
        """기술적 분석 차트 내부 렌더링"""
        if not indicators:
            st.warning("기술적 지표를 계산할 수 없습니다")
            return

        # 이동평균선
        if 'moving_averages' in indicators:
            st.subheader("이동평균선")
            st.line_chart(indicators['moving_averages'])

        # 볼린저 밴드
        if 'bollinger_bands' in indicators:
            st.subheader("볼린저 밴드")
            bb_df = pd.DataFrame({
                'Upper': indicators['bollinger_bands']['upper'],
                'Middle': indicators['bollinger_bands']['middle'],
                'Lower': indicators['bollinger_bands']['lower']
            })
            st.line_chart(bb_df)

        # MACD
        if 'macd' in indicators:
            st.subheader("MACD")
            macd_df = pd.DataFrame({
                'MACD': indicators['macd']['macd'],
                'Signal': indicators['macd']['signal'],
                'Histogram': indicators['macd']['hist']
            })
            st.line_chart(macd_df)

        # RSI
        if 'rsi' in indicators:
            st.subheader("RSI")
            st.line_chart(indicators['rsi'])

        # 스토캐스틱
        if 'stochastic' in indicators:
            st.subheader("스토캐스틱")
            stoch_df = pd.DataFrame({
                'Fast %K': indicators['stochastic']['fast']['k'],
                'Fast %D': indicators['stochastic']['fast']['d'],
                'Slow %K': indicators['stochastic']['slow']['k'],
                'Slow %D': indicators['stochastic']['slow']['d']
            })
            st.line_chart(stoch_df)

    def run(self):
        """앱 실행"""
        # 사이드바에 메뉴 추가
        with st.sidebar:
            st.title("📊 메뉴")
            page = st.radio(
                "페이지 선택:",
                ["📈 기본 트레이딩", "🤖 AI 포트폴리오", "📰 뉴스 분석", "📊 차트 분석"],
                index=0
            )
        
        # 세션 상태에 API 객체 저장 (포트폴리오에서 사용)
        if 'upbit_api' not in st.session_state:
            st.session_state.upbit_api = self.trading_manager.api
        
        if 'news_api' not in st.session_state:
            from src.api.news import CryptoNewsAPI
            st.session_state.news_api = CryptoNewsAPI()
        
        if page == "📈 기본 트레이딩":
            self.render_basic_trading()
        elif page == "🤖 AI 포트폴리오":
            from src.ui.components.portfolio_dashboard import create_portfolio_dashboard
            create_portfolio_dashboard()
        elif page == "📰 뉴스 분석":
            self.render_news_analysis()
        elif page == "📊 차트 분석":
            self.render_chart_analysis()
    
    def render_basic_trading(self):
        """기본 트레이딩 페이지"""
        # 마켓 선택
        selected_market = self.render_market_selector()
        
        # 시장 데이터 표시
        self.render_market_data(selected_market)
        
        # 차트 및 기술적 분석
        self.render_technical_analysis(selected_market)
        
        # 거래 인터페이스
        self.render_trading_interface(selected_market)
    
    def render_news_analysis(self):
        """뉴스 분석 페이지"""
        from src.ui.components.news import render_news_section
        
        st.title("📰 뉴스 분석")
        
        # 마켓 선택
        selected_market = self.render_market_selector()
        
        # 현재 시장 데이터 가져오기
        market_data = self.trading_manager.get_market_data(selected_market)
        change_rate = market_data.get('signed_change_rate', 0) * 100 if market_data else 0
        
        # 뉴스 섹션 렌더링
        render_news_section(selected_market, change_rate)
    
    def render_chart_analysis(self):
        """차트 분석 페이지"""
        st.title("📊 차트 패턴 분석")
        
        # 마켓 선택
        selected_market = self.render_market_selector()
        
        # 차트 패턴 분석
        from src.ui.components.chart_analysis import render_chart_analysis_section
        render_chart_analysis_section(self.trading_manager, selected_market)

if __name__ == "__main__":
    app = AutoCoinApp()
    app.run() 