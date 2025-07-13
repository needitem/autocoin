"""
Streamlit 기반 암호화폐 트레이딩 애플리케이션
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.api.upbit import UpbitTradingSystem
from src.core.trading import TradingManager
import numpy as np

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
        col1, col2, col3 = st.columns([2, 1, 1])
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
        ohlcv = self.trading_manager.get_ohlcv(market)
        if ohlcv is None or ohlcv.empty:
            st.warning("차트 데이터를 불러올 수 없습니다")
            return

        # 기술적 지표 계산
        indicators = self.trading_manager.calculate_indicators(ohlcv)
        
        # 차트 표시
        st.line_chart(ohlcv[['close']])
        
        # 주요 지표 표시
        col1, col2 = st.columns(2)
        with col1:
            if 'ma' in indicators:
                st.line_chart(indicators['ma'])
        with col2:
            if 'rsi' in indicators:
                st.line_chart(indicators['rsi'])

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
        # 마켓 선택
        selected_market = self.render_market_selector()
        
        # 시장 데이터 표시
        self.render_market_data(selected_market)
        
        # 차트 및 기술적 분석
        self.render_technical_analysis(selected_market)
        
        # 거래 인터페이스
        self.render_trading_interface(selected_market)

if __name__ == "__main__":
    app = AutoCoinApp()
    app.run() 