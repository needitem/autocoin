"""
Market Metrics Component

This component displays detailed market metrics and technical analysis data.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from datetime import datetime

class MarketMetricsComponent:
    def render(self, market_data: Dict[str, Any]):
        """Render market metrics interface."""
        if not market_data:
            st.warning("시장 데이터를 불러올 수 없습니다.")
            return
        
        # Price Information
        st.subheader("가격 정보")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "현재가",
                f"₩{market_data.get('current_price', 0):,.0f}",
                f"{market_data.get('price_change_24h', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "거래량 (24H)",
                f"₩{market_data.get('volume_24h', 0):,.0f}",
                f"{market_data.get('volume_change_24h', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                "시가총액",
                f"₩{market_data.get('market_cap', 0):,.0f}"
            )
        
        # Technical Analysis
        st.subheader("기술적 분석")
        
        # Moving Averages
        ma_col1, ma_col2, ma_col3 = st.columns(3)
        
        with ma_col1:
            ma20 = market_data.get('moving_averages', {}).get('ma_20', 0)
            st.metric(
                "20일 이동평균",
                f"₩{ma20:,.0f}",
                f"현재가 대비 {((market_data.get('current_price', 0) - ma20) / ma20 * 100):.2f}%"
            )
        
        with ma_col2:
            ma50 = market_data.get('moving_averages', {}).get('ma_50', 0)
            st.metric(
                "50일 이동평균",
                f"₩{ma50:,.0f}",
                f"현재가 대비 {((market_data.get('current_price', 0) - ma50) / ma50 * 100):.2f}%"
            )
        
        with ma_col3:
            ma200 = market_data.get('moving_averages', {}).get('ma_200', 0)
            st.metric(
                "200일 이동평균",
                f"₩{ma200:,.0f}",
                f"현재가 대비 {((market_data.get('current_price', 0) - ma200) / ma200 * 100):.2f}%"
            )
        
        # Oscillators and Momentum
        st.subheader("보조지표")
        osc_col1, osc_col2, osc_col3, osc_col4 = st.columns(4)
        
        with osc_col1:
            rsi = market_data.get('oscillators', {}).get('rsi', 0)
            st.metric(
                "RSI (14)",
                f"{rsi:.1f}",
                "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립")
            )
        
        with osc_col2:
            macd = market_data.get('oscillators', {}).get('macd', {})
            st.metric(
                "MACD",
                f"{macd.get('value', 0):.1f}",
                f"Signal: {macd.get('signal', 0):.1f}"
            )
        
        with osc_col3:
            bb = market_data.get('volatility', {}).get('bollinger_bands', {})
            st.metric(
                "볼린저 밴드",
                f"폭: {bb.get('width', 0):.1f}%",
                f"위치: {bb.get('position', 'N/A')}"
            )
        
        with osc_col4:
            st.metric(
                "거래량 강도",
                f"{market_data.get('volume_analysis', {}).get('strength', 0):.1f}",
                "상승" if market_data.get('volume_analysis', {}).get('is_rising', False) else "하락"
            )
        
        # Market Trend and Signals
        st.subheader("시장 동향")
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            trend = market_data.get('market_conditions', {}).get('trend', 'NEUTRAL')
            trend_map = {
                'STRONG_UPTREND': '강한 상승세',
                'UPTREND': '상승세',
                'SIDEWAYS': '횡보',
                'DOWNTREND': '하락세',
                'STRONG_DOWNTREND': '강한 하락세'
            }
            st.metric(
                "추세",
                trend_map.get(trend, '알 수 없음'),
                f"모멘텀: {market_data.get('market_conditions', {}).get('momentum', 0):.1f}%"
            )
        
        with trend_col2:
            signals = market_data.get('signals', {})
            signal_strength = len([s for s in signals.values() if s in ['BUY', 'BULLISH']])
            total_signals = len(signals)
            if total_signals > 0:
                buy_ratio = signal_strength / total_signals * 100
                st.metric(
                    "매수 신호 강도",
                    f"{buy_ratio:.1f}%",
                    f"{signal_strength}/{total_signals} 지표 매수 신호"
                )
        
        # Support and Resistance
        st.subheader("지지/저항 레벨")
        sr_col1, sr_col2 = st.columns(2)
        
        with sr_col1:
            resistance = market_data.get('support_resistance', {}).get('resistance', [])
            if resistance:
                st.write("저항 레벨:")
                for level in sorted(resistance)[:3]:
                    st.text(f"₩{level:,.0f}")
        
        with sr_col2:
            support = market_data.get('support_resistance', {}).get('support', [])
            if support:
                st.write("지지 레벨:")
                for level in sorted(support, reverse=True)[:3]:
                    st.text(f"₩{level:,.0f}")
        
        # Volume Profile
        if market_data.get('volume_profile'):
            st.subheader("거래량 프로파일")
            volume_profile = pd.DataFrame(market_data['volume_profile'])
            st.bar_chart(volume_profile.set_index('price')['volume']) 