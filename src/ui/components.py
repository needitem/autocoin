"""
UI Components Module

This module contains all the Streamlit UI components used in the application.
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from src.config import AppConfig
import plotly.graph_objects as go

class MarketMetricsComponent:
    def render(self, data: Dict[str, Any]) -> None:
        if not data:
            st.warning('시장 데이터를 불러올 수 없습니다.')
            return
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                '현재가',
                f"{data['current_price']:,.0f}원",
                f"{data['daily_change']:.2f}%"
            )
        with col2:
            st.metric(
                '매수 세력',
                f"{data['buy_pressure']:.1f}%"
            )
        with col3:
            st.metric(
                '거래량',
                f"{data.get('volume', 0):,.0f}원"
            )

class VirtualTradingComponent:
    def render(self, data: Dict[str, Any]) -> None:
        if not data:
            st.warning('거래 데이터를 불러올 수 없습니다.')
            return
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                '계좌 잔고',
                f"{data['balance']:,.0f}원",
                f"{data['total_profit']:,.0f}원"
            )
        with col2:
            st.metric(
                '승률',
                f"{data['win_rate']:.1f}%"
            )
        with col3:
            st.metric(
                '거래 횟수',
                f"{len(data['trade_history']):,d}회"
            )
        
        if data['trade_history']:
            self._render_trade_history(data['trade_history'])
    
    def _render_trade_history(self, history: List[Dict[str, Any]]) -> None:
        st.subheader('최근 거래 내역')
        for trade in history:
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text(trade['timestamp'][:16])
                with col2:
                    color = 'blue' if trade['action'] == 'BUY' else 'red'
                    st.markdown(
                        f"<span style='color: {color}'>{trade['action']}</span> "
                        f"{trade['amount']:,.0f}원",
                        unsafe_allow_html=True
                    )

class StrategyAnalysisComponent:
    def render(self, data: Dict[str, Any]) -> None:
        if not data:
            st.warning('분석 데이터를 불러올 수 없습니다.')
            return
            
        st.subheader('전략 분석')
        
        # 전략 신호
        strategy = data['strategy']
        st.markdown(
            f"**매매 신호**: {strategy['action']} "
            f"(신뢰도: {strategy['confidence']:.1f}%)"
        )
        if strategy['reason']:
            st.text(f"사유: {strategy['reason']}")
        
        # 기술적 지표
        technical = data['technical']
        st.markdown(
            f"**추세**: {technical['trend']} "
            f"(강도: {technical['strength']:.1f})"
        )
        if technical['indicators']:
            self._render_indicators(technical['indicators'])
        
        # 시장 분석
        market = data['market']
        st.markdown(
            f"**시장 상황**: {market['sentiment']} "
            f"(변동성: {market['volatility']:.1f}%)"
        )
        if market['volume_profile']:
            self._render_volume_profile(market['volume_profile'])
    
    def _render_indicators(self, indicators: Dict[str, Any]) -> None:
        st.markdown('**기술적 지표**')
        for name, value in indicators.items():
            st.text(f"{name}: {value}")
    
    def _render_volume_profile(self, profile: Dict[str, Any]) -> None:
        fig = go.Figure(data=[
            go.Bar(
                x=list(profile.keys()),
                y=list(profile.values()),
                name='거래량'
            )
        ])
        fig.update_layout(
            title='거래량 프로파일',
            showlegend=False,
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)

class SidebarComponent:
    def render(self, markets: List[str]) -> str:
        st.sidebar.title('설정')
        
        selected_market = st.sidebar.selectbox(
            '거래 시장',
            markets,
            index=0 if markets else None
        )
        
        st.sidebar.markdown('---')
        
        if st.sidebar.checkbox('자동 매매 활성화', value=False):
            confidence = st.sidebar.slider(
                '신뢰도 임계값',
                min_value=0,
                max_value=100,
                value=50
            )
            st.session_state['auto_trading'] = {
                'enabled': True,
                'confidence': confidence / 100
            }
        else:
            st.session_state['auto_trading'] = {
                'enabled': False,
                'confidence': 0
            }
        
        return selected_market 