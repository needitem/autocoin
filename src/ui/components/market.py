"""
Market data display component
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def render_market_data(data: dict):
    """마켓 데이터 렌더링"""
    try:
        if not data:
            st.warning("시장 데이터를 불러올 수 없습니다")
            return

        # 필수 필드 확인
        required_fields = ['current_price', 'open', 'high', 'low', 'volume', 'change_rate']
        if not all(field in data for field in required_fields):
            st.warning("일부 시장 데이터가 누락되었습니다")
            return

        # 데이터 유효성 검사
        for field in required_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                st.warning("일부 시장 데이터가 유효하지 않습니다")
                return

        # 컬럼 생성
        col1, col2, col3 = st.columns(3)

        # 현재가 및 변동률
        with col1:
            st.metric(
                "현재가",
                f"{data['current_price']:,.0f}원",
                f"{data['change_rate']:+.2f}%"
            )

        # 거래량
        with col2:
            st.metric(
                "24시간 거래량",
                f"{data['volume']:,.4f} BTC"
            )

        # 고가/저가
        with col3:
            st.metric(
                "고가/저가",
                f"{data['high']:,.0f}원",
                f"저가: {data['low']:,.0f}원"
            )

        # 기술적 지표 표시
        if 'indicators' in data and data['indicators']:
            st.subheader("기술적 지표")
            
            # 이동평균선
            ma_col1, ma_col2, ma_col3 = st.columns(3)
            with ma_col1:
                ma5 = data['indicators']['moving_averages']['MA5'].iloc[-1]
                st.metric("MA5", f"{ma5:,.0f}원")
            with ma_col2:
                ma10 = data['indicators']['moving_averages']['MA10'].iloc[-1]
                st.metric("MA10", f"{ma10:,.0f}원")
            with ma_col3:
                ma20 = data['indicators']['moving_averages']['MA20'].iloc[-1]
                st.metric("MA20", f"{ma20:,.0f}원")

            # 볼린저 밴드
            bb_col1, bb_col2, bb_col3 = st.columns(3)
            with bb_col1:
                upper = data['indicators']['bollinger_bands']['upper'].iloc[-1]
                st.metric("상단 밴드", f"{upper:,.0f}원")
            with bb_col2:
                middle = data['indicators']['bollinger_bands']['middle'].iloc[-1]
                st.metric("중간 밴드", f"{middle:,.0f}원")
            with bb_col3:
                lower = data['indicators']['bollinger_bands']['lower'].iloc[-1]
                st.metric("하단 밴드", f"{lower:,.0f}원")

            # RSI
            rsi_col1, rsi_col2 = st.columns(2)
            with rsi_col1:
                rsi = data['indicators']['rsi'].iloc[-1]
                st.metric("RSI", f"{rsi:.2f}")
            with rsi_col2:
                volatility = data['indicators']['bollinger_bands']['volatility']
                st.metric("변동성", f"{volatility:.2f}%")

            # MACD
            macd_col1, macd_col2, macd_col3 = st.columns(3)
            with macd_col1:
                macd = data['indicators']['macd']['macd'].iloc[-1]
                st.metric("MACD", f"{macd:.2f}")
            with macd_col2:
                signal = data['indicators']['macd']['signal'].iloc[-1]
                st.metric("시그널", f"{signal:.2f}")
            with macd_col3:
                histogram = data['indicators']['macd']['histogram'].iloc[-1]
                st.metric("히스토그램", f"{histogram:.2f}")

    except Exception as e:
        st.error(f"마켓 데이터 렌더링 중 오류 발생: {str(e)}")

def render_market_selector(markets: list) -> str:
    """거래 시장 선택기 렌더링
    
    Args:
        markets (list): 마켓 목록
        
    Returns:
        str: 선택된 마켓 코드
    """
    try:
        if not markets:
            st.error("거래 시장 목록을 불러올 수 없습니다")
            return "KRW-BTC"  # 기본값 반환
            
        # markets가 딕셔너리 리스트인 경우와 문자열 리스트인 경우를 모두 처리
        def format_market(x):
            if isinstance(x, dict):
                return f"{x.get('korean_name', '')} ({x.get('market', '')})"
            return x.split('-')[1] if isinstance(x, str) and '-' in x else x

        selected = st.selectbox(
            "거래 시장 선택",
            options=markets,
            format_func=format_market,
            index=0  # 첫 번째 항목을 기본 선택
        )
        
        # 딕셔너리인 경우 market 값을 반환
        if isinstance(selected, dict):
            return selected.get('market', "KRW-BTC")
        return selected
        
    except Exception as e:
        st.error(f"거래 시장 선택기 오류: {str(e)}")
        return "KRW-BTC"  # 오류 발생 시 기본값 반환 