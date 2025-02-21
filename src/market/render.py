import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def render_market_data(market_data: Optional[Dict[str, Any]] = None):
    """시장 데이터 렌더링
    
    Args:
        market_data (Optional[Dict[str, Any]]): 시장 데이터 딕셔너리
    """
    if not market_data:
        st.warning("시장 데이터가 없습니다.")
        return
        
    try:
        # 필수 필드 확인
        required_fields = ['market', 'current_price', 'change_rate', 
                         'high_price', 'low_price', 'trade_volume']
        if not all(field in market_data for field in required_fields):
            st.error("시장 데이터 형식이 올바르지 않습니다.")
            return
            
        # 데이터 타입 검증 및 변환
        try:
            current_price = float(market_data['current_price'])
            change_rate = float(market_data['change_rate'])
            high_price = float(market_data['high_price'])
            low_price = float(market_data['low_price'])
            trade_volume = float(market_data['trade_volume'])
        except (ValueError, TypeError) as e:
            st.error("시장 데이터 값이 올바르지 않습니다.")
            logger.error(f"데이터 변환 오류: {e}")
            return
            
        # 레이아웃 설정
        col1, col2, col3 = st.columns(3)
        
        # 현재가 및 변동률
        with col1:
            st.metric(
                label="현재가",
                value=f"{current_price:,.0f}원",
                delta=f"{change_rate:.2f}%"
            )
            
        # 고가/저가
        with col2:
            st.metric(
                label="고가",
                value=f"{high_price:,.0f}원",
                delta=f"{((high_price/current_price)-1)*100:.2f}%"
            )
            st.metric(
                label="저가",
                value=f"{low_price:,.0f}원",
                delta=f"{((low_price/current_price)-1)*100:.2f}%"
            )
            
        # 거래량
        with col3:
            st.metric(
                label="거래량",
                value=f"{trade_volume:.3f} BTC"
            )
            
        # 기술적 지표 표시 (있는 경우)
        if 'indicators' in market_data:
            indicators = market_data['indicators']
            
            st.subheader("기술적 지표")
            ind_col1, ind_col2 = st.columns(2)
            
            with ind_col1:
                if 'rsi' in indicators:
                    st.metric(
                        label="RSI",
                        value=f"{indicators['rsi']:.2f}"
                    )
                if 'macd' in indicators:
                    st.metric(
                        label="MACD",
                        value=f"{indicators['macd']:.2f}"
                    )
                    
            with ind_col2:
                if 'bollinger_bands' in indicators:
                    bb = indicators['bollinger_bands']
                    st.metric(
                        label="볼린저 밴드",
                        value=f"중심: {bb['middle']:.0f}",
                        delta=f"상단: {bb['upper']:.0f} / 하단: {bb['lower']:.0f}"
                    )
                    
        # 차트 데이터가 있는 경우 캔들스틱 차트 표시
        if 'chart_data' in market_data:
            chart_data = market_data['chart_data']
            if isinstance(chart_data, pd.DataFrame) and len(chart_data) > 0:
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['open'],
                        high=chart_data['high'],
                        low=chart_data['low'],
                        close=chart_data['close']
                    )
                ])
                
                fig.update_layout(
                    title=f"{market_data['market']} 가격 차트",
                    yaxis_title="가격",
                    xaxis_title="시간"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error("시장 데이터 표시 중 오류가 발생했습니다.")
        logger.error(f"렌더링 오류: {e}") 