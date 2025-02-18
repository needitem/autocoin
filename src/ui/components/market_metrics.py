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