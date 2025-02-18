"""
Sidebar Component

This component handles the sidebar interface for market selection and settings.
"""

import streamlit as st
from typing import List, Optional

class SidebarComponent:
    def render(self, markets: List[str]) -> Optional[str]:
        """Render sidebar interface."""
        with st.sidebar:
            st.header("시장 선택")
            
            selected_market = st.selectbox(
                "거래 시장",
                markets,
                index=0 if markets else None,
                format_func=lambda x: x.replace('KRW-', '')
            )
            
            st.divider()
            
            st.subheader("자동매매 설정")
            
            # Auto Trading Settings
            auto_trading = st.toggle("자동매매 활성화", value=False)
            if auto_trading:
                st.info("자동매매가 활성화되었습니다. 매매 신호에 따라 자동으로 거래가 실행됩니다.")
                
                st.markdown("#### 매매 조건")
                
                # Risk threshold
                risk_threshold = st.slider(
                    "리스크 임계값",
                    min_value=1.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.1,
                    help="이 값을 초과하는 리스크 점수에서는 매매가 실행되지 않습니다."
                )
                
                # Trade amount
                trade_amount = st.slider(
                    "매매 금액",
                    min_value=10000,
                    max_value=1000000,
                    value=100000,
                    step=10000,
                    format="%d원",
                    help="한 번에 매매할 금액입니다."
                )
                
                # Stop loss
                stop_loss = st.slider(
                    "손절 비율",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    format="%.1f%%",
                    help="이 비율만큼 손실이 발생하면 자동으로 매도합니다."
                )
                
                # Take profit
                take_profit = st.slider(
                    "익절 비율",
                    min_value=1.0,
                    max_value=20.0,
                    value=10.0,
                    step=0.1,
                    format="%.1f%%",
                    help="이 비율만큼 수익이 발생하면 자동으로 매도합니다."
                )
                
                # Update session state
                st.session_state.auto_trading = {
                    'enabled': True,
                    'risk_threshold': risk_threshold,
                    'trade_amount': trade_amount,
                    'stop_loss': stop_loss / 100,
                    'take_profit': take_profit / 100
                }
            else:
                st.session_state.auto_trading = {
                    'enabled': False,
                    'risk_threshold': 7.0,
                    'trade_amount': 100000,
                    'stop_loss': 0.05,
                    'take_profit': 0.10
                }
            
            return selected_market