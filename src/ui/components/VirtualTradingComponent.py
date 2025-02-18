"""
Virtual Trading Component

This component handles the virtual trading interface and functionality.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from datetime import datetime

class VirtualTradingComponent:
    def __init__(self):
        """Initialize virtual trading component."""
        if 'virtual_balance' not in st.session_state:
            st.session_state.virtual_balance = 10000000  # 1천만원 초기 자금
        if 'virtual_positions' not in st.session_state:
            st.session_state.virtual_positions = {}
        if 'trade_history' not in st.session_state:
            st.session_state.trade_history = []
    
    def _calculate_position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current position value."""
        position = st.session_state.virtual_positions.get(symbol, {})
        return position.get('quantity', 0) * current_price
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total profit/loss."""
        total_pnl = 0
        for symbol, position in st.session_state.virtual_positions.items():
            if 'pnl' in position:
                total_pnl += position['pnl']
        return total_pnl
    
    def _update_position(self, symbol: str, price: float, quantity: float, is_buy: bool):
        """Update virtual position."""
        if symbol not in st.session_state.virtual_positions:
            st.session_state.virtual_positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'pnl': 0
            }
        
        position = st.session_state.virtual_positions[symbol]
        cost = price * quantity
        
        if is_buy:
            new_quantity = position['quantity'] + quantity
            new_cost = (position['quantity'] * position['avg_price']) + cost
            position['avg_price'] = new_cost / new_quantity
            position['quantity'] = new_quantity
            st.session_state.virtual_balance -= cost
        else:
            position['pnl'] += (price - position['avg_price']) * quantity
            position['quantity'] -= quantity
            st.session_state.virtual_balance += cost
        
        # Record trade in history
        st.session_state.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY' if is_buy else 'SELL',
            'price': price,
            'quantity': quantity,
            'total': cost
        })
    
    def render(self, trading_data: Dict[str, Any]):
        """Render virtual trading interface."""
        st.subheader("가상 매매")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "가상 계좌 잔고",
                f"₩{st.session_state.virtual_balance:,.0f}",
                f"₩{self._calculate_total_pnl():,.0f}"
            )
        
        with col2:
            if trading_data.get('current_price'):
                position_value = self._calculate_position_value(
                    trading_data['symbol'],
                    trading_data['current_price']
                )
                st.metric(
                    "보유 포지션 가치",
                    f"₩{position_value:,.0f}"
                )
        
        # Trading Interface
        st.subheader("매매 실행")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            action = st.selectbox("매수/매도", ["매수", "매도"])
        
        with col2:
            quantity = st.number_input(
                "수량",
                min_value=0.0,
                step=0.1,
                format="%.1f"
            )
        
        with col3:
            if st.button("주문 실행"):
                if trading_data.get('current_price'):
                    is_buy = action == "매수"
                    price = trading_data['current_price']
                    cost = price * quantity
                    
                    if is_buy and cost > st.session_state.virtual_balance:
                        st.error("잔고가 부족합니다.")
                    elif not is_buy and quantity > st.session_state.virtual_positions.get(
                        trading_data['symbol'], {}).get('quantity', 0):
                        st.error("매도 가능 수량을 초과했습니다.")
                    else:
                        self._update_position(
                            trading_data['symbol'],
                            price,
                            quantity,
                            is_buy
                        )
                        st.success(f"{action} 주문이 체결되었습니다.")
        
        # Position Information
        if trading_data.get('symbol') in st.session_state.virtual_positions:
            position = st.session_state.virtual_positions[trading_data['symbol']]
            st.subheader("현재 포지션")
            
            metrics = {
                "보유 수량": f"{position['quantity']:.1f}",
                "평균 매수가": f"₩{position['avg_price']:,.0f}",
                "실현 손익": f"₩{position['pnl']:,.0f}"
            }
            
            cols = st.columns(len(metrics))
            for col, (label, value) in zip(cols, metrics.items()):
                col.metric(label, value)
        
        # Trade History
        if st.session_state.trade_history:
            st.subheader("거래 내역")
            df = pd.DataFrame(st.session_state.trade_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            st.dataframe(
                df,
                column_config={
                    'timestamp': '시간',
                    'symbol': '심볼',
                    'action': '매수/매도',
                    'price': '가격',
                    'quantity': '수량',
                    'total': '총액'
                },
                hide_index=True
            ) 