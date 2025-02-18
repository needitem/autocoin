"""
Virtual Trading Component

This component handles the virtual trading interface and functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from src.data.db_manager import DBManager

class VirtualTradingComponent:
    def __init__(self):
        """Initialize virtual trading component."""
        self.db = DBManager()
        
        if 'virtual_balance' not in st.session_state:
            st.session_state.virtual_balance = 10000000  # 1천만원 초기 자금
        if 'virtual_positions' not in st.session_state:
            st.session_state.virtual_positions = {}
    
    def _calculate_position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current position value."""
        position = st.session_state.virtual_positions.get(symbol, {})
        return position.get('quantity', 0) * current_price
    
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
        fee = cost * 0.0005  # 0.05% fee
        
        if is_buy:
            new_quantity = position['quantity'] + quantity
            new_cost = (position['quantity'] * position['avg_price']) + cost
            position['avg_price'] = new_cost / new_quantity
            position['quantity'] = new_quantity
            st.session_state.virtual_balance -= (cost + fee)
            trade_pnl = -fee
        else:
            trade_pnl = (price - position['avg_price']) * quantity - fee
            position['pnl'] += trade_pnl
            position['quantity'] -= quantity
            st.session_state.virtual_balance += (cost - fee)
        
        # Save trade to database
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY' if is_buy else 'SELL',
            'price': price,
            'quantity': quantity,
            'amount': cost,
            'fee': fee,
            'pnl': trade_pnl
        }
        self.db.save_trade(trade_data)
        
        # Calculate and save performance metrics
        trades = self.db.get_trades()
        if trades:
            trades_df = pd.DataFrame(trades)
            
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
            total_pnl = trades_df['pnl'].sum()
            return_rate = total_pnl / 10000000 * 100  # Initial balance
            
            if len(trades_df) > 1:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                daily_returns = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl'].sum() / 10000000
                volatility = daily_returns.std() * np.sqrt(252) * 100
                
                excess_returns = daily_returns.mean() * 252 - 0.03  # 3% risk-free rate
                sharpe_ratio = excess_returns / (volatility / 100) if volatility > 0 else 0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            trading_days = len(trades_df['timestamp'].dt.date.unique())
            
            perf_data = {
                'win_rate': win_rate,
                'return_rate': return_rate,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'trading_days': trading_days
            }
            self.db.save_performance(perf_data)
    
    def render(self, trading_data: Dict[str, Any]):
        """Render virtual trading interface."""
        st.subheader("가상 매매")
        
        # Reset button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("초기화", type="secondary"):
                self.db.clear_data()
                st.session_state.virtual_balance = 10000000
                st.session_state.virtual_positions = {}
                st.experimental_rerun()
        
        # Account Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "가상 계좌 잔고",
                f"₩{st.session_state.virtual_balance:,.0f}"
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
        
        with col3:
            total_value = st.session_state.virtual_balance
            if trading_data.get('current_price'):
                total_value += self._calculate_position_value(
                    trading_data['symbol'],
                    trading_data['current_price']
                )
            st.metric(
                "총 자산",
                f"₩{total_value:,.0f}"
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
                format="%.4f"
            )
        
        with col3:
            if st.button("주문 실행"):
                if trading_data.get('current_price'):
                    is_buy = action == "매수"
                    price = trading_data['current_price']
                    cost = price * quantity
                    fee = cost * 0.0005
                    
                    if is_buy and (cost + fee) > st.session_state.virtual_balance:
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
            if position['quantity'] > 0:
                st.subheader("현재 포지션")
                
                pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                
                with pos_col1:
                    st.metric("보유 수량", f"{position['quantity']:.4f}")
                
                with pos_col2:
                    st.metric("평균 매수가", f"₩{position['avg_price']:,.0f}")
                
                with pos_col3:
                    current_value = position['quantity'] * trading_data.get('current_price', 0)
                    st.metric("평가 금액", f"₩{current_value:,.0f}")
                
                with pos_col4:
                    st.metric("실현 손익", f"₩{position['pnl']:,.0f}")
        
        # Trade History
        trades = self.db.get_trades()
        if trades:
            st.subheader("거래 내역")
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Format numbers for display
            df['price'] = df['price'].apply(lambda x: f"₩{x:,.0f}")
            df['amount'] = df['amount'].apply(lambda x: f"₩{x:,.0f}")
            df['fee'] = df['fee'].apply(lambda x: f"₩{x:,.0f}")
            df['pnl'] = df['pnl'].apply(lambda x: f"₩{x:,.0f}")
            
            st.dataframe(
                df,
                column_config={
                    'timestamp': '시간',
                    'symbol': '심볼',
                    'action': '매수/매도',
                    'price': '가격',
                    'quantity': '수량',
                    'amount': '거래금액',
                    'fee': '수수료',
                    'pnl': '손익'
                },
                hide_index=True
            ) 