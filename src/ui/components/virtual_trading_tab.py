"""
Virtual Trading Tab Component

This component handles the virtual trading interface.
"""

import streamlit as st
from typing import Dict, Any
from datetime import datetime

class VirtualTradingTab:
    def __init__(self):
        """Initialize virtual trading tab."""
        if 'virtual_portfolio' not in st.session_state:
            st.session_state.virtual_portfolio = {
                'balance': 10000000,  # 1천만원
                'positions': {},
                'trades': [],
                'performance': {
                    'win_rate': 0.0,
                    'return_rate': 0.0,
                    'total_profit': 0.0,
                    'total_trades': 0
                }
            }
    
    def render(self, market_data: Dict[str, Any]):
        """Render virtual trading interface."""
        st.header("가상 매매")
        
        if not market_data.get('current_price'):
            st.warning("시장 데이터를 불러올 수 없습니다.")
            return
            
        # Display current market info
        st.subheader(f"현재 시장: {market_data.get('market', '').replace('KRW-', '')}")
        st.metric(
            "현재가",
            f"₩{market_data.get('current_price', 0):,.0f}"
        )
        
        # Account Overview
        self._render_account_overview()
        
        # Trading Interface
        self._render_trading_interface(market_data)
        
        # Position Information
        self._render_positions()
        
        # Trade History
        self._render_trade_history()
        
        # Performance Metrics
        self._render_performance_metrics()
        
        # Reset button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("초기화", type="secondary", key="reset_portfolio_button"):
                self.db.clear_data()
                st.session_state.virtual_balance = 10000000
                st.session_state.virtual_positions = {}
                st.experimental_rerun()
    
    def _render_account_overview(self):
        """Render account overview section."""
        st.subheader("계좌 현황")
        
        portfolio = st.session_state.virtual_portfolio
        total_position_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in portfolio['positions'].values()
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "보유 현금",
                f"₩{portfolio['balance']:,.0f}"
            )
        
        with col2:
            st.metric(
                "포지션 가치",
                f"₩{total_position_value:,.0f}"
            )
        
        with col3:
            st.metric(
                "총 자산",
                f"₩{(portfolio['balance'] + total_position_value):,.0f}"
            )
    
    def _render_trading_interface(self, market_data: Dict[str, Any]):
        """Render trading interface."""
        st.subheader("매매")
        
        col1, col2 = st.columns(2)
        
        with col1:
            action = st.radio(
                "매매 유형",
                options=["매수", "매도"],
                horizontal=True,
                key="trade_action_radio"
            )
        
        with col2:
            amount = st.number_input(
                "주문 금액",
                min_value=5000,
                max_value=1000000,
                value=100000,
                step=1000,
                format="%d",
                key="trade_amount_input"
            )
        
        if st.button("주문 실행", key="execute_trade_button"):
            self._execute_trade(
                action,
                amount,
                market_data.get('market', ''),
                market_data.get('current_price', 0)
            )
    
    def _render_positions(self):
        """Render positions information."""
        st.subheader("보유 포지션")
        
        positions = st.session_state.virtual_portfolio['positions']
        
        if not positions:
            st.info("현재 보유중인 포지션이 없습니다.")
            return
        
        for market, position in positions.items():
            current_value = position['quantity'] * position['current_price']
            profit_loss = current_value - (position['quantity'] * position['avg_price'])
            profit_loss_pct = (profit_loss / (position['quantity'] * position['avg_price'])) * 100
            
            st.markdown(f"**{market}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("수량", f"{position['quantity']:.8f}")
            with col2:
                st.metric("평균단가", f"₩{position['avg_price']:,.0f}")
            with col3:
                st.metric("현재가치", f"₩{current_value:,.0f}")
            with col4:
                st.metric("손익", f"₩{profit_loss:,.0f} ({profit_loss_pct:.2f}%)")
    
    def _render_trade_history(self):
        """Render trade history."""
        st.subheader("거래 내역")
        
        trades = st.session_state.virtual_portfolio['trades']
        
        if not trades:
            st.info("거래 내역이 없습니다.")
            return
        
        for trade in reversed(trades[-10:]):  # Show last 10 trades
            timestamp = datetime.fromisoformat(trade['timestamp'])
            
            st.markdown(
                f"**{trade['market']}** - {trade['action']} "
                f"({timestamp.strftime('%Y-%m-%d %H:%M')})"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("수량", f"{trade['quantity']:.8f}")
            with col2:
                st.metric("가격", f"₩{trade['price']:,.0f}")
            with col3:
                st.metric("금액", f"₩{trade['amount']:,.0f}")
    
    def _render_performance_metrics(self):
        """Render performance metrics."""
        st.subheader("성과 지표")
        
        perf = st.session_state.virtual_portfolio['performance']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "승률",
                f"{perf['win_rate']:.1f}%"
            )
        
        with col2:
            st.metric(
                "수익률",
                f"{perf['return_rate']:.1f}%"
            )
        
        with col3:
            st.metric(
                "총 손익",
                f"₩{perf['total_profit']:,.0f}"
            )
        
        with col4:
            st.metric(
                "총 거래수",
                f"{perf['total_trades']}"
            )
    
    def _execute_trade(self, action: str, amount: float, market: str, price: float):
        """Execute a trade."""
        if not market or not price:
            st.error("시장 데이터를 불러올 수 없습니다.")
            return
        
        portfolio = st.session_state.virtual_portfolio
        
        # Calculate quantity
        quantity = amount / price
        
        if action == "매수":
            if amount > portfolio['balance']:
                st.error("잔액이 부족합니다.")
                return
            
            # Update balance
            portfolio['balance'] -= amount
            
            # Update position
            if market not in portfolio['positions']:
                portfolio['positions'][market] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'current_price': price
                }
            else:
                position = portfolio['positions'][market]
                total_quantity = position['quantity'] + quantity
                total_cost = (position['quantity'] * position['avg_price']) + amount
                position['avg_price'] = total_cost / total_quantity
                position['quantity'] = total_quantity
                position['current_price'] = price
        
        else:  # 매도
            if market not in portfolio['positions']:
                st.error("해당 자산을 보유하고 있지 않습니다.")
                return
            
            position = portfolio['positions'][market]
            if quantity > position['quantity']:
                st.error("보유 수량이 부족합니다.")
                return
            
            # Update balance
            portfolio['balance'] += amount
            
            # Update position
            position['quantity'] -= quantity
            position['current_price'] = price
            
            if position['quantity'] < 0.00000001:  # Remove if quantity is negligible
                del portfolio['positions'][market]
        
        # Record trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'action': action,
            'quantity': quantity,
            'price': price,
            'amount': amount
        }
        portfolio['trades'].append(trade)
        
        # Update performance metrics
        self._update_performance_metrics(trade)
        
        st.success(f"{action} 주문이 체결되었습니다.")
    
    def _update_performance_metrics(self, trade: Dict[str, Any]):
        """Update performance metrics after a trade."""
        portfolio = st.session_state.virtual_portfolio
        perf = portfolio['performance']
        
        # Update total trades
        perf['total_trades'] += 1
        
        # Calculate profit/loss for this trade
        if trade['action'] == "매도":
            position = portfolio['positions'].get(trade['market'])
            if position:
                profit = trade['amount'] - (trade['quantity'] * position['avg_price'])
                perf['total_profit'] += profit
                
                # Update win rate
                if profit > 0:
                    perf['win_rate'] = (
                        (perf['win_rate'] * (perf['total_trades'] - 1) + 100) /
                        perf['total_trades']
                    )
                else:
                    perf['win_rate'] = (
                        (perf['win_rate'] * (perf['total_trades'] - 1)) /
                        perf['total_trades']
                    )
        
        # Calculate return rate
        initial_balance = 10000000  # 1천만원
        current_total = portfolio['balance'] + sum(
            pos['quantity'] * pos['current_price']
            for pos in portfolio['positions'].values()
        )
        perf['return_rate'] = ((current_total / initial_balance) - 1) * 100 