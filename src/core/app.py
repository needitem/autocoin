"""
Main Streamlit Application
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import time

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.exchange.exchange_api import UpbitAPI
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.technical_analyzer import TechnicalAnalyzer
from src.analysis.performance import PerformanceAnalyzer
from src.strategies.trader_ai_strategy import TraderAIStrategy
from src.trading.virtual_trading import VirtualTrading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
upbit_api = UpbitAPI()
market_analyzer = MarketAnalyzer()
technical_analyzer = TechnicalAnalyzer()
performance_analyzer = PerformanceAnalyzer()
trader_ai = TraderAIStrategy()

# Initialize virtual trading in session state
if 'virtual_trading' not in st.session_state:
    st.session_state.virtual_trading = VirtualTrading()
    st.session_state.trading_enabled = False

def get_trend_emoji(trend: str) -> str:
    """Get emoji for market trend."""
    trend_emojis = {
        'STRONG_UPTREND': '🚀',
        'UPTREND': '📈',
        'SIDEWAYS': '➡️',
        'DOWNTREND': '📉',
        'STRONG_DOWNTREND': '🔻'
    }
    return trend_emojis.get(trend, '➡️')

def get_action_color(action: str) -> str:
    """Get color for trading action."""
    action_colors = {
        'BUY': 'green',
        'SELL': 'red',
        'HOLD': 'blue'
    }
    return action_colors.get(action, 'gray')

def get_confidence_description(confidence: float) -> str:
    """Get description for confidence level."""
    if confidence >= 0.8:
        return "매우 확신"
    elif confidence >= 0.6:
        return "확신"
    elif confidence >= 0.4:
        return "중간"
    else:
        return "낮음"

def get_market_summary(conditions: Dict[str, Any]) -> str:
    """Get simple market summary."""
    trend = conditions.get('trend', 'SIDEWAYS')
    volatility = conditions.get('volatility', 0)
    volume_change = conditions.get('volume_change', 0)
    
    summaries = {
        'STRONG_UPTREND': '시장이 매우 강세입니다',
        'UPTREND': '시장이 상승 중입니다',
        'SIDEWAYS': '시장이 횡보 중입니다',
        'DOWNTREND': '시장이 하락 중입니다',
        'STRONG_DOWNTREND': '시장이 매우 약세입니다'
    }
    
    base_summary = summaries.get(trend, '시장이 횡보 중입니다')
    
    if volatility > 5:
        base_summary += ". 변동성이 매우 큽니다"
    elif volatility > 2:
        base_summary += ". 변동성이 있습니다"
    
    if volume_change > 50:
        base_summary += ". 거래량이 매우 증가했습니다"
    elif volume_change > 20:
        base_summary += ". 거래량이 증가했습니다"
    elif volume_change < -50:
        base_summary += ". 거래량이 매우 감소했습니다"
    elif volume_change < -20:
        base_summary += ". 거래량이 감소했습니다"
    
    return base_summary

def main():
    # Auto-refresh every second
    st.set_page_config(page_title='Upbit 암호화폐 트레이딩 도우미', layout='wide')
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= 1:
        st.session_state.last_refresh = current_time
        time.sleep(1)
        st.rerun()

    st.title('Upbit 암호화폐 트레이딩 도우미 🤖')
    
    # Sidebar configuration
    st.sidebar.header('설정')
    
    # Get available markets
    markets = upbit_api.get_market_all()
    krw_markets = [market['market'] for market in markets if market['market'].startswith('KRW-')]
    
    # Market selection
    selected_market = st.sidebar.selectbox('마켓 선택', krw_markets, index=0)
    timeframe = st.sidebar.selectbox('시간프레임', ['1', '3', '5', '15', '30', '60', '240'], index=3)

    # Trading Strategy Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("매매 전략 설정")
    
    # Risk Level
    risk_level = st.sidebar.select_slider(
        "위험도",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate",
        help="매매 위험도를 설정합니다. Conservative: 보수적, Moderate: 중립적, Aggressive: 공격적"
    )
    
    # Position Size
    max_position_size = st.sidebar.slider(
        "최대 포지션 크기",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="총 자산 대비 단일 포지션의 최대 크기 (%)"
    )
    
    # Minimum Confidence
    min_confidence = st.sidebar.slider(
        "최소 신뢰도",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="매매 신호의 최소 신뢰도 (0.0 ~ 1.0)"
    )
    
    # Minimum Trade Amount
    min_trade_amount = st.sidebar.number_input(
        "최소 거래금액",
        min_value=5000,
        max_value=100000,
        value=10000,
        step=5000,
        help="최소 거래금액 (원)"
    )

    # Trading Hours
    st.sidebar.markdown("---")
    st.sidebar.subheader("거래 시간 설정")
    trading_hours_enabled = st.sidebar.checkbox(
        "거래시간 제한",
        value=False,
        help="특정 시간대에만 거래를 실행합니다."
    )
    
    if trading_hours_enabled:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_time = st.time_input("시작 시간", value=datetime.strptime("09:00", "%H:%M").time())
        with col2:
            end_time = st.time_input("종료 시간", value=datetime.strptime("23:00", "%H:%M").time())

    # Update session state with new settings
    if 'trading_settings' not in st.session_state:
        st.session_state.trading_settings = {}
    
    st.session_state.trading_settings.update({
        'risk_level': risk_level,
        'max_position_size': max_position_size,
        'min_confidence': min_confidence,
        'min_trade_amount': min_trade_amount,
        'trading_hours_enabled': trading_hours_enabled,
        'start_time': start_time if trading_hours_enabled else None,
        'end_time': end_time if trading_hours_enabled else None
    })

    try:
        # Fetch market data
        candles = upbit_api.get_candles_minutes(selected_market, unit=int(timeframe))
        market_index = upbit_api.get_market_index(selected_market)
        orderbook = upbit_api.get_orderbook(selected_market)
        
        if not candles:
            st.error(f"{selected_market}의 데이터를 가져올 수 없습니다.")
            return
        
        # Create DataFrame for visualization
        df = pd.DataFrame(candles)
        
        # Display current market information
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = float(df['trade_price'].iloc[-1])
        opening_price = float(df['opening_price'].iloc[-1])
        price_change = current_price - opening_price
        price_change_pct = (price_change / opening_price) * 100
        
        with col1:
            st.metric("현재가", f"₩{current_price:,.0f}")
        with col2:
            st.metric("24시간 변동", f"{price_change_pct:.2f}%", 
                     delta_color="normal" if abs(price_change_pct) < 0.1 else ("inverse" if price_change_pct < 0 else "normal"))
        with col3:
            st.metric("24시간 거래량", f"₩{float(df['candle_acc_trade_price'].iloc[-1]):,.0f}")
        with col4:
            if market_index:
                st.metric("매수/매도 압력", f"{market_index.get('buy_sell_pressure', 0):.1f}")
        
        # AI Trading Analysis
        st.subheader('AI 트레이더의 분석 💡')
        
        # Get AI analysis
        market_data = {'ohlcv': candles}
        ai_analysis = trader_ai.analyze_market(market_data)
        
        # Display AI's decision
        action = ai_analysis['action']
        confidence = ai_analysis['confidence']
        market_conditions = ai_analysis['market_conditions']
        
        # Create three columns for AI analysis
        ai_col1, ai_col2, ai_col3 = st.columns(3)
        
        with ai_col1:
            st.markdown(f"### 트레이딩 신호 {get_trend_emoji(market_conditions.get('trend', 'SIDEWAYS'))}")
            st.markdown(f"<h2 style='color: {get_action_color(action)}'>{action}</h2>", unsafe_allow_html=True)
        
        with ai_col2:
            st.markdown("### 확신도")
            st.markdown(f"<h2>{get_confidence_description(confidence)}</h2>", unsafe_allow_html=True)
            st.progress(confidence)
        
        with ai_col3:
            st.markdown("### 시장 상태")
            st.markdown(f"<h4>{get_market_summary(market_conditions)}</h4>", unsafe_allow_html=True)
        
        # Virtual Trading Section
        st.markdown("---")
        st.subheader("가상 매매 💰")
        
        # Virtual trading controls
        trading_col1, trading_col2 = st.columns([1, 3])
        
        with trading_col1:
            trading_enabled = st.checkbox("가상 매매 활성화", value=st.session_state.trading_enabled)
            st.session_state.trading_enabled = trading_enabled
            
            if trading_enabled:
                # Check trading hours if enabled
                can_trade = True
                if st.session_state.trading_settings['trading_hours_enabled']:
                    current_time = datetime.now().time()
                    start_time = st.session_state.trading_settings['start_time']
                    end_time = st.session_state.trading_settings['end_time']
                    can_trade = start_time <= current_time <= end_time
                
                # Check confidence threshold
                confidence_sufficient = confidence >= st.session_state.trading_settings['min_confidence']
                
                if can_trade and confidence_sufficient and action != 'HOLD':
                    trade_result = st.session_state.virtual_trading.execute_trade(
                        market=selected_market,
                        action=action,
                        current_price=current_price,
                        confidence=confidence,
                        max_position_size=st.session_state.trading_settings['max_position_size'],
                        min_trade_amount=st.session_state.trading_settings['min_trade_amount'],
                        orderbook=orderbook
                    )
                    st.success(trade_result['message'])
                elif not can_trade:
                    st.warning("현재 거래 시간이 아닙니다.")
                elif not confidence_sufficient:
                    st.warning(f"신뢰도({confidence:.1f})가 최소 기준({st.session_state.trading_settings['min_confidence']:.1f})보다 낮습니다.")
                elif action == 'HOLD':
                    st.info("현재 관망 신호입니다.")
        
        with trading_col2:
            # Display portfolio status
            portfolio = st.session_state.virtual_trading.get_portfolio_status()
            
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)
            
            with status_col1:
                st.metric("초기 자본", f"₩{portfolio['initial_balance']:,.0f}")
            with status_col2:
                st.metric("현재 현금", f"₩{portfolio['current_balance']:,.0f}")
            with status_col3:
                st.metric("포트폴리오 가치", f"₩{portfolio['total_value']:,.0f}")
            with status_col4:
                st.metric("총 수익률", f"{portfolio['total_return']:.2f}%")
            
            # Display holdings
            if portfolio['holdings']:
                st.markdown("### 보유 자산")
                for holding in portfolio['holdings']:
                    st.text(f"{holding['market']}: {holding['amount']:.8f} (평균단가: ₩{holding['avg_price']:,.0f})")

        # Trading Dashboard
        st.markdown("---")
        st.subheader("트레이딩 대시보드 📊")
        
        # Create three columns for the dashboard
        dash_col1, dash_col2, dash_col3 = st.columns(3)
        
        with dash_col1:
            st.markdown("### 거래 요약")
            trades = st.session_state.virtual_trading.get_trade_history()
            if trades:
                total_trades = len(trades)
                profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                
                st.metric("총 거래 횟수", f"{total_trades}회")
                st.metric("승률", f"{win_rate:.1f}%")
                
                # Calculate average profit/loss
                total_profit = sum(trade.get('profit', 0) for trade in trades)
                avg_profit = total_profit / total_trades if total_trades > 0 else 0
                st.metric("평균 수익", f"₩{avg_profit:,.0f}")
            else:
                st.info("아직 거래 내역이 없습니다.")
        
        with dash_col2:
            st.markdown("### 최근 거래")
            if trades:
                recent_trades = trades[-5:]  # Get last 5 trades
                for trade in reversed(recent_trades):
                    action = trade.get('action', '')
                    market = trade.get('market', '')
                    price = trade.get('price', 0)
                    amount = trade.get('amount', 0)
                    
                    # Calculate profit/loss
                    if action == 'SELL':
                        profit = trade.get('revenue', 0) - trade.get('cost', 0)
                    else:
                        profit = 0
                    
                    color = 'green' if profit > 0 else 'red' if profit < 0 else 'gray'
                    emoji = '📈' if profit > 0 else '📉' if profit < 0 else '➡️'
                    
                    st.markdown(
                        f"""
                        <div style='
                            padding: 10px;
                            border-left: 4px solid {color};
                            background-color: rgba(0, 0, 0, 0.05);
                            border-radius: 4px;
                            margin-bottom: 10px;
                        '>
                            <div style='color: {color}; font-weight: bold;'>
                                {emoji} {action} {market}
                            </div>
                            <div style='font-size: 0.9em; margin-top: 5px;'>
                                💰 가격: ₩{price:,.0f}<br>
                                📊 수량: {amount:.8f}
                            </div>
                            {f'<div style="color: {color}; margin-top: 5px;">손익: ₩{profit:,.0f}</div>' if action == 'SELL' else ''}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("아직 거래 내역이 없습니다.")
        
        with dash_col3:
            st.markdown("### 성과 지표")
            if trades:
                # Calculate performance metrics
                total_profit = sum(trade.get('profit', 0) for trade in trades)
                max_drawdown = st.session_state.virtual_trading.get_max_drawdown()
                profit_factor = st.session_state.virtual_trading.get_profit_factor()
                
                st.metric("총 손익", f"₩{total_profit:,.0f}")
                st.metric("최대 낙폭", f"{max_drawdown:.1f}%")
                st.metric("수익 팩터", f"{profit_factor:.2f}")
                
                # Risk-adjusted return
                if max_drawdown != 0:
                    risk_adjusted_return = (total_profit / portfolio['initial_balance']) / (max_drawdown / 100)
                    st.metric("위험조정수익률", f"{risk_adjusted_return:.2f}")
            else:
                st.info("아직 성과 지표를 계산할 수 없습니다.")
        
        # Display detailed analysis and price chart (existing code)
        st.markdown("---")
        st.markdown("### 상세 분석")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("#### 기술적 지표")
            st.markdown(f"- 추세: {market_conditions.get('trend', 'SIDEWAYS')} {get_trend_emoji(market_conditions.get('trend', 'SIDEWAYS'))}")
            st.markdown(f"- 변동성: {market_conditions.get('volatility', 0):.2f}%")
            st.markdown(f"- 거래량 변화: {market_conditions.get('volume_change', 0):.2f}%")
        
        with detail_col2:
            st.markdown("#### 투자 조언")
            if action == 'BUY':
                st.markdown("✅ 매수하기 좋은 시점입니다")
                st.markdown("- 적절한 매수 가격을 설정하세요")
                st.markdown("- 분할 매수를 고려하세요")
            elif action == 'SELL':
                st.markdown("🚫 매도하기 좋은 시점입니다")
                st.markdown("- 적절한 매도 가격을 설정하세요")
                st.markdown("- 분할 매도를 고려하세요")
            else:
                st.markdown("⏳ 관망하기 좋은 시점입니다")
                st.markdown("- 시장의 방향성을 좀 더 지켜보세요")
                st.markdown("- 무리한 거래는 피하세요")
        
        # Price Chart
        st.markdown("---")
        st.subheader('가격 차트 📊')
        try:
            fig = go.Figure(data=[go.Candlestick(x=df['candle_date_time_kst'],
                                                open=df['opening_price'],
                                                high=df['high_price'],
                                                low=df['low_price'],
                                                close=df['trade_price'])])
            
            fig.update_layout(xaxis_rangeslider_visible=False,
                            title=f'{selected_market} 가격 차트 ({timeframe}분)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            st.warning("차트를 일시적으로 사용할 수 없습니다.")
            
    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error("대시보드 로딩 중 오류가 발생했습니다. 나중에 다시 시도해주세요.")

if __name__ == "__main__":
    main() 