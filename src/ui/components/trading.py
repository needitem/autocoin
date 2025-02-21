"""
Trading interface component
"""

import streamlit as st
from typing import Dict, Any
from src.core.trading import TradingManager
import re

def validate_input(value: Any) -> bool:
    """입력값 검증
    
    Args:
        value (Any): 검증할 입력값
        
    Returns:
        bool: 유효한 입력값인지 여부
    """
    if not isinstance(value, (str, int, float)):
        return True  # 숫자형은 안전하다고 가정
        
    if isinstance(value, (int, float)):
        return True  # 숫자형은 안전하다고 가정
    
    # 문자열로 변환
    value_str = str(value).lower()
    
    # 위험한 패턴 목록
    dangerous_patterns = [
        # SQL 인젝션
        r"'.*--",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"update\s+.*set",
        r"select\s+.*from",
        
        # XSS
        r"<script.*>",
        r"javascript:",
        r"onerror=",
        r"onload=",
        r"onclick=",
        r"<img.*>",
        r"<iframe.*>",
        
        # 경로 탐색
        r"\.\.\/",
        r"\.\.\\",
        r"\/etc\/",
        r"\\windows\\",
        
        # 명령어 실행
        r"\$\(.*\)",
        r"`.*`",
        r"&&.*",
        r"\|\|.*",
        r";\s*.*",
        r"rm\s+-rf",
        r"del\s+\/",
        r"format\s+c:",
        
        # 파일 접근
        r"\.env",
        r"config\..*",
        r"passwd",
        r"shadow"
    ]
    
    # 패턴 검사
    for pattern in dangerous_patterns:
        if re.search(pattern, value_str, re.IGNORECASE):
            return False
    
    return True

def render_trading_interface(trading_manager, market: str):
    """거래 인터페이스 렌더링
    
    Args:
        trading_manager: 거래 관리자 인스턴스
        market (str): 거래 시장 코드
    """
    try:
        if trading_manager is None:
            st.error("거래 인터페이스 렌더링 중 오류 발생")
            return
            
        st.markdown("### 거래")
        
        # 거래 모드 선택
        trade_mode = st.radio(
            "거래 모드",
            ["자동", "수동"],
            horizontal=True
        )
        
        # 거래 설정
        cols = st.columns(3)
        
        with cols[0]:
            if trade_mode == "자동":
                investment_ratio = st.number_input(
                    "투자 비율 (%)",
                    min_value=1,
                    max_value=100,
                    value=50,
                    step=1,
                    key="investment_ratio"
                )
            else:
                volume = st.number_input(
                    "주문 수량",
                    min_value=0.0,
                    value=0.0,
                    step=0.0001,
                    format="%.4f",
                    key="order_volume"
                )
                if not validate_input(str(volume)):
                    st.error("유효하지 않은 입력값입니다")
                    return
                
        with cols[1]:
            if trade_mode == "수동":
                price = st.number_input(
                    "주문 가격",
                    min_value=0,
                    value=0,
                    step=1000,
                    key="order_price"
                )
                if not validate_input(str(price)):
                    st.error("유효하지 않은 입력값입니다")
                    return
                
        with cols[2]:
            if trade_mode == "자동":
                strategy = st.selectbox(
                    "전략",
                    ["볼린저 밴드", "이동평균", "RSI"],
                    key="strategy"
                )
            else:
                order_type = st.selectbox(
                    "주문 유형",
                    ["지정가", "시장가"],
                    key="order_type"
                )
                
        # 주문 버튼
        order_cols = st.columns(2)
        
        with order_cols[0]:
            if st.button("매수", type="primary"):
                if not validate_input("매수"):
                    st.error("유효하지 않은 입력값입니다")
                    return
                if trade_mode == "자동":
                    st.info("자동 매수 전략 실행 중...")
                else:
                    volume = st.session_state.order_volume
                    price = st.session_state.order_price if st.session_state.order_type == "지정가" else None
                    result = trading_manager.place_order(market, "bid", volume, price)
                    if result:
                        st.success("매수 주문이 접수되었습니다")
                    else:
                        st.error("매수 주문 실패")
                        
        with order_cols[1]:
            if st.button("매도", type="primary"):
                if not validate_input("매도"):
                    st.error("유효하지 않은 입력값입니다")
                    return
                if trade_mode == "자동":
                    st.info("자동 매도 전략 실행 중...")
                else:
                    volume = st.session_state.order_volume
                    price = st.session_state.order_price if st.session_state.order_type == "지정가" else None
                    result = trading_manager.place_order(market, "ask", volume, price)
                    if result:
                        st.success("매도 주문이 접수되었습니다")
                    else:
                        st.error("매도 주문 실패")
                        
    except Exception as e:
        st.error(f"거래 인터페이스 렌더링 중 오류 발생: {str(e)}")

def render_auto_trading(trading_manager: TradingManager, market: str, market_data: Dict):
    """자동 매매 인터페이스 렌더링"""
    try:
        st.markdown("### 자동 매매")
        
        # 현재가 표시
        if market_data and 'current_price' in market_data:
            current_price = float(market_data['current_price'])
            st.metric("현재가", f"₩{current_price:,.0f}")
        else:
            st.warning("현재가를 불러올 수 없습니다")
            return
        
        # 기술적 지표 표시
        if market_data and 'indicators' in market_data:
            indicators = market_data['indicators']
            
            # 지표 표시를 위한 컬럼 생성
            col1, col2, col3 = st.columns(3)
            
            # RSI 표시
            with col1:
                if 'rsi' in indicators and len(indicators['rsi']) > 0:
                    rsi_value = float(indicators['rsi'].iloc[-1])
                    rsi_color = "🔴" if rsi_value > 70 else "🟢" if rsi_value < 30 else "⚪"
                    st.metric("RSI", f"{rsi_color} {rsi_value:.1f}")
                else:
                    st.metric("RSI", "N/A")
            
            # MACD 표시
            with col2:
                if ('macd' in indicators and 
                    isinstance(indicators['macd'], dict) and 
                    'macd' in indicators['macd'] and 
                    'signal' in indicators['macd'] and
                    len(indicators['macd']['macd']) > 0 and
                    len(indicators['macd']['signal']) > 0):
                    
                    macd_value = float(indicators['macd']['macd'].iloc[-1])
                    signal_value = float(indicators['macd']['signal'].iloc[-1])
                    macd_color = "🟢" if macd_value > signal_value else "🔴"
                    st.metric("MACD", f"{macd_color} {macd_value:.1f}")
                else:
                    st.metric("MACD", "N/A")
            
            # 볼린저 밴드 표시
            with col3:
                if ('bollinger_bands' in indicators and 
                    'upper' in indicators['bollinger_bands'] and 
                    'middle' in indicators['bollinger_bands'] and 
                    'lower' in indicators['bollinger_bands'] and
                    len(indicators['bollinger_bands']['middle']) > 0):
                    
                    bb_middle = float(indicators['bollinger_bands']['middle'].iloc[-1])
                    bb_position = ((current_price - bb_middle) / bb_middle) * 100
                    bb_color = "🟢" if bb_position > 0 else "🔴"
                    st.metric("볼린저 밴드", f"{bb_color} {bb_position:.1f}%")
                else:
                    st.metric("볼린저 밴드", "N/A")
        
        # 전략 실행
        try:
            result = trading_manager.execute_strategy(market, current_price)
            
            if result and isinstance(result, dict) and 'action' in result:
                if result['action'] != 'HOLD':
                    action_kr = "매수" if result['action'] == 'BUY' else "매도"
                    amount = float(result.get('amount', 0))
                    confidence = float(result.get('confidence', 0))
                    st.success(
                        f"자동 {action_kr} 실행: "
                        f"₩{amount:,.0f} "
                        f"(신뢰도: {confidence:.1%})"
                    )
                else:
                    st.info("현재 매매 신호 없음")
            else:
                st.warning("전략 실행 결과가 유효하지 않습니다")
        except Exception as e:
            st.error(f"전략 실행 중 오류 발생: {str(e)}")
            if hasattr(trading_manager, 'logger'):
                trading_manager.logger.error(f"Strategy execution error: {e}")
        
        # 메모리 정리
        if 'current_price' in locals(): del current_price
        if 'indicators' in locals(): del indicators
        if 'result' in locals(): del result
        if 'rsi_value' in locals(): del rsi_value
        if 'macd_value' in locals(): del macd_value
        if 'signal_value' in locals(): del signal_value
        if 'bb_middle' in locals(): del bb_middle
        if 'bb_position' in locals(): del bb_position
        
    except Exception as e:
        st.error(f"자동 매매 인터페이스 오류: {str(e)}")
        if hasattr(trading_manager, 'logger'):
            trading_manager.logger.error(f"Auto trading interface error: {e}")

def render_manual_trading(trading_manager: TradingManager, market: str, market_data: Dict):
    """수동 매매 인터페이스 렌더링"""
    try:
        st.markdown("### 수동 매매")
        
        # 현재가 표시
        if market_data and 'current_price' in market_data:
            current_price = float(market_data['current_price'])
            st.metric("현재가", f"₩{current_price:,.0f}")
        else:
            st.warning("현재가를 불러올 수 없습니다")
            return
        
        # 매수/매도 선택
        action = st.radio("매매 유형", ["매수", "매도"], horizontal=True)
        
        # 금액 입력
        try:
            if action == "매수":
                available = trading_manager.get_balance("KRW")
                max_amount = min(available, 1000000)  # 최대 100만원
                amount = st.number_input(
                    "매수 금액 (원)",
                    min_value=5000,
                    max_value=int(max_amount),
                    value=10000,
                    step=1000
                )
            else:  # 매도
                holdings = trading_manager.get_balance(market.split('-')[1])
                max_amount = holdings * current_price
                amount = st.number_input(
                    "매도 수량",
                    min_value=0.0,
                    max_value=float(holdings),
                    value=float(holdings),
                    step=0.0001,
                    format="%.4f"
                )
            
            # 주문 실행 버튼
            if st.button(f"{action} 실행"):
                try:
                    order_type = "buy" if action == "매수" else "sell"
                    result = trading_manager.execute_order(market, order_type, amount)
                    
                    if result and isinstance(result, dict) and result.get('status') == 'success':
                        st.success(f"{action} 주문이 성공적으로 실행되었습니다")
                    else:
                        st.error(f"{action} 주문 실행 중 오류가 발생했습니다")
                except Exception as e:
                    st.error(f"주문 실행 중 오류 발생: {str(e)}")
                    if hasattr(trading_manager, 'logger'):
                        trading_manager.logger.error(f"Order execution error: {e}")
        
        except Exception as e:
            st.error(f"거래 금액 설정 중 오류 발생: {str(e)}")
            if hasattr(trading_manager, 'logger'):
                trading_manager.logger.error(f"Amount setting error: {e}")
        
        # 메모리 정리
        if 'available' in locals(): del available
        if 'max_amount' in locals(): del max_amount
        if 'amount' in locals(): del amount
        if 'holdings' in locals(): del holdings
        if 'current_price' in locals(): del current_price
        
    except Exception as e:
        st.error(f"수동 매매 인터페이스 오류: {str(e)}")
        if hasattr(trading_manager, 'logger'):
            trading_manager.logger.error(f"Manual trading interface error: {e}")

def render_balance(trading_manager: TradingManager):
    """Render balance information."""
    try:
        # KRW 잔고 표시
        krw_balance = trading_manager.api.get_balance()
        if krw_balance:
            st.metric("KRW 잔고", f"₩{krw_balance:,.0f}")

        # 보유 자산 표시
        st.markdown("### 보유 자산")
        balances = trading_manager.api.get_accounts()
        if balances:
            for balance in balances:
                if balance['currency'] != 'KRW' and float(balance['balance']) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            f"{balance['currency']}",
                            f"{float(balance['balance']):.8f}",
                        )
                    with col2:
                        avg_price = float(balance['avg_buy_price'])
                        st.metric(
                            "평균 매수가",
                            f"₩{avg_price:,.0f}" if avg_price > 0 else "N/A"
                        )

        # 미체결 주문 표시
        st.markdown("### 미체결 주문")
        orders = trading_manager.api.get_pending_orders(market=None)
        if orders:
            for order in orders:
                side = "매수" if order['side'] == 'bid' else "매도"
                volume = float(order['remaining_volume'])
                price = float(order['price'])
                total = volume * price
                st.markdown(
                    f"**{order['market']}** - {side}\n"
                    f"- 수량: {volume:.8f}\n"
                    f"- 가격: ₩{price:,.0f}\n"
                    f"- 총액: ₩{total:,.0f}"
                )
                st.markdown("---")
        else:
            st.info("미체결 주문이 없습니다.")

    except Exception as e:
        st.error(f"Error rendering balance: {e}") 