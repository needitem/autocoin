import streamlit as st

def render_balance(balances):
    """잔고 정보 렌더링
    
    Args:
        balances (list): 계좌 잔고 정보 리스트
    """
    try:
        if not balances:
            st.error("잔고 정보를 불러올 수 없습니다")
            return
            
        st.markdown("### 보유 자산")
        
        for balance in balances:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=balance['currency'],
                    value=f"{float(balance['balance']):,.8f}"
                )
                
            with col2:
                if float(balance['locked']) > 0:
                    st.metric(
                        label="주문 중",
                        value=f"{float(balance['locked']):,.8f}"
                    )
                    
            with col3:
                if float(balance['avg_buy_price']) > 0:
                    st.metric(
                        label="평균 매수가",
                        value=f"{float(balance['avg_buy_price']):,.0f}"
                    )
                    
    except Exception as e:
        st.error(f"잔고 정보 렌더링 중 오류 발생: {str(e)}") 