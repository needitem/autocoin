"""
AutoCoin Trading Application

This is the main application file for the AutoCoin trading platform.
"""

import streamlit as st
from src.ui.app import AutoCoinApp
from src.core.trading import TradingManager
from src.api.upbit import UpbitTradingSystem

if __name__ == "__main__":
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="AutoCoin Trading Bot",
        page_icon="📈",
        layout="wide"
    )
    
    # 시스템 초기화
    @st.cache_resource
    def init_system():
        # 세션 상태에서 거래소 가져오기 (기본값: Upbit)
        default_exchange = 'upbit'
        trading_manager = TradingManager(exchange=default_exchange, verbose=False)
        return trading_manager
    
    # Trading Manager 초기화
    trading_manager = init_system()
    
    # 앱 실행
    app = AutoCoinApp(trading_manager=trading_manager)
    app.run() 