"""
AutoCoin Trading Application

This is the main application file for the AutoCoin trading platform.
"""

import streamlit as st
from src.core.session_manager import SessionManager
from src.core.trading_manager import TradingManager
from src.core.strategy_manager import StrategyManager
from src.core.error_handler import ErrorHandler
from src.ui.components.sidebar import SidebarComponent
from src.ui.components.market_metrics import MarketMetricsComponent
from src.ui.components.virtual_trading import VirtualTradingComponent
from src.ui.components.strategy_analysis import StrategyAnalysisComponent
from src.ui.components.virtual_trading_tab import VirtualTradingTab

class AutoCoinApp:
    def __init__(self):
        """Initialize the application."""
        st.set_page_config(
            page_title="AutoCoin",
            page_icon="📈",
            layout="wide"
        )
        
        self.session_manager = SessionManager()
        self.trading_manager = TradingManager()
        self.strategy_manager = StrategyManager()
        self.error_handler = ErrorHandler()
        
        # UI Components
        self.sidebar = SidebarComponent()
        self.market_metrics = MarketMetricsComponent()
        self.virtual_trading = VirtualTradingComponent()
        self.strategy_analysis = StrategyAnalysisComponent()
        self.virtual_trading_tab = VirtualTradingTab()
    
    def run(self):
        """Run the application."""
        try:
            # Initialize session state
            self.session_manager.initialize_session_state()
            
            # Render sidebar and get selected market
            market = self.sidebar.render(self.session_manager.get_markets())
            
            # Render main content
            self.render_main_content(market)
            
        except Exception as e:
            self.error_handler.handle_error("애플리케이션 실행 실패", e)
    
    def render_main_content(self, market: str):
        """Render main content area."""
        try:
            # Get market data once
            market_data = self.session_manager.get_market_data()
            
            # Create tabs
            tab1, tab2 = st.tabs(["실제 매매", "가상 매매"])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    self._render_trading_section(market)
                with col2:
                    analysis = self.strategy_manager.get_analysis(market)
                    self.strategy_analysis.render(analysis)
            
            with tab2:
                # Pass both market and market_data
                self.virtual_trading_tab.render({
                    'market': market,
                    'current_price': market_data.get('current_price', 0),
                    'symbol': market
                })
                
        except Exception as e:
            self.error_handler.handle_error("화면 렌더링 실패", e)
    
    def _render_trading_section(self, market: str):
        """Render trading section."""
        try:
            # Update market data first
            market_data = self.trading_manager.get_market_data(market)
            if market_data:
                self.session_manager.update_market_data(market_data)
            
            # Get updated market data
            market_data = self.session_manager.get_market_data()
            
            # Render market metrics if data is available
            if market_data and market_data.get('current_price'):
                self.market_metrics.render(market_data)
                
                # Execute trading based on mode
                price = market_data.get('current_price')
                if self.session_manager.is_auto_trading_enabled():
                    # Get strategy analysis
                    analysis = self.strategy_manager.get_analysis(market)
                    
                    # Execute auto trading strategy
                    result = self.trading_manager.execute_strategy(market, price)
                    
                    if result.get('action') != 'HOLD':
                        if result.get('amount', 0) > 0:
                            st.success(
                                f"자동 {'매수' if result['action'] == 'BUY' else '매도'} 실행: "
                                f"₩{result['amount']:,.0f} "
                                f"(신뢰도: {result['confidence']:.1%})"
                            )
                else:
                    self.trading_manager.execute_manual_trade(market, price)
            else:
                st.warning("시장 데이터를 불러올 수 없습니다.")
                    
        except Exception as e:
            self.error_handler.handle_error("거래 섹션 렌더링 실패", e)

if __name__ == "__main__":
    app = AutoCoinApp()
    app.run() 