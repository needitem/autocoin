"""
Session Manager

This module manages the application session state.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from src.config import AppConfig
from src.trading.virtual_trading import VirtualTrading
from src.core.market_manager import MarketManager

class SessionManager:
    """Class for managing application session state."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.market_manager = MarketManager()
    
    def initialize_session_state(self) -> bool:
        """Initialize session state variables."""
        try:
            if 'market_data' not in st.session_state:
                st.session_state.market_data = {}
            
            if 'markets' not in st.session_state:
                st.session_state.markets = self.market_manager.get_markets()
            
            if 'auto_trading' not in st.session_state:
                st.session_state.auto_trading = {
                    'enabled': False,
                    'risk_threshold': 7.0,
                    'trade_amount': 100000,
                    'stop_loss': 0.05,
                    'take_profit': 0.10
                }
            
            if 'analysis_cache' not in st.session_state:
                st.session_state.analysis_cache = {
                    'last_analysis': None,
                    'last_analysis_time': None,
                    'analysis_result': None
                }
            
            return True
        except Exception as e:
            return False
    
    def update_market_data(self, data: Dict[str, Any]) -> None:
        """Update market data in session state."""
        st.session_state.market_data = data
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data from session state."""
        return st.session_state.market_data
    
    def get_virtual_trading(self) -> VirtualTrading:
        """Get virtual trading instance from session state."""
        return st.session_state.virtual_trading
    
    def get_analysis_cache(self) -> Dict[str, Any]:
        """Get analysis cache from session state."""
        return st.session_state.analysis_cache
    
    def update_analysis_cache(self, market: str, result: Dict[str, Any]):
        """Update analysis cache in session state."""
        st.session_state.analysis_cache.update({
            'last_analysis': market,
            'last_analysis_time': datetime.now(),
            'analysis_result': result
        })
    
    def is_auto_trading_enabled(self) -> bool:
        """Check if auto trading is enabled."""
        return st.session_state.auto_trading.get('enabled', False)
    
    def get_auto_trading_settings(self) -> Dict[str, Any]:
        """Get auto trading settings."""
        return st.session_state.auto_trading
    
    def update_auto_trading_settings(self, settings: Dict[str, Any]) -> None:
        """Update auto trading settings."""
        st.session_state.auto_trading.update(settings)
    
    def get_markets(self) -> list:
        """Get list of available markets."""
        return st.session_state.markets
    
    def update_markets(self, markets: list):
        """Update list of available markets."""
        st.session_state.markets = markets
    
    def clear_session(self):
        """Clear all session state data."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self.initialize_session_state() 