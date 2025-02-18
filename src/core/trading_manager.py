"""
Trading Manager

This module handles trading operations and management.
"""

import logging
from typing import Dict, Any, Optional
import streamlit as st
from datetime import datetime
from src.config import AppConfig
from src.trading.virtual_trading import VirtualTrading
from src.strategies.upbit_trading_strategy import UpbitTradingStrategy
import pyupbit

class TradingManager:
    """Class for managing trading operations."""
    
    def __init__(self):
        """Initialize the trading manager."""
        self.logger = logging.getLogger(__name__)
        self.strategy = UpbitTradingStrategy()
        self.virtual_trading = VirtualTrading(initial_balance=AppConfig.INITIAL_BALANCE)
    
    def execute_strategy(self, market: str, current_price: float) -> Dict[str, Any]:
        """Execute trading strategy for the specified market."""
        try:
            # Get signals from strategy analysis
            signals = self.strategy.analyze_market(market)
            if not signals or 'signals' not in signals:
                return self._create_trade_result(market, 'HOLD', 0)
            
            # Extract trading signals
            ma_signal = signals['signals']['ma']['signal']
            rsi_signal = signals['signals']['rsi']['signal']
            macd_signal = signals['signals']['macd']['signal']
            
            # Calculate overall signal
            bullish_count = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] if signal == 'BULLISH')
            bearish_count = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] if signal == 'BEARISH')
            
            # Determine action
            if bullish_count >= 2:
                action = 'BUY'
                confidence = bullish_count / 3
            elif bearish_count >= 2:
                action = 'SELL'
                confidence = bearish_count / 3
            else:
                return self._create_trade_result(market, 'HOLD', 0)
            
            # Check risk metrics
            risk_metrics = signals['risk']
            if risk_metrics['risk_score'] > AppConfig.RISK_THRESHOLD:
                return self._create_trade_result(market, 'HOLD', confidence)
            
            # Calculate trade amount
            trade_amount = self._calculate_trade_amount(current_price)
            if not trade_amount:
                return self._create_trade_result(market, 'HOLD', confidence)
            
            # Execute trade
            success = False
            if action == 'BUY':
                success = self.virtual_trading.buy(market, current_price, trade_amount)
            else:  # SELL
                success = self.virtual_trading.sell(market, current_price, trade_amount)
            
            return self._create_trade_result(
                market,
                action if success else 'HOLD',
                confidence,
                trade_amount if success else 0
            )
            
        except Exception as e:
            self.logger.error(f"자동매매 실행 실패: {str(e)}")
            return self._create_trade_result(market, 'HOLD', 0)
    
    def execute_manual_trade(self, market: str, price: float) -> bool:
        """Execute manual trade."""
        try:
            # Implement actual trading logic here
            return True
        except Exception as e:
            self.logger.error(f"매매 실행 실패: {str(e)}")
            return False
    
    def _calculate_trade_amount(self, current_price: float) -> Optional[float]:
        """Calculate trade amount based on current balance and price."""
        try:
            balance = self.virtual_trading.balance
            
            # Check minimum trade amount
            min_amount = AppConfig.MIN_TRADE_AMOUNT
            if balance < min_amount:
                return None
            
            # Calculate trade amount (10% of balance)
            trade_amount = balance * AppConfig.TRADE_AMOUNT_PERCENTAGE
            
            # Ensure trade amount is between min and max limits
            trade_amount = max(min_amount, min(trade_amount, AppConfig.MAX_TRADE_AMOUNT))
            
            return trade_amount if balance >= current_price else None
            
        except Exception as e:
            self.logger.error(f"거래금액 계산 실패: {str(e)}")
            return None
    
    def _create_trade_result(self, market: str, action: str, confidence: float, amount: float = 0) -> Dict[str, Any]:
        """Create trade result dictionary."""
        return {
            'market': market,
            'action': action,
            'confidence': confidence,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_trading_data(self) -> Dict[str, Any]:
        """Get current trading data."""
        return {
            'symbol': '',
            'current_price': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_win_rate(self) -> float:
        total_trades = self.virtual_trading.win_count + self.virtual_trading.loss_count
        return (self.virtual_trading.win_count / total_trades * 100) if total_trades > 0 else 0
    
    def get_market_data(self, market: str) -> Optional[Dict[str, Any]]:
        """Get current market data."""
        try:
            # Get current ticker
            ticker = pyupbit.get_current_price(market)
            if not ticker:
                return None
            
            # Get daily OHLCV
            df = pyupbit.get_ohlcv(market, interval="day", count=1)
            if df is None or df.empty:
                return None
            
            return {
                'market': market,
                'symbol': market,
                'current_price': float(ticker),
                'open': float(df['open'].iloc[-1]),
                'high': float(df['high'].iloc[-1]),
                'low': float(df['low'].iloc[-1]),
                'volume_24h': float(df['volume'].iloc[-1]),
                'value_24h': float(df['value'].iloc[-1]),
                'price_change_24h': float((ticker - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100),
                'volume_change_24h': 0.0,  # Need historical data for comparison
                'market_cap': float(df['value'].iloc[-1]),  # Using trading value as market cap
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            return None 