"""
Virtual Trading Module

This module handles virtual trading functionality based on AI trading signals.
"""

from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import logging

class VirtualTrading:
    """Class for managing virtual trading."""
    
    def __init__(self, initial_balance: float = 10_000_000):
        """Initialize virtual trading with initial balance."""
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> None:
        """Reset trading state."""
        self.balance = self.initial_balance
        self.holdings = {}  # {market: {'amount': float, 'avg_price': float}}
        self.trade_history = []
        self.start_time = datetime.now()
    
    def execute_trade(self, 
                     market: str, 
                     action: str, 
                     current_price: float,
                     confidence: float,
                     max_position_size: float = 50.0,
                     min_trade_amount: float = 5000.0) -> Dict[str, Any]:
        """
        Execute a virtual trade based on AI signal.
        
        Args:
            market (str): Market symbol
            action (str): Trading action (BUY/SELL/HOLD)
            current_price (float): Current market price
            confidence (float): AI confidence level
            max_position_size (float): Maximum position size as percentage of total balance
            min_trade_amount (float): Minimum trade amount in KRW
            
        Returns:
            Dict[str, Any]: Trade result
        """
        try:
            if action == 'HOLD':
                return self._get_trading_status(market, "관망 중입니다")
            
            # Calculate trade amount based on confidence and max position size
            if action == 'BUY':
                return self._execute_buy(market, current_price, confidence, max_position_size, min_trade_amount)
            elif action == 'SELL':
                return self._execute_sell(market, current_price, confidence)
            
            return self._get_trading_status(market, "잘못된 거래 신호입니다")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return self._get_trading_status(market, f"거래 실행 오류: {str(e)}")
    
    def _execute_buy(self, 
                    market: str, 
                    current_price: float, 
                    confidence: float,
                    max_position_size: float,
                    min_trade_amount: float,
                    orderbook: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute buy order with orderbook simulation."""
        try:
            # Calculate maximum amount based on position size limit
            max_position_amount = self.balance * (max_position_size / 100.0)
            max_amount = min(max_position_amount, self.balance * confidence)
            
            if max_amount < min_trade_amount:
                return self._get_trading_status(market, f"거래금액({max_amount:,.0f}원)이 최소 거래금액({min_trade_amount:,.0f}원)보다 작습니다")
            
            # Simulate order execution with orderbook
            executed_orders = []
            remaining_amount = max_amount
            avg_price = 0
            total_quantity = 0
            
            if orderbook and orderbook[0].get('orderbook_units'):
                for order in orderbook[0]['orderbook_units']:
                    ask_price = float(order['ask_price'])
                    ask_size = float(order['ask_size'])
                    
                    # Calculate how much we can buy at this price level
                    possible_amount = min(remaining_amount, ask_price * ask_size)
                    if possible_amount >= min_trade_amount:
                        quantity = possible_amount / ask_price
                        executed_orders.append({
                            'price': ask_price,
                            'quantity': quantity,
                            'amount': possible_amount
                        })
                        
                        avg_price = ((avg_price * total_quantity) + (ask_price * quantity)) / (total_quantity + quantity)
                        total_quantity += quantity
                        remaining_amount -= possible_amount
                        
                        if remaining_amount < min_trade_amount:
                            break
            else:
                # Fallback to current price if no orderbook
                quantity = max_amount / current_price
                executed_orders.append({
                    'price': current_price,
                    'quantity': quantity,
                    'amount': max_amount
                })
                avg_price = current_price
                total_quantity = quantity
            
            if not executed_orders:
                return self._get_trading_status(market, "호가창에서 적절한 매도 물량을 찾을 수 없습니다")
            
            # Execute the trade
            total_cost = sum(order['amount'] for order in executed_orders)
            self.balance -= total_cost
            
            # Update holdings
            if market in self.holdings:
                total_amount = self.holdings[market]['amount'] + total_quantity
                total_cost_with_existing = (self.holdings[market]['amount'] * self.holdings[market]['avg_price']) + total_cost
                self.holdings[market] = {
                    'amount': total_amount,
                    'avg_price': total_cost_with_existing / total_amount
                }
            else:
                self.holdings[market] = {
                    'amount': total_quantity,
                    'avg_price': avg_price
                }
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'market': market,
                'action': 'BUY',
                'price': avg_price,
                'amount': total_quantity,
                'cost': total_cost,
                'confidence': confidence,
                'balance': self.balance,
                'executed_orders': executed_orders
            })
            
            executed_details = [f"{order['quantity']:.8f} @ ₩{order['price']:,.0f}" for order in executed_orders]
            return self._get_trading_status(
                market, 
                f"매수 체결 완료:\n" + "\n".join(executed_details) + f"\n평균단가: ₩{avg_price:,.0f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            return self._get_trading_status(market, f"매수 주문 실행 오류: {str(e)}")
    
    def _execute_sell(self, 
                     market: str, 
                     current_price: float, 
                     confidence: float,
                     orderbook: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute sell order with orderbook simulation."""
        try:
            if market not in self.holdings or self.holdings[market]['amount'] <= 0:
                return self._get_trading_status(market, "보유 수량이 없습니다")
            
            # Calculate sell amount based on confidence
            available_amount = self.holdings[market]['amount']
            sell_quantity = available_amount * confidence
            
            # Simulate order execution with orderbook
            executed_orders = []
            remaining_quantity = sell_quantity
            avg_price = 0
            total_amount = 0
            
            if orderbook and orderbook[0].get('orderbook_units'):
                for order in orderbook[0]['orderbook_units']:
                    bid_price = float(order['bid_price'])
                    bid_size = float(order['bid_size'])
                    
                    # Calculate how much we can sell at this price level
                    quantity = min(remaining_quantity, bid_size)
                    amount = quantity * bid_price
                    
                    executed_orders.append({
                        'price': bid_price,
                        'quantity': quantity,
                        'amount': amount
                    })
                    
                    avg_price = ((avg_price * total_amount) + (bid_price * amount)) / (total_amount + amount)
                    total_amount += amount
                    remaining_quantity -= quantity
                    
                    if remaining_quantity <= 0:
                        break
            else:
                # Fallback to current price if no orderbook
                amount = sell_quantity * current_price
                executed_orders.append({
                    'price': current_price,
                    'quantity': sell_quantity,
                    'amount': amount
                })
                avg_price = current_price
                total_amount = amount
            
            if not executed_orders:
                return self._get_trading_status(market, "호가창에서 적절한 매수 물량을 찾을 수 없습니다")
            
            # Execute the trade
            total_quantity = sum(order['quantity'] for order in executed_orders)
            self.balance += total_amount
            
            # Update holdings
            self.holdings[market]['amount'] -= total_quantity
            if self.holdings[market]['amount'] <= 0:
                del self.holdings[market]
            
            # Calculate profit/loss
            avg_buy_price = self.holdings.get(market, {}).get('avg_price', current_price)
            profit_loss = ((avg_price - avg_buy_price) / avg_buy_price) * 100
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'market': market,
                'action': 'SELL',
                'price': avg_price,
                'amount': total_quantity,
                'revenue': total_amount,
                'profit_loss': profit_loss,
                'confidence': confidence,
                'balance': self.balance,
                'executed_orders': executed_orders
            })
            
            executed_details = [f"{order['quantity']:.8f} @ ₩{order['price']:,.0f}" for order in executed_orders]
            return self._get_trading_status(
                market, 
                f"매도 체결 완료:\n" + "\n".join(executed_details) + 
                f"\n평균단가: ₩{avg_price:,.0f} (수익률: {profit_loss:.2f}%)"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            return self._get_trading_status(market, f"매도 주문 실행 오류: {str(e)}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        total_value = self.balance
        holdings_detail = []
        
        for market, holding in self.holdings.items():
            holdings_detail.append({
                'market': market,
                'amount': holding['amount'],
                'avg_price': holding['avg_price']
            })
            total_value += holding['amount'] * holding['avg_price']
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_value': total_value,
            'total_return': ((total_value - self.initial_balance) / self.initial_balance) * 100,
            'holdings': holdings_detail,
            'trade_count': len(self.trade_history),
            'start_time': self.start_time,
            'current_time': datetime.now()
        }
    
    def _get_trading_status(self, market: str, message: str) -> Dict[str, Any]:
        """Get current trading status."""
        portfolio = self.get_portfolio_status()
        return {
            'message': message,
            'market': market,
            'balance': self.balance,
            'total_value': portfolio['total_value'],
            'total_return': portfolio['total_return'],
            'holdings': portfolio['holdings']
        } 