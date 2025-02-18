"""
Virtual Trading Module for Strategy Testing

This module simulates trading with virtual money to test trading strategies.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import logging

class VirtualTrading:
    """가상 거래를 통한 전략 테스트를 위한 클래스입니다."""
    
    def __init__(self, initial_balance: float = 10_000_000):
        """초기 잔고로 가상 거래를 초기화합니다."""
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> None:
        """거래 상태를 초기화합니다."""
        self.balance = self.initial_balance
        self.holdings = {}  # market -> {quantity, avg_price}
        self.trade_history = []
        self.total_profit = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.start_time = datetime.now()
    
    def buy(self, market: str, price: float, amount: float) -> bool:
        if amount <= 0 or price <= 0 or amount > self.balance:
            return False
            
        quantity = amount / price
        if quantity <= 0:
            return False
            
        holding = self.holdings.get(market, {'quantity': 0, 'avg_price': 0})
        total_quantity = holding['quantity'] + quantity
        total_cost = (holding['quantity'] * holding['avg_price']) + amount
        
        self.holdings[market] = {
            'quantity': total_quantity,
            'avg_price': total_cost / total_quantity if total_quantity > 0 else 0
        }
        
        self.balance -= amount
        self._add_trade_history('BUY', market, price, quantity, amount)
        
        return True
    
    def sell(self, market: str, price: float, amount: float) -> bool:
        if amount <= 0 or price <= 0:
            return False
            
        holding = self.holdings.get(market)
        if not holding:
            return False
            
        quantity = amount / price
        if quantity <= 0 or quantity > holding['quantity']:
            return False
            
        profit = (price - holding['avg_price']) * quantity
        self.total_profit += profit
        
        if profit > 0:
            self.win_count += 1
        elif profit < 0:
            self.loss_count += 1
        
        remaining_quantity = holding['quantity'] - quantity
        if remaining_quantity > 0:
            self.holdings[market]['quantity'] = remaining_quantity
        else:
            del self.holdings[market]
        
        self.balance += amount
        self._add_trade_history('SELL', market, price, quantity, amount, profit)
        
        return True
    
    def _add_trade_history(self, action: str, market: str, price: float,
                          quantity: float, amount: float, profit: float = 0) -> None:
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'market': market,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'profit': profit
        })
        
        # Keep only the last 100 trades to save memory
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_holding(self, market: str) -> Dict[str, float]:
        return self.holdings.get(market, {'quantity': 0, 'avg_price': 0})
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """전략 성능 지표를 반환합니다."""
        total_trades = len(self.trade_history)
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        # 최대 손실폭(MDD) 계산
        balance_history = []
        current_balance = self.initial_balance
        for trade in self.trade_history:
            if trade['action'] == 'BUY':
                current_balance -= trade['amount']
            else:  # SELL
                current_balance += trade['amount']
            balance_history.append(current_balance)
        
        if balance_history:
            running_max = pd.Series(balance_history).expanding().max()
            drawdowns = (running_max - balance_history) / running_max * 100
            max_drawdown = float(drawdowns.max())
        else:
            max_drawdown = 0.0
        
        # 총 자산 계산
        total_value = self.balance
        if self.holdings:
            last_price = self.trade_history[-1]['price'] if self.trade_history else 0
            for market, holding in self.holdings.items():
                total_value += holding['quantity'] * last_price
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_value': total_value,
            'total_return': ((total_value - self.initial_balance) / self.initial_balance) * 100,
            'total_profit': self.total_profit,
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """거래 기록을 반환합니다."""
        return self.trade_history 