"""
Database Manager for Virtual Trading

This module handles the SQLite database operations for virtual trading.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class DBManager:
    def __init__(self):
        """Initialize database manager."""
        self.db_path = "virtual_trading.db"
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    amount REAL NOT NULL,
                    fee REAL NOT NULL,
                    pnl REAL NOT NULL
                )
            """)
            
            # Create performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    win_rate REAL NOT NULL,
                    return_rate REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    volatility REAL NOT NULL,
                    trading_days INTEGER NOT NULL
                )
            """)
            
            conn.commit()
    
    def save_trade(self, trade_data: Dict[str, Any]):
        """Save trade data to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, action, price,
                    quantity, amount, fee, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['timestamp'],
                trade_data['symbol'],
                trade_data['action'],
                trade_data['price'],
                trade_data['quantity'],
                trade_data['amount'],
                trade_data['fee'],
                trade_data['pnl']
            ))
            
            conn.commit()
    
    def save_performance(self, perf_data: Dict[str, Any]):
        """Save performance metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance (
                    timestamp, win_rate, return_rate,
                    sharpe_ratio, volatility, trading_days
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                perf_data['win_rate'],
                perf_data['return_rate'],
                perf_data['sharpe_ratio'],
                perf_data['volatility'],
                perf_data['trading_days']
            ))
            
            conn.commit()
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, symbol, action, price,
                       quantity, amount, fee, pnl
                FROM trades
                ORDER BY timestamp DESC
            """)
            
            columns = ['timestamp', 'symbol', 'action', 'price',
                      'quantity', 'amount', 'fee', 'pnl']
            trades = []
            
            for row in cursor.fetchall():
                trades.append(dict(zip(columns, row)))
            
            return trades
    
    def get_latest_performance(self) -> Optional[Dict[str, Any]]:
        """Get latest performance metrics from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT win_rate, return_rate, sharpe_ratio,
                       volatility, trading_days
                FROM performance
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'win_rate': row[0],
                    'return_rate': row[1],
                    'sharpe_ratio': row[2],
                    'volatility': row[3],
                    'trading_days': row[4]
                }
            
            return None
    
    def clear_data(self):
        """Clear all data from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM performance")
            
            conn.commit() 