"""
Database Module

This module handles database operations for the cryptocurrency trading application,
including storing market data, trading signals, and performance metrics.
"""

import sqlite3
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import os

class Database:
    """Class for handling database operations."""
    
    def __init__(self) -> None:
        """Initialize the Database with default settings."""
        self.logger = logging.getLogger(__name__)
        self.db_path = "data/trading.db"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create market data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        data JSON NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create trading signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        signal_data JSON NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_time TEXT NOT NULL,
                        exit_time TEXT,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        position_size REAL NOT NULL,
                        trade_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        pnl REAL,
                        metadata JSON,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        metadata JSON,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise Exception(f"Failed to initialize database: {str(e)}")

    def store_market_data(self, 
                         symbol: str,
                         timeframe: str,
                         data: Dict[str, Any]) -> None:
        """
        Store market data in the database.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            data (Dict[str, Any]): Market data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO market_data (symbol, timeframe, timestamp, data)
                    VALUES (?, ?, ?, ?)
                """, (
                    symbol,
                    timeframe,
                    datetime.now().isoformat(),
                    json.dumps(data)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            raise Exception(f"Failed to store market data: {str(e)}")

    def store_trading_signal(self,
                           symbol: str,
                           signal_type: str,
                           signal_data: Dict[str, Any]) -> None:
        """
        Store trading signal in the database.
        
        Args:
            symbol (str): Trading pair symbol
            signal_type (str): Type of trading signal
            signal_data (Dict[str, Any]): Signal data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_signals (symbol, timestamp, signal_type, signal_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now().isoformat(),
                    signal_type,
                    json.dumps(signal_data)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing trading signal: {str(e)}")
            raise Exception(f"Failed to store trading signal: {str(e)}")

    def record_trade(self,
                    symbol: str,
                    entry_time: str,
                    entry_price: float,
                    position_size: float,
                    trade_type: str,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record a new trade in the database.
        
        Args:
            symbol (str): Trading pair symbol
            entry_time (str): Trade entry timestamp
            entry_price (float): Entry price
            position_size (float): Position size
            trade_type (str): Type of trade (e.g., 'LONG', 'SHORT')
            metadata (Optional[Dict[str, Any]]): Additional trade metadata
            
        Returns:
            int: Trade ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, entry_time, entry_price, position_size,
                        trade_type, status, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    entry_time,
                    entry_price,
                    position_size,
                    trade_type,
                    'OPEN',
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
            raise Exception(f"Failed to record trade: {str(e)}")

    def update_trade(self,
                    trade_id: int,
                    exit_time: str,
                    exit_price: float,
                    pnl: float) -> None:
        """
        Update a trade record with exit information.
        
        Args:
            trade_id (int): Trade ID
            exit_time (str): Trade exit timestamp
            exit_price (float): Exit price
            pnl (float): Profit/Loss amount
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE trades
                    SET exit_time = ?, exit_price = ?, pnl = ?, status = 'CLOSED'
                    WHERE id = ?
                """, (
                    exit_time,
                    exit_price,
                    pnl,
                    trade_id
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            raise Exception(f"Failed to update trade: {str(e)}")

    def store_performance_metric(self,
                               metric_type: str,
                               value: float,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a performance metric in the database.
        
        Args:
            metric_type (str): Type of metric
            value (float): Metric value
            metadata (Optional[Dict[str, Any]]): Additional metric metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_metrics (timestamp, metric_type, value, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    metric_type,
                    value,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing performance metric: {str(e)}")
            raise Exception(f"Failed to store performance metric: {str(e)}")

    def get_recent_market_data(self,
                             symbol: str,
                             timeframe: str,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent market data from the database.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            limit (int): Number of records to retrieve
            
        Returns:
            List[Dict[str, Any]]: Recent market data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT data
                    FROM market_data
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, timeframe, limit))
                
                rows = cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {str(e)}")
            return []

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get all open trades from the database.
        
        Returns:
            List[Dict[str, Any]]: Open trades
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT *
                    FROM trades
                    WHERE status = 'OPEN'
                    ORDER BY entry_time DESC
                """)
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade = dict(zip(columns, row))
                    trade['metadata'] = json.loads(trade['metadata'])
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            self.logger.error(f"Error retrieving open trades: {str(e)}")
            return []

    def get_trade_history(self,
                         symbol: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history from the database.
        
        Args:
            symbol (Optional[str]): Trading pair symbol to filter by
            limit (int): Number of trades to retrieve
            
        Returns:
            List[Dict[str, Any]]: Trade history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                        SELECT *
                        FROM trades
                        WHERE symbol = ?
                        ORDER BY entry_time DESC
                        LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                        SELECT *
                        FROM trades
                        ORDER BY entry_time DESC
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade = dict(zip(columns, row))
                    trade['metadata'] = json.loads(trade['metadata'])
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            self.logger.error(f"Error retrieving trade history: {str(e)}")
            return []

    def get_performance_metrics(self,
                              metric_type: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics from the database.
        
        Args:
            metric_type (Optional[str]): Type of metric to filter by
            limit (int): Number of metrics to retrieve
            
        Returns:
            List[Dict[str, Any]]: Performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if metric_type:
                    cursor.execute("""
                        SELECT *
                        FROM performance_metrics
                        WHERE metric_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (metric_type, limit))
                else:
                    cursor.execute("""
                        SELECT *
                        FROM performance_metrics
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metric = dict(zip(columns, row))
                    metric['metadata'] = json.loads(metric['metadata'])
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error retrieving performance metrics: {str(e)}")
            return []

    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep (int): Number of days of data to retain
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate cutoff date
                cutoff_date = (
                    datetime.now() - timedelta(days=days_to_keep)
                ).isoformat()
                
                # Delete old market data
                cursor.execute("""
                    DELETE FROM market_data
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Delete old trading signals
                cursor.execute("""
                    DELETE FROM trading_signals
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Delete old performance metrics
                cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            raise Exception(f"Failed to clean up old data: {str(e)}")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                for table in ['market_data', 'trading_signals', 'trades', 'performance_metrics']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Get database file size
                stats['database_size'] = os.path.getsize(self.db_path)
                
                # Get latest record timestamps
                for table in ['market_data', 'trading_signals', 'trades', 'performance_metrics']:
                    cursor.execute(f"""
                        SELECT timestamp
                        FROM {table}
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    stats[f"{table}_latest"] = result[0] if result else None
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {} 