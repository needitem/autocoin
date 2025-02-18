"""
Data Fetcher Module

This module handles fetching cryptocurrency market data from various exchanges,
including historical OHLCV data, order book data, and trade history.
"""

import ccxt
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataFetcher:
    """Class for fetching cryptocurrency market data."""
    
    def __init__(self) -> None:
        """Initialize the DataFetcher with default settings."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange connections
        self.exchanges = {
            'binance': ccxt.binance({
                'enableRateLimit': True
            })
        }
        
        self.default_exchange = 'binance'

    def fetch_ohlcv(self,
                   symbol: str,
                   timeframe: str = '1h',
                   limit: int = 100,
                   exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for data (e.g., '1m', '1h', '1d')
            limit (int): Number of candles to fetch
            exchange (Optional[str]): Exchange to use (default: self.default_exchange)
            
        Returns:
            Dict[str, Any]: OHLCV data and metadata
        """
        try:
            exchange = exchange or self.default_exchange
            exchange_instance = self.exchanges[exchange]
            
            # Fetch OHLCV data
            ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure all required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Calculate additional metrics
            df['volume_24h'] = df['volume'].rolling(window=24).sum()
            df['price_change_24h'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24) * 100
            
            # Convert DataFrame to dict format
            ohlcv_data = []
            for _, row in df.iterrows():
                ohlcv_data.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'volume_24h': float(row['volume_24h']) if not pd.isna(row['volume_24h']) else 0.0,
                    'price_change_24h': float(row['price_change_24h']) if not pd.isna(row['price_change_24h']) else 0.0
                })
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'ohlcv': ohlcv_data,
                'current_price': float(df['close'].iloc[-1]),
                'volume_24h': float(df['volume_24h'].iloc[-1]) if not pd.isna(df['volume_24h'].iloc[-1]) else 0.0,
                'price_change_24h': float(df['price_change_24h'].iloc[-1]) if not pd.isna(df['price_change_24h'].iloc[-1]) else 0.0,
                'metadata': {
                    'start_time': df['timestamp'].min().isoformat(),
                    'end_time': df['timestamp'].max().isoformat(),
                    'num_candles': len(df)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            # Return empty but valid data structure
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange or self.default_exchange,
                'ohlcv': [],
                'current_price': 0.0,
                'volume_24h': 0.0,
                'price_change_24h': 0.0,
                'metadata': {
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'num_candles': 0,
                    'error': str(e)
                }
            }

    def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol (str): Trading pair symbol
            exchange (Optional[str]): Exchange to use
            
        Returns:
            Dict[str, Any]: Ticker information
        """
        try:
            exchange = exchange or self.default_exchange
            exchange_instance = self.exchanges[exchange]
            
            # Fetch ticker
            ticker = exchange_instance.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': datetime.fromtimestamp(ticker['timestamp']/1000).isoformat(),
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'quote_volume': ticker['quoteVolume'],
                'percentage': ticker['percentage'],
                'metadata': {
                    'trading_pairs': symbol,
                    'update_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            raise Exception(f"Failed to fetch ticker: {str(e)}")

    def get_order_book(self, symbol: str, limit: int = 20, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current order book.
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of orders to fetch
            exchange (Optional[str]): Exchange to use
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            exchange = exchange or self.default_exchange
            exchange_instance = self.exchanges[exchange]
            
            # Fetch order book
            order_book = exchange_instance.fetch_order_book(symbol, limit=limit)
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'spread': order_book['asks'][0][0] - order_book['bids'][0][0],
                'metadata': {
                    'bid_levels': len(order_book['bids']),
                    'ask_levels': len(order_book['asks'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            raise Exception(f"Failed to fetch order book: {str(e)}")

    def get_market_info(self, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Get market information.
        
        Args:
            symbol (Optional[str]): Trading pair symbol
            exchange (Optional[str]): Exchange to use
            
        Returns:
            Dict[str, Any]: Market information
        """
        try:
            exchange = exchange or self.default_exchange
            exchange_instance = self.exchanges[exchange]
            
            # Fetch markets
            markets = exchange_instance.fetch_markets()
            
            # Filter by symbol if provided
            if symbol:
                markets = [m for m in markets if m['symbol'] == symbol]
            
            processed_markets = []
            for market in markets:
                processed_markets.append({
                    'symbol': market['symbol'],
                    'base': market['base'],
                    'quote': market['quote'],
                    'active': market['active'],
                    'precision': market['precision'],
                    'limits': market['limits'],
                    'info': market['info']
                })
            
            return {
                'exchange': exchange,
                'markets': processed_markets,
                'metadata': {
                    'num_markets': len(processed_markets),
                    'update_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market info: {str(e)}")
            raise Exception(f"Failed to fetch market info: {str(e)}") 