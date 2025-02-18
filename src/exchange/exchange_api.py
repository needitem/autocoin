"""
Exchange API Module

This module handles API interactions with various cryptocurrency exchanges
including Upbit and Binance.
"""

import ccxt
import pyupbit
import jwt
import uuid
import hashlib
from urllib.parse import urlencode
import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd

class ExchangeAPI:
    """Class for handling exchange API interactions."""
    
    def __init__(self) -> None:
        """Initialize exchange connections with API credentials."""
        self.logger = logging.getLogger(__name__)
        load_dotenv()
        
        # Initialize exchanges
        self.upbit = self._init_upbit()
        self.binance = self._init_binance()
        self.current_exchange = 'upbit'  # Default exchange

    def _init_upbit(self) -> Any:
        """
        Initialize Upbit exchange connection.
        
        Returns:
            Any: Authenticated Upbit instance
        """
        try:
            access_key = os.getenv('UPBIT_ACCESS_KEY')
            secret_key = os.getenv('UPBIT_SECRET_KEY')
            
            if not access_key or not secret_key:
                self.logger.warning("Upbit API keys not found. Running in public API mode.")
                return None
                
            return pyupbit.Upbit(access_key, secret_key)
            
        except Exception as e:
            self.logger.error(f"Error initializing Upbit: {str(e)}")
            return None

    def _init_binance(self) -> ccxt.binance:
        """
        Initialize Binance exchange connection.
        
        Returns:
            ccxt.binance: Authenticated Binance exchange instance
        """
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_SECRET_KEY')
            
            binance = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
            
            return binance
            
        except Exception as e:
            self.logger.error(f"Error initializing Binance: {str(e)}")
            return ccxt.binance()

    def set_exchange(self, exchange_name: str) -> None:
        """
        Set the current exchange to use.
        
        Args:
            exchange_name (str): Name of the exchange ('upbit' or 'binance')
        """
        if exchange_name.lower() in ['upbit', 'binance']:
            self.current_exchange = exchange_name.lower()
            self.logger.info(f"Exchange set to: {exchange_name}")
        else:
            raise ValueError("Unsupported exchange. Use 'upbit' or 'binance'")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Ticker information
        """
        try:
            if self.current_exchange == 'upbit':
                ticker = pyupbit.get_current_price(symbol)
                return {
                    'symbol': symbol,
                    'price': ticker,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                ticker = self.binance.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'change': ticker['percentage'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise Exception(f"Failed to fetch ticker: {str(e)}")

    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """
        Get current orderbook.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Orderbook data
        """
        try:
            if self.current_exchange == 'upbit':
                orderbook = pyupbit.get_orderbook(symbol)
                return {
                    'symbol': symbol,
                    'bids': orderbook['bids'],
                    'asks': orderbook['asks'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                orderbook = self.binance.fetch_order_book(symbol)
                return {
                    'symbol': symbol,
                    'bids': orderbook['bids'],
                    'asks': orderbook['asks'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol}: {str(e)}")
            raise Exception(f"Failed to fetch orderbook: {str(e)}")

    def get_balance(self) -> Dict[str, float]:
        """
        Get account balances.
        
        Returns:
            Dict[str, float]: Account balances
        """
        try:
            if self.current_exchange == 'upbit':
                if self.upbit is not None:
                    balances = self.upbit.get_balances()
                    return {
                        balance['currency']: float(balance['balance'])
                        for balance in balances
                    }
                else:
                    raise Exception("Upbit not authenticated")
            else:
                balances = self.binance.fetch_balance()
                return {
                    currency: float(balance['free'])
                    for currency, balance in balances['total'].items()
                    if float(balance['free']) > 0
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching balances: {str(e)}")
            raise Exception(f"Failed to fetch balances: {str(e)}")

    def place_order(self,
                   symbol: str,
                   order_type: str,
                   side: str,
                   amount: float,
                   price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol (str): Trading pair symbol
            order_type (str): Type of order ('limit' or 'market')
            side (str): Order side ('buy' or 'sell')
            amount (float): Order amount
            price (Optional[float]): Order price (required for limit orders)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            if self.current_exchange == 'upbit':
                if self.upbit is None:
                    raise Exception("Upbit not authenticated")
                    
                if order_type == 'limit':
                    if side == 'buy':
                        order = self.upbit.buy_limit_order(symbol, price, amount)
                    else:
                        order = self.upbit.sell_limit_order(symbol, price, amount)
                else:
                    if side == 'buy':
                        order = self.upbit.buy_market_order(symbol, amount)
                    else:
                        order = self.upbit.sell_market_order(symbol, amount)
                        
                return {
                    'id': order['uuid'],
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'open',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                order = self.binance.create_order(
                    symbol,
                    order_type,
                    side,
                    amount,
                    price
                )
                
                return {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order['price']) if order['price'] else None,
                    'status': order['status'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise Exception(f"Failed to place order: {str(e)}")

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id (str): Order ID to cancel
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Cancellation result
        """
        try:
            if self.current_exchange == 'upbit':
                if self.upbit is None:
                    raise Exception("Upbit not authenticated")
                    
                result = self.upbit.cancel_order(order_id)
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'status': 'canceled',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = self.binance.cancel_order(order_id, symbol)
                return {
                    'id': result['id'],
                    'symbol': result['symbol'],
                    'status': result['status'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            raise Exception(f"Failed to cancel order: {str(e)}")

    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id (str): Order ID to check
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Order status information
        """
        try:
            if self.current_exchange == 'upbit':
                if self.upbit is None:
                    raise Exception("Upbit not authenticated")
                    
                order = self.upbit.get_order(order_id)
                return {
                    'id': order['uuid'],
                    'symbol': symbol,
                    'type': order['ord_type'],
                    'side': order['side'],
                    'amount': float(order['volume']),
                    'price': float(order['price']),
                    'status': order['state'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                order = self.binance.fetch_order(order_id, symbol)
                return {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order['price']) if order['price'] else None,
                    'status': order['status'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching order status for {order_id}: {str(e)}")
            raise Exception(f"Failed to fetch order status: {str(e)}")

    def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trading history.
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of trades to fetch
            
        Returns:
            List[Dict[str, Any]]: List of historical trades
        """
        try:
            if self.current_exchange == 'upbit':
                if self.upbit is None:
                    raise Exception("Upbit not authenticated")
                    
                trades = self.upbit.get_order_list()
                return [{
                    'id': trade['uuid'],
                    'symbol': trade['market'],
                    'type': trade['ord_type'],
                    'side': trade['side'],
                    'amount': float(trade['volume']),
                    'price': float(trade['price']),
                    'status': trade['state'],
                    'timestamp': trade['created_at']
                } for trade in trades[:limit]]
            else:
                trades = self.binance.fetch_my_trades(symbol, limit=limit)
                return [{
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'type': trade['type'],
                    'side': trade['side'],
                    'amount': float(trade['amount']),
                    'price': float(trade['price']),
                    'cost': float(trade['cost']),
                    'fee': trade['fee'],
                    'timestamp': trade['timestamp']
                } for trade in trades]
                
        except Exception as e:
            self.logger.error(f"Error fetching trade history for {symbol}: {str(e)}")
            raise Exception(f"Failed to fetch trade history: {str(e)}")

    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information and trading rules.
        
        Returns:
            Dict[str, Any]: Exchange information
        """
        try:
            if self.current_exchange == 'upbit':
                markets = pyupbit.get_tickers()
                return {
                    'name': 'Upbit',
                    'markets': markets,
                    'status': 'active',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                info = self.binance.load_markets()
                return {
                    'name': 'Binance',
                    'markets': list(info.keys()),
                    'status': 'active',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching exchange info: {str(e)}")
            raise Exception(f"Failed to fetch exchange info: {str(e)}")

    def get_server_time(self) -> Dict[str, Any]:
        """
        Get exchange server time.
        
        Returns:
            Dict[str, Any]: Server time information
        """
        try:
            if self.current_exchange == 'upbit':
                return {
                    'timestamp': datetime.now().isoformat(),
                    'exchange': 'Upbit'
                }
            else:
                time = self.binance.fetch_time()
                return {
                    'timestamp': datetime.fromtimestamp(time/1000).isoformat(),
                    'exchange': 'Binance'
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching server time: {str(e)}")
            raise Exception(f"Failed to fetch server time: {str(e)}")

class UpbitAPI:
    """Class for interacting with the Upbit exchange API."""
    
    def __init__(self) -> None:
        """Initialize the UpbitAPI with API configuration."""
        self.logger = logging.getLogger(__name__)
        self.access_key = os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = os.getenv('UPBIT_SECRET_KEY')
        self.server_url = 'https://api.upbit.com'
        
    def _get_jwt_token(self, query: Optional[Dict[str, Any]] = None) -> str:
        """Generate JWT token for API authentication."""
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4())
        }
        
        if query:
            query_string = urlencode(query)
            m = hashlib.sha512()
            m.update(query_string.encode())
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        
        return jwt.encode(payload, self.secret_key)
    
    def get_market_all(self) -> List[Dict[str, Any]]:
        """Get all available markets."""
        try:
            url = f"{self.server_url}/v1/market/all"
            headers = {'Accept': 'application/json'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting markets: {str(e)}")
            return []
    
    def get_ticker(self, market: str) -> List[Dict[str, Any]]:
        """Get current price ticker."""
        try:
            url = f"{self.server_url}/v1/ticker"
            params = {'markets': market}
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting ticker: {str(e)}")
            return []
    
    def get_orderbook(self, market: str) -> List[Dict[str, Any]]:
        """Get current orderbook."""
        try:
            url = f"{self.server_url}/v1/orderbook"
            params = {'markets': market}
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Process response
            data = response.json()
            processed_data = []
            
            for item in data:
                orderbook_units = item.get('orderbook_units', [])
                
                # Calculate total sizes
                total_bid_size = sum(float(unit.get('bid_size', 0)) for unit in orderbook_units)
                total_ask_size = sum(float(unit.get('ask_size', 0)) for unit in orderbook_units)
                
                # Calculate spread
                if orderbook_units:
                    best_bid = float(orderbook_units[0].get('bid_price', 0))
                    best_ask = float(orderbook_units[0].get('ask_price', 0))
                    spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
                else:
                    best_bid = 0
                    best_ask = 0
                    spread = 0
                
                processed_item = {
                    'market': item.get('market', ''),
                    'timestamp': item.get('timestamp', ''),
                    'total_ask_size': float(total_ask_size),
                    'total_bid_size': float(total_bid_size),
                    'orderbook_units': [{
                        'ask_price': float(unit.get('ask_price', 0)),
                        'bid_price': float(unit.get('bid_price', 0)),
                        'ask_size': float(unit.get('ask_size', 0)),
                        'bid_size': float(unit.get('bid_size', 0))
                    } for unit in orderbook_units],
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': spread
                }
                
                processed_data.append(processed_item)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {str(e)}")
            return []
    
    def get_trades_ticks(self, market: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            url = f"{self.server_url}/v1/trades/ticks"
            params = {'market': market, 'count': count}
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting trades: {str(e)}")
            return []
    
    def get_candles_minutes(self, market: str, unit: int = 1, count: int = 200) -> List[Dict[str, Any]]:
        """Get minute candle data."""
        try:
            url = f"{self.server_url}/v1/candles/minutes/{unit}"
            params = {'market': market, 'count': count}
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting minute candles: {str(e)}")
            return []
    
    def get_candles_days(self, market: str, count: int = 200) -> List[Dict[str, Any]]:
        """Get daily candle data."""
        try:
            url = f"{self.server_url}/v1/candles/days"
            params = {'market': market, 'count': count}
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting daily candles: {str(e)}")
            return []
    
    def get_market_index(self, market: str) -> Dict[str, Any]:
        """Get market index data."""
        try:
            # Get recent trades
            trades = self.get_trades_ticks(market, count=100)
            
            # Get candles
            candles = self.get_candles_minutes(market, unit=1, count=1440)  # Last 24 hours
            
            # Calculate market index
            if not trades or not candles:
                return {}
            
            # Calculate volatility
            prices = [float(candle['trade_price']) for candle in candles]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum(r * r for r in returns) / len(returns)) ** 0.5 * 100
            
            # Calculate volume change
            volume_24h = sum(float(candle['candle_acc_trade_volume']) for candle in candles[:24])
            volume_prev_24h = sum(float(candle['candle_acc_trade_volume']) for candle in candles[24:48])
            volume_change = ((volume_24h - volume_prev_24h) / volume_prev_24h * 100) if volume_prev_24h > 0 else 0
            
            # Calculate price change
            price_change = ((prices[0] - prices[-1]) / prices[-1] * 100) if prices[-1] > 0 else 0
            
            # Calculate buy/sell pressure
            buy_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'BID')
            sell_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'ASK')
            pressure = ((buy_volume - sell_volume) / (buy_volume + sell_volume) * 100) if (buy_volume + sell_volume) > 0 else 0
            
            return {
                'market': market,
                'timestamp': datetime.now().isoformat(),
                'volatility_24h': float(volatility),
                'volume_change_24h': float(volume_change),
                'price_change_24h': float(price_change),
                'buy_sell_pressure': float(pressure)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market index: {str(e)}")
            return {} 