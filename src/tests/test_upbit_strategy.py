"""
Test module for Upbit trading strategy.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from ..strategies.upbit_trading_strategy import UpbitTradingStrategy

class TestUpbitTradingStrategy(unittest.TestCase):
    """Test cases for UpbitTradingStrategy class."""
    
    def setUp(self):
        """Set up test environment."""
        self.strategy = UpbitTradingStrategy()
        
        # Mock market data
        self.mock_candles = [
            {
                'market': 'KRW-BTC',
                'candle_date_time_utc': '2024-03-20T00:00:00',
                'opening_price': 100000.0,
                'high_price': 101000.0,
                'low_price': 99000.0,
                'trade_price': 100500.0,
                'timestamp': 1234567890,
                'candle_acc_trade_price': 1000000.0,
                'candle_acc_trade_volume': 10.0
            }
        ] * 200  # Duplicate for 200 candles
        
        # Add some variation to prices for testing
        for i, candle in enumerate(self.mock_candles):
            multiplier = 1 + (i - 100) * 0.001  # Create a trend
            candle['trade_price'] *= multiplier
            candle['opening_price'] *= multiplier
            candle['high_price'] *= multiplier
            candle['low_price'] *= multiplier
            candle['candle_acc_trade_volume'] = 10.0 + i * 0.1
        
        self.mock_market_index = {
            'volatility_24h': 3.5,
            'volume_change_24h': 25.0,
            'price_change_24h': 2.5,
            'buy_sell_pressure': 15.0
        }
        
        self.mock_orderbook = [{
            'market': 'KRW-BTC',
            'timestamp': 1234567890,
            'total_ask_size': 10.0,
            'total_bid_size': 12.0,
            'orderbook_units': [
                {
                    'ask_price': 101000.0,
                    'bid_price': 100000.0,
                    'ask_size': 1.0,
                    'bid_size': 1.2
                }
            ]
        }]
        
        self.mock_account = [
            {
                'currency': 'KRW',
                'balance': '1000000.0',
                'locked': '0.0',
                'avg_buy_price': '0',
                'avg_buy_price_modified': True
            },
            {
                'currency': 'BTC',
                'balance': '0.1',
                'locked': '0.0',
                'avg_buy_price': '100000000',
                'avg_buy_price_modified': False
            }
        ]
    
    @patch('src.exchange.exchange_api.UpbitAPI')
    def test_analyze_market(self, mock_upbit):
        """Test market analysis functionality."""
        # Configure mock
        mock_upbit.get_candles_minutes.return_value = self.mock_candles
        mock_upbit.get_market_index.return_value = self.mock_market_index
        mock_upbit.get_orderbook.return_value = self.mock_orderbook
        
        # Set mock as strategy's upbit instance
        self.strategy.upbit = mock_upbit
        
        # Test analysis
        result = self.strategy.analyze_market('KRW-BTC')
        
        # Verify result structure
        self.assertIn('market', result)
        self.assertIn('indicators', result)
        self.assertIn('market_index', result)
        self.assertIn('signals', result)
        self.assertIn('timestamp', result)
        
        # Verify indicators
        indicators = result['indicators']
        self.assertIn('moving_averages', indicators)
        self.assertIn('bollinger_bands', indicators)
        self.assertIn('rsi', indicators)
        self.assertIn('macd', indicators)
        self.assertIn('volume', indicators)
    
    @patch('src.exchange.exchange_api.UpbitAPI')
    def test_execute_strategy(self, mock_upbit):
        """Test strategy execution."""
        # Configure mocks
        mock_upbit.get_candles_minutes.return_value = self.mock_candles
        mock_upbit.get_market_index.return_value = self.mock_market_index
        mock_upbit.get_orderbook.return_value = self.mock_orderbook
        mock_upbit.get_account.return_value = self.mock_account
        mock_upbit.get_ticker.return_value = [{'trade_price': 100000.0}]
        
        # Mock order placement
        mock_upbit.place_order.return_value = {
            'uuid': '123456789',
            'side': 'bid',
            'ord_type': 'limit',
            'price': '100000.0',
            'state': 'wait',
            'market': 'KRW-BTC',
            'created_at': '2024-03-20T00:00:00+00:00',
            'volume': '0.01',
            'remaining_volume': '0.01',
            'reserved_fee': '5.0',
            'remaining_fee': '5.0',
            'paid_fee': '0.0',
            'locked': '1000.0',
            'executed_volume': '0.0',
            'trades_count': 0
        }
        
        # Set mock as strategy's upbit instance
        self.strategy.upbit = mock_upbit
        
        # Test strategy execution
        result = self.strategy.execute_strategy('KRW-BTC')
        
        # Verify result structure
        self.assertIn('market', result)
        self.assertIn('action', result)
        self.assertIn('confidence', result)
        self.assertIn('timestamp', result)
        
        # If order was placed, verify order details
        if 'order' in result:
            order = result['order']
            self.assertIn('uuid', order)
            self.assertIn('side', order)
            self.assertIn('price', order)
            self.assertIn('volume', order)
    
    def test_calculate_risk_level(self):
        """Test risk level calculation."""
        # Create test DataFrame
        df = pd.DataFrame(self.mock_candles)
        
        # Test LOW risk
        market_index = {
            'volatility_24h': 3.0,
            'volume_change_24h': 20.0,
            'price_change_24h': 2.0
        }
        risk_level = self.strategy._calculate_risk_level(df, market_index)
        self.assertEqual(risk_level, 'LOW')
        
        # Test MEDIUM risk
        market_index = {
            'volatility_24h': 7.0,
            'volume_change_24h': 60.0,
            'price_change_24h': 7.0
        }
        risk_level = self.strategy._calculate_risk_level(df, market_index)
        self.assertEqual(risk_level, 'MEDIUM')
        
        # Test HIGH risk
        market_index = {
            'volatility_24h': 12.0,
            'volume_change_24h': 120.0,
            'price_change_24h': 12.0
        }
        risk_level = self.strategy._calculate_risk_level(df, market_index)
        self.assertEqual(risk_level, 'HIGH')
    
    def test_generate_signals(self):
        """Test trading signal generation."""
        # Create test DataFrame
        df = pd.DataFrame(self.mock_candles)
        
        # Calculate indicators
        indicators = self.strategy._calculate_indicators(df)
        
        # Generate signals
        signals = self.strategy._generate_signals(
            df,
            indicators,
            self.mock_market_index,
            self.mock_orderbook[0]
        )
        
        # Verify signal structure
        self.assertIn('action', signals)
        self.assertIn('confidence', signals)
        self.assertIn('signals', signals)
        self.assertIn('risk_level', signals)
        
        # Verify signal types
        signal_types = {s['type'] for s in signals['signals']}
        expected_types = {'MA_CROSS', 'RSI', 'MACD', 'BB', 'VOLUME', 'PRESSURE'}
        self.assertTrue(signal_types.issubset(expected_types))
        
        # Verify signal actions
        for signal in signals['signals']:
            self.assertIn(signal['action'], {'BUY', 'SELL'})
            self.assertTrue(0 <= signal['strength'] <= 1)

if __name__ == '__main__':
    unittest.main() 