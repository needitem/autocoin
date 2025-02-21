"""
Pytest configuration file
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_ohlcv_data():
    """기본 OHLCV 데이터 생성"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.abs(np.random.normal(1000, 100, 100))
    }, index=dates)
    
    # high가 open, close보다 높고, low가 open, close보다 낮도록 조정
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

@pytest.fixture
def sample_market_data(sample_ohlcv_data):
    """통합 시장 데이터 생성"""
    # 이동평균선
    ma_data = {
        'MA5': sample_ohlcv_data['close'].rolling(window=5).mean(),
        'MA10': sample_ohlcv_data['close'].rolling(window=10).mean(),
        'MA20': sample_ohlcv_data['close'].rolling(window=20).mean(),
        'MA60': sample_ohlcv_data['close'].rolling(window=60).mean(),
        'MA120': sample_ohlcv_data['close'].rolling(window=120).mean()
    }
    
    # RSI
    delta = sample_ohlcv_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    bb_period = 20
    bb_std = 2
    middle_band = sample_ohlcv_data['close'].rolling(window=bb_period).mean()
    std = sample_ohlcv_data['close'].rolling(window=bb_period).std()
    
    return {
        'current_price': float(sample_ohlcv_data['close'].iloc[-1]),
        'ohlcv_df': sample_ohlcv_data,
        'volume': sample_ohlcv_data['volume'].tolist(),
        'volatility': float(sample_ohlcv_data['close'].pct_change().std() * 100),
        'indicators': {
            'moving_averages': ma_data,
            'rsi': rsi,
            'bollinger_bands': {
                'upper': middle_band + (std * bb_std),
                'middle': middle_band,
                'lower': middle_band - (std * bb_std)
            },
            'macd': {
                'macd': sample_ohlcv_data['close'].ewm(span=12, adjust=False).mean() - 
                       sample_ohlcv_data['close'].ewm(span=26, adjust=False).mean(),
                'signal': (sample_ohlcv_data['close'].ewm(span=12, adjust=False).mean() - 
                          sample_ohlcv_data['close'].ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean(),
                'hist': None  # 실제 계산시 추가
            },
            'stochastic': {
                'fast': {
                    'k': pd.Series(dtype=float),
                    'd': pd.Series(dtype=float)
                },
                'slow': {
                    'k': pd.Series(dtype=float),
                    'd': pd.Series(dtype=float)
                }
            }
        }
    }

@pytest.fixture
def mock_upbit_response():
    """Upbit API 응답 모의"""
    return {
        'market': 'KRW-BTC',
        'candle_date_time_utc': '2024-01-01T00:00:00',
        'opening_price': 50000000.0,
        'high_price': 51000000.0,
        'low_price': 49000000.0,
        'trade_price': 50500000.0,
        'candle_acc_trade_volume': 1.5,
        'unit': 1
    }

@pytest.fixture
def empty_market_data():
    """빈 시장 데이터"""
    return {
        'current_price': 0,
        'ohlcv_df': pd.DataFrame(),
        'volume': [],
        'volatility': 0,
        'indicators': {}
    }

@pytest.fixture
def invalid_market_data():
    """잘못된 형식의 시장 데이터"""
    return {
        'current_price': None,
        'ohlcv_df': None,
        'volume': None,
        'volatility': None,
        'indicators': None
    }

@pytest.fixture
def mock_streamlit():
    """Streamlit 모의 객체"""
    from unittest.mock import MagicMock
    mock_st = MagicMock()
    mock_st.selectbox = MagicMock(return_value="KRW-BTC")
    mock_st.empty = MagicMock(return_value=MagicMock())
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    mock_st.expander = MagicMock(return_value=MagicMock())
    mock_st.radio = MagicMock(return_value="자동")
    return mock_st

@pytest.fixture
def trading_manager():
    """실제 TradingManager 인스턴스"""
    from src.core.trading import TradingManager
    return TradingManager(verbose=True)

@pytest.fixture
def market_data(trading_manager):
    """실제 시장 데이터"""
    market = "KRW-BTC"
    ohlcv_df = trading_manager.api.fetch_ohlcv(market)
    ticker = trading_manager.api.get_ticker(market)
    
    if isinstance(ticker, list):
        ticker = ticker[0]
    
    from src.api.calculate import TechnicalAnalysis
    technical = TechnicalAnalysis()
    indicators = technical.calculate_all_indicators(ohlcv_df)
    
    return {
        'current_price': float(ticker.get('trade_price', 0)),
        'open': float(ticker.get('opening_price', 0)),
        'high': float(ticker.get('high_price', 0)),
        'low': float(ticker.get('low_price', 0)),
        'volume': float(ticker.get('acc_trade_volume_24h', 0)),
        'change_rate': float(ticker.get('signed_change_rate', 0)) * 100,
        'ohlcv_df': ohlcv_df,
        'indicators': indicators
    } 