"""
UI components test module
"""

import pytest
import time
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from src.core.trading import TradingManager
from src.api.calculate import TechnicalAnalysis
from unittest.mock import patch, MagicMock

@pytest.fixture
def trading_manager():
    """TradingManager 인스턴스"""
    return TradingManager(verbose=True)

@pytest.fixture
def market_data(trading_manager):
    """시장 데이터"""
    market = "KRW-BTC"
    ohlcv_df = trading_manager.api.fetch_ohlcv(market)
    ticker = trading_manager.api.get_ticker(market)
    
    if isinstance(ticker, list):
        ticker = ticker[0]
    
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

@pytest.fixture
def mock_market_data():
    """Mock 시장 데이터"""
    import pandas as pd
    import numpy as np
    
    # 날짜 범위 생성
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # OHLCV 데이터 생성
    ohlcv_df = pd.DataFrame({
        'open': np.random.normal(50000000, 1000000, len(dates)),
        'high': np.random.normal(51000000, 1000000, len(dates)),
        'low': np.random.normal(49000000, 1000000, len(dates)),
        'close': np.random.normal(50000000, 1000000, len(dates)),
        'volume': np.random.normal(100, 10, len(dates))
    }, index=dates)
    
    # 이동평균선
    ma_data = {
        'MA5': ohlcv_df['close'].rolling(window=5).mean(),
        'MA10': ohlcv_df['close'].rolling(window=10).mean(),
        'MA20': ohlcv_df['close'].rolling(window=20).mean()
    }
    
    # 볼린저 밴드
    middle = ohlcv_df['close'].rolling(window=20).mean()
    std = ohlcv_df['close'].rolling(window=20).std()
    upper = middle + (std * 2)
    lower = middle - (std * 2)
    
    # RSI
    delta = ohlcv_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = ohlcv_df['close'].ewm(span=12, adjust=False).mean()
    exp2 = ohlcv_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    return {
        'current_price': float(ohlcv_df['close'].iloc[-1]),
        'indicators': {
            'moving_averages': ma_data,
            'bollinger_bands': {
                'upper': upper,
                'middle': middle,
                'lower': lower
            },
            'rsi': rsi,
            'macd': {
                'macd': macd_line,
                'signal': signal_line,
                'hist': macd_line - signal_line
            }
        }
    }

@pytest.fixture
def mock_trading_manager():
    """Mock 거래 관리자"""
    class MockTradingManager:
        def get_balance(self, currency):
            if currency == "KRW":
                return 1000000  # 100만원
            return 0.1  # 0.1 BTC
        
        def execute_order(self, market, order_type, amount):
            return {'status': 'success'}
        
        def execute_strategy(self, market, current_price):
            return {
                'action': 'BUY',
                'amount': 100000,  # 10만원
                'confidence': 0.8
            }
        
        def logger(self):
            return None
    
    return MockTradingManager()

def test_render_market_selector(trading_manager):
    """마켓 선택기 렌더링 테스트"""
    from src.ui.components.market import render_market_selector
    
    # 실제 마켓 목록 가져오기
    markets = trading_manager.get_markets()
    assert len(markets) > 0
    assert any(market['market'].startswith('KRW-') for market in markets)

def test_render_market_data(market_data):
    """마켓 데이터 렌더링 테스트"""
    from src.ui.components.market import render_market_data
    
    # 정상 데이터 테스트
    assert market_data is not None
    assert 'current_price' in market_data
    assert 'open' in market_data
    assert 'high' in market_data
    assert 'low' in market_data
    assert 'volume' in market_data
    assert 'change_rate' in market_data
    
    # 데이터 타입 검증
    assert isinstance(market_data['current_price'], (int, float))
    assert isinstance(market_data['volume'], (int, float))
    assert isinstance(market_data['change_rate'], (int, float))

def test_render_trading_interface(trading_manager):
    """거래 인터페이스 렌더링 테스트"""
    from src.ui.components.trading import render_trading_interface
    
    # 거래 관리자 검증
    assert trading_manager is not None
    assert hasattr(trading_manager, 'api')
    assert hasattr(trading_manager, 'strategy')

def test_render_auto_trading(mock_trading_manager, mock_market_data):
    """자동 매매 인터페이스 테스트"""
    try:
        from src.ui.components.trading import render_auto_trading
        
        # 시장 데이터 검증
        assert isinstance(mock_market_data, dict)
        assert 'current_price' in mock_market_data
        assert isinstance(mock_market_data['current_price'], (int, float))
        
        # 거래 관리자 검증
        assert mock_trading_manager is not None
        assert hasattr(mock_trading_manager, 'execute_order')
        
        # 시장 데이터 검증
        market = "KRW-BTC"  # 테스트용 마켓
        
        # 자동 매매 인터페이스 렌더링
        try:
            render_auto_trading(mock_trading_manager, market, mock_market_data)
        except Exception as e:
            pytest.fail(f"자동 매매 인터페이스 렌더링 실패: {str(e)}")
        
        # 메모리 정리
        del mock_market_data
        
    except Exception as e:
        pytest.fail(f"자동 매매 테스트 실패: {str(e)}")

def test_render_manual_trading(mock_trading_manager):
    """수동 매매 인터페이스 테스트"""
    try:
        from src.ui.components.trading import render_manual_trading
        
        # Mock 시장 데이터
        market_data = {'current_price': 50000000}  # 5천만원
        market = "KRW-BTC"
        
        # 수동 매매 인터페이스 렌더링
        try:
            render_manual_trading(mock_trading_manager, market, market_data)
        except Exception as e:
            pytest.fail(f"수동 매매 인터페이스 렌더링 실패: {str(e)}")
        
        # 메모리 정리
        del market_data
        
    except Exception as e:
        pytest.fail(f"수동 매매 테스트 실패: {str(e)}")

def test_technical_indicators_display(mock_market_data):
    """기술적 지표 표시 테스트"""
    try:
        from src.ui.components.market import render_market_data
        
        # 필수 지표 존재 확인
        assert 'indicators' in mock_market_data
        indicators = mock_market_data['indicators']
        
        # 이동평균선 검증
        assert 'moving_averages' in indicators
        ma_data = indicators['moving_averages']
        assert 'MA5' in ma_data
        assert 'MA10' in ma_data
        assert 'MA20' in ma_data
        
        # 볼린저 밴드 검증
        assert 'bollinger_bands' in indicators
        bb_data = indicators['bollinger_bands']
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        
        # RSI 검증
        assert 'rsi' in indicators
        assert isinstance(indicators['rsi'], pd.Series)
        assert 0 <= indicators['rsi'].iloc[-1] <= 100
        
        # MACD 검증
        assert 'macd' in indicators
        macd_data = indicators['macd']
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'hist' in macd_data
        
        # 메모리 정리
        del indicators
        del ma_data
        del bb_data
        del macd_data
        
    except Exception as e:
        pytest.fail(f"기술적 지표 표시 테스트 실패: {str(e)}")

def test_real_time_updates(trading_manager):
    """실시간 업데이트 테스트"""
    market = "KRW-BTC"
    
    # 첫 번째 데이터 조회
    initial_ticker = trading_manager.api.get_ticker(market)
    time.sleep(1)  # API 호출 제한 고려
    
    # 두 번째 데이터 조회
    updated_ticker = trading_manager.api.get_ticker(market)
    
    # 데이터가 다른지 확인 (실시간 시장이므로 가격이 변경되었을 것)
    assert initial_ticker != updated_ticker

def test_component_interaction(mock_trading_manager, mock_market_data):
    """컴포넌트 상호작용 테스트"""
    try:
        from src.ui.components.market import render_market_data
        from src.ui.components.trading import render_auto_trading, render_manual_trading, render_trading_interface
        
        # 시장 데이터 검증
        assert isinstance(mock_market_data, dict), "시장 데이터가 딕셔너리 형식이 아닙니다"
        assert 'current_price' in mock_market_data, "현재가 데이터가 없습니다"
        assert isinstance(mock_market_data['current_price'], (int, float)), "현재가가 숫자 형식이 아닙니다"
        
        # 거래 관리자 검증
        assert mock_trading_manager is not None, "거래 관리자가 None입니다"
        assert hasattr(mock_trading_manager, 'execute_order'), "거래 관리자에 execute_order 메서드가 없습니다"
        assert hasattr(mock_trading_manager, 'execute_strategy'), "거래 관리자에 execute_strategy 메서드가 없습니다"
        
        # 시장 데이터 검증
        market = "KRW-BTC"  # 테스트용 마켓
        
        # 컴포넌트 렌더링 테스트
        try:
            # 시장 데이터 렌더링
            render_market_data(mock_market_data)
            
            # 자동 매매 인터페이스 렌더링
            render_auto_trading(mock_trading_manager, market, mock_market_data)
            
            # 수동 매매 인터페이스 렌더링
            render_manual_trading(mock_trading_manager, market, mock_market_data)
            
            # 거래 인터페이스 렌더링
            render_trading_interface(mock_trading_manager, market, mock_market_data)
            
        except Exception as e:
            pytest.fail(f"컴포넌트 렌더링 실패: {str(e)}")
            
        # 메모리 정리
        del mock_market_data
        
    except Exception as e:
        pytest.fail(f"컴포넌트 상호작용 테스트 실패: {str(e)}")
    finally:
        # 추가 메모리 정리
        import gc
        gc.collect()

def test_large_number_formatting(market_data):
    """큰 숫자 포맷팅 테스트"""
    # 데이터 형식 검증
    assert isinstance(market_data['current_price'], (int, float))
    assert isinstance(market_data['volume'], (int, float))
    
    # 가격이 정수형태인지 확인
    assert market_data['current_price'].is_integer()
    
    # 거래량이 소수점 이하 자리를 가질 수 있는지 확인
    volume_str = f"{market_data['volume']:.4f}"
    assert '.' in volume_str

if __name__ == "__main__":
    pytest.main(["-v"]) 