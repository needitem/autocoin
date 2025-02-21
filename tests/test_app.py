"""
Streamlit 애플리케이션 테스트
"""

import os
import sys
import pytest
import time
import psutil
import threading
import queue
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import gc

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.app import AutoCoinApp
from src.core.trading import TradingManager
from src.api.upbit import UpbitTradingSystem

# Fixtures
@pytest.fixture
def trading_manager():
    """실제 TradingManager 인스턴스"""
    return TradingManager(verbose=True)

@pytest.fixture
def app(trading_manager):
    """실제 AutoCoinApp 인스턴스"""
    return AutoCoinApp(trading_manager)

@pytest.fixture
def mock_streamlit():
    """Streamlit 모의 객체"""
    mock_st = MagicMock()
    mock_st.selectbox = MagicMock(return_value="KRW-BTC")
    mock_st.empty = MagicMock(return_value=MagicMock())
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    mock_st.expander = MagicMock(return_value=MagicMock())
    mock_st.radio = MagicMock(return_value="자동")
    return mock_st

# 기본 기능 테스트
def test_app_initialization(app, trading_manager):
    """앱 초기화 테스트"""
    assert isinstance(app.trading_manager, TradingManager)
    assert isinstance(app.trading_manager.api, UpbitTradingSystem)
    assert hasattr(app, 'technical_analysis')

# 실제 API 테스트
def test_real_api_calls(trading_manager):
    """실제 API 호출 테스트"""
    try:
        # 마켓 조회
        markets = trading_manager.get_markets()
        assert isinstance(markets, list), "마켓 목록이 리스트가 아닙니다"
        assert len(markets) > 0, "마켓 목록이 비어있습니다"
        
        # KRW-BTC 마켓이 있는지 확인
        btc_market = next((market for market in markets 
                          if isinstance(market, dict) and market.get('market') == 'KRW-BTC'), None)
        assert btc_market is not None, "KRW-BTC 마켓을 찾을 수 없습니다"

        # API 호출 간격 준수
        time.sleep(0.1)

        # OHLCV 데이터 조회
        ohlcv = trading_manager.api.fetch_ohlcv('KRW-BTC', interval='minute1', count=10)
        assert isinstance(ohlcv, pd.DataFrame), "OHLCV 데이터가 DataFrame이 아닙니다"
        assert len(ohlcv) > 0, "OHLCV 데이터가 비어있습니다"
        assert all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
            "OHLCV 데이터에 필요한 컬럼이 없습니다"

        # API 호출 간격 준수
        time.sleep(0.1)

        # 현재가 조회
        ticker = trading_manager.api.get_ticker('KRW-BTC')
        assert isinstance(ticker, dict), "현재가 데이터가 딕셔너리가 아닙니다"
        assert 'trade_price' in ticker, "현재가 데이터에 trade_price가 없습니다"
        assert isinstance(ticker['trade_price'], (int, float)), "trade_price가 숫자가 아닙니다"
        assert ticker['trade_price'] > 0, "trade_price가 0 이하입니다"

    except Exception as e:
        pytest.fail(f"API 호출 테스트 실패: {str(e)}")

def test_api_error_handling(trading_manager):
    """API 에러 처리 테스트"""
    # 잘못된 마켓 코드
    invalid_market = "INVALID-MARKET"
    with pytest.raises(Exception):
        trading_manager.api.get_ticker(invalid_market)

    # 네트워크 타임아웃
    with patch('src.api.upbit.UpbitTradingSystem.fetch_ohlcv',
              side_effect=TimeoutError("Network timeout")):
        with pytest.raises(TimeoutError):
            trading_manager.api.fetch_ohlcv('KRW-BTC')

# 비정상적인 상황 테스트
def test_invalid_market_data_handling(mock_streamlit):
    """비정상적인 마켓 데이터 처리 테스트"""
    from src.ui.components.market import render_market_data
    
    test_cases = [
        (None, "시장 데이터를 불러올 수 없습니다"),
        ({}, "일부 시장 데이터가 누락되었습니다"),
        (
            {'current_price': -1000, 'open': 1000, 'high': 1000, 
             'low': 1000, 'volume': 1000, 'change_rate': 0},
            "일부 시장 데이터가 누락되었습니다"
        ),
        (
            {'current_price': float('inf'), 'open': 1000, 'high': 1000, 
             'low': 1000, 'volume': 1000, 'change_rate': 0},
            "일부 시장 데이터가 누락되었습니다"
        ),
        (
            {'current_price': 'invalid', 'open': 1000, 'high': 1000, 
             'low': 1000, 'volume': 1000, 'change_rate': 0},
            "일부 시장 데이터가 누락되었습니다"
        )
    ]
    
    for data, expected_message in test_cases:
        with patch('streamlit.warning') as mock_warning:
            render_market_data(data)
            mock_warning.assert_called_with(expected_message)

# 동시성 테스트
def test_concurrent_order_placement(trading_manager):
    """동시 주문 처리 테스트"""
    from src.ui.components.trading import render_trading_interface
    
    order_queue = queue.Queue()
    
    def place_order():
        with patch('streamlit.button') as mock_button:
            mock_button.return_value = True
            render_trading_interface(trading_manager, "KRW-BTC")
            order_queue.put(True)
    
    threads = [threading.Thread(target=place_order) for _ in range(5)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    assert order_queue.qsize() == 5

# 메모리 누수 테스트
def test_memory_leak(app):
    """메모리 누수 테스트"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    for _ in range(100):
        with patch('streamlit.line_chart'):
            app._render_technical_analysis({
                'moving_averages': pd.DataFrame(np.random.randn(1000, 4)),
                'bollinger_bands': {
                    'upper': pd.Series(np.random.randn(1000)),
                    'middle': pd.Series(np.random.randn(1000)),
                    'lower': pd.Series(np.random.randn(1000))
                },
                'macd': {
                    'macd': pd.Series(np.random.randn(1000)),
                    'signal': pd.Series(np.random.randn(1000)),
                    'hist': pd.Series(np.random.randn(1000))
                },
                'rsi': pd.Series(np.random.randn(1000)),
                'stochastic': {
                    'fast': {'k': pd.Series(np.random.randn(1000)),
                            'd': pd.Series(np.random.randn(1000))},
                    'slow': {'k': pd.Series(np.random.randn(1000)),
                            'd': pd.Series(np.random.randn(1000))}
                }
            })
    
    final_memory = process.memory_info().rss
    # 메모리 증가가 10MB 이하여야 함
    assert (final_memory - initial_memory) < 10 * 1024 * 1024

# 대량 데이터 처리 테스트
def test_large_dataset_handling(app):
    """대량 데이터 처리 테스트"""
    # 대량의 테스트 데이터 생성
    large_data = {
        'ohlcv_df': pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10000, freq='1min'),
            'open': np.random.randn(10000),
            'high': np.random.randn(10000),
            'low': np.random.randn(10000),
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000)
        })
    }
    
    start_time = time.time()
    
    with patch('streamlit.line_chart'):
        app._render_technical_analysis(
            app.technical_analysis.calculate_all_indicators(large_data['ohlcv_df'])
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 처리 시간이 5초를 넘지 않아야 함
    assert processing_time < 5.0

# 보안 테스트
def test_api_key_security(trading_manager):
    """API 키 보안 테스트"""
    import re
    
    with patch('logging.Logger.info') as mock_logger:
        trading_manager.api.get_balance()
        
        for call in mock_logger.call_args_list:
            log_message = call[0][0]
            assert not re.search(r'[a-f0-9]{32}', log_message)

def test_input_sanitization(trading_manager):
    """입력값 검증 테스트"""
    from src.ui.components.trading import validate_input
    
    dangerous_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "../../../etc/passwd",
        "$(rm -rf /)",
        "`rm -rf /`",
        "'; DELETE FROM market_data; --",
        "<img src=x onerror=alert('XSS')>",
        "../../.env",
        "$(cat /etc/passwd)",
        "&& rm -rf /"
    ]
    
    # 위험한 입력값 테스트
    for dangerous_input in dangerous_inputs:
        assert not validate_input(dangerous_input), \
            f"위험한 입력값이 통과됨: {dangerous_input}"
    
    # 안전한 입력값 테스트
    safe_inputs = [
        "KRW-BTC",
        "1234567890",
        "0.12345",
        "Test Market",
        "50%",
        "-123.45",
        "1e-10",
        "한글입력",
        "  spaces  ",
        "under_score"
    ]
    
    for safe_input in safe_inputs:
        assert validate_input(safe_input), \
            f"안전한 입력값이 거부됨: {safe_input}"

# 성능 테스트
def test_performance_degradation(app):
    """성능 저하 테스트"""
    import gc
    import time
    import psutil
    import os
    
    try:
        # 초기 메모리 사용량 측정
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
        
        # 초기 응답 시간 측정
        start_time = time.time()
        app.get_market_data("KRW-BTC")
        initial_response_time = time.time() - start_time
        
        # 10회 반복 테스트
        response_times = []
        memory_usages = []
        
        for i in range(10):
            # 가비지 컬렉션 실행
            gc.collect()
            
            # 응답 시간 측정
            start_time = time.time()
            app.get_market_data("KRW-BTC")
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # 메모리 사용량 측정
            memory_usage = process.memory_info().rss / 1024 / 1024
            memory_usages.append(memory_usage)
            
            # 응답 시간이 초기값의 2배 이상인 경우 경고
            if response_time > initial_response_time * 2:
                pytest.warns(UserWarning, match=f"응답 시간이 크게 증가했습니다: {response_time:.2f}초")
            
            # 메모리 사용량이 초기값의 1.5배 이상인 경우 경고
            if memory_usage > initial_memory * 1.5:
                pytest.warns(UserWarning, match=f"메모리 사용량이 크게 증가했습니다: {memory_usage:.2f}MB")
        
        # 최종 검증
        final_memory = process.memory_info().rss / 1024 / 1024
        avg_response_time = sum(response_times) / len(response_times)
        
        # 메모리 누수 검사
        assert final_memory < initial_memory * 2, f"메모리 사용량이 초기값의 2배를 초과했습니다: {final_memory:.2f}MB"
        
        # 응답 시간 저하 검사
        assert avg_response_time < initial_response_time * 3, f"평균 응답 시간이 초기값의 3배를 초과했습니다: {avg_response_time:.2f}초"
        
        # 메모리 정리
        del response_times
        del memory_usages
        gc.collect()
        
    except Exception as e:
        pytest.fail(f"성능 테스트 중 오류 발생: {str(e)}")

# 실시간 데이터 테스트
def test_realtime_data_updates(trading_manager):
    """실시간 데이터 업데이트 테스트"""
    import time
    from datetime import datetime, timedelta
    
    try:
        market = "KRW-BTC"
        update_count = 5  # 업데이트 횟수
        min_update_interval = 0.2  # 최소 업데이트 간격 (초)
        max_price_change = 10  # 최대 허용 가격 변동 (%)
        
        # 초기 데이터 검증 함수
        def validate_ticker_data(ticker, context=""):
            assert isinstance(ticker, dict), f"{context} 현재가 데이터가 딕셔너리가 아닙니다"
            
            required_fields = ['trade_price', 'acc_trade_volume_24h', 'acc_trade_price_24h', 
                             'high_price', 'low_price', 'prev_closing_price']
            
            for field in required_fields:
                assert field in ticker, f"{context} 데이터에 {field}가 없습니다"
                assert isinstance(ticker[field], (int, float)), f"{context} {field}가 숫자가 아닙니다"
                assert ticker[field] >= 0, f"{context} {field}가 음수입니다"
        
        # 데이터 일관성 검증 함수
        def check_data_consistency(prev_ticker, curr_ticker):
            # 가격 변동 검증
            prev_price = float(prev_ticker['trade_price'])
            curr_price = float(curr_ticker['trade_price'])
            price_change = abs((curr_price - prev_price) / prev_price * 100)
            
            assert price_change <= max_price_change, \
                f"가격 변동이 너무 큽니다: {price_change:.2f}% (최대 허용: {max_price_change}%)"
            
            # 거래량 증가 검증
            assert float(curr_ticker['acc_trade_volume_24h']) >= float(prev_ticker['acc_trade_volume_24h']), \
                "24시간 누적 거래량이 감소했습니다"
            
            # 고가/저가 범위 검증
            assert float(curr_ticker['high_price']) >= float(curr_ticker['trade_price']), \
                "현재가가 고가를 초과했습니다"
            assert float(curr_ticker['low_price']) <= float(curr_ticker['trade_price']), \
                "현재가가 저가 미만입니다"
        
        # 초기 데이터 조회
        initial_ticker = trading_manager.api.get_ticker(market)
        validate_ticker_data(initial_ticker, "초기")
        
        previous_ticker = initial_ticker
        update_times = []
        
        # 여러 번의 업데이트 테스트
        for i in range(update_count):
            update_start = time.time()
            
            # API 호출 간격 준수
            time.sleep(min_update_interval)
            
            try:
                current_ticker = trading_manager.api.get_ticker(market)
                validate_ticker_data(current_ticker, f"{i+1}번째 업데이트")
                
                # 데이터 일관성 검증
                check_data_consistency(previous_ticker, current_ticker)
                
                # 업데이트 시간 기록
                update_times.append(time.time() - update_start)
                
                previous_ticker = current_ticker
                
            except Exception as e:
                pytest.fail(f"{i+1}번째 업데이트 중 오류 발생: {str(e)}")
        
        # 업데이트 간격 검증
        avg_update_time = sum(update_times) / len(update_times)
        assert avg_update_time >= min_update_interval, \
            f"평균 업데이트 간격이 너무 짧습니다: {avg_update_time:.3f}초"
        
        # 메모리 정리
        del initial_ticker
        del previous_ticker
        del current_ticker
        del update_times
        
    except Exception as e:
        pytest.fail(f"실시간 데이터 업데이트 테스트 실패: {str(e)}")
    finally:
        # 추가 메모리 정리
        import gc
        gc.collect()

# 장기 실행 테스트
@pytest.mark.skip(reason="장시간 실행 테스트")
def test_long_running_operation(app):
    """장기 실행 테스트"""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)
    
    while datetime.now() < end_time:
        with patch('streamlit.line_chart'):
            app.run()
        time.sleep(60)  # 1분 대기
        
        # 메모리 사용량 확인
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 500  # 500MB 이하 유지

if __name__ == "__main__":
    pytest.main(["-v"]) 