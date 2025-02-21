import os
import sys
import time
from threading import Thread
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sqlite3
import redis
import requests
from src.api.upbit import UpbitTradingSystem
import jwt

# Fake Redis 구현
class FakeRedis:
    def __init__(self):
        self.data = {}
        
    def get(self, key):
        return self.data.get(key)
        
    def set(self, key, value, ex=None):
        self.data[key] = value
        
    def delete(self, key):
        if key in self.data:
            del self.data[key]
            
    def ping(self):
        return True

@pytest.fixture
def fake_redis():
    return FakeRedis()

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.json.return_value = [{
        'market': 'KRW-BTC',
        'candle_date_time_utc': '2024-01-01T00:00:00',
        'opening_price': 50000000.0,
        'high_price': 51000000.0,
        'low_price': 49000000.0,
        'trade_price': 50500000.0,
        'candle_acc_trade_volume': 1.5,
        'unit': 1
    }]
    mock.status_code = 200
    return mock

@pytest.fixture
def trading_system(fake_redis):
    with patch('redis.Redis', return_value=fake_redis):
        system = UpbitTradingSystem(db_path=':memory:', redis_host='mock-redis')
        return system

@pytest.fixture
def sample_ohlcv():
    """샘플 OHLCV 데이터 생성"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1min')
    data = {
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'timestamp': dates
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_initialization(trading_system):
    """시스템 초기화 테스트"""
    assert trading_system.db_conn is not None
    assert trading_system.redis is not None
    assert trading_system.access_key is not None
    assert trading_system.secret_key is not None

    # Check table creation
    cursor = trading_system.db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
    assert cursor.fetchone() is not None

def test_get_markets(trading_system):
    """마켓 코드 조회 테스트"""
    mock_response = [
        {"market": "KRW-BTC", "korean_name": "비트코인", "english_name": "Bitcoin"},
        {"market": "KRW-ETH", "korean_name": "이더리움", "english_name": "Ethereum"}
    ]
    
    with patch.object(trading_system, '_send_request', return_value=mock_response):
        markets = trading_system.get_markets()
        assert len(markets) == 2
        assert markets[0]['market'] == 'KRW-BTC'
        assert markets[1]['market'] == 'KRW-ETH'

def test_get_minute_candles(trading_system, mock_response):
    """분봉 데이터 조회 테스트"""
    with patch('requests.get', return_value=mock_response):
        candles = trading_system.get_minute_candles('KRW-BTC')
        assert isinstance(candles, list)
        assert len(candles) > 0
        assert 'market' in candles[0]
        assert 'candle_date_time_utc' in candles[0]
        assert 'opening_price' in candles[0]

def test_get_daily_ohlcv(trading_system, mock_response):
    """일봉 데이터 조회 테스트"""
    with patch('requests.get', return_value=mock_response):
        ohlcv = trading_system.get_daily_ohlcv('KRW-BTC')
        assert isinstance(ohlcv, pd.DataFrame)
        assert not ohlcv.empty
        assert all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_get_current_price(trading_system, mock_response):
    """현재가 조회 테스트"""
    mock_response.json.return_value = [{
        'market': 'KRW-BTC',
        'trade_price': 50000000.0,
        'signed_change_rate': 0.01
    }]
    
    with patch('requests.get', return_value=mock_response):
        price = trading_system.get_current_price('KRW-BTC')
        assert isinstance(price, dict)
        assert 'trade_price' in price
        assert 'signed_change_rate' in price

def test_get_orderbook(trading_system, mock_response):
    """호가 정보 조회 테스트"""
    mock_response.json.return_value = [{
        'market': 'KRW-BTC',
        'orderbook_units': [
            {'ask_price': 50100000.0, 'bid_price': 50000000.0, 'ask_size': 0.1, 'bid_size': 0.1}
        ]
    }]
    
    with patch('requests.get', return_value=mock_response):
        orderbook = trading_system.get_orderbook('KRW-BTC')
        assert isinstance(orderbook, dict)
        assert 'orderbook_units' in orderbook
        assert len(orderbook['orderbook_units']) > 0

def test_get_accounts(trading_system, mock_response):
    """계좌 정보 조회 테스트"""
    mock_response.json.return_value = [{
        'currency': 'KRW',
        'balance': '1000000.0',
        'locked': '0.0',
        'avg_buy_price': '0',
        'avg_buy_price_modified': False
    }]
    
    with patch('requests.get', return_value=mock_response):
        accounts = trading_system.get_accounts()
        assert isinstance(accounts, list)
        assert len(accounts) > 0
        assert 'currency' in accounts[0]
        assert 'balance' in accounts[0]

def test_place_order(trading_system, mock_response):
    """주문 테스트"""
    mock_response.json.return_value = {
        'uuid': 'test-uuid',
        'side': 'bid',
        'ord_type': 'limit',
        'price': '50000000.0',
        'state': 'wait',
        'market': 'KRW-BTC'
    }
    
    with patch('requests.post', return_value=mock_response):
        order = trading_system.place_order('KRW-BTC', 'bid', 'limit', price=50000000.0, volume=0.001)
        assert isinstance(order, dict)
        assert 'uuid' in order
        assert 'state' in order

def test_error_handling(trading_system):
    """에러 처리 테스트"""
    with patch('requests.get', side_effect=Exception('API Error')):
        result = trading_system.get_current_price('KRW-BTC')
        assert result is None or isinstance(result, dict)

def test_cache_invalidation(trading_system, fake_redis):
    """캐시 무효화 테스트"""
    cache_key = 'test_key'
    fake_redis.set(cache_key, 'test_value')
    trading_system.invalidate_cache(cache_key)
    assert fake_redis.get(cache_key) is None

def test_rate_limiting(trading_system):
    """요청 속도 제한 테스트"""
    start_time = datetime.now()
    for _ in range(3):
        trading_system._wait_for_rate_limit()
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    assert time_diff >= 0.2  # 최소 200ms 간격

def test_data_consistency(trading_system, sample_ohlcv_data, mock_upbit_response):
    """데이터 일관성 테스트"""
    # API 응답 데이터가 DataFrame으로 올바르게 변환되는지 테스트
    with patch('requests.get', return_value=MagicMock(status_code=200, json=lambda: [mock_upbit_response])):
        ohlcv = trading_system.get_daily_ohlcv('KRW-BTC')
        assert isinstance(ohlcv, pd.DataFrame)
        assert all(col in ohlcv.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # 데이터 타입 검증
        assert ohlcv['open'].dtype == np.float64
        assert ohlcv['high'].dtype == np.float64
        assert ohlcv['low'].dtype == np.float64
        assert ohlcv['close'].dtype == np.float64
        assert ohlcv['volume'].dtype == np.float64
        
        # 데이터 관계 검증
        assert (ohlcv['high'] >= ohlcv['low']).all()
        assert (ohlcv['high'] >= ohlcv['open']).all()
        assert (ohlcv['high'] >= ohlcv['close']).all()
        assert (ohlcv['low'] <= ohlcv['open']).all()
        assert (ohlcv['low'] <= ohlcv['close']).all()
        assert (ohlcv['volume'] >= 0).all()

def test_database_operations(trading_system, sample_ohlcv_data):
    """데이터베이스 작업 테스트"""
    # 테스트 데이터 저장
    market = 'KRW-BTC'
    trading_system.save_market_data(market, sample_ohlcv_data)
    
    # 저장된 데이터 조회
    saved_data = trading_system.get_market_data(market)
    assert isinstance(saved_data, pd.DataFrame)
    assert not saved_data.empty
    assert all(col in saved_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # 데이터 일치 확인
    pd.testing.assert_frame_equal(
        sample_ohlcv_data.sort_index(),
        saved_data.sort_index(),
        check_dtype=True
    )

def test_api_error_recovery(trading_system):
    """API 에러 복구 테스트"""
    error_count = 0
    max_retries = 3
    
    def mock_api_call_with_recovery(*args, **kwargs):
        nonlocal error_count
        if error_count < 2:
            error_count += 1
            raise requests.exceptions.RequestException("API Error")
        return MagicMock(status_code=200, json=lambda: [{'market': 'KRW-BTC', 'trade_price': 50000000.0}])
    
    with patch('requests.get', side_effect=mock_api_call_with_recovery):
        result = trading_system.get_current_price('KRW-BTC', max_retries=max_retries)
        assert result is not None
        assert isinstance(result, dict)
        assert 'trade_price' in result 

def test_performance_large_dataset(trading_system, sample_ohlcv_data):
    """대량 데이터 처리 성능 테스트"""
    # 대량의 테스트 데이터 생성
    large_data = pd.concat([sample_ohlcv_data] * 100, axis=0)
    large_data.index = pd.date_range(start='2024-01-01', periods=len(large_data), freq='1min')
    
    start_time = datetime.now()
    
    # 데이터 저장
    market = 'KRW-BTC'
    trading_system.save_market_data(market, large_data)
    
    # 데이터 조회
    saved_data = trading_system.get_market_data(market)
    
    # 기술적 지표 계산
    indicators = trading_system.calculate_technical_indicators(saved_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 10,000개 데이터 처리에 10초 이내
    assert processing_time < 10
    assert len(saved_data) == len(large_data)
    assert isinstance(indicators, dict)

def test_concurrent_requests(trading_system, mock_upbit_response):
    """동시성 테스트"""
    def make_request():
        with patch('requests.get', return_value=MagicMock(status_code=200, json=lambda: [mock_upbit_response])):
            return trading_system.get_current_price('KRW-BTC')
    
    # 여러 스레드에서 동시에 요청
    threads = []
    results = []
    
    for _ in range(5):
        thread = Thread(target=lambda: results.append(make_request()))
        threads.append(thread)
        thread.start()
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    # 결과 검증
    assert len(results) == 5
    assert all(isinstance(result, dict) for result in results)
    assert all('trade_price' in result for result in results)

def test_network_latency(trading_system):
    """네트워크 지연 시뮬레이션 테스트"""
    def delayed_response(*args, **kwargs):
        time.sleep(1)  # 1초 지연
        return MagicMock(status_code=200, json=lambda: [{'market': 'KRW-BTC', 'trade_price': 50000000.0}])
    
    with patch('requests.get', side_effect=delayed_response):
        start_time = datetime.now()
        result = trading_system.get_current_price('KRW-BTC', timeout=2)
        end_time = datetime.now()
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'trade_price' in result
        assert (end_time - start_time).total_seconds() >= 1

def test_memory_usage(trading_system, sample_ohlcv_data):
    """메모리 사용량 테스트"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 대량의 데이터 생성 및 처리
    large_data = pd.concat([sample_ohlcv_data] * 1000, axis=0)
    large_data.index = pd.date_range(start='2024-01-01', periods=len(large_data), freq='1min')
    
    market = 'KRW-BTC'
    trading_system.save_market_data(market, large_data)
    saved_data = trading_system.get_market_data(market)
    indicators = trading_system.calculate_technical_indicators(saved_data)
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
    
    # 메모리 증가가 1GB 이내여야 함
    assert memory_increase < 1024  # 1GB
    
    # 메모리 정리
    del large_data
    del saved_data
    del indicators 

def test_check_data_consistency(trading_system, sample_ohlcv):
    """데이터 일관성 검사 테스트"""
    # 정상 데이터 테스트
    assert trading_system.check_data_consistency(sample_ohlcv)
    
    # 잘못된 데이터 테스트
    invalid_data = sample_ohlcv.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = invalid_data.loc[invalid_data.index[0], 'low'] - 1
    assert not trading_system.check_data_consistency(invalid_data)
    
    # 음수 값 테스트
    negative_data = sample_ohlcv.copy()
    negative_data.loc[negative_data.index[0], 'volume'] = -1
    assert not trading_system.check_data_consistency(negative_data)

def test_database_operations(trading_system, sample_ohlcv):
    """데이터베이스 작업 테스트"""
    market = 'KRW-BTC'
    interval = 'minute1'
    
    # 데이터 저장 테스트
    assert trading_system.save_to_database(sample_ohlcv, market, interval)
    
    # 데이터 로드 테스트
    loaded_data = trading_system.load_from_database(market, interval)
    assert loaded_data is not None
    assert len(loaded_data) == len(sample_ohlcv)
    
    # 시간 범위 지정 테스트
    start_time = sample_ohlcv.index[0]
    end_time = sample_ohlcv.index[-1]
    ranged_data = trading_system.load_from_database(market, interval, start_time, end_time)
    assert ranged_data is not None
    assert len(ranged_data) == len(sample_ohlcv)

def test_handle_api_error(trading_system):
    """API 에러 처리 테스트"""
    # 일반적인 API 에러 테스트
    test_error = requests.exceptions.RequestException("Test error")
    result = trading_system.handle_api_error(test_error, retry_count=2, retry_delay=0.1)
    assert result is None
    
    # JWT 에러 테스트
    jwt_error = jwt.InvalidTokenError()
    result = trading_system.handle_api_error(jwt_error)
    assert result is None
    
    # 재시도 성공 시나리오 테스트
    with patch.object(trading_system, '_send_request', side_effect=[
        requests.exceptions.RequestException("Temporary error"),
        {'success': True}
    ]):
        result = trading_system.handle_api_error(
            requests.exceptions.RequestException("Initial error"),
            retry_count=2,
            retry_delay=0.1
        )
        assert result == {'success': True} 