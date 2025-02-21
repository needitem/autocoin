"""
데이터베이스 관리 모듈 테스트
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.db.database import DatabaseManager

@pytest.fixture
def db_manager():
    """테스트용 DatabaseManager 인스턴스"""
    # 테스트용 임시 데이터베이스 파일 생성
    test_db_path = 'test_market_data.db'
    manager = DatabaseManager(db_path=test_db_path, verbose=True)
    yield manager

    # 테스트 후 정리
    manager.close()  # 연결 종료
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except PermissionError:
        # Windows에서 파일이 사용 중인 경우 대기 후 재시도
        import time
        time.sleep(1)
        try:
            os.remove(test_db_path)
        except:
            pass

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
    # 테스트 데이터 생성
    dates = pd.date_range(start=datetime.now(), periods=200, freq='1min')
    data = pd.DataFrame({
        'open': [50000000 + i * 100 for i in range(200)],
        'high': [51000000 + i * 100 for i in range(200)],
        'low': [49000000 + i * 100 for i in range(200)],
        'close': [50500000 + i * 100 for i in range(200)],
        'volume': [i * 0.01 for i in range(200)]
    }, index=dates)
    return data

def test_database_initialization(db_manager):
    """데이터베이스 초기화 테스트"""
    assert db_manager.db_conn is not None
    assert db_manager.redis is not None

def test_data_consistency_check(db_manager, sample_data):
    """데이터 일관성 검사 테스트"""
    # 정상 데이터
    assert db_manager.check_data_consistency(sample_data)

    # 필수 컬럼 누락
    invalid_data = sample_data.drop(columns=['open'])
    assert not db_manager.check_data_consistency(invalid_data)

    # 음수 값 포함
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[0], 'volume'] = -1
    assert not db_manager.check_data_consistency(invalid_data)

def test_save_and_load_market_data(db_manager, sample_data):
    """시장 데이터 저장 및 로드 테스트"""
    market = "KRW-BTC"

    # 데이터 저장
    assert db_manager.save_market_data(market, sample_data)

    # 데이터 로드
    loaded_data = db_manager.load_market_data(market)
    assert loaded_data is not None
    assert len(loaded_data) == len(sample_data)
    assert all(col in loaded_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_save_and_load_with_interval(db_manager, sample_data):
    """시간 간격을 지정한 데이터 저장 및 로드 테스트"""
    market = "KRW-BTC:minute1"

    # 데이터 저장
    assert db_manager.save_market_data(market, sample_data)

    # 전체 데이터 로드
    loaded_data = db_manager.load_market_data(market)
    assert loaded_data is not None
    assert len(loaded_data) == len(sample_data)

    # 특정 기간 데이터 로드
    start_time = sample_data.index[0]
    end_time = sample_data.index[-1]
    interval_data = db_manager.load_market_data(market, start_time, end_time)
    assert interval_data is not None
    assert len(interval_data) == len(sample_data)

def test_cache_invalidation(db_manager, sample_data):
    """캐시 무효화 테스트"""
    market = "KRW-BTC:minute1"

    # 데이터 저장
    assert db_manager.save_market_data(market, sample_data)

    # 캐시 무효화
    assert db_manager.invalidate_cache(market)

    # 데이터가 없어야 함
    assert db_manager.load_market_data(market) is None

def test_error_handling(db_manager):
    """에러 처리 테스트"""
    # 잘못된 데이터 저장 시도
    invalid_data = pd.DataFrame()
    assert not db_manager.save_market_data("INVALID-MARKET", invalid_data)

    # 존재하지 않는 데이터 로드 시도
    assert db_manager.load_market_data("INVALID-MARKET") is None

    # 잘못된 데이터 일관성 검사
    assert not db_manager.check_data_consistency(invalid_data)

def test_concurrent_access(db_manager, sample_data):
    """동시 접근 테스트"""
    import threading

    def save_data(market_suffix):
        market = f"KRW-BTC-{market_suffix}:minute1"
        assert db_manager.save_market_data(market, sample_data)

    # 여러 스레드에서 동시에 데이터 저장
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_data, args=(i,))
        threads.append(thread)
        thread.start()

    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()

def test_large_dataset(db_manager, sample_data):
    """대용량 데이터 처리 테스트"""
    market = "KRW-BTC:minute1"

    # 대용량 데이터 생성
    large_data = pd.concat([sample_data] * 5)
    large_data.index = pd.date_range(start=datetime.now(), periods=len(large_data), freq='1min')

    # 데이터 저장
    assert db_manager.save_market_data(market, large_data)

    # 데이터 로드
    loaded_data = db_manager.load_market_data(market)
    assert loaded_data is not None
    assert len(loaded_data) == len(large_data)

if __name__ == "__main__":
    pytest.main(["-v"]) 