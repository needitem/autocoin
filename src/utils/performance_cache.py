"""
고성능 캐싱 시스템
"""

import functools
import hashlib
import json
import time
from typing import Any, Dict, Optional, Callable
import pickle
import threading
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceCache:
    """고성능 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """캐시 키 생성"""
        try:
            # 함수명 + 인수를 해시로 변환
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': sorted(kwargs.items())
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception:
            # 직렬화 실패시 시간 기반 키 사용
            return f"{func_name}_{int(time.time() * 1000)}"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # TTL 확인
                if time.time() - timestamp < self.ttl:
                    self.access_times[key] = time.time()
                    return value
                else:
                    # 만료된 항목 제거
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        with self.lock:
            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, time.time())
            self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """LRU 방식으로 오래된 항목 제거"""
        if not self.access_times:
            return
        
        # 가장 오래된 항목 찾기
        oldest_key = min(self.access_times, key=self.access_times.get)
        
        # 제거
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """캐시 전체 삭제"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict:
        """캐시 통계"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'usage': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }

# 전역 캐시 인스턴스 (캐시 크기와 TTL 증가)
_global_cache = PerformanceCache(max_size=1000, ttl=300)  # 5분 TTL, 1000개 항목

def cached(ttl: int = 180, cache_instance: Optional[PerformanceCache] = None):
    """캐싱 데코레이터"""
    cache = cache_instance or _global_cache
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # 캐시에서 확인
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # 캐시 미스 - 함수 실행
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 결과 캐시
            cache.set(cache_key, result)
            
            logger.debug(f"Cache miss for {func.__name__} (executed in {execution_time:.3f}s)")
            return result
        
        return wrapper
    return decorator

def cache_key_for_market_data(market: str, timeframe: str, count: int) -> str:
    """시장 데이터용 캐시 키 생성"""
    # 시간 기반 키 (1분 단위로 캐시)
    time_bucket = int(time.time() / 60)
    return f"market_data_{market}_{timeframe}_{count}_{time_bucket}"

def get_cached_market_data(market: str, timeframe: str, count: int) -> Optional[Any]:
    """시장 데이터 캐시 조회"""
    cache_key = cache_key_for_market_data(market, timeframe, count)
    return _global_cache.get(cache_key)

def set_cached_market_data(market: str, timeframe: str, count: int, data: Any) -> None:
    """시장 데이터 캐시 저장"""
    cache_key = cache_key_for_market_data(market, timeframe, count)
    _global_cache.set(cache_key, data)

def clear_cache():
    """전체 캐시 삭제"""
    _global_cache.clear()

def get_cache_stats() -> Dict:
    """캐시 통계 조회"""
    return _global_cache.stats()