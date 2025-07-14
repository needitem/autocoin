"""
차트 분석 성능 최적화 모듈
"""

import time
import functools
from typing import Any, Dict, Optional, Callable
import hashlib
import json
import threading
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AnalysisCache:
    """분석 결과 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        
        # 정리 스레드 시작
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """캐시 키 생성"""
        try:
            # 함수명과 인자들을 기반으로 해시 생성
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"캐시 키 생성 오류: {str(e)}")
            return f"{func_name}_{int(time.time())}"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # TTL 확인
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    entry['last_access'] = time.time()
                    return entry['data']
                else:
                    # 만료된 항목 제거
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        with self.lock:
            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = {
                'data': value,
                'timestamp': time.time(),
                'last_access': time.time()
            }
    
    def _evict_lru(self) -> None:
        """LRU 방식으로 항목 제거"""
        if not self.cache:
            return
        
        # 가장 오래된 접근 시간을 가진 항목 찾기
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k]['last_access'])
        del self.cache[oldest_key]
    
    def _cleanup_expired(self) -> None:
        """만료된 항목 정리 (백그라운드)"""
        while True:
            try:
                time.sleep(60)  # 1분마다 정리
                current_time = time.time()
                
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if current_time - entry['timestamp'] > self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                
                if expired_keys:
                    logger.debug(f"만료된 캐시 항목 {len(expired_keys)}개 정리")
                        
            except Exception as e:
                logger.error(f"캐시 정리 오류: {str(e)}")
    
    def clear(self) -> None:
        """캐시 전체 비우기"""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict:
        """캐시 통계"""
        with self.lock:
            current_time = time.time()
            active_items = sum(
                1 for entry in self.cache.values()
                if current_time - entry['timestamp'] < self.ttl_seconds
            )
            
            return {
                'total_items': len(self.cache),
                'active_items': active_items,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }

# 전역 캐시 인스턴스
analysis_cache = AnalysisCache(max_size=500, ttl_seconds=180)  # 3분 TTL

def cached_analysis(ttl_seconds: Optional[int] = None):
    """분석 결과 캐싱 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = analysis_cache._generate_key(func.__name__, *args, **kwargs)
            
            # 캐시에서 확인
            cached_result = analysis_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"캐시 히트: {func.__name__}")
                return cached_result
            
            # 캐시 미스 - 실제 함수 실행
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 결과 캐싱
            analysis_cache.set(cache_key, result)
            
            logger.debug(f"함수 실행: {func.__name__} ({execution_time:.3f}초)")
            return result
            
        return wrapper
    return decorator

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record_execution(self, func_name: str, execution_time: float, 
                        cache_hit: bool = False) -> None:
        """실행 메트릭 기록"""
        with self.lock:
            if func_name not in self.metrics:
                self.metrics[func_name] = {
                    'total_calls': 0,
                    'total_time': 0.0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'avg_time': 0.0,
                    'max_time': 0.0,
                    'min_time': float('inf')
                }
            
            metric = self.metrics[func_name]
            metric['total_calls'] += 1
            
            if cache_hit:
                metric['cache_hits'] += 1
            else:
                metric['cache_misses'] += 1
                metric['total_time'] += execution_time
                metric['max_time'] = max(metric['max_time'], execution_time)
                metric['min_time'] = min(metric['min_time'], execution_time)
                
                # 평균 시간 업데이트 (캐시 미스만 고려)
                if metric['cache_misses'] > 0:
                    metric['avg_time'] = metric['total_time'] / metric['cache_misses']
    
    def get_stats(self) -> Dict:
        """성능 통계 반환"""
        with self.lock:
            stats = {}
            for func_name, metric in self.metrics.items():
                cache_hit_rate = (metric['cache_hits'] / metric['total_calls'] * 100 
                                if metric['total_calls'] > 0 else 0)
                
                stats[func_name] = {
                    'total_calls': metric['total_calls'],
                    'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                    'avg_time': f"{metric['avg_time']:.3f}s",
                    'max_time': f"{metric['max_time']:.3f}s",
                    'min_time': f"{metric['min_time']:.3f}s" if metric['min_time'] != float('inf') else "0.000s"
                }
            
            return stats
    
    def reset(self) -> None:
        """통계 초기화"""
        with self.lock:
            self.metrics.clear()

# 전역 성능 모니터
performance_monitor = PerformanceMonitor()

def monitored_execution(func: Callable) -> Callable:
    """성능 모니터링 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = analysis_cache._generate_key(func.__name__, *args, **kwargs)
        
        # 캐시 확인
        cached_result = analysis_cache.get(cache_key)
        if cached_result is not None:
            performance_monitor.record_execution(func.__name__, 0.0, cache_hit=True)
            return cached_result
        
        # 실제 실행
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # 성능 기록
        performance_monitor.record_execution(func.__name__, execution_time, cache_hit=False)
        
        # 캐싱
        analysis_cache.set(cache_key, result)
        
        return result
    
    return wrapper

class DataFrameOptimizer:
    """DataFrame 최적화 유틸리티"""
    
    @staticmethod
    def optimize_memory(df):
        """DataFrame 메모리 최적화"""
        try:
            # 숫자형 컬럼 최적화
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = df[col].astype('int32')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
            
            return df
        except Exception as e:
            logger.error(f"DataFrame 메모리 최적화 오류: {str(e)}")
            return df
    
    @staticmethod
    def batch_process(data, func, batch_size: int = 100):
        """배치 처리를 통한 대용량 데이터 최적화"""
        try:
            if len(data) <= batch_size:
                return func(data)
            
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_result = func(batch)
                results.append(batch_result)
            
            # 결과 병합 (타입에 따라)
            if hasattr(results[0], 'concat'):  # pandas DataFrame/Series
                import pandas as pd
                return pd.concat(results, ignore_index=True)
            elif isinstance(results[0], list):
                return [item for sublist in results for item in sublist]
            else:
                return results
                
        except Exception as e:
            logger.error(f"배치 처리 오류: {str(e)}")
            return func(data)

class AsyncAnalysisPool:
    """비동기 분석 풀"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = None
    
    def __enter__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit_analysis(self, func, *args, **kwargs):
        """분석 작업 제출"""
        if self.executor:
            return self.executor.submit(func, *args, **kwargs)
        else:
            # 동기 실행
            return func(*args, **kwargs)

def optimize_chart_analysis():
    """차트 분석 최적화 설정"""
    
    # 캐시 설정 최적화
    analysis_cache.max_size = 1000
    analysis_cache.ttl_seconds = 300  # 5분
    
    # 로깅 레벨 조정 (운영 환경에서는 WARNING 이상만)
    import os
    if os.getenv('ENVIRONMENT') == 'production':
        logging.getLogger('src.core').setLevel(logging.WARNING)
    
    logger.info("차트 분석 성능 최적화 완료")

def get_performance_report() -> Dict:
    """종합 성능 리포트"""
    return {
        'cache_stats': analysis_cache.stats(),
        'performance_stats': performance_monitor.get_stats(),
        'timestamp': datetime.now().isoformat()
    }

# 자동 최적화 실행
optimize_chart_analysis()