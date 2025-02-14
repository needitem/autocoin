import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from functools import wraps
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, cache_dir: str = "cache"):
        """캐시 초기화"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_config = self._load_cache_config()
        
    def _load_cache_config(self) -> Dict:
        """캐시 설정 로드"""
        config_path = self.cache_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "max_age": {
                "market_data": 300,  # 5분
                "news": 1800,        # 30분
                "analysis": 3600     # 1시간
            },
            "max_size": {
                "memory": 1000,      # 메모리 캐시 최대 항목 수
                "disk": 10000        # 디스크 캐시 최대 항목 수
            }
        }
        
    def _get_cache_key(self, prefix: str, key: str) -> str:
        """캐시 키 생성"""
        return f"{prefix}_{key}"
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """캐시 파일 경로 생성"""
        return self.cache_dir / f"{cache_key}.json"
        
    def get(self, prefix: str, key: str) -> Optional[Any]:
        """캐시된 데이터 조회"""
        cache_key = self._get_cache_key(prefix, key)
        
        # 메모리 캐시 확인
        if cache_key in self.memory_cache:
            data = self.memory_cache[cache_key]
            if datetime.now().timestamp() - data['timestamp'] < self.cache_config['max_age'][prefix]:
                return data['value']
            else:
                del self.memory_cache[cache_key]
        
        # 디스크 캐시 확인
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                if datetime.now().timestamp() - data['timestamp'] < self.cache_config['max_age'][prefix]:
                    # 자주 사용되는 데이터는 메모리 캐시에 추가
                    self.memory_cache[cache_key] = data
                    return data['value']
                else:
                    cache_path.unlink()  # 만료된 캐시 삭제
            except Exception as e:
                logger.error(f"캐시 읽기 실패: {e}")
                
        return None
        
    def set(self, prefix: str, key: str, value: Any):
        """데이터 캐시 저장"""
        cache_key = self._get_cache_key(prefix, key)
        data = {
            'timestamp': datetime.now().timestamp(),
            'value': value
        }
        
        # 메모리 캐시 저장
        if len(self.memory_cache) >= self.cache_config['max_size']['memory']:
            # LRU 방식으로 오래된 항목 제거
            oldest_key = min(self.memory_cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.memory_cache[oldest_key]
        self.memory_cache[cache_key] = data
        
        # 디스크 캐시 저장
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            
    def clear(self, prefix: Optional[str] = None):
        """캐시 삭제"""
        if prefix:
            # 특정 prefix의 캐시만 삭제
            for key in list(self.memory_cache.keys()):
                if key.startswith(f"{prefix}_"):
                    del self.memory_cache[key]
            
            for path in self.cache_dir.glob(f"{prefix}_*.json"):
                path.unlink()
        else:
            # 전체 캐시 삭제
            self.memory_cache.clear()
            for path in self.cache_dir.glob("*.json"):
                path.unlink()

class RateLimiter:
    def __init__(self, calls: int, period: int):
        """
        레이트 리미터 초기화
        calls: 허용된 호출 횟수
        period: 기간 (초)
        """
        self.calls = calls
        self.period = period
        self.timestamps = []
        
    async def acquire(self):
        """요청 허용 여부 확인"""
        now = time.time()
        
        # 만료된 타임스탬프 제거
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
        
        if len(self.timestamps) >= self.calls:
            # 다음 요청 가능 시간까지 대기
            sleep_time = self.timestamps[0] + self.period - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
                
        self.timestamps.append(now)
        return True

class APIClient:
    def __init__(self, base_url: str, rate_limit_calls: int = 30, rate_limit_period: int = 60):
        """API 클라이언트 초기화"""
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)
        self.session = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
            
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET 요청 수행"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API 요청 실패: {e}")
            raise

def async_timed():
    """비동기 함수 실행 시간 측정 데코레이터"""
    def wrapper(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end = time.time()
                total = end - start
                logger.info(f"{func.__name__} 실행 시간: {total:.2f}초")
        return wrapped
    return wrapper

class PerformanceMonitor:
    def __init__(self):
        """성능 모니터 초기화"""
        self.metrics = {
            'api_calls': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'execution_times': {},
            'memory_usage': [],
            'error_counts': {}
        }
        
    def record_api_call(self, endpoint: str, success: bool, response_time: float):
        """API 호출 기록"""
        if endpoint not in self.metrics['api_calls']:
            self.metrics['api_calls'][endpoint] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_time': 0,
                'average_time': 0
            }
            
        metrics = self.metrics['api_calls'][endpoint]
        metrics['total_calls'] += 1
        if success:
            metrics['successful_calls'] += 1
        else:
            metrics['failed_calls'] += 1
        metrics['total_time'] += response_time
        metrics['average_time'] = metrics['total_time'] / metrics['total_calls']
        
    def record_cache_access(self, hit: bool):
        """캐시 접근 기록"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
            
    def record_execution_time(self, function_name: str, execution_time: float):
        """함수 실행 시간 기록"""
        if function_name not in self.metrics['execution_times']:
            self.metrics['execution_times'][function_name] = {
                'total_time': 0,
                'calls': 0,
                'average_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
            
        metrics = self.metrics['execution_times'][function_name]
        metrics['total_time'] += execution_time
        metrics['calls'] += 1
        metrics['average_time'] = metrics['total_time'] / metrics['calls']
        metrics['min_time'] = min(metrics['min_time'], execution_time)
        metrics['max_time'] = max(metrics['max_time'], execution_time)
        
    def record_error(self, error_type: str):
        """에러 발생 기록"""
        if error_type not in self.metrics['error_counts']:
            self.metrics['error_counts'][error_type] = 0
        self.metrics['error_counts'][error_type] += 1
        
    def get_metrics(self) -> Dict:
        """성능 지표 조회"""
        return {
            'api_performance': {
                endpoint: {
                    'success_rate': metrics['successful_calls'] / metrics['total_calls'] * 100,
                    'average_response_time': metrics['average_time'],
                    'total_calls': metrics['total_calls']
                }
                for endpoint, metrics in self.metrics['api_calls'].items()
            },
            'cache_performance': {
                'hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100
                if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0 else 0
            },
            'function_performance': {
                name: {
                    'average_time': metrics['average_time'],
                    'total_calls': metrics['calls'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time']
                }
                for name, metrics in self.metrics['execution_times'].items()
            },
            'error_statistics': self.metrics['error_counts']
        }
        
    def print_metrics(self):
        """성능 지표 출력"""
        metrics = self.get_metrics()
        
        print("\n=== 성능 모니터링 리포트 ===")
        print(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n[API 성능]")
        for endpoint, data in metrics['api_performance'].items():
            print(f"\n엔드포인트: {endpoint}")
            print(f"성공률: {data['success_rate']:.1f}%")
            print(f"평균 응답 시간: {data['average_response_time']:.3f}초")
            print(f"총 호출 수: {data['total_calls']:,}회")
            
        print(f"\n[캐시 성능]")
        print(f"캐시 히트율: {metrics['cache_performance']['hit_rate']:.1f}%")
        
        print("\n[함수 성능]")
        for name, data in metrics['function_performance'].items():
            print(f"\n함수: {name}")
            print(f"평균 실행 시간: {data['average_time']:.3f}초")
            print(f"최소 실행 시간: {data['min_time']:.3f}초")
            print(f"최대 실행 시간: {data['max_time']:.3f}초")
            print(f"총 호출 수: {data['total_calls']:,}회")
            
        if metrics['error_statistics']:
            print("\n[에러 통계]")
            for error_type, count in metrics['error_statistics'].items():
                print(f"{error_type}: {count:,}회")

# 전역 인스턴스
cache = Cache()
monitor = PerformanceMonitor()

def cached(prefix: str, key_func=None):
    """캐시 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            # 캐시 키 생성
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{args}_{kwargs}"
                
            # 캐시 확인
            cached_data = cache.get(prefix, cache_key)
            if cached_data is not None:
                monitor.record_cache_access(hit=True)
                return cached_data
                
            monitor.record_cache_access(hit=False)
            
            # 함수 실행 및 결과 캐시
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            monitor.record_execution_time(func.__name__, execution_time)
            cache.set(prefix, cache_key, result)
            
            return result
        return wrapped
    return decorator

def timed(func):
    """실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            monitor.record_execution_time(func.__name__, execution_time)
            return result
        except Exception as e:
            monitor.record_error(type(e).__name__)
            raise
    return wrapped 