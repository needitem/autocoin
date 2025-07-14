"""
비동기 처리 최적화 헬퍼
"""

import asyncio
import concurrent.futures
import time
from typing import List, Callable, Any, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class AsyncBatchProcessor:
    """배치 비동기 처리"""
    
    def __init__(self, max_workers: int = 5, timeout: float = 30.0):
        self.max_workers = max_workers
        self.timeout = timeout
    
    async def process_batch(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """여러 작업을 비동기로 병렬 처리"""
        if not tasks:
            return []
        
        try:
            # 세마포어로 동시 실행 수 제한
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def limited_task(task):
                async with semaphore:
                    return await self._run_task_safely(task, *args, **kwargs)
            
            # 모든 작업을 병렬로 실행
            results = await asyncio.gather(
                *[limited_task(task) for task in tasks],
                return_exceptions=True
            )
            
            # 예외 결과 필터링
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"태스크 {i} 실행 실패: {str(result)}")
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"배치 처리 오류: {str(e)}")
            return []
    
    async def _run_task_safely(self, task: Callable, *args, **kwargs) -> Any:
        """안전한 태스크 실행"""
        try:
            if asyncio.iscoroutinefunction(task):
                return await asyncio.wait_for(task(*args, **kwargs), timeout=self.timeout)
            else:
                # 동기 함수를 비동기로 실행
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, task, *args, **kwargs)
        except asyncio.TimeoutError:
            logger.warning(f"태스크 타임아웃: {task.__name__}")
            return None
        except Exception as e:
            logger.error(f"태스크 실행 오류: {task.__name__} - {str(e)}")
            return None

def parallel_execute(functions: List[Callable], max_workers: int = 3, timeout: float = 10.0) -> List[Any]:
    """동기 함수들을 병렬로 실행"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 함수를 제출
        future_to_func = {
            executor.submit(func): func for func in functions
        }
        
        try:
            # 완료된 것부터 결과 수집
            for future in concurrent.futures.as_completed(future_to_func, timeout=timeout):
                func = future_to_func[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"함수 {func.__name__} 완료")
                except Exception as e:
                    logger.warning(f"함수 {func.__name__} 실행 실패: {str(e)}")
                    results.append(None)
        
        except concurrent.futures.TimeoutError:
            logger.warning(f"일부 함수가 {timeout}초 내에 완료되지 않음")
    
    return results

def async_cache_warmer(cache_functions: List[Callable], interval: int = 300):
    """캐시 워밍 스케줄러"""
    def run_warmer():
        while True:
            try:
                logger.info("캐시 워밍 시작...")
                start_time = time.time()
                
                # 병렬로 캐시 함수들 실행
                results = parallel_execute(cache_functions, max_workers=3, timeout=60)
                
                execution_time = time.time() - start_time
                success_count = sum(1 for r in results if r is not None)
                
                logger.info(f"캐시 워밍 완료: {success_count}/{len(cache_functions)} 성공 ({execution_time:.2f}초)")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"캐시 워밍 오류: {str(e)}")
                time.sleep(60)  # 오류시 1분 대기
    
    # 백그라운드 스레드에서 실행
    import threading
    thread = threading.Thread(target=run_warmer, daemon=True)
    thread.start()
    return thread

def rate_limited(calls_per_second: float = 1.0):
    """함수 호출 속도 제한 데코레이터"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def timeout_wrapper(timeout_seconds: float = 30.0):
    """함수 실행 타임아웃 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"함수 {func.__name__} 타임아웃 ({timeout_seconds}초)")
                    return None
        return wrapper
    return decorator

class BackgroundTaskManager:
    """백그라운드 태스크 관리자"""
    
    def __init__(self):
        self.tasks = []
        self.running = False
    
    def add_task(self, func: Callable, interval: int, *args, **kwargs):
        """주기적 실행 태스크 추가"""
        self.tasks.append({
            'func': func,
            'interval': interval,
            'args': args,
            'kwargs': kwargs,
            'last_run': 0
        })
    
    def start(self):
        """백그라운드 태스크 시작"""
        if self.running:
            return
        
        self.running = True
        
        def run_tasks():
            while self.running:
                current_time = time.time()
                
                for task in self.tasks:
                    if current_time - task['last_run'] >= task['interval']:
                        try:
                            logger.debug(f"백그라운드 태스크 실행: {task['func'].__name__}")
                            task['func'](*task['args'], **task['kwargs'])
                            task['last_run'] = current_time
                        except Exception as e:
                            logger.error(f"백그라운드 태스크 오류: {task['func'].__name__} - {str(e)}")
                
                time.sleep(1)  # 1초마다 확인
        
        import threading
        self.thread = threading.Thread(target=run_tasks, daemon=True)
        self.thread.start()
        logger.info("백그라운드 태스크 매니저 시작")
    
    def stop(self):
        """백그라운드 태스크 중지"""
        self.running = False
        logger.info("백그라운드 태스크 매니저 중지")

# 전역 인스턴스
batch_processor = AsyncBatchProcessor(max_workers=5)
task_manager = BackgroundTaskManager()

def fast_parallel_map(func: Callable, items: List[Any], max_workers: int = 5) -> List[Any]:
    """리스트 아이템들에 함수를 병렬 적용"""
    if not items:
        return []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results = []
        
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"병렬 처리 중 오류: {str(e)}")
                results.append(None)
        
        return results