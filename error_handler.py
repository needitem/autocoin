import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import json
from pathlib import Path
import os
import sys
from performance import monitor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    def __init__(self, error_log_dir: str = "error_logs"):
        """에러 핸들러 초기화"""
        self.error_log_dir = Path(error_log_dir)
        self.error_log_dir.mkdir(exist_ok=True)
        self.error_counts = {}
        self.error_thresholds = {
            'API_ERROR': 5,      # API 오류 임계값
            'NETWORK_ERROR': 3,  # 네트워크 오류 임계값
            'DATA_ERROR': 10,    # 데이터 오류 임계값
            'SYSTEM_ERROR': 1    # 시스템 오류 임계값
        }
        self.alert_callbacks = []
        
    def handle_error(self, error: Exception, context: Dict = None):
        """에러 처리"""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 에러 카운트 증가
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # 에러 로그 저장
        self._save_error_log(error_type, error_message, stack_trace, context)
        
        # 성능 모니터에 에러 기록
        monitor.record_error(error_type)
        
        # 임계값 체크 및 알림
        self._check_thresholds(error_type)
        
        # 로깅
        logger.error(f"Error: {error_type} - {error_message}")
        logger.debug(f"Stack trace: {stack_trace}")
        if context:
            logger.debug(f"Context: {context}")
            
    def _save_error_log(self, error_type: str, error_message: str, 
                       stack_trace: str, context: Optional[Dict] = None):
        """에러 로그 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.error_log_dir / f"error_{timestamp}_{error_type}.json"
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': error_message,
                'stack_trace': stack_trace,
                'context': context or {},
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'cwd': os.getcwd()
                }
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"에러 로그 저장 실패: {e}")
            
    def _check_thresholds(self, error_type: str):
        """에러 임계값 체크"""
        if error_type in self.error_thresholds:
            threshold = self.error_thresholds[error_type]
            if self.error_counts[error_type] >= threshold:
                self._send_alerts(error_type, self.error_counts[error_type])
                
    def _send_alerts(self, error_type: str, count: int):
        """알림 발송"""
        alert_message = f"경고: {error_type} 에러가 임계값을 초과했습니다. (발생 횟수: {count})"
        logger.warning(alert_message)
        
        for callback in self.alert_callbacks:
            try:
                callback(error_type, count, alert_message)
            except Exception as e:
                logger.error(f"알림 발송 실패: {e}")
                
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
        
    def get_error_summary(self) -> Dict:
        """에러 발생 요약"""
        return {
            'error_counts': self.error_counts,
            'thresholds': self.error_thresholds
        }
        
    def reset_error_counts(self):
        """에러 카운트 초기화"""
        self.error_counts.clear()
        
    def set_threshold(self, error_type: str, threshold: int):
        """에러 임계값 설정"""
        self.error_thresholds[error_type] = threshold

class CustomError(Exception):
    """사용자 정의 에러"""
    def __init__(self, message: str, error_type: str = None, context: Dict = None):
        super().__init__(message)
        self.error_type = error_type or type(self).__name__
        self.context = context or {}

class APIError(CustomError):
    """API 관련 에러"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, "API_ERROR", context)

class NetworkError(CustomError):
    """네트워크 관련 에러"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, "NETWORK_ERROR", context)

class DataError(CustomError):
    """데이터 관련 에러"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, "DATA_ERROR", context)

class SystemError(CustomError):
    """시스템 관련 에러"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, "SYSTEM_ERROR", context)

# 전역 에러 핸들러 인스턴스
error_handler = ErrorHandler()

def handle_errors(func):
    """에러 처리 데코레이터"""
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            error_handler.handle_error(e, context)
            raise
    return wrapped

def async_handle_errors(func):
    """비동기 에러 처리 데코레이터"""
    @wraps(func)
    async def wrapped(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            error_handler.handle_error(e, context)
            raise
    return wrapped

class ErrorMonitor:
    def __init__(self):
        """에러 모니터 초기화"""
        self.error_patterns = {}
        self.error_trends = {}
        
    def analyze_errors(self, error_logs_dir: str) -> Dict:
        """에러 로그 분석"""
        try:
            error_dir = Path(error_logs_dir)
            if not error_dir.exists():
                return {}
                
            error_files = list(error_dir.glob("error_*.json"))
            error_data = []
            
            for file in error_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        error_data.append(json.load(f))
                except Exception as e:
                    logger.error(f"에러 로그 파일 읽기 실패 ({file}): {e}")
                    
            return self._analyze_error_data(error_data)
            
        except Exception as e:
            logger.error(f"에러 분석 실패: {e}")
            return {}
            
    def _analyze_error_data(self, error_data: List[Dict]) -> Dict:
        """에러 데이터 분석"""
        analysis = {
            'error_types': {},       # 에러 유형별 통계
            'time_distribution': {}, # 시간대별 분포
            'common_patterns': [],   # 공통 패턴
            'critical_errors': [],   # 중요 에러
            'recommendations': []    # 개선 제안
        }
        
        # 에러 유형별 통계
        for error in error_data:
            error_type = error['type']
            if error_type not in analysis['error_types']:
                analysis['error_types'][error_type] = {
                    'count': 0,
                    'messages': set(),
                    'contexts': []
                }
            
            analysis['error_types'][error_type]['count'] += 1
            analysis['error_types'][error_type]['messages'].add(error['message'])
            if error.get('context'):
                analysis['error_types'][error_type]['contexts'].append(error['context'])
                
        # 시간대별 분포 분석
        for error in error_data:
            timestamp = datetime.fromisoformat(error['timestamp'])
            hour = timestamp.hour
            if hour not in analysis['time_distribution']:
                analysis['time_distribution'][hour] = 0
            analysis['time_distribution'][hour] += 1
            
        # 공통 패턴 분석
        for error_type, data in analysis['error_types'].items():
            if data['count'] >= 3:  # 3회 이상 발생한 에러
                common_contexts = self._find_common_contexts(data['contexts'])
                if common_contexts:
                    analysis['common_patterns'].append({
                        'error_type': error_type,
                        'count': data['count'],
                        'common_contexts': common_contexts
                    })
                    
        # 중요 에러 식별
        for error_type, data in analysis['error_types'].items():
            if (error_type in ['SYSTEM_ERROR', 'API_ERROR'] or 
                data['count'] >= 5):
                analysis['critical_errors'].append({
                    'error_type': error_type,
                    'count': data['count'],
                    'messages': list(data['messages'])
                })
                
        # 개선 제안 생성
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
        
    def _find_common_contexts(self, contexts: List[Dict]) -> Dict:
        """공통 컨텍스트 찾기"""
        if not contexts:
            return {}
            
        common = {}
        first = contexts[0]
        
        for key in first:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if len(values) == len(contexts):  # 모든 컨텍스트에 존재
                if all(v == values[0] for v in values):  # 모든 값이 동일
                    common[key] = values[0]
                    
        return common
        
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """개선 제안 생성"""
        recommendations = []
        
        # 에러 빈도 기반 제안
        for error_type, data in analysis['error_types'].items():
            if data['count'] >= 10:
                recommendations.append(
                    f"{error_type} 에러가 자주 발생합니다. "
                    f"관련 시스템 점검이 필요합니다."
                )
                
        # 시간대별 분포 기반 제안
        peak_hour = max(analysis['time_distribution'].items(), 
                       key=lambda x: x[1])[0]
        if analysis['time_distribution'][peak_hour] > 5:
            recommendations.append(
                f"{peak_hour}시에 에러가 집중됩니다. "
                f"해당 시간대 시스템 부하 확인이 필요합니다."
            )
            
        # 공통 패턴 기반 제안
        for pattern in analysis['common_patterns']:
            recommendations.append(
                f"{pattern['error_type']} 에러의 공통 패턴이 발견되었습니다. "
                f"관련 컨텍스트를 확인해주세요: {pattern['common_contexts']}"
            )
            
        return recommendations
        
    def print_analysis(self, analysis: Dict):
        """분석 결과 출력"""
        print("\n=== 에러 분석 리포트 ===")
        print(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n[에러 유형별 통계]")
        for error_type, data in analysis['error_types'].items():
            print(f"\n{error_type}:")
            print(f"발생 횟수: {data['count']}회")
            print("에러 메시지:")
            for msg in data['messages']:
                print(f"- {msg}")
                
        print("\n[시간대별 분포]")
        for hour in sorted(analysis['time_distribution'].keys()):
            count = analysis['time_distribution'][hour]
            print(f"{hour:02d}시: {'#' * count} ({count}회)")
            
        if analysis['common_patterns']:
            print("\n[공통 패턴]")
            for pattern in analysis['common_patterns']:
                print(f"\n{pattern['error_type']} (발생: {pattern['count']}회)")
                print(f"공통 컨텍스트: {pattern['common_contexts']}")
                
        if analysis['critical_errors']:
            print("\n[중요 에러]")
            for error in analysis['critical_errors']:
                print(f"\n{error['error_type']} (발생: {error['count']}회)")
                print("메시지:")
                for msg in error['messages']:
                    print(f"- {msg}")
                    
        if analysis['recommendations']:
            print("\n[개선 제안]")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"{i}. {rec}")

# 전역 에러 모니터 인스턴스
error_monitor = ErrorMonitor() 