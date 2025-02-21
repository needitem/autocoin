"""
중앙 집중식 로깅 시스템

Features:
1. 파일 및 콘솔 로깅 지원
2. 로그 레벨 관리
3. 로그 포맷팅
4. 로그 회전
5. 컴포넌트별 로거 제공
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Optional

# 로깅 레벨 상수 export
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

class Logger:
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logging()
        return cls._instance
    
    def _initialize_logging(self):
        """로깅 시스템 초기화"""
        # 로그 디렉토리 생성
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 기본 로그 파일 설정
        self.log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
        
        # 기본 포맷터 설정
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 루트 로거 설정
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self.formatter)
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        console_handler.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 및 새 핸들러 추가
        root_logger.handlers = []
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """컴포넌트별 로거 반환
        
        Args:
            name (str): 로거 이름
            level (int, optional): 로깅 레벨. Defaults to None.
            
        Returns:
            logging.Logger: 설정된 로거
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            
            # 로깅 레벨 설정
            if level is not None:
                logger.setLevel(level)
            
            # 핸들러가 없는 경우에만 추가
            if not logger.handlers:
                # 파일 핸들러
                file_handler = RotatingFileHandler(
                    os.path.join('logs', f'{name}.log'),
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                file_handler.setFormatter(self.formatter)
                logger.addHandler(file_handler)
                
                # 콘솔 핸들러
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self.formatter)
                logger.addHandler(console_handler)
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_level(self, name: str, level: int):
        """로거의 로깅 레벨 설정
        
        Args:
            name (str): 로거 이름
            level (int): 로깅 레벨
        """
        if name in self._loggers:
            self._loggers[name].setLevel(level)
    
    def debug(self, name: str, message: str):
        """DEBUG 레벨 로그 기록"""
        self.get_logger(name).debug(message)
    
    def info(self, name: str, message: str):
        """INFO 레벨 로그 기록"""
        self.get_logger(name).info(message)
    
    def warning(self, name: str, message: str):
        """WARNING 레벨 로그 기록"""
        self.get_logger(name).warning(message)
    
    def error(self, name: str, message: str):
        """ERROR 레벨 로그 기록"""
        self.get_logger(name).error(message)
    
    def critical(self, name: str, message: str):
        """CRITICAL 레벨 로그 기록"""
        self.get_logger(name).critical(message)

# 싱글톤 인스턴스 생성
_logger = Logger()

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """로거 인스턴스 반환
    
    Args:
        name (str): 로거 이름
        level (int, optional): 로깅 레벨. Defaults to None.
        
    Returns:
        logging.Logger: 설정된 로거
    """
    return _logger.get_logger(name, level) 