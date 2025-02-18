"""
Application Configuration

This module contains application configuration settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """Application configuration settings."""
    
    # Application settings
    APP_TITLE = "AutoCoin Trading"
    PAGE_TITLE = "AutoCoin"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # Trading settings
    INITIAL_BALANCE = 10000000  # 1천만원
    DEFAULT_MARKET = "KRW-BTC"
    TRADE_AMOUNT_PERCENTAGE = 0.1  # 10% of balance per trade
    
    # Update intervals
    UPDATE_INTERVAL = 1.0  # seconds
    ANALYSIS_CACHE_DURATION = 60  # seconds
    
    # API settings
    API_RATE_LIMIT = 10  # requests per second
    API_TIMEOUT = 10  # seconds
    
    # Trading strategy settings
    CONFIDENCE_THRESHOLD = 0.5
    RISK_THRESHOLD = 7.0  # Risk score threshold (0-10)
    
    # Auto trading settings
    MIN_TRADE_AMOUNT = 5000  # 최소 거래금액 (KRW)
    MAX_TRADE_AMOUNT = 1000000  # 최대 거래금액 (KRW)
    DEFAULT_STOP_LOSS = 0.05  # 5% stop loss
    DEFAULT_TAKE_PROFIT = 0.10  # 10% take profit
    MIN_TRADE_VOLUME = 1000000  # 최소 거래량 (KRW)
    MAX_TRADE_VOLUME_RATIO = 0.01  # 최대 거래량 비율 (1%)
    
    # UI settings
    CHART_HEIGHT = 800
    CHART_WIDTH = 1200
    
    # Logging settings
    LOG_LEVEL = "WARNING"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_DIR = "cache"
    CACHE_MAX_SIZE = 1000  # entries
    CACHE_EXPIRY = 300  # 캐시 만료 시간 (초)
    
    # Database settings
    DB_PATH = "data/trading.db"
    DB_BACKUP_ENABLED = True
    DB_BACKUP_INTERVAL = 86400  # 24 hours
    
    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    ERROR_NOTIFICATION_ENABLED = True
    
    # Upbit API settings
    UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
    UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
    
    # Trading history
    MAX_HISTORY_SIZE = 100  # 최대 거래 기록 수
    
    # Performance metrics
    METRICS_UPDATE_INTERVAL = 60  # 성과 지표 업데이트 간격 (초)
    
    # Debug mode
    DEBUG = bool(os.getenv('DEBUG', 'False').lower() == 'true')
    
    # Strategy settings
    MIN_CONFIDENCE = 60  # 최소 신뢰도 (%) 