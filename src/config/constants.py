"""
기술적 분석 관련 상수 정의
"""

class TechnicalConstants:
    """기술적 분석 상수들"""
    
    # RSI 관련 상수
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_DEFAULT_PERIOD = 14
    
    # 볼린저 밴드 관련 상수
    BOLLINGER_STD = 2
    BOLLINGER_DEFAULT_PERIOD = 20
    
    # 이동평균 기간
    MA_PERIODS = [5, 10, 20, 60, 120]
    
    # MACD 관련 상수
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    
    # 스토캐스틱 관련 상수
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    STOCH_SMOOTH_K = 3
    
    # 거래량 관련 상수
    VOLUME_MA_PERIOD = 20
    
    # 가격 변화율 임계값
    PRICE_CHANGE_THRESHOLD = 0.05  # 5%
    
    # 신뢰도 관련 상수
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.6
    LOW_CONFIDENCE = 0.4