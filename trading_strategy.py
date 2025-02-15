from enum import Enum

class TradingStrategy(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"      # 단타 매매
    SWING = "swing"            # 스윙 매매
    POSITION = "position"      # 포지션 매매 