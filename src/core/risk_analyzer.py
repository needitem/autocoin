"""
실시간 위험도 분석 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
from .performance_optimizer import cached_analysis, monitored_execution

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """위험도 레벨"""
    VERY_LOW = "매우 낮음"
    LOW = "낮음" 
    MEDIUM = "보통"
    HIGH = "높음"
    VERY_HIGH = "매우 높음"
    EXTREME = "극도로 높음"

class RiskType(Enum):
    """위험 유형"""
    VOLATILITY = "변동성"
    LIQUIDITY = "유동성"
    TECHNICAL = "기술적"
    MOMENTUM = "모멘텀"
    DRAWDOWN = "하락폭"
    CORRELATION = "상관관계"

@dataclass
class RiskMetric:
    """위험 메트릭"""
    risk_type: RiskType
    current_value: float
    threshold_value: float
    risk_level: RiskLevel
    description: str
    recommendation: str
    timestamp: datetime

@dataclass
class PositionRisk:
    """포지션 위험도"""
    market: str
    position_size: float  # 포지션 크기 (%)
    current_price: float
    entry_price: Optional[float]
    unrealized_pnl: float
    unrealized_pnl_pct: float
    var_1d: float  # 1일 VaR (Value at Risk)
    var_7d: float  # 7일 VaR
    max_drawdown: float
    risk_level: RiskLevel
    stop_loss_suggestion: float
    position_size_suggestion: float

@dataclass
class PortfolioRisk:
    """포트폴리오 위험도"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    portfolio_var: float
    portfolio_volatility: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    risk_level: RiskLevel
    diversification_score: float
    correlation_risk: float
    concentration_risk: float

class RealTimeRiskAnalyzer:
    """실시간 위험도 분석기"""
    
    def __init__(self):
        self.risk_metrics_history = {}
        self.portfolio_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 위험도 임계값 설정
        self.risk_thresholds = {
            'volatility_daily': 0.05,  # 5% 일일 변동성
            'volatility_weekly': 0.15,  # 15% 주간 변동성
            'var_1d_threshold': 0.02,  # 2% 1일 VaR
            'var_7d_threshold': 0.08,  # 8% 7일 VaR
            'drawdown_threshold': 0.10,  # 10% 최대 하락폭
            'correlation_threshold': 0.8,  # 80% 상관관계
            'concentration_threshold': 0.3,  # 30% 집중도
        }
        
        # 포지션 제한
        self.position_limits = {
            'max_single_position': 0.2,  # 단일 포지션 최대 20%
            'max_sector_exposure': 0.4,  # 섹터 최대 노출 40%
            'min_diversification': 5,  # 최소 5개 자산
        }
    
    @monitored_execution
    def analyze_market_risk(self, market: str, ohlcv_data: pd.DataFrame, 
                           current_price: float, position_size: float = 0) -> List[RiskMetric]:
        """시장 위험도 분석"""
        try:
            if ohlcv_data.empty or len(ohlcv_data) < 30:
                return self._get_default_risk_metrics(market)
            
            risk_metrics = []
            
            # 1. 변동성 위험
            volatility_metrics = self._calculate_volatility_risk(ohlcv_data)
            risk_metrics.extend(volatility_metrics)
            
            # 2. 유동성 위험
            liquidity_metric = self._calculate_liquidity_risk(ohlcv_data)
            if liquidity_metric:
                risk_metrics.append(liquidity_metric)
            
            # 3. 기술적 위험
            technical_metrics = self._calculate_technical_risk(ohlcv_data, current_price)
            risk_metrics.extend(technical_metrics)
            
            # 4. 모멘텀 위험
            momentum_metric = self._calculate_momentum_risk(ohlcv_data)
            if momentum_metric:
                risk_metrics.append(momentum_metric)
            
            # 5. 하락폭 위험
            drawdown_metric = self._calculate_drawdown_risk(ohlcv_data)
            if drawdown_metric:
                risk_metrics.append(drawdown_metric)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"시장 위험도 분석 오류: {str(e)}")
            return self._get_default_risk_metrics(market)
    
    @cached_analysis(ttl_seconds=60)
    def _calculate_volatility_risk(self, ohlcv_data: pd.DataFrame) -> List[RiskMetric]:
        """변동성 위험 계산"""
        risk_metrics = []
        
        try:
            # 일일 수익률 계산
            returns = ohlcv_data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return risk_metrics
            
            # 일일 변동성
            daily_vol = returns.std()
            daily_vol_annualized = daily_vol * np.sqrt(252)
            
            # 주간 변동성 (최근 7일)
            weekly_returns = returns.tail(7)
            weekly_vol = weekly_returns.std() * np.sqrt(7) if len(weekly_returns) > 1 else daily_vol
            
            # 일일 변동성 위험 평가
            if daily_vol > self.risk_thresholds['volatility_daily']:
                risk_level = RiskLevel.HIGH if daily_vol > 0.08 else RiskLevel.MEDIUM
                recommendation = "포지션 크기 축소 권장" if risk_level == RiskLevel.HIGH else "주의 깊은 모니터링 필요"
            else:
                risk_level = RiskLevel.LOW
                recommendation = "변동성이 안정적인 수준"
            
            risk_metrics.append(RiskMetric(
                risk_type=RiskType.VOLATILITY,
                current_value=daily_vol,
                threshold_value=self.risk_thresholds['volatility_daily'],
                risk_level=risk_level,
                description=f"일일 변동성: {daily_vol*100:.2f}% (연환산: {daily_vol_annualized*100:.1f}%)",
                recommendation=recommendation,
                timestamp=datetime.now()
            ))
            
            # 주간 변동성 위험 평가
            if weekly_vol > self.risk_thresholds['volatility_weekly']:
                risk_level = RiskLevel.VERY_HIGH if weekly_vol > 0.25 else RiskLevel.HIGH
                recommendation = "단기 거래 위험 높음" if risk_level == RiskLevel.VERY_HIGH else "주간 변동성 주의"
            else:
                risk_level = RiskLevel.MEDIUM if weekly_vol > 0.08 else RiskLevel.LOW
                recommendation = "주간 변동성 양호" if risk_level == RiskLevel.LOW else "보통 수준"
            
            risk_metrics.append(RiskMetric(
                risk_type=RiskType.VOLATILITY,
                current_value=weekly_vol,
                threshold_value=self.risk_thresholds['volatility_weekly'],
                risk_level=risk_level,
                description=f"주간 변동성: {weekly_vol*100:.2f}%",
                recommendation=recommendation,
                timestamp=datetime.now()
            ))
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"변동성 위험 계산 오류: {str(e)}")
            return []
    
    def _calculate_liquidity_risk(self, ohlcv_data: pd.DataFrame) -> Optional[RiskMetric]:
        """유동성 위험 계산"""
        try:
            if 'volume' not in ohlcv_data.columns or len(ohlcv_data) < 20:
                return None
            
            # 평균 거래량 대비 현재 거래량
            recent_volume = ohlcv_data['volume'].tail(5).mean()
            avg_volume = ohlcv_data['volume'].tail(20).mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Amihud 비유동성 지표 (간소화 버전)
            returns = ohlcv_data['close'].pct_change().abs()
            volume_usd = ohlcv_data['volume'] * ohlcv_data['close']
            
            illiquidity = (returns / volume_usd).replace([np.inf, -np.inf], np.nan).dropna()
            avg_illiquidity = illiquidity.tail(20).mean() if len(illiquidity) > 0 else 0
            
            # 위험도 평가
            if volume_ratio < 0.3:  # 거래량이 평균의 30% 미만
                risk_level = RiskLevel.HIGH
                recommendation = "유동성 부족 - 대량 거래 시 주의"
            elif volume_ratio < 0.6:
                risk_level = RiskLevel.MEDIUM
                recommendation = "유동성 다소 부족 - 거래 규모 고려"
            else:
                risk_level = RiskLevel.LOW
                recommendation = "유동성 양호"
            
            return RiskMetric(
                risk_type=RiskType.LIQUIDITY,
                current_value=volume_ratio,
                threshold_value=0.6,
                risk_level=risk_level,
                description=f"거래량 비율: {volume_ratio:.2f} (평균 대비 {volume_ratio*100:.0f}%)",
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"유동성 위험 계산 오류: {str(e)}")
            return None
    
    def _calculate_technical_risk(self, ohlcv_data: pd.DataFrame, current_price: float) -> List[RiskMetric]:
        """기술적 위험 계산"""
        risk_metrics = []
        
        try:
            # RSI 과매수/과매도 위험
            if len(ohlcv_data) >= 14:
                delta = ohlcv_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                if current_rsi > 80:
                    risk_level = RiskLevel.HIGH
                    recommendation = "과매수 구간 - 조정 위험 높음"
                elif current_rsi < 20:
                    risk_level = RiskLevel.HIGH
                    recommendation = "과매도 구간 - 반등 가능하나 추가 하락 위험"
                elif current_rsi > 70 or current_rsi < 30:
                    risk_level = RiskLevel.MEDIUM
                    recommendation = "과매수/과매도 근접 - 주의 필요"
                else:
                    risk_level = RiskLevel.LOW
                    recommendation = "RSI 정상 범위"
                
                risk_metrics.append(RiskMetric(
                    risk_type=RiskType.TECHNICAL,
                    current_value=current_rsi,
                    threshold_value=70,
                    risk_level=risk_level,
                    description=f"RSI: {current_rsi:.1f}",
                    recommendation=recommendation,
                    timestamp=datetime.now()
                ))
            
            # 볼린저 밴드 위험
            if len(ohlcv_data) >= 20:
                bb_period = 20
                close_prices = ohlcv_data['close']
                bb_middle = close_prices.rolling(window=bb_period).mean()
                bb_std = close_prices.rolling(window=bb_period).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                
                if bb_position > 0.9:
                    risk_level = RiskLevel.HIGH
                    recommendation = "상단 밴드 근접 - 과매수 위험"
                elif bb_position < 0.1:
                    risk_level = RiskLevel.HIGH
                    recommendation = "하단 밴드 근접 - 과매도 상태"
                elif bb_position > 0.8 or bb_position < 0.2:
                    risk_level = RiskLevel.MEDIUM
                    recommendation = "볼린저 밴드 극단 근접"
                else:
                    risk_level = RiskLevel.LOW
                    recommendation = "볼린저 밴드 정상 범위"
                
                risk_metrics.append(RiskMetric(
                    risk_type=RiskType.TECHNICAL,
                    current_value=bb_position,
                    threshold_value=0.8,
                    risk_level=risk_level,
                    description=f"볼린저 밴드 위치: {bb_position*100:.1f}%",
                    recommendation=recommendation,
                    timestamp=datetime.now()
                ))
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"기술적 위험 계산 오류: {str(e)}")
            return []
    
    def _calculate_momentum_risk(self, ohlcv_data: pd.DataFrame) -> Optional[RiskMetric]:
        """모멘텀 위험 계산"""
        try:
            if len(ohlcv_data) < 30:
                return None
            
            # 가격 모멘텀 (최근 5일 vs 이전 25일)
            recent_returns = ohlcv_data['close'].pct_change().tail(5).mean()
            longer_returns = ohlcv_data['close'].pct_change().tail(30).head(25).mean()
            
            momentum_strength = abs(recent_returns - longer_returns)
            momentum_direction = "상승" if recent_returns > longer_returns else "하락"
            
            # 모멘텀 위험 평가
            if momentum_strength > 0.05:  # 5% 이상 모멘텀
                risk_level = RiskLevel.HIGH if momentum_strength > 0.1 else RiskLevel.MEDIUM
                recommendation = f"강한 {momentum_direction} 모멘텀 - 반전 위험 주의"
            elif momentum_strength > 0.02:
                risk_level = RiskLevel.MEDIUM
                recommendation = f"보통 {momentum_direction} 모멘텀"
            else:
                risk_level = RiskLevel.LOW
                recommendation = "모멘텀 안정적"
            
            return RiskMetric(
                risk_type=RiskType.MOMENTUM,
                current_value=momentum_strength,
                threshold_value=0.05,
                risk_level=risk_level,
                description=f"모멘텀 강도: {momentum_strength*100:.2f}% ({momentum_direction})",
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"모멘텀 위험 계산 오류: {str(e)}")
            return None
    
    def _calculate_drawdown_risk(self, ohlcv_data: pd.DataFrame) -> Optional[RiskMetric]:
        """하락폭 위험 계산"""
        try:
            if len(ohlcv_data) < 20:
                return None
            
            # 최대 하락폭 계산
            prices = ohlcv_data['close']
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 현재 하락폭
            current_drawdown = drawdown.iloc[-1]
            
            # 위험도 평가
            if abs(max_drawdown) > self.risk_thresholds['drawdown_threshold']:
                risk_level = RiskLevel.VERY_HIGH if abs(max_drawdown) > 0.2 else RiskLevel.HIGH
                recommendation = "높은 하락폭 위험 - 손절매 고려"
            elif abs(current_drawdown) > 0.05:
                risk_level = RiskLevel.MEDIUM
                recommendation = "현재 하락폭 주의"
            else:
                risk_level = RiskLevel.LOW
                recommendation = "하락폭 위험 낮음"
            
            return RiskMetric(
                risk_type=RiskType.DRAWDOWN,
                current_value=abs(max_drawdown),
                threshold_value=self.risk_thresholds['drawdown_threshold'],
                risk_level=risk_level,
                description=f"최대 하락폭: {max_drawdown*100:.2f}%, 현재: {current_drawdown*100:.2f}%",
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"하락폭 위험 계산 오류: {str(e)}")
            return None
    
    def calculate_position_risk(self, market: str, ohlcv_data: pd.DataFrame, 
                              position_size: float, entry_price: Optional[float] = None) -> PositionRisk:
        """포지션 위험도 계산"""
        try:
            current_price = ohlcv_data['close'].iloc[-1]
            returns = ohlcv_data['close'].pct_change().dropna()
            
            # PnL 계산
            unrealized_pnl = 0
            unrealized_pnl_pct = 0
            if entry_price:
                unrealized_pnl = (current_price - entry_price) * position_size
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            # VaR 계산 (95% 신뢰구간)
            if len(returns) >= 30:
                var_1d = np.percentile(returns, 5) * position_size * current_price
                var_7d = np.percentile(returns, 5) * np.sqrt(7) * position_size * current_price
            else:
                var_1d = var_7d = 0
            
            # 최대 하락폭
            prices = ohlcv_data['close']
            rolling_max = prices.expanding().max()
            drawdown = ((prices - rolling_max) / rolling_max).min()
            max_drawdown = abs(drawdown) * position_size
            
            # 위험도 평가
            risk_factors = []
            if abs(var_1d) > abs(position_size * current_price * 0.02):
                risk_factors.append("높은 VaR")
            if max_drawdown > 0.1:
                risk_factors.append("높은 하락폭")
            if position_size > self.position_limits['max_single_position']:
                risk_factors.append("과도한 포지션")
            
            if len(risk_factors) >= 2:
                risk_level = RiskLevel.VERY_HIGH
            elif len(risk_factors) == 1:
                risk_level = RiskLevel.HIGH
            elif max_drawdown > 0.05 or abs(var_1d) > abs(position_size * current_price * 0.01):
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # 권장사항 계산
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                recommended_position = position_size * 0.5  # 절반으로 축소
                stop_loss = current_price * 0.95 if entry_price is None else min(current_price * 0.95, entry_price * 0.9)
            else:
                recommended_position = min(position_size, self.position_limits['max_single_position'])
                stop_loss = current_price * 0.92 if entry_price is None else entry_price * 0.92
            
            return PositionRisk(
                market=market,
                position_size=position_size,
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                var_1d=var_1d,
                var_7d=var_7d,
                max_drawdown=max_drawdown,
                risk_level=risk_level,
                stop_loss_suggestion=stop_loss,
                position_size_suggestion=recommended_position
            )
            
        except Exception as e:
            logger.error(f"포지션 위험도 계산 오류: {str(e)}")
            return self._get_default_position_risk(market, ohlcv_data)
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """VaR (Value at Risk) 계산"""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Historical VaR
            return np.percentile(returns, (1 - confidence_level) * 100)
            
        except Exception as e:
            logger.error(f"VaR 계산 오류: {str(e)}")
            return 0.0
    
    def get_overall_risk_score(self, risk_metrics: List[RiskMetric]) -> Tuple[float, RiskLevel]:
        """전체 위험도 점수 계산"""
        try:
            if not risk_metrics:
                return 0.5, RiskLevel.MEDIUM
            
            # 위험도 레벨을 숫자로 변환
            risk_scores = {
                RiskLevel.VERY_LOW: 0.1,
                RiskLevel.LOW: 0.3,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.7,
                RiskLevel.VERY_HIGH: 0.9,
                RiskLevel.EXTREME: 1.0
            }
            
            # 가중평균 계산 (변동성과 기술적 위험에 더 높은 가중치)
            weights = {
                RiskType.VOLATILITY: 0.3,
                RiskType.TECHNICAL: 0.25,
                RiskType.MOMENTUM: 0.2,
                RiskType.DRAWDOWN: 0.15,
                RiskType.LIQUIDITY: 0.1
            }
            
            total_score = 0
            total_weight = 0
            
            for metric in risk_metrics:
                weight = weights.get(metric.risk_type, 0.1)
                score = risk_scores.get(metric.risk_level, 0.5)
                total_score += score * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.5
            
            # 전체 위험도 레벨 결정
            if overall_score >= 0.8:
                overall_level = RiskLevel.VERY_HIGH
            elif overall_score >= 0.6:
                overall_level = RiskLevel.HIGH
            elif overall_score >= 0.4:
                overall_level = RiskLevel.MEDIUM
            elif overall_score >= 0.2:
                overall_level = RiskLevel.LOW
            else:
                overall_level = RiskLevel.VERY_LOW
            
            return overall_score, overall_level
            
        except Exception as e:
            logger.error(f"전체 위험도 점수 계산 오류: {str(e)}")
            return 0.5, RiskLevel.MEDIUM
    
    def _get_default_risk_metrics(self, market: str) -> List[RiskMetric]:
        """기본 위험 메트릭"""
        return [
            RiskMetric(
                risk_type=RiskType.VOLATILITY,
                current_value=0.03,
                threshold_value=0.05,
                risk_level=RiskLevel.MEDIUM,
                description="데이터 부족으로 기본값 적용",
                recommendation="충분한 데이터 수집 후 재분석 필요",
                timestamp=datetime.now()
            )
        ]
    
    def _get_default_position_risk(self, market: str, ohlcv_data: pd.DataFrame) -> PositionRisk:
        """기본 포지션 위험도"""
        current_price = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0
        
        return PositionRisk(
            market=market,
            position_size=0,
            current_price=current_price,
            entry_price=None,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            var_1d=0,
            var_7d=0,
            max_drawdown=0,
            risk_level=RiskLevel.MEDIUM,
            stop_loss_suggestion=current_price * 0.95,
            position_size_suggestion=0.05
        )