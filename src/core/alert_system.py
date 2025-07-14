"""
실시간 차트 분석 알림 시스템
"""

import time
import threading
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from .chart_analysis import ChartAnalyzer, PatternType, SignalStrength, ChartPattern
from .risk_analyzer import RealTimeRiskAnalyzer, RiskLevel, RiskType

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """알림 타입"""
    PATTERN_DETECTED = "패턴감지"
    PRICE_BREAKOUT = "가격돌파"
    VOLUME_SPIKE = "거래량급증"
    SUPPORT_RESISTANCE = "지지저항"
    MOMENTUM_CHANGE = "모멘텀변화"
    RISK_WARNING = "위험경고"
    RISK_CRITICAL = "위험긴급"

class AlertPriority(Enum):
    """알림 우선순위"""
    LOW = "낮음"
    MEDIUM = "보통"
    HIGH = "높음"
    CRITICAL = "긴급"

@dataclass
class Alert:
    """알림 데이터 클래스"""
    id: str
    market: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    pattern: Optional[ChartPattern] = None
    price: Optional[float] = None
    is_read: bool = False
    is_active: bool = True

class AlertSystem:
    """실시간 알림 시스템"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.monitoring_markets: Dict[str, bool] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 알림 설정
        self.pattern_alerts_enabled = True
        self.price_alerts_enabled = True
        self.volume_alerts_enabled = True
        self.risk_alerts_enabled = True
        
        # 위험도 분석기 초기화
        self.risk_analyzer = RealTimeRiskAnalyzer()
        
        # 알림 임계값
        self.volume_spike_threshold = 2.0  # 평균 대비 2배 이상
        self.price_breakout_threshold = 0.02  # 2% 이상 돌파
        
        # 중요 패턴 목록
        self.critical_patterns = [
            PatternType.ENGULFING_BULLISH,
            PatternType.ENGULFING_BEARISH,
            PatternType.HAMMER,
            PatternType.SHOOTING_STAR,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM
        ]
        
    def add_market_monitor(self, market: str):
        """모니터링 마켓 추가"""
        self.monitoring_markets[market] = True
        logger.info(f"마켓 {market} 모니터링 시작")
        
    def remove_market_monitor(self, market: str):
        """모니터링 마켓 제거"""
        if market in self.monitoring_markets:
            del self.monitoring_markets[market]
            logger.info(f"마켓 {market} 모니터링 중지")
    
    def start_monitoring(self, trading_manager, interval_seconds: int = 60):
        """실시간 모니터링 시작"""
        if self.is_monitoring:
            logger.warning("이미 모니터링이 실행 중입니다.")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(trading_manager, interval_seconds),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"실시간 모니터링 시작 (간격: {interval_seconds}초)")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("실시간 모니터링 중지")
    
    def _monitoring_loop(self, trading_manager, interval_seconds: int):
        """모니터링 메인 루프"""
        chart_analyzer = ChartAnalyzer()
        
        while self.is_monitoring:
            try:
                for market in list(self.monitoring_markets.keys()):
                    if not self.monitoring_markets.get(market, False):
                        continue
                        
                    # 차트 데이터 가져오기
                    ohlcv_data = trading_manager.get_ohlcv(market, count=100)
                    if ohlcv_data is None or ohlcv_data.empty:
                        continue
                        
                    indicators = trading_manager.calculate_indicators(ohlcv_data)
                    
                    # 차트 분석
                    analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
                    
                    # 알림 체크
                    self._check_pattern_alerts(market, analysis.patterns, ohlcv_data)
                    self._check_price_alerts(market, ohlcv_data, analysis.support_resistance)
                    self._check_volume_alerts(market, ohlcv_data)
                    self._check_momentum_alerts(market, analysis.momentum_signals)
                    self._check_risk_alerts(market, ohlcv_data)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(interval_seconds)
    
    def _check_pattern_alerts(self, market: str, patterns: List[ChartPattern], ohlcv_data):
        """패턴 알림 체크"""
        if not self.pattern_alerts_enabled:
            return
            
        for pattern in patterns:
            # 최근 패턴만 알림
            pattern_date = datetime.strptime(pattern.end_date, '%Y-%m-%d')
            if (datetime.now() - pattern_date).days > 1:
                continue
                
            # 중요 패턴 체크
            priority = AlertPriority.CRITICAL if pattern.pattern_type in self.critical_patterns else AlertPriority.HIGH
            
            # 신뢰도 기반 우선순위 조정
            if pattern.confidence < 0.7:
                priority = AlertPriority.MEDIUM
            elif pattern.confidence < 0.5:
                priority = AlertPriority.LOW
                
            alert = Alert(
                id=f"pattern_{market}_{pattern.pattern_type.value}_{int(time.time())}",
                market=market,
                alert_type=AlertType.PATTERN_DETECTED,
                priority=priority,
                title=f"{market} {pattern.pattern_type.value} 패턴 감지",
                message=f"신뢰도: {pattern.confidence*100:.1f}% | {pattern.description}",
                timestamp=datetime.now(),
                pattern=pattern,
                price=ohlcv_data['close'].iloc[-1]
            )
            
            self._add_alert(alert)
    
    def _check_price_alerts(self, market: str, ohlcv_data, support_resistance: Dict):
        """가격 돌파 알림 체크"""
        if not self.price_alerts_enabled:
            return
            
        current_price = ohlcv_data['close'].iloc[-1]
        prev_price = ohlcv_data['close'].iloc[-2] if len(ohlcv_data) > 1 else current_price
        
        # 지지선 돌파 체크
        for support_level in support_resistance.get('support', []):
            if prev_price > support_level and current_price < support_level:
                # 지지선 하향 돌파
                alert = Alert(
                    id=f"support_break_{market}_{int(time.time())}",
                    market=market,
                    alert_type=AlertType.SUPPORT_RESISTANCE,
                    priority=AlertPriority.HIGH,
                    title=f"{market} 지지선 하향 돌파",
                    message=f"지지선 {support_level:,.0f}원 하향 돌파 (현재가: {current_price:,.0f}원)",
                    timestamp=datetime.now(),
                    price=current_price
                )
                self._add_alert(alert)
        
        # 저항선 돌파 체크
        for resistance_level in support_resistance.get('resistance', []):
            if prev_price < resistance_level and current_price > resistance_level:
                # 저항선 상향 돌파
                alert = Alert(
                    id=f"resistance_break_{market}_{int(time.time())}",
                    market=market,
                    alert_type=AlertType.SUPPORT_RESISTANCE,
                    priority=AlertPriority.HIGH,
                    title=f"{market} 저항선 상향 돌파",
                    message=f"저항선 {resistance_level:,.0f}원 상향 돌파 (현재가: {current_price:,.0f}원)",
                    timestamp=datetime.now(),
                    price=current_price
                )
                self._add_alert(alert)
    
    def _check_volume_alerts(self, market: str, ohlcv_data):
        """거래량 급증 알림 체크"""
        if not self.volume_alerts_enabled or len(ohlcv_data) < 20:
            return
            
        current_volume = ohlcv_data['volume'].iloc[-1]
        avg_volume = ohlcv_data['volume'].tail(20).mean()
        
        if current_volume > avg_volume * self.volume_spike_threshold:
            volume_ratio = current_volume / avg_volume
            priority = AlertPriority.CRITICAL if volume_ratio > 5.0 else AlertPriority.HIGH
            
            alert = Alert(
                id=f"volume_spike_{market}_{int(time.time())}",
                market=market,
                alert_type=AlertType.VOLUME_SPIKE,
                priority=priority,
                title=f"{market} 거래량 급증",
                message=f"거래량이 평균 대비 {volume_ratio:.1f}배 증가 (현재: {current_volume:,.0f})",
                timestamp=datetime.now(),
                price=ohlcv_data['close'].iloc[-1]
            )
            self._add_alert(alert)
    
    def _check_momentum_alerts(self, market: str, momentum_signals: Dict):
        """모멘텀 변화 알림 체크"""
        strong_signals = ['강한 상승', '강한 하락', '상승 확인', '하락 확인']
        
        for indicator, signal in momentum_signals.items():
            if signal in strong_signals:
                priority = AlertPriority.HIGH if '강한' in signal else AlertPriority.MEDIUM
                
                alert = Alert(
                    id=f"momentum_{market}_{indicator}_{int(time.time())}",
                    market=market,
                    alert_type=AlertType.MOMENTUM_CHANGE,
                    priority=priority,
                    title=f"{market} {indicator} 모멘텀 변화",
                    message=f"{indicator}: {signal}",
                    timestamp=datetime.now()
                )
                self._add_alert(alert)
    
    def _check_risk_alerts(self, market: str, ohlcv_data):
        """위험도 알림 체크"""
        if not self.risk_alerts_enabled or len(ohlcv_data) < 20:
            return
        
        try:
            current_price = ohlcv_data['close'].iloc[-1]
            
            # 시장 위험도 분석
            risk_metrics = self.risk_analyzer.analyze_market_risk(
                market, ohlcv_data, current_price
            )
            
            # 전체 위험도 점수 계산
            overall_score, overall_level = self.risk_analyzer.get_overall_risk_score(risk_metrics)
            
            # 위험도 레벨별 알림 생성
            if overall_level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                # 긴급 위험 알림
                alert = Alert(
                    id=f"risk_critical_{market}_{int(time.time())}",
                    market=market,
                    alert_type=AlertType.RISK_CRITICAL,
                    priority=AlertPriority.CRITICAL,
                    title=f"{market} 긴급 위험 경고",
                    message=f"위험도: {overall_level.value} (점수: {overall_score*100:.0f}/100)",
                    timestamp=datetime.now(),
                    price=current_price
                )
                self._add_alert(alert)
                
                # 상세 위험 요소 알림
                high_risk_metrics = [m for m in risk_metrics if m.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
                for metric in high_risk_metrics[:2]:  # 최대 2개까지
                    detail_alert = Alert(
                        id=f"risk_detail_{market}_{metric.risk_type.value}_{int(time.time())}",
                        market=market,
                        alert_type=AlertType.RISK_WARNING,
                        priority=AlertPriority.HIGH,
                        title=f"{market} {metric.risk_type.value} 위험",
                        message=f"{metric.description} | {metric.recommendation}",
                        timestamp=datetime.now(),
                        price=current_price
                    )
                    self._add_alert(detail_alert)
            
            elif overall_level == RiskLevel.HIGH:
                # 일반 위험 경고
                alert = Alert(
                    id=f"risk_warning_{market}_{int(time.time())}",
                    market=market,
                    alert_type=AlertType.RISK_WARNING,
                    priority=AlertPriority.HIGH,
                    title=f"{market} 위험도 증가",
                    message=f"위험도: {overall_level.value} | 주의 깊은 모니터링 필요",
                    timestamp=datetime.now(),
                    price=current_price
                )
                self._add_alert(alert)
            
            # 개별 위험 요소 체크
            for metric in risk_metrics:
                if metric.risk_level == RiskLevel.VERY_HIGH:
                    specific_alert = Alert(
                        id=f"risk_{metric.risk_type.value}_{market}_{int(time.time())}",
                        market=market,
                        alert_type=AlertType.RISK_WARNING,
                        priority=AlertPriority.HIGH,
                        title=f"{market} {metric.risk_type.value} 위험 급증",
                        message=f"{metric.description} | {metric.recommendation}",
                        timestamp=datetime.now(),
                        price=current_price
                    )
                    self._add_alert(specific_alert)
        
        except Exception as e:
            logger.error(f"위험도 알림 체크 오류: {str(e)}")
    
    def _add_alert(self, alert: Alert):
        """알림 추가"""
        # 중복 알림 체크 (같은 타입, 같은 마켓, 10분 이내)
        recent_alerts = [
            a for a in self.alerts 
            if a.market == alert.market 
            and a.alert_type == alert.alert_type
            and (datetime.now() - a.timestamp).total_seconds() < 600
        ]
        
        if recent_alerts:
            logger.debug(f"중복 알림 무시: {alert.title}")
            return
        
        self.alerts.append(alert)
        
        # 최대 1000개 알림 유지
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # 콜백 호출
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"알림 콜백 오류: {str(e)}")
        
        logger.info(f"새 알림: {alert.title}")
    
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def get_alerts(self, market: str = None, limit: int = 50, unread_only: bool = False) -> List[Alert]:
        """알림 조회"""
        filtered_alerts = self.alerts
        
        if market:
            filtered_alerts = [a for a in filtered_alerts if a.market == market]
        
        if unread_only:
            filtered_alerts = [a for a in filtered_alerts if not a.is_read]
        
        # 최신순 정렬
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_alerts[:limit]
    
    def mark_alert_read(self, alert_id: str):
        """알림 읽음 처리"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.is_read = True
                break
    
    def mark_all_read(self, market: str = None):
        """모든 알림 읽음 처리"""
        for alert in self.alerts:
            if market is None or alert.market == market:
                alert.is_read = True
    
    def clear_alerts(self, market: str = None, days: int = 7):
        """오래된 알림 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        before_count = len(self.alerts)
        
        if market:
            self.alerts = [
                a for a in self.alerts 
                if not (a.market == market and a.timestamp < cutoff_date)
            ]
        else:
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
        
        after_count = len(self.alerts)
        logger.info(f"알림 정리 완료: {before_count - after_count}개 삭제")
    
    def get_alert_stats(self) -> Dict:
        """알림 통계"""
        total_alerts = len(self.alerts)
        unread_alerts = len([a for a in self.alerts if not a.is_read])
        
        # 타입별 통계
        type_stats = {}
        for alert_type in AlertType:
            count = len([a for a in self.alerts if a.alert_type == alert_type])
            type_stats[alert_type.value] = count
        
        # 우선순위별 통계
        priority_stats = {}
        for priority in AlertPriority:
            count = len([a for a in self.alerts if a.priority == priority])
            priority_stats[priority.value] = count
        
        return {
            'total': total_alerts,
            'unread': unread_alerts,
            'by_type': type_stats,
            'by_priority': priority_stats,
            'monitoring_markets': list(self.monitoring_markets.keys())
        }

# 전역 알림 시스템 인스턴스
alert_system = AlertSystem()