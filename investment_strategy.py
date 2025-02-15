from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
from enum import Enum
from trading_strategy import TradingStrategy

@dataclass
class InvestmentLevel:
    price: float
    ratio: float  # 전체 투자금 대비 비율
    description: str
    reasons: List[str] = field(default_factory=list)  # 진입 근거 리스트 추가

@dataclass
class StrategyRecommendation:
    entry_levels: List[InvestmentLevel]  # 매수 진입 레벨
    exit_levels: List[InvestmentLevel]   # 매도 청산 레벨
    stop_loss: float                     # 손절가
    risk_ratio: float                    # 리스크 비율
    investment_amount: float             # 추천 투자 금액 (총 자산 대비)
    holding_period: str                  # 추천 보유 기간
    strategy_type: str                   # 전략 유형
    confidence_score: float              # 전략 신뢰도

class TradingStrategy(Enum):
    SCALPING = "SCALPING"
    SWING = "SWING"
    POSITION = "POSITION"
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"

class InvestmentStrategy:
    def __init__(self, strategy_type: TradingStrategy):
        self.strategy_type = strategy_type
        self.strategy_configs = {
            TradingStrategy.SCALPING: {
                'holding_period': '1-4시간',
                'risk_ratio': 0.01,
                'target_profit': 0.02,
                'stop_loss': 0.01,
                'entry_points': [0.995, 0.99, 0.985],
                'exit_points': [1.01, 1.02, 1.03],
                'position_sizes': [0.4, 0.3, 0.3]
            },
            TradingStrategy.SWING: {
                'holding_period': '1-5일',
                'risk_ratio': 0.02,
                'target_profit': 0.05,
                'stop_loss': 0.03,
                'entry_points': [0.98, 0.96, 0.94],
                'exit_points': [1.03, 1.05, 1.07],
                'position_sizes': [0.4, 0.3, 0.3]
            },
            TradingStrategy.POSITION: {
                'holding_period': '1-4주',
                'risk_ratio': 0.05,
                'target_profit': 0.15,
                'stop_loss': 0.07,
                'entry_points': [0.97, 0.94, 0.91],
                'exit_points': [1.05, 1.10, 1.15],
                'position_sizes': [0.5, 0.3, 0.2]
            },
            TradingStrategy.CONSERVATIVE: {
                'holding_period': '1-2주',
                'risk_ratio': 0.01,
                'target_profit': 0.03,
                'stop_loss': 0.02,
                'entry_points': [0.99, 0.98, 0.97],
                'exit_points': [1.02, 1.03, 1.04],
                'position_sizes': [0.5, 0.3, 0.2]
            },
            TradingStrategy.MODERATE: {
                'holding_period': '3-7일',
                'risk_ratio': 0.02,
                'target_profit': 0.05,
                'stop_loss': 0.03,
                'entry_points': [0.98, 0.96, 0.94],
                'exit_points': [1.03, 1.05, 1.07],
                'position_sizes': [0.4, 0.3, 0.3]
            },
            TradingStrategy.AGGRESSIVE: {
                'holding_period': '1-3일',
                'risk_ratio': 0.03,
                'target_profit': 0.08,
                'stop_loss': 0.05,
                'entry_points': [0.97, 0.94, 0.91],
                'exit_points': [1.04, 1.08, 1.12],
                'position_sizes': [0.3, 0.4, 0.3]
            }
        }
        
        # 전략 설정 가져오기
        self.config = self.strategy_configs[strategy_type]
        
    def get_entry_points(self, current_price: float) -> list:
        """진입 가격 계산"""
        return [current_price * point for point in self.config['entry_points']]
    
    def get_exit_points(self, current_price: float) -> list:
        """청산 가격 계산"""
        return [current_price * point for point in self.config['exit_points']]
    
    def get_stop_loss(self, current_price: float) -> float:
        """손절가 계산"""
        return current_price * (1 - self.config['stop_loss'])
    
    def get_position_sizes(self) -> list:
        """포지션 크기 비율"""
        return self.config['position_sizes']
    
    def get_risk_ratio(self) -> float:
        """리스크 비율"""
        return self.config['risk_ratio']
    
    def get_holding_period(self) -> str:
        """추천 보유기간"""
        return self.config['holding_period']
    
    def get_base_position_size(self) -> float:
        """기본 포지션 크기 반환"""
        if self.strategy_type == TradingStrategy.CONSERVATIVE:
            return 0.1  # 10%
        elif self.strategy_type == TradingStrategy.MODERATE:
            return 0.3  # 30%
        elif self.strategy_type == TradingStrategy.AGGRESSIVE:
            return 0.5  # 50%
        return 0.1  # 기본값
    
    def calculate_risk_ratio(self, fg_index: float, rsi: float, volatility: float) -> float:
        """리스크 비율 계산"""
        # 공포탐욕지수 기반 리스크 조정
        if fg_index <= 20:
            fg_risk = 0.8  # 낮은 리스크
        elif fg_index >= 80:
            fg_risk = 0.2  # 높은 리스크
        else:
            fg_risk = 0.5  # 중간 리스크
            
        # RSI 기반 리스크 조정
        if rsi <= 30:
            rsi_risk = 0.8
        elif rsi >= 70:
            rsi_risk = 0.2
        else:
            rsi_risk = 0.5
            
        # 변동성 기반 리스크 조정 (변동성이 높을수록 리스크 높음)
        vol_risk = 1 - min(volatility / 100, 1)
        
        # 종합 리스크 점수 계산
        return (fg_risk * 0.4 + rsi_risk * 0.3 + vol_risk * 0.3)
    
    def calculate_investment_amount(self, total_assets: float, risk_ratio: float) -> float:
        """투자 금액 계산"""
        base_ratio = self.config['risk_ratio'] * risk_ratio
        return total_assets * base_ratio
    
    def get_holding_period(self, fg_index: float, trend_strength: str) -> str:
        """보유 기간 추천"""
        if fg_index <= 20 or fg_index >= 80:
            return "1-3일 (단기 트레이딩)"
        elif trend_strength == "강함":
            return "7-14일 (스윙 트레이딩)"
        else:
            return "14-30일 (포지션 트레이딩)"
    
    def calculate_entry_levels(self, current_price: float, 
                             orderbook_analysis: dict,
                             technical_indicators: dict,
                             patterns: list,
                             support_resistance: dict) -> List[Dict]:
        """진입 가격 계산 및 근거 도출"""
        try:
            entry_levels = []
            base_levels = self.config['entry_points']
            position_sizes = self.config['position_sizes']
            
            # 기본 진입 근거
            reasons = []
            price_adjustments = []
            
            # 호가 분석 기반 조정
            if orderbook_analysis:
                bid_ratio = orderbook_analysis.get('bid_ask_ratio', 1.0)
                bid_concentration = orderbook_analysis.get('bid_concentration', 0.0)
                
                if bid_ratio > 1.2:
                    price_adjustments.append(0.002)
                    reasons.append(f"매수세 우위 (매수/매도 비율: {bid_ratio:.2f})")
                elif bid_ratio < 0.8:
                    price_adjustments.append(-0.003)
                    reasons.append(f"매도세 우위 (매수/매도 비율: {bid_ratio:.2f})")
                
                if bid_concentration > 0.5:
                    price_adjustments.append(0.001)
                    reasons.append(f"높은 매수 집중도 ({bid_concentration:.1%})")
            
            # 기술적 지표 기반 조정
            rsi = technical_indicators.get('rsi', 50)
            if rsi < 30:
                price_adjustments.append(0.002)
                reasons.append(f"과매도 구간 (RSI: {rsi:.1f})")
            elif rsi > 70:
                price_adjustments.append(-0.002)
                reasons.append(f"과매수 구간 (RSI: {rsi:.1f})")
            
            # MACD 신호 기반 조정
            macd = technical_indicators.get('macd', {})
            if macd and macd.get('macd', 0) > macd.get('signal', 0):
                price_adjustments.append(0.001)
                reasons.append("MACD 매수 신호")
            elif macd:
                price_adjustments.append(-0.001)
                reasons.append("MACD 매도 신호")
            
            # 패턴 기반 조정
            if patterns:
                for pattern in patterns:
                    if pattern.get('pattern_type') == 'bullish':
                        price_adjustments.append(0.002)
                        reasons.append(f"상승 패턴 감지: {pattern.get('name')}")
                    else:
                        price_adjustments.append(-0.002)
                        reasons.append(f"하락 패턴 감지: {pattern.get('name')}")
            
            # 지지/저항선 고려
            if support_resistance:
                support_levels = support_resistance.get('support_levels', [])
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
                    support_distance = (current_price - nearest_support) / current_price
                    if support_distance < 0.02:
                        price_adjustments.append(0.001)
                        reasons.append(f"지지선 근접 (거리: {support_distance:.1%})")
            
            # 조정이 없는 경우 기본 근거 추가
            if not reasons:
                reasons.append("기본 분할 매수 전략")
            
            # 최종 가격 조정률 계산
            total_adjustment = sum(price_adjustments) if price_adjustments else 0
            
            # 진입 레벨 설정
            for i, (base_level, size) in enumerate(zip(base_levels, position_sizes)):
                adjusted_level = base_level * (1 + total_adjustment)
                price = current_price * adjusted_level
                
                level_reasons = reasons.copy()
                level_reasons.append(f"{i+1}차 분할 매수 지점 (현재가 대비 {(adjusted_level-1)*100:.1f}%)")
                
                entry_levels.append({
                    'price': price,
                    'ratio': size,
                    'description': f"{i+1}차 진입 (현재가 대비 {(adjusted_level-1)*100:.1f}%)",
                    'reasons': level_reasons  # 각 레벨별 근거 포함
                })
            
            return entry_levels
            
        except Exception as e:
            logger.error(f"진입 레벨 계산 중 오류: {str(e)}")
            return [
                {
                    'price': current_price * 0.99,
                    'ratio': 0.4,
                    'description': "1차 진입 (현재가 대비 -1%)",
                    'reasons': ["기본 분할 매수 전략"]
                },
                {
                    'price': current_price * 0.97,
                    'ratio': 0.3,
                    'description': "2차 진입 (현재가 대비 -3%)",
                    'reasons': ["기본 분할 매수 전략"]
                },
                {
                    'price': current_price * 0.95,
                    'ratio': 0.3,
                    'description': "3차 진입 (현재가 대비 -5%)",
                    'reasons': ["기본 분할 매수 전략"]
                }
            ]

    def _analyze_market_condition(self, fg_index: float, rsi: float, ma_data: dict) -> str:
        """시장 상황 분석"""
        # MA 데이터 분석
        ma5 = ma_data.get('MA5', 0)
        ma20 = ma_data.get('MA20', 0)
        ma60 = ma_data.get('MA60', 0)
        
        # 이동평균선 배열 확인 (하락 강도 추가)
        if ma5 < ma20 < ma60:
            if (ma5/ma60 - 1) * 100 < -5:  # 5일선이 60일선보다 5% 이상 낮음
                return "강한 하락"
            else:
                return "약한 하락"
        elif ma5 > ma20 > ma60:
            return "상승"
        else:
            return "중립"

    def _find_nearest_support(self, target_price: float, support_levels: List[float]) -> float:
        """가장 가까운 지지선 찾기"""
        if not support_levels:
            return target_price
        
        # 타겟 가격보다 낮은 지지선 중 가장 가까운 것 선택
        lower_supports = [p for p in support_levels if p < target_price]
        if lower_supports:
            return max(lower_supports)
        return target_price

    def _calculate_ma_based_price(self, current_price: float, ma_data: dict) -> float:
        """이동평균선 기반 매수가 계산"""
        ma5 = ma_data.get('MA5', current_price)
        ma20 = ma_data.get('MA20', current_price)
        ma60 = ma_data.get('MA60', current_price)
        
        # 이동평균선 배열 확인
        if current_price < ma5 < ma20:  # 하락 추세
            return min(ma5, current_price * 1.01)  # 5일선 또는 1% 위
        elif ma5 < current_price < ma20:  # 5일선 위, 20일선 아래
            return min(ma20, current_price * 1.02)  # 20일선 또는 2% 위
        elif ma20 < current_price < ma60:  # 20일선 위, 60일선 아래
            return min(ma60, current_price * 1.03)  # 60일선 또는 3% 위
        else:
            return current_price

    def calculate_exit_levels(self, current_price: float, 
                            fg_index: float,
                            resistance_levels: List[float] = None) -> List[InvestmentLevel]:
        """매도 청산 레벨 계산"""
        levels = []
        
        # 시장 상황에 따른 매도 가격 조정
        if fg_index >= 80:  # 극단적 탐욕
            profit_targets = [x * 0.995 for x in self.config['exit_points']]  # 더 낮은 가격에 매도
        elif fg_index >= 60:  # 탐욕
            profit_targets = self.config['exit_points']
        else:
            profit_targets = [x * 1.005 for x in self.config['exit_points']]  # 더 높은 가격에 매도
        
        position_sizes = self.config['position_sizes']
        
        for i, (target, ratio) in enumerate(zip(profit_targets, position_sizes)):
            price = current_price * target
            
            if self.strategy_type == TradingStrategy.SCALPING:
                description = f"단타 매도 {i+1}차 (익절 {(target-1)*100:.1f}%)"
            elif self.strategy_type == TradingStrategy.SWING:
                description = f"스윙 매도 {i+1}차 (익절 {(target-1)*100:.1f}%)"
            else:
                description = f"{i+1}차 매도 포인트 (보유량의 {ratio*100}%)"
            
            levels.append(InvestmentLevel(price, ratio, description))
        
        return levels
    
    def get_strategy_recommendation(self, 
                                  current_price: float,
                                  fg_index: float,
                                  rsi: float,
                                  volatility: float,
                                  trend_strength: str,
                                  ma_data: dict,
                                  orderbook_analysis: dict = None,
                                  pattern_analysis: dict = None,
                                  total_assets: float = 10000000
                                  ) -> dict:
        """종합 투자 전략 추천"""
        try:
            # 리스크 비율 계산
            risk_ratio = self.calculate_risk_ratio(fg_index, rsi, volatility)
            
            # 호가 분석 기반 신뢰도 조정
            confidence_score = 0.7  # 기본 신뢰도
            if orderbook_analysis:
                bid_ratio = orderbook_analysis.get('bid_ask_ratio', 1.0)
                bid_concentration = orderbook_analysis.get('bid_concentration', 0.0)
                
                # 매수세가 강하고 집중도가 높은 경우
                if bid_ratio > 1.2 and bid_concentration > 0.5:
                    confidence_score *= 1.2
                    risk_ratio *= 1.1
                # 매도세가 강하고 집중도가 높은 경우
                elif bid_ratio < 0.8 and bid_concentration > 0.5:
                    confidence_score *= 0.8
                    risk_ratio *= 0.9
            
            # 차트 패턴 기반 전략 조정
            if pattern_analysis and pattern_analysis.get('patterns'):
                for pattern in pattern_analysis['patterns']:
                    pattern_type = pattern.get('pattern_type')
                    reliability = pattern.get('reliability', 'medium')
                    
                    # 신뢰도 조정
                    if reliability == 'high':
                        confidence_score *= 1.2
                    elif reliability == 'low':
                        confidence_score *= 0.8
                    
                    # 매수/매도 전략 조정
                    if pattern_type == 'bullish':
                        risk_ratio *= 1.1
                        holding_period = min(self.config['holding_period'].split('-'))  # 짧은 보유기간
                    elif pattern_type == 'bearish':
                        risk_ratio *= 0.9
                        holding_period = max(self.config['holding_period'].split('-'))  # 긴 보유기간
            
            # 투자 금액 계산
            base_amount = self.calculate_investment_amount(total_assets, risk_ratio)
            
            # 시장 상황별 투자 금액 조정
            if fg_index <= 20 and rsi <= 30:  # 매우 강한 매수 신호
                investment_amount = base_amount * 1.2  # 20% 증액
            elif fg_index >= 80 or rsi >= 70:  # 매우 강한 매도 신호
                investment_amount = base_amount * 0.8  # 20% 감액
            else:
                investment_amount = base_amount
            
            # 최대 투자 한도 체크
            investment_amount = min(investment_amount, total_assets * self.config['risk_ratio'])
            
            # 매수/매도 레벨 계산
            entry_levels = self.calculate_entry_levels(
                current_price=current_price,
                orderbook_analysis=orderbook_analysis,
                technical_indicators={
                    'rsi': rsi,
                    'macd': ma_data.get('macd', {}),
                    'moving_averages': ma_data
                },
                patterns=pattern_analysis.get('patterns', []),
                support_resistance=pattern_analysis.get('support_resistance', {})
            )
            
            exit_levels = self.calculate_exit_levels(
                current_price=current_price,
                fg_index=fg_index,
                resistance_levels=pattern_analysis.get('support_resistance', {}).get('resistance', []) if pattern_analysis else []
            )
            
            # 손절가 설정 (패턴 분석 반영)
            lowest_entry = min(level['price'] for level in entry_levels)
            ma_support = min(ma_data.get('MA5', current_price),
                            ma_data.get('MA20', current_price))
            pattern_support = min(pattern_analysis.get('support_resistance', {}).get('support', [current_price * 0.95])) if pattern_analysis else current_price * 0.95
            
            stop_loss = max(
                lowest_entry * 0.97,  # 최저 매수가의 97%
                ma_support * 0.98,    # 주요 이평선의 98%
                pattern_support * 0.99 # 지지선의 99%
            )
            
            return {
                'entry_levels': [
                    {
                        'price': level['price'],
                        'ratio': level['ratio'],
                        'description': level['description'],
                        'reasons': level.get('reasons', ["기본 분할 매수 전략"])
                    } for level in entry_levels
                ],
                'exit_levels': [
                    {
                        'price': level['price'] if isinstance(level, dict) else level.price,
                        'ratio': level['ratio'] if isinstance(level, dict) else level.ratio,
                        'description': level['description'] if isinstance(level, dict) else level.description,
                        'reasons': level.get('reasons', ["기본 청산 전략"]) if isinstance(level, dict) else ["기본 청산 전략"]
                    } for level in exit_levels
                ],
                'stop_loss': stop_loss,
                'risk_ratio': risk_ratio,
                'investment_amount': investment_amount,
                'holding_period': self.config['holding_period'],
                'strategy_type': self.strategy_type.value,
                'confidence_score': min(confidence_score, 1.0)
            }
            
        except Exception as e:
            logger.error(f"전략 추천 생성 실패: {str(e)}")
            # 기본값 반환
            return {
                'entry_levels': [
                    {
                        'price': current_price * 0.99,
                        'ratio': 0.4,
                        'description': "1차 진입 (현재가 대비 -1%)",
                        'reasons': ["기본 분할 매수 전략"]
                    },
                    {
                        'price': current_price * 0.97,
                        'ratio': 0.3,
                        'description': "2차 진입 (현재가 대비 -3%)",
                        'reasons': ["기본 분할 매수 전략"]
                    },
                    {
                        'price': current_price * 0.95,
                        'ratio': 0.3,
                        'description': "3차 진입 (현재가 대비 -5%)",
                        'reasons': ["기본 분할 매수 전략"]
                    }
                ],
                'exit_levels': [
                    {
                        'price': current_price * 1.02,
                        'ratio': 0.3,
                        'description': "1차 청산 (현재가 대비 +2%)",
                        'reasons': ["기본 청산 전략"]
                    },
                    {
                        'price': current_price * 1.05,
                        'ratio': 0.4,
                        'description': "2차 청산 (현재가 대비 +5%)",
                        'reasons': ["기본 청산 전략"]
                    },
                    {
                        'price': current_price * 1.08,
                        'ratio': 0.3,
                        'description': "3차 청산 (현재가 대비 +8%)",
                        'reasons': ["기본 청산 전략"]
                    }
                ],
                'stop_loss': current_price * 0.95,
                'risk_ratio': 0.02,
                'investment_amount': total_assets * 0.1,
                'holding_period': "1-3일",
                'strategy_type': self.strategy_type.value,
                'confidence_score': 0.5
            }

    def recommend_strategy_type(self, fg_index: float, rsi: float, 
                              volatility: float, trend_strength: str) -> TradingStrategy:
        """시장 상황에 따른 최적 전략 추천"""
        
        # 변동성이 매우 높은 경우 (20% 이상)
        if volatility > 20:
            if fg_index <= 30 or fg_index >= 70:  # 극단적 상황
                return TradingStrategy.SCALPING  # 변동성이 높고 극단적일 때는 스캘핑
            else:
                return TradingStrategy.SWING  # 변동성이 높을 때는 스윙
        
        # 추세가 강한 경우
        if trend_strength == "강함":
            if 40 <= fg_index <= 60:  # 중립적 상황
                return TradingStrategy.POSITION  # 강한 추세 + 중립적 = 포지션
        
        # RSI 기반 추천
        if rsi <= 30 or rsi >= 70:  # 과매수/과매도
            if volatility > 15:
                return TradingStrategy.SWING
            else:
                return TradingStrategy.POSITION
        
        # 공포탐욕지수 기반 추천
        if fg_index <= 20 or fg_index >= 80:  # 극단적 공포/탐욕
            return TradingStrategy.SCALPING if volatility > 15 else TradingStrategy.SWING
        elif fg_index <= 40 or fg_index >= 60:  # 공포/탐욕
            return TradingStrategy.POSITION
        
        # 기본값: 안정적인 시장
        return TradingStrategy.POSITION

    def get_strategy_description(self, fg_index: float, rsi: float, 
                               volatility: float, trend_strength: str) -> str:
        """전략 추천 이유 설명"""
        recommended = self.recommend_strategy_type(fg_index, rsi, volatility, trend_strength)
        
        reasons = []
        if volatility > 20:
            reasons.append(f"높은 변동성 ({volatility:.1f}%)")
        elif volatility < 10:
            reasons.append(f"낮은 변동성 ({volatility:.1f}%)")
        
        if fg_index <= 20:
            reasons.append("극단적 공포 상태")
        elif fg_index >= 80:
            reasons.append("극단적 탐욕 상태")
        
        if rsi <= 30:
            reasons.append("과매도 구간")
        elif rsi >= 70:
            reasons.append("과매수 구간")
        
        if trend_strength == "강함":
            reasons.append("강한 추세")
        
        strategy_descriptions = {
            TradingStrategy.SCALPING: "빠른 진입/청산으로 변동성 활용",
            TradingStrategy.SWING: "추세를 활용한 중기 포지션",
            TradingStrategy.POSITION: "안정적인 장기 포지션"
        }
        
        return f"""
💡 추천 전략: {recommended.value}
- 추천 이유: {', '.join(reasons)}
- 전략 특징: {strategy_descriptions[recommended]}
- 현재 선택: {self.strategy_type.value}
    """

def format_strategy_message(strategy: dict, total_assets: float) -> str:
    """전략 메시지 포맷팅"""
    message = [
        f"💡 투자 전략 추천 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"\n전략 유형: {strategy['strategy_type']} (신뢰도: {strategy['confidence_score']*100:.1f}%)",
        f"리스크 비율: {strategy['risk_ratio']*100:.1f}%",
        f"추천 투자 금액: {strategy['investment_amount']:,.0f}원 (총자산의 {strategy['investment_amount']/total_assets*100:.1f}%)",
        f"추천 보유 기간: {strategy['holding_period']}",
        "\n매수 전략:",
    ]
    
    for level in strategy['entry_levels']:
        message.append(f"- {level['description']}")
        message.append(f"  가격: {level['price']:,.0f}원 (투자금액: {level['ratio'] * strategy['investment_amount']:,.0f}원)")
    
    message.append("\n매도 전략:")
    for level in strategy['exit_levels']:
        message.append(f"- {level['description']}")
        message.append(f"  가격: {level['price']:,.0f}원")
    
    message.append(f"\n손절가: {strategy['stop_loss']:,.0f}원")
    
    message.append("\n⚠️ 주의사항:")
    message.append("- 투자는 본인의 판단과 책임하에 진행하세요.")
    message.append("- 시장 상황에 따라 전략을 유연하게 조정하세요.")
    message.append("- 설정된 손절가를 반드시 준수하세요.")
    
    return "\n".join(message) 